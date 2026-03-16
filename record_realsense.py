#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(
        description="Record RGB video from RealSense D435i/D455 with IMU gravity calibration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/records/<timestamp>)",
    )
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--imu-samples",
        type=int,
        default=100,
        help="Number of IMU samples for gravity calibration (default: 100)",
    )
    args = parser.parse_args()

    if args.imu_samples < 10:
        parser.error("--imu-samples must be at least 10")

    import time as _time
    output_dir = Path(args.output_dir) if args.output_dir else Path("output/records") / _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "recording.mp4"
    intrinsics_path = output_dir / "recording.json"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(video_path), fourcc, args.fps, (args.width, args.height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    print(f"Recording: {video_path} ({args.width}x{args.height} @ {args.fps}fps)")
    print(
        f"Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
        f"cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}"
    )
    print(f"Calibrating gravity ({args.imu_samples} samples)... ", end="", flush=True)

    accel_samples = []
    while len(accel_samples) < args.imu_samples:
        frames = pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            accel_samples.append([accel_data.x, accel_data.y, accel_data.z])

    accel_array = np.array(accel_samples, dtype=np.float64)
    gravity_avg = -np.mean(accel_array, axis=0)
    gravity_norm = np.linalg.norm(gravity_avg)
    if gravity_norm < 1e-6:
        raise RuntimeError("Failed to calibrate gravity")
    gravity_unit = (gravity_avg / gravity_norm).tolist()

    print(f"done ({gravity_norm:.2f} m/s^2)")
    print(
        f"Gravity: [{gravity_unit[0]:+.3f}, {gravity_unit[1]:+.3f}, {gravity_unit[2]:+.3f}]"
    )

    intrinsics_data = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "width": args.width,
        "height": args.height,
        "camera_matrix": [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        "gravity": gravity_unit,
    }
    with open(intrinsics_path, "w") as f:
        json.dump(intrinsics_data, f, indent=2)

    print("Recording... (Ctrl+C to stop)")
    frame_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            out.write(np.asanyarray(color_frame.get_data()))
            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"\r{frame_count} frames ({frame_count / args.fps:.1f}s)",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        pipeline.stop()
        out.release()
        print(f"\nSaved: {video_path} ({frame_count} frames, {frame_count / args.fps:.1f}s)")
        print(f"JSON:  {intrinsics_path}")


if __name__ == "__main__":
    main()
