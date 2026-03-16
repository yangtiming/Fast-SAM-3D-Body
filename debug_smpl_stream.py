import argparse
import time
from collections import deque

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from mocap.realtime.constants import ZMQ_HEADER_SIZE
from mocap.utils.smpl_render_utils import render_smpl_records_video


SMPL_JOINTS_PRE_ROT_MAT = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
SMPL_BASE_RECOVERY_QUAT_XYZW = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
MESH_EXTRA_ROT_X_DEG = 180.0

RENDER_WIDTH = 640
RENDER_HEIGHT = 960


class PoseSubscriber:
    def __init__(self, zmq_addr="tcp://localhost:5556", topic="pose"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(zmq_addr)
        self.topic = topic
        self.topic_bytes = topic.encode("utf-8")
        self.topic_len = len(self.topic_bytes)
        self.sock.subscribe(self.topic_bytes)

    def receive(self):
        msg = self.sock.recv()
        offset = self.topic_len + ZMQ_HEADER_SIZE

        required = 8 + 32 + 576 + 504
        if len(msg) < offset + required:
            raise RuntimeError(
                f"Pose message too short: got={len(msg)} bytes, need>={offset + required}"
            )

        frame_idx = int(np.frombuffer(msg[offset : offset + 8], dtype=np.int64)[0])
        offset += 8
        body_quat = np.frombuffer(msg[offset : offset + 32], dtype=np.float64).reshape(
            4
        )
        offset += 32
        smpl_joints = np.frombuffer(
            msg[offset : offset + 576], dtype=np.float64
        ).reshape(24, 3)
        offset += 576
        smpl_pose = np.frombuffer(msg[offset : offset + 504], dtype=np.float64).reshape(
            21, 3
        )

        return {
            "frame_index": frame_idx,
            "body_quat": body_quat,
            "smpl_joints": smpl_joints,
            "smpl_pose": smpl_pose,
            "recv_time": time.perf_counter(),
        }

    def close(self):
        self.sock.close(0)
        self.ctx.term()


def _body_quat_wxyz_to_xyzw(body_quat):
    quat = np.array(body_quat, dtype=np.float64, copy=True).reshape(4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def _transform_body_quat_for_render(quat_xyzw, frame_rot, base_recovery_rot):
    return (
        (frame_rot * Rotation.from_quat(quat_xyzw) * base_recovery_rot)
        .as_quat()
        .astype(np.float64)
    )


def prepare_render_records(records):
    converted = []
    joints_rot = Rotation.from_matrix(SMPL_JOINTS_PRE_ROT_MAT)
    base_recovery_rot = Rotation.from_quat(SMPL_BASE_RECOVERY_QUAT_XYZW)
    mesh_extra_rot = Rotation.from_euler("x", MESH_EXTRA_ROT_X_DEG, degrees=True)

    for record in records:
        quat_xyzw = _body_quat_wxyz_to_xyzw(record["body_quat"])
        quat_transformed = _transform_body_quat_for_render(
            quat_xyzw, joints_rot, base_recovery_rot
        )
        quat_transformed = (
            (mesh_extra_rot * Rotation.from_quat(quat_transformed))
            .as_quat()
            .astype(np.float64)
        )
        converted_record = dict(record)
        converted_record["body_quat"] = quat_transformed
        converted.append(converted_record)

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Collect ZMQ pose frames and render SMPL mesh debug video"
    )
    parser.add_argument(
        "--addr", type=str, default="tcp://localhost:5556", help="ZMQ address"
    )
    parser.add_argument("--topic", type=str, default="pose", help="ZMQ topic")
    parser.add_argument(
        "--fps-window", type=int, default=200, help="Window size for avg FPS"
    )
    parser.add_argument(
        "--num-frames", type=int, default=500, help="Number of frames to collect"
    )
    parser.add_argument(
        "--render-output",
        type=str,
        default="output/smpl_debug.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--render-fps", type=float, default=50.0, help="Render video FPS"
    )
    parser.add_argument(
        "--smpl-model-path", type=str, required=True, help="SMPL model file path"
    )
    parser.add_argument(
        "--show-joints", action="store_true", default=False, help="Overlay skeleton joints"
    )
    args = parser.parse_args()

    if args.fps_window <= 1:
        parser.error("--fps-window must be > 1")
    if args.num_frames <= 0:
        parser.error("--num-frames must be > 0")
    if args.render_fps <= 0:
        parser.error("--render-fps must be > 0")

    subscriber = PoseSubscriber(zmq_addr=args.addr, topic=args.topic)
    print(f"Collecting {args.num_frames} frames on topic '{args.topic}' ...")
    print("Press Ctrl+C to stop early")

    recv_intervals = deque(maxlen=args.fps_window)
    last_recv_time = None
    last_frame_idx = None
    total_messages = 0
    missing_frames = 0
    start_time = time.perf_counter()
    records = []

    try:
        while total_messages < args.num_frames:
            data = subscriber.receive()
            now = data["recv_time"]

            if last_recv_time is not None:
                recv_intervals.append(now - last_recv_time)
            last_recv_time = now

            frame_idx = data["frame_index"]
            if last_frame_idx is not None and frame_idx > last_frame_idx + 1:
                missing_frames += frame_idx - last_frame_idx - 1
            last_frame_idx = frame_idx

            total_messages += 1
            records.append(data)

            if total_messages % 25 == 0:
                elapsed = now - start_time
                inst_fps = 1.0 / recv_intervals[-1] if recv_intervals else 0.0
                avg_fps = (
                    1.0 / (sum(recv_intervals) / len(recv_intervals))
                    if recv_intervals
                    else 0.0
                )
                print(
                    f"topic={args.topic} msg={total_messages:6d}/{args.num_frames} "
                    f"frame={frame_idx:6d} inst_fps={inst_fps:6.2f} avg_fps={avg_fps:6.2f} "
                    f"missing={missing_frames:5d} elapsed={elapsed:7.2f}s"
                )

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        total_elapsed = time.perf_counter() - start_time
        mean_fps = total_messages / total_elapsed if total_elapsed > 0 else 0.0
        print(
            f"Summary: topic={args.topic}, msgs={total_messages}, elapsed={total_elapsed:.2f}s, "
            f"mean_fps={mean_fps:.2f}, missing_frames={missing_frames}"
        )

        if records:
            render_records = prepare_render_records(records)
            print(f"Rendering {len(render_records)} frames to {args.render_output} ...")
            render_smpl_records_video(
                render_records,
                output_path=args.render_output,
                smpl_model_path=args.smpl_model_path,
                fps=args.render_fps,
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                show_joints=args.show_joints,
            )
            print(f"Saved: {args.render_output}")

        subscriber.close()


if __name__ == "__main__":
    main()
