import cv2
from abc import ABC, abstractmethod
import json
import time
from pathlib import Path

import numpy as np


class VideoSource(ABC):
    @abstractmethod
    def get_frame(self):
        """Return (frame_bgr, timestamp) or (None, None) if end"""
        ...

    @abstractmethod
    def release(self): ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @abstractmethod
    def get_camera_intrinsics(self) -> np.ndarray: ...

    @abstractmethod
    def get_frame_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_gravity_direction(self) -> np.ndarray: ...


class RealSenseSource(VideoSource):

    def __init__(self, width: int = 848, height: int = 480, fps: float = 30):
        import pyrealsense2 as rs

        self.rs = rs
        self.width = int(width)
        self.height = int(height)
        self._fps = float(fps)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, int(self._fps)
        )

        # Enable IMU for gravity calibration
        config.enable_stream(rs.stream.accel)

        profile = self.pipeline.start(config)

        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.cam_intrinsics = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )[None, ...]

        print("RealSense camera intrinsics:")
        print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        print(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

        # Calibrate gravity direction (once, assuming camera is static)
        print("Calibrating gravity direction (keep camera steady)...")
        accel_samples = []
        num_samples = 100

        for _ in range(num_samples):
            frames = self.pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame and len(accel_samples) < num_samples:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                accel_samples.append([accel_data.x, accel_data.y, accel_data.z])

        if len(accel_samples) < num_samples:
            raise RuntimeError(
                f"Failed to collect enough IMU samples: {len(accel_samples)}/{num_samples}"
            )

        accel_array = np.array(accel_samples, dtype=np.float64)
        gravity_avg = -np.mean(accel_array, axis=0)
        gravity_norm = np.linalg.norm(gravity_avg)

        if gravity_norm < 1e-6:
            raise RuntimeError("Gravity magnitude is near zero - invalid IMU data")

        self.gravity_direction = gravity_avg / gravity_norm
        print(
            f"  Gravity direction: [{self.gravity_direction[0]:.3f}, {self.gravity_direction[1]:.3f}, {self.gravity_direction[2]:.3f}]"
        )
        print(f"  Magnitude: {gravity_norm:.2f} m/s^2")

    def get_frame(self) -> tuple[np.ndarray, float]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None
        frame = np.asanyarray(color_frame.get_data())
        return frame, time.time()

    def release(self):
        self.pipeline.stop()
        self.pipeline = None

    @property
    def fps(self) -> float:
        return self._fps

    def get_camera_intrinsics(self) -> np.ndarray:
        return self.cam_intrinsics

    def get_frame_size(self) -> tuple[int, int]:
        return self.width, self.height

    def get_gravity_direction(self) -> np.ndarray:
        return self.gravity_direction


class VideoFileSource(VideoSource):
    def __init__(self, video_path, intrinsics_path, loop=False):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.loop = bool(loop)
        self.video_path = video_path
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_time = 1.0 / self._fps
        self.start_time = None
        self.frame_count = 0

        print(f"Video: {video_path}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self._fps:.1f}")
        print(f"  Total frames: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        intrinsics_path = Path(intrinsics_path)
        if not intrinsics_path.exists():
            raise RuntimeError(f"Intrinsics JSON not found: {intrinsics_path}")

        with open(intrinsics_path, "r") as f:
            intrinsics_data = json.load(f)

        cam_matrix = np.array(intrinsics_data["camera_matrix"], dtype=np.float32)
        self.cam_intrinsics = cam_matrix[None, ...]
        print(f"  Loaded camera intrinsics from: {intrinsics_path}")
        if "fx" in intrinsics_data:
            print(f"    fx={intrinsics_data['fx']:.2f}, fy={intrinsics_data['fy']:.2f}")
            print(f"    cx={intrinsics_data['cx']:.2f}, cy={intrinsics_data['cy']:.2f}")

        gravity = np.array(intrinsics_data["gravity"], dtype=np.float64)
        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm < 1e-6:
            raise RuntimeError("Gravity magnitude is near zero - invalid data")

        self.gravity_direction = gravity / gravity_norm
        print(
            f"  Gravity direction: [{self.gravity_direction[0]:.3f}, {self.gravity_direction[1]:.3f}, {self.gravity_direction[2]:.3f}]"
        )

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None, None
                self.start_time = None
                self.frame_count = 0
            else:
                return None, None

        if self.start_time is None:
            self.start_time = time.time()

        timestamp = self.start_time + self.frame_count * self.frame_time
        self.frame_count += 1
        return frame, timestamp

    def release(self):
        self.cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    def get_camera_intrinsics(self):
        return self.cam_intrinsics

    def get_frame_size(self):
        return self.width, self.height

    def get_gravity_direction(self):
        return self.gravity_direction


def create_video_source(source_type, **kwargs):
    if source_type == "camera":
        return RealSenseSource(**kwargs)
    if source_type == "video":
        return VideoFileSource(**kwargs)
    raise ValueError(f"Unknown source type: {source_type}")
