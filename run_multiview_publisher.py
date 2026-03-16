import argparse
import json
import os
import sys
import threading
import queue
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation

from mocap.core.gravity_alignment import build_camera_to_world_rotation
from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
from mocap.core.setup_estimator import build_default_estimator
from mocap.realtime.interpolator import PoseInterpolator
from mocap.realtime.publisher import ZMQPublisher
from mocap.utils.pose_protocol import prepare_publish_pose
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR

FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True
YOLO_MODEL_PATH = str(REPO_ROOT / "checkpoints" / "yolo" / "yolo11m-pose.engine")

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)


def pick_main_bbox(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    return boxes[idx : idx + 1]


class MultiCameraSource:
    def get_frames(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    def get_frame_size(self):
        raise NotImplementedError

    def get_camera_intrinsics(self):
        raise NotImplementedError

    def get_gravity_directions(self):
        raise NotImplementedError

    def get_camera_names(self):
        raise NotImplementedError


class MultiVideoFileSource(MultiCameraSource):
    def __init__(self, video_paths, intrinsics_paths, loop=False):
        if len(video_paths) < 1:
            raise RuntimeError("At least one video path is required")
        if len(video_paths) != len(intrinsics_paths):
            raise RuntimeError("--videos and --intrinsics must have the same length")

        self.video_paths = [str(Path(p)) for p in video_paths]
        self.intrinsics_paths = [str(Path(p)) for p in intrinsics_paths]
        self.loop = bool(loop)
        self.caps = []
        self.cam_intrinsics = []
        self.gravity_directions = []
        self.camera_names = []
        self.frame_count = 0
        self.start_time = None
        self.frame_time = None
        self.width = None
        self.height = None
        self._fps = None

        for idx, (video_path, intr_path) in enumerate(
            zip(self.video_paths, self.intrinsics_paths)
        ):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self.caps.append(cap)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise RuntimeError(
                    f"Invalid FPS reported by video: {video_path} (fps={fps})"
                )

            if self.width is None:
                self.width = width
                self.height = height
                self._fps = fps
                self.frame_time = 1.0 / fps
            else:
                if width != self.width or height != self.height:
                    raise RuntimeError("All input videos must have the same resolution")
                if abs(fps - self._fps) > 1e-3:
                    raise RuntimeError("All input videos must have the same FPS")

            with open(intr_path, "r") as f:
                intr_data = json.load(f)
            if "camera_matrix" not in intr_data or "gravity" not in intr_data:
                raise RuntimeError(
                    f"Intrinsics JSON must contain camera_matrix and gravity: {intr_path}"
                )

            cam_matrix = np.asarray(intr_data["camera_matrix"], dtype=np.float32)
            gravity = np.asarray(intr_data["gravity"], dtype=np.float64)
            gravity_norm = np.linalg.norm(gravity)
            if gravity_norm <= 1e-12:
                raise RuntimeError(f"Invalid gravity vector in intrinsics: {intr_path}")
            gravity /= gravity_norm

            self.cam_intrinsics.append(cam_matrix[None, ...])
            self.gravity_directions.append(gravity)
            self.camera_names.append(Path(video_path).stem or f"cam{idx}")

        logger.info(
            f"Loaded {len(self.caps)} video streams at {self.width}x{self.height} {self._fps:.1f}fps"
        )

    def get_frames(self):
        frames = []
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                if not self.loop:
                    return None, None
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    return None, None
                self.start_time = None
                self.frame_count = 0
            frames.append(frame)

        if self.start_time is None:
            self.start_time = time.time()
        ts = self.start_time + self.frame_count * self.frame_time
        self.frame_count += 1
        return frames, ts

    def release(self):
        for cap in self.caps:
            cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    def get_frame_size(self):
        return self.width, self.height

    def get_camera_intrinsics(self):
        return self.cam_intrinsics

    def get_gravity_directions(self):
        return self.gravity_directions

    def get_camera_names(self):
        return self.camera_names


class MultiRealSenseSource(MultiCameraSource):
    def __init__(self, width=848, height=480, fps=30, serials=None, imu_samples=100):
        import pyrealsense2 as rs

        self.rs = rs
        self.width = int(width)
        self.height = int(height)
        self._fps = float(fps)
        self.imu_samples = int(imu_samples)
        self.ctx = rs.context()

        wanted_serials = [s.strip() for s in serials if s.strip()] if serials else None

        devices = list(self.ctx.query_devices())
        if not devices:
            raise RuntimeError("No RealSense devices found")

        dev_map = {dev.get_info(rs.camera_info.serial_number): dev for dev in devices}
        if wanted_serials is None:
            raise RuntimeError("serials must be provided explicitly for camera source")
        else:
            missing = [sn for sn in wanted_serials if sn not in dev_map]
            if missing:
                raise RuntimeError(f"Requested RealSense serials not found: {missing}")
            serial_list = wanted_serials

        self.serials = serial_list
        self.pipelines = []
        self.cam_intrinsics = []
        self.gravity_directions = []

        logger.info(f"Starting {len(self.serials)} RealSense streams: {self.serials}")
        for serial in self.serials:
            pipe = rs.pipeline(self.ctx)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, int(self._fps)
            )
            config.enable_stream(rs.stream.accel)
            profile = pipe.start(config)
            self.pipelines.append(pipe)

            color_stream = profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            cam_matrix = np.array(
                [
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self.cam_intrinsics.append(cam_matrix[None, ...])

        logger.info(
            f"Calibrating gravity for {len(self.serials)} cameras ({self.imu_samples} samples)"
        )
        accel_samples = [[] for _ in self.pipelines]
        while any(len(samples) < self.imu_samples for samples in accel_samples):
            for i, pipe in enumerate(self.pipelines):
                if len(accel_samples[i]) >= self.imu_samples:
                    continue
                frames = pipe.wait_for_frames()
                accel_frame = frames.first_or_default(rs.stream.accel)
                if accel_frame:
                    data = accel_frame.as_motion_frame().get_motion_data()
                    accel_samples[i].append([data.x, data.y, data.z])

        for serial, samples in zip(self.serials, accel_samples):
            accel = np.asarray(samples, dtype=np.float64)
            gravity = -np.mean(accel, axis=0)
            gravity_norm = np.linalg.norm(gravity)
            if gravity_norm <= 1e-12:
                raise RuntimeError(f"Invalid gravity calibration for camera: {serial}")
            gravity /= gravity_norm
            self.gravity_directions.append(gravity)
            logger.info(
                f"Gravity {serial}: [{gravity[0]:+.3f}, {gravity[1]:+.3f}, {gravity[2]:+.3f}]"
            )

    def get_frames(self):
        frames = []
        for pipe in self.pipelines:
            fs = pipe.wait_for_frames()
            color_frame = fs.get_color_frame()
            if not color_frame:
                raise RuntimeError("RealSense source returned an empty color frame")
            frames.append(np.asanyarray(color_frame.get_data()))
        return frames, time.time()

    def release(self):
        for pipe in self.pipelines:
            pipe.stop()

    @property
    def fps(self) -> float:
        return self._fps

    def get_frame_size(self):
        return self.width, self.height

    def get_camera_intrinsics(self):
        return self.cam_intrinsics

    def get_gravity_directions(self):
        return self.gravity_directions

    def get_camera_names(self):
        return self.serials


class RealtimeMultiViewPublisher:
    def __init__(
        self,
        source,
        main_camera,
        publish_hz,
        interpolate_lag_ms,
        smpl_model_path,
        nn_model_dir,
        mhr2smpl_mapping_path,
        mhr_mesh_path=None,
        smoother_dir=None,
        addr="tcp://*:5556",
        image_size=512,
        yolo_model_path=YOLO_MODEL_PATH,
        min_person_confidence=0.75,
        record=False,
        record_dir="output/records",
        device=None,
    ):
        self.source = source
        self.main_camera = int(main_camera)
        self.publish_hz = float(publish_hz)
        self.publish_dt = 1.0 / self.publish_hz
        self.interpolate_lag_s = float(interpolate_lag_ms) / 1000.0
        self.min_person_confidence = float(min_person_confidence)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.camera_names = self.source.get_camera_names()
        self.cam_intrinsics = self.source.get_camera_intrinsics()
        self.gravity_directions = self.source.get_gravity_directions()
        if len(self.camera_names) != 2:
            raise RuntimeError(
                f"Expected exactly 2 cameras, got {len(self.camera_names)}"
            )
        if not (0 <= self.main_camera < len(self.camera_names)):
            raise RuntimeError(
                f"--main-camera index out of range: {self.main_camera}, num_cameras={len(self.camera_names)}"
            )

        logger.info(
            f"Using main camera index={self.main_camera} name={self.camera_names[self.main_camera]}"
        )

        self.R_world_cam = build_camera_to_world_rotation(
            self.gravity_directions[self.main_camera]
        )
        R_zup_adjustment = np.array(
            [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64
        )
        self.R_world_cam = R_zup_adjustment @ self.R_world_cam

        self.record = record
        self.record_dir = record_dir
        if self.record:
            os.makedirs(self.record_dir, exist_ok=True)
            session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join(self.record_dir, session_id)
            os.makedirs(self.session_dir, exist_ok=True)
            self.video_out_paths = [
                os.path.join(self.session_dir, f"raw_video_{i}.mp4")
                for i in range(len(self.camera_names))
            ]
            self.smpl_out_path = os.path.join(self.session_dir, "smpl_data.npz")
            logger.info(f"Recording enabled. Saving to: {self.session_dir}")

            frame_size = self.source.get_frame_size()
            for i, (gravity, cam_matrix) in enumerate(
                zip(self.gravity_directions, self.cam_intrinsics or [None] * len(self.gravity_directions))
            ):
                intr_data = {"gravity": gravity.tolist()}
                if cam_matrix is not None:
                    intr_data["camera_matrix"] = cam_matrix[0].tolist()
                if frame_size is not None:
                    intr_data["width"] = frame_size[0]
                    intr_data["height"] = frame_size[1]
                json_path = os.path.join(self.session_dir, f"raw_video_{i}.json")
                with open(json_path, "w") as f:
                    json.dump(intr_data, f, indent=2)

            self.video_queue = queue.Queue(maxsize=300)
            self.smpl_queue = queue.Queue(maxsize=300)
            self.smpl_data_list = []

        logger.info("Loading SAM 3D estimator...")
        self.estimator = build_default_estimator(
            image_size=image_size,
            yolo_model_path=yolo_model_path,
            fov_model_size=FOV_MODEL_SIZE,
            fov_resolution_level=FOV_RESOLUTION_LEVEL,
            fov_fixed_size=FOV_FIXED_SIZE,
            fov_fast_mode=FOV_FAST_MODE,
        )

        self._warmup()

        self.fusion_runner = MultiViewFusionRunner(
            smpl_model_path=smpl_model_path,
            model_dir=nn_model_dir,
            mapping_path=mhr2smpl_mapping_path,
            mhr_mesh_path=mhr_mesh_path,
            device=self.device,
            smoother_dir=smoother_dir,
        )

        self.interpolator = PoseInterpolator()
        self.publisher = ZMQPublisher(addr)
        self.running = False
        self.video_ended = False
        self.capture_thread = None
        self.worker_thread = None
        self.publish_thread = None
        self.recording_thread = None
        self._latest_frames = None
        self._latest_frames_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._pose_clock_lock = threading.Lock()
        self._latest_pose_source_ts = None
        self._latest_pose_perf_ts = None
        self._last_warn_reason = None
        self._last_live_log_perf = time.perf_counter()
        self._thread_error = None
        self._interp_not_ready_warned = False
        self._live_log_interval_s = 2.0
        self._capture_wall_base = None
        self._capture_ts_base = None
        self.stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "processed_count": 0,
            "published_count": 0,
            "publish_intervals": deque(maxlen=500),
            "worker_total_s": 0.0,
            "worker_total_ms": deque(maxlen=200),
            "stage1_total_s": 0.0,
            "stage1_total_ms": deque(maxlen=200),
            "detect_cam0_ms": deque(maxlen=200),
            "detect_cam1_ms": deque(maxlen=200),
            "stage1_body_ms": deque(maxlen=200),
            "fusion_total_s": 0.0,
            "fusion_total_ms": deque(maxlen=200),
        }
        self._live_prev_stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "processed_count": 0,
            "published_count": 0,
            "worker_total_s": 0.0,
            "stage1_total_s": 0.0,
            "fusion_total_s": 0.0,
        }

    def _warmup(self):
        frame_size = self.source.get_frame_size()
        if frame_size is None:
            raise RuntimeError("video source returned frame_size=None during warmup")
        width, height = frame_size
        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        warmup_bbox = np.array(
            [[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32
        )
        warmup_cam_int = torch.from_numpy(
            np.asarray(self.cam_intrinsics[self.main_camera], dtype=np.float32)
        )
        for _ in range(2):
            _ = self.estimator.process_one_image(
                dummy_img,
                cam_int=warmup_cam_int,
                bboxes=warmup_bbox,
                hand_box_source="body_decoder",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _log_skip(self, reason):
        if reason != self._last_warn_reason:
            logger.warning(reason)
            self._last_warn_reason = reason

    def _maybe_log_live_stats(self, now_perf):
        elapsed = now_perf - self._last_live_log_perf
        if elapsed < self._live_log_interval_s:
            return

        curr = {
            "capture_count": self.stats["capture_count"],
            "dropped_capture_count": self.stats["dropped_capture_count"],
            "processed_count": self.stats["processed_count"],
            "published_count": self.stats["published_count"],
            "worker_total_s": self.stats["worker_total_s"],
            "stage1_total_s": self.stats["stage1_total_s"],
            "fusion_total_s": self.stats["fusion_total_s"],
        }
        prev = self._live_prev_stats

        d_capture = curr["capture_count"] - prev["capture_count"]
        d_dropped = curr["dropped_capture_count"] - prev["dropped_capture_count"]
        d_processed = curr["processed_count"] - prev["processed_count"]
        d_published = curr["published_count"] - prev["published_count"]
        d_worker_total_s = curr["worker_total_s"] - prev["worker_total_s"]
        d_stage1_total_s = curr["stage1_total_s"] - prev["stage1_total_s"]
        d_fusion_total_s = curr["fusion_total_s"] - prev["fusion_total_s"]

        capture_fps = d_capture / elapsed
        process_fps = d_processed / elapsed
        publish_fps = d_published / elapsed
        worker_ms = (
            (d_worker_total_s / d_processed * 1000.0)
            if d_processed > 0
            else float("nan")
        )
        stage1_ms = (
            (d_stage1_total_s / d_processed * 1000.0)
            if d_processed > 0
            else float("nan")
        )
        fusion_ms = (
            (d_fusion_total_s / d_processed * 1000.0)
            if d_processed > 0
            else float("nan")
        )

        def fmt_ms(x):
            return f"{x:.1f}" if np.isfinite(x) else "n/a"

        cam0_ms = (
            np.mean(self.stats["detect_cam0_ms"])
            if self.stats["detect_cam0_ms"]
            else float("nan")
        )
        cam1_ms = (
            np.mean(self.stats["detect_cam1_ms"])
            if self.stats["detect_cam1_ms"]
            else float("nan")
        )
        body_ms = (
            np.mean(self.stats["stage1_body_ms"])
            if self.stats["stage1_body_ms"]
            else float("nan")
        )

        logger.info(
            "Live: "
            f"capture={capture_fps:.1f}fps drop+={d_dropped}, "
            f"infer throughput={process_fps:.1f}fps, "
            f"publish={publish_fps:.1f}Hz, "
            f"worker={fmt_ms(worker_ms)}ms, "
            f"stage1={fmt_ms(stage1_ms)}ms "
            f"(det0={fmt_ms(cam0_ms)}ms, det1={fmt_ms(cam1_ms)}ms, body={fmt_ms(body_ms)}ms), "
            f"fusion={fmt_ms(fusion_ms)}ms"
        )

        self._live_prev_stats = curr
        self._last_live_log_perf = now_perf

    def _log_final_stats(self):
        worker_ms = (
            np.mean(self.stats["worker_total_ms"])
            if self.stats["worker_total_ms"]
            else float("nan")
        )
        stage1_ms = (
            np.mean(self.stats["stage1_total_ms"])
            if self.stats["stage1_total_ms"]
            else float("nan")
        )
        fusion_ms = (
            np.mean(self.stats["fusion_total_ms"])
            if self.stats["fusion_total_ms"]
            else float("nan")
        )
        cam0_ms = (
            np.mean(self.stats["detect_cam0_ms"])
            if self.stats["detect_cam0_ms"]
            else float("nan")
        )
        cam1_ms = (
            np.mean(self.stats["detect_cam1_ms"])
            if self.stats["detect_cam1_ms"]
            else float("nan")
        )
        body_ms = (
            np.mean(self.stats["stage1_body_ms"])
            if self.stats["stage1_body_ms"]
            else float("nan")
        )
        publish_hz = (
            1.0 / np.mean(self.stats["publish_intervals"])
            if self.stats["publish_intervals"]
            else float("nan")
        )

        def fmt_ms(x):
            return f"{x:.1f}" if np.isfinite(x) else "n/a"

        publish_hz_str = f"{publish_hz:.1f}" if np.isfinite(publish_hz) else "n/a"
        logger.info(
            "Final stats: "
            f"captured={self.stats['capture_count']}, "
            f"capture_drop={self.stats['dropped_capture_count']}, "
            f"processed={self.stats['processed_count']}, "
            f"published={self.stats['published_count']}, "
            f"publish={publish_hz_str}Hz, "
            f"worker={fmt_ms(worker_ms)}ms, "
            f"stage1={fmt_ms(stage1_ms)}ms "
            f"(det0={fmt_ms(cam0_ms)}ms, det1={fmt_ms(cam1_ms)}ms, body={fmt_ms(body_ms)}ms), "
            f"fusion={fmt_ms(fusion_ms)}ms"
        )

    @torch.no_grad()
    def _process_two_images_batched(self, img_rgb0, img_rgb1):
        batches = []
        detect_times_s = []

        for cam_idx, (img_rgb, cam_intr) in enumerate(
            zip((img_rgb0, img_rgb1), self.cam_intrinsics)
        ):
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            det_t0 = time.perf_counter()
            if self.estimator.detector is not None:
                det = self.estimator.detector.run_human_detection(
                    img_bgr,
                    det_cat_id=0,
                    bbox_thr=0.5,
                    nms_thr=0.3,
                    default_to_full_image=False,
                )
                boxes = det["boxes"] if isinstance(det, dict) else det
            else:
                raise RuntimeError(
                    "estimator.detector is required for batched multiview inference"
                )
            detect_times_s.append(time.perf_counter() - det_t0)

            boxes = pick_main_bbox(boxes)
            if boxes is None or len(boxes) != 1:
                if cam_idx == self.main_camera:
                    return None, detect_times_s, None
                else:
                    batches.append(None)
                    continue

            batch = prepare_batch(img_rgb, self.estimator.transform, boxes)
            cam_int_t = torch.from_numpy(np.asarray(cam_intr, dtype=np.float32))
            batch["cam_int"] = cam_int_t.to(batch["img"])
            batch["cam_idx"] = cam_idx
            batches.append(batch)

        valid_batches = [b for b in batches if b is not None]
        if not valid_batches:
            return None, detect_times_s, None

        tensor_keys = [
            "img",
            "img_size",
            "ori_img_size",
            "bbox_center",
            "bbox_scale",
            "bbox",
            "affine_trans",
            "mask",
            "mask_score",
            "cam_int",
            "person_valid",
        ]
        merged = {}
        for key in tensor_keys:
            merged[key] = torch.cat([b[key] for b in valid_batches if key in b], dim=0)

        img_ori = []
        for b in valid_batches:
            img_ori.extend(b["img_ori"])
        merged["img_ori"] = img_ori

        model_t0 = time.perf_counter()
        merged = recursive_to(merged, self.device)
        self.estimator.model._initialize_batch(merged)
        pose_output = self.estimator.model.forward_step(merged, decoder_type="body")
        out = recursive_to(recursive_to(pose_output["mhr"], "cpu"), "numpy")
        model_dt = time.perf_counter() - model_t0

        preds = []
        valid_out_idx = 0
        for idx in range(2):
            if batches[idx] is None:
                preds.append(None)
            else:
                preds.append(
                    {
                        "pred_vertices": out["pred_vertices"][valid_out_idx],
                        "pred_cam_t": out["pred_cam_t"][valid_out_idx],
                        "pred_joint_coords": out["pred_joint_coords"][valid_out_idx],
                        "global_rot": out["global_rot"][valid_out_idx],
                    }
                )
                valid_out_idx += 1
        return preds, detect_times_s, model_dt

    def _compute_main_body_quat(self, main_out):
        global_rot = np.asarray(main_out["global_rot"], dtype=np.float64).reshape(3)
        rot = Rotation.from_euler("ZYX", global_rot)
        x180 = Rotation.from_euler("x", 180.0, degrees=True)
        return (x180 * rot).as_quat().astype(np.float64)

    def _prepare_publish_pose(self, body_quat_xyzw, canonical_joints, smpl_pose):
        return prepare_publish_pose(
            body_quat_xyzw,
            canonical_joints,
            smpl_pose,
            self.R_world_cam,
        )

    def _capture_loop(self):
        while self.running:
            frames, frame_ts = self.source.get_frames()
            if frames is None or frame_ts is None:
                self.video_ended = True
                self._frame_event.set()
                break

            if self._capture_wall_base is None:
                self._capture_wall_base = time.perf_counter()
                self._capture_ts_base = frame_ts
            else:
                target_wall = self._capture_wall_base + (
                    frame_ts - self._capture_ts_base
                )
                now_wall = time.perf_counter()
                delay = target_wall - now_wall
                if delay > 0:
                    time.sleep(delay)

            self.stats["capture_count"] += 1
            if self.record:
                try:
                    self.video_queue.put_nowait((frame_ts, frames))
                except queue.Full:
                    logger.warning(
                        "Video recording queue full, dropping frame for recording"
                    )

            with self._latest_frames_lock:
                if self._latest_frames is not None:
                    self.stats["dropped_capture_count"] += 1
                self._latest_frames = (frames, frame_ts)
            self._frame_event.set()

    def _worker_loop(self):
        while self.running:
            self._frame_event.wait(timeout=0.05)
            self._frame_event.clear()

            with self._latest_frames_lock:
                item = self._latest_frames
                self._latest_frames = None

            if item is None:
                if self.video_ended:
                    break
                continue

            frames, frame_ts = item

            t0 = time.perf_counter()
            frame_rgb0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            frame_rgb1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)
            outputs, detect_times_s, body_model_dt = self._process_two_images_batched(
                frame_rgb0, frame_rgb1
            )

            for cam_idx, det_dt in enumerate(detect_times_s):
                self.stats[f"detect_cam{cam_idx}_ms"].append(det_dt * 1000.0)

            if outputs is None:
                self._log_skip(
                    "Skipping publish: batched Stage1 failed to produce person on main camera"
                )
                continue

            self._last_warn_reason = None
            valid_views = []
            for out in outputs:
                if out is None:
                    valid_views.append(None)
                else:
                    valid_views.append(
                        (
                            np.asarray(out["pred_vertices"], dtype=np.float32),
                            np.asarray(out["pred_cam_t"], dtype=np.float32),
                        )
                    )

            if outputs[self.main_camera] is None:
                self._log_skip("Main camera returned None. Skipping frame.")
                continue

            stage1_total_s = sum(detect_times_s) + body_model_dt
            self.stats["stage1_body_ms"].append(body_model_dt * 1000.0)
            fusion_t0 = time.perf_counter()
            body_quat_xyzw = self._compute_main_body_quat(outputs[self.main_camera])
            smpl_pose, canonical_joints, _betas, view_weights = (
                self.fusion_runner.infer(valid_views)
            )
            body_quat, smpl_joints, smpl_pose = self._prepare_publish_pose(
                body_quat_xyzw, canonical_joints, smpl_pose
            )
            fusion_dt = time.perf_counter() - fusion_t0
            self.interpolator.add_pose(frame_ts, body_quat, smpl_joints, smpl_pose)

            if self.record:
                try:
                    self.smpl_queue.put_nowait(
                        (frame_ts, body_quat, smpl_joints, smpl_pose)
                    )
                except queue.Full:
                    logger.warning(
                        "SMPL recording queue full, dropping pose for recording"
                    )

            worker_dt = time.perf_counter() - t0
            self.stats["processed_count"] += 1
            self.stats["worker_total_s"] += worker_dt
            self.stats["worker_total_ms"].append(worker_dt * 1000.0)
            self.stats["stage1_total_s"] += stage1_total_s
            self.stats["stage1_total_ms"].append(stage1_total_s * 1000.0)
            self.stats["fusion_total_s"] += fusion_dt
            self.stats["fusion_total_ms"].append(fusion_dt * 1000.0)

            with self._pose_clock_lock:
                self._latest_pose_source_ts = frame_ts
                self._latest_pose_perf_ts = time.perf_counter()

            now_perf = time.perf_counter()
            self._maybe_log_live_stats(now_perf)

    def _publish_loop(self):
        next_publish = time.perf_counter()
        last_publish_perf = None
        while self.running:
            now_perf = time.perf_counter()
            wait_time = next_publish - now_perf
            if wait_time > 0:
                time.sleep(min(wait_time, 0.0015))
                continue

            with self._pose_clock_lock:
                latest_pose_source_ts = self._latest_pose_source_ts
                latest_pose_perf_ts = self._latest_pose_perf_ts

            if latest_pose_source_ts is None or latest_pose_perf_ts is None:
                next_publish += self.publish_dt
                continue

            source_now_est = latest_pose_source_ts + (now_perf - latest_pose_perf_ts)
            query_ts = source_now_est - self.interpolate_lag_s
            result = self.interpolator.interpolate(query_ts)
            if result is None:
                latest_pose = self.interpolator.get_latest_pose()
                if latest_pose is not None:
                    if not self._interp_not_ready_warned:
                        logger.warning(
                            "Interpolator not ready yet; falling back to latest pose until two timestamps are available."
                        )
                        self._interp_not_ready_warned = True
                    result = latest_pose

            if result is not None:
                self.publisher.publish(*result)
                self.stats["published_count"] += 1
                if last_publish_perf is not None:
                    self.stats["publish_intervals"].append(now_perf - last_publish_perf)
                last_publish_perf = now_perf
                self._interp_not_ready_warned = False

            next_publish += self.publish_dt
            if next_publish < now_perf - self.publish_dt:
                missed = int((now_perf - next_publish) / self.publish_dt) + 1
                next_publish += missed * self.publish_dt

            if self.video_ended and latest_pose_source_ts is not None:
                break

    def _recording_loop(self):
        video_writers = [None for _ in range(len(self.camera_names))]
        fps = self.source.fps

        while self.running:
            try:
                frame_ts, frames = self.video_queue.get(timeout=0.05)
                for idx, frame in enumerate(frames):
                    if video_writers[idx] is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"avc1")
                        video_writers[idx] = cv2.VideoWriter(
                            self.video_out_paths[idx], fourcc, fps, (w, h)
                        )
                    if video_writers[idx] is not None:
                        video_writers[idx].write(frame)
            except queue.Empty:
                pass

            try:
                while True:
                    smpl_ts, body_quat, smpl_joints, smpl_pose = (
                        self.smpl_queue.get_nowait()
                    )
                    self.smpl_data_list.append(
                        {
                            "timestamp": smpl_ts,
                            "body_quat": body_quat,
                            "smpl_joints": smpl_joints,
                            "smpl_pose": smpl_pose,
                        }
                    )
            except queue.Empty:
                pass

        # Drain queues on exit
        if self.record:
            logger.info("Flushing recording queues to disk. Please wait...")
            while not self.video_queue.empty():
                try:
                    frame_ts, frames = self.video_queue.get_nowait()
                    for idx, frame in enumerate(frames):
                        if video_writers[idx] is None:
                            h, w = frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"avc1")
                            video_writers[idx] = cv2.VideoWriter(
                                self.video_out_paths[idx], fourcc, fps, (w, h)
                            )
                        if video_writers[idx] is not None:
                            video_writers[idx].write(frame)
                except queue.Empty:
                    break

            while not self.smpl_queue.empty():
                try:
                    smpl_ts, body_quat, smpl_joints, smpl_pose = (
                        self.smpl_queue.get_nowait()
                    )
                    self.smpl_data_list.append(
                        {
                            "timestamp": smpl_ts,
                            "body_quat": body_quat,
                            "smpl_joints": smpl_joints,
                            "smpl_pose": smpl_pose,
                        }
                    )
                except queue.Empty:
                    break

            for vw in video_writers:
                if vw is not None:
                    vw.release()
            logger.info(f"Finished writing videos to {self.video_out_paths}")

            if self.smpl_data_list:
                timestamps = np.array([d["timestamp"] for d in self.smpl_data_list])
                body_quats = np.array([d["body_quat"] for d in self.smpl_data_list])
                smpl_joints = np.array([d["smpl_joints"] for d in self.smpl_data_list])
                smpl_poses = np.array([d["smpl_pose"] for d in self.smpl_data_list])
                np.savez(
                    self.smpl_out_path,
                    timestamps=timestamps,
                    body_quats=body_quats,
                    smpl_joints=smpl_joints,
                    smpl_poses=smpl_poses,
                )
                logger.info(f"Finished writing SMPL data to {self.smpl_out_path}")

    def _thread_main(self, target):
        try:
            target()
        except BaseException as exc:
            self._thread_error = exc
            self.running = False
            self.video_ended = True
            raise

    def start(self):
        logger.info("Starting realtime multi-view publisher (Press Ctrl+C to stop)")
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._thread_main, args=(self._capture_loop,), daemon=True
        )
        self.worker_thread = threading.Thread(
            target=self._thread_main, args=(self._worker_loop,), daemon=True
        )
        self.publish_thread = threading.Thread(
            target=self._thread_main, args=(self._publish_loop,), daemon=True
        )
        self.capture_thread.start()
        self.worker_thread.start()
        self.publish_thread.start()

        if self.record:
            self.recording_thread = threading.Thread(
                target=self._thread_main, args=(self._recording_loop,), daemon=True
            )
            self.recording_thread.start()

        while self.running:
            if self._thread_error is not None:
                raise RuntimeError("publisher thread failed") from self._thread_error
            if self.video_ended and not self.worker_thread.is_alive():
                self.running = False
                break
            if not self.capture_thread.is_alive() and not self.worker_thread.is_alive():
                self.running = False
                break
            time.sleep(0.05)

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)
        if (
            self.record
            and self.recording_thread is not None
            and self.recording_thread.is_alive()
        ):
            self.recording_thread.join(timeout=5.0)

    def stop(self):
        self.running = False
        self.source.release()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.publish_thread is not None and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)
        if (
            self.record
            and self.recording_thread is not None
            and self.recording_thread.is_alive()
        ):
            logger.info("Waiting for recording thread to finish writing to disk...")
            self.recording_thread.join(timeout=5.0)
        self.publisher.close()
        self._log_final_stats()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish multi-view SMPL over ZMQ using main-camera quaternion + multi-view MHR fusion"
    )
    parser.add_argument("--source", choices=["camera", "video"], default="camera")
    parser.add_argument(
        "--main-camera",
        type=int,
        default=0,
        help="Main camera index for body quaternion",
    )
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--interp-lag-ms", type=float, default=140.0)
    parser.add_argument("--addr", type=str, default="tcp://*:5556")
    parser.add_argument("--image-size", type=int, default=512, choices=[256, 384, 512])
    parser.add_argument("--yolo-model", type=str, default=YOLO_MODEL_PATH)
    parser.add_argument("--smpl-model-path", type=str, required=True)
    parser.add_argument("--nn-model-dir", type=str, required=True)
    parser.add_argument(
        "--mhr2smpl-mapping-path",
        type=str,
        required=True,
        help="Path to mhr2smpl_mapping.npz (mhr_vert_ids or triangle_ids format)",
    )
    parser.add_argument(
        "--mhr-mesh-path",
        type=str,
        default=None,
        help="Path to MHR mesh PLY (required when mapping uses triangle_ids format)",
    )
    parser.add_argument("--smoother-dir", type=str, default=None)
    parser.add_argument("--min-person-confidence", type=float, default=0.75)
    parser.add_argument(
        "--record", action="store_true", help="Record raw video and parsed SMPL data"
    )
    parser.add_argument(
        "--record-dir", type=str, default="output/records", help="Directory to save recordings"
    )

    parser.add_argument(
        "--serials",
        type=str,
        default=None,
        help="Comma-separated RealSense serials to use. Default: all detected cameras.",
    )

    parser.add_argument(
        "--videos", nargs="+", default=None, help="Video paths for --source video"
    )
    parser.add_argument(
        "--intrinsics",
        nargs="+",
        default=None,
        help="Per-video intrinsics/gravity JSON paths for --source video",
    )
    parser.add_argument("--no-loop", action="store_true")

    args = parser.parse_args()
    if args.publish_hz <= 0:
        parser.error("--publish-hz must be > 0")
    if args.interp_lag_ms < 0:
        parser.error("--interp-lag-ms must be >= 0")

    return args


def build_source(args):
    if args.source == "camera":
        if args.serials is None:
            raise RuntimeError("--serials is required when --source camera")
        serials = args.serials.split(",")
        return MultiRealSenseSource(
            width=848,
            height=480,
            fps=30,
            serials=serials,
            imu_samples=100,
        )

    if not args.videos:
        raise RuntimeError("--videos is required when --source video")
    if not args.intrinsics:
        raise RuntimeError("--intrinsics is required when --source video")
    return MultiVideoFileSource(args.videos, args.intrinsics, loop=not args.no_loop)


def main():
    args = parse_args()

    source = build_source(args)
    publisher = RealtimeMultiViewPublisher(
        source=source,
        main_camera=args.main_camera,
        publish_hz=args.publish_hz,
        interpolate_lag_ms=args.interp_lag_ms,
        smpl_model_path=args.smpl_model_path,
        nn_model_dir=args.nn_model_dir,
        mhr2smpl_mapping_path=args.mhr2smpl_mapping_path,
        mhr_mesh_path=args.mhr_mesh_path,
        smoother_dir=args.smoother_dir,
        addr=args.addr,
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        min_person_confidence=args.min_person_confidence,
        record=args.record,
        record_dir=args.record_dir,
    )

    try:
        publisher.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        publisher.stop()
        logger.success("Stopped.")


if __name__ == "__main__":
    main()
