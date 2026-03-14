# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path

import numpy as np
import torch


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device
        self.name = name

        if name == "vitdet":
            print("########### Using human detector: ViTDet...")
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        elif name == "yolo" or name.startswith("yolo11"):
            print(f"########### Using human detector: YOLO11...")
            model_name = kwargs.get("model", "./checkpoints/yolo/yolo11n.pt")
            self.detector = load_yolo11(model_name, device=device)
            self.detector_func = run_yolo11
        elif name == "yolo_pose":
            print(f"########### Using human detector: YOLO11-Pose (with keypoints)...")
            model_name = kwargs.get("model", "./checkpoints/yolo/yolo11m-pose.pt")
            self.detector = load_yolo11(model_name, device=device)
            self.detector_func = run_yolo_pose
        else:
            raise NotImplementedError(f"Detector '{name}' not supported. Use 'vitdet', 'yolo', or 'yolo_pose'.")

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)


def load_detectron2_vitdet(path=""):
    """
    Load vitdet detector similar to 4D-Humans demo.py approach.
    Checkpoint is automatically downloaded from the hardcoded URL.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    # Get config file from tools directory (same folder as this file)
    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)

    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    # Print model dtype info
    first_param = next(detector.parameters())
    print(f"          [DEBUG] Model: ViTDet (Human Detector), param_dtype: {first_param.dtype}, input_dtype: {img_transformed.dtype}")

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    valid_idx = (det_instances.pred_classes == det_cat_id) & (
        det_instances.scores > bbox_thr
    )
    if valid_idx.sum() == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
    else:
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    # Sort boxes to keep a consistent output order
    sorted_indices = np.lexsort(
        (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
    )  # shape: [len(boxes),]
    boxes = boxes[sorted_indices]
    return boxes


def load_yolo11(model_name="yolo11n.pt", device="cuda", task=None):
    """
    Load YOLO11 model from ultralytics.

    Args:
        model_name: Model name (e.g., "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
                   Also supports TensorRT format (e.g., "yolo11n.engine")
        device: Device to use ("cuda" or "cpu")
        task: Task type ("detect", "pose", etc.). Auto-detected from model name if not specified.

    Returns:
        YOLO model instance
    """
    from ultralytics import YOLO

    # Auto-detect task from model name if not specified
    if task is None:
        if 'pose' in model_name.lower():
            task = 'pose'
        else:
            task = 'detect'

    # TensorRT engines need explicit task specification
    is_engine = model_name.endswith('.engine') or model_name.endswith('.onnx')
    if is_engine:
        model = YOLO(model_name, task=task)
        print(f"Loading {model_name} for TensorRT inference (task={task})...")
    else:
        model = YOLO(model_name)

    if not is_engine:
        model.to(device)

    # Store device and task for inference
    model._device = device
    model._task = task

    # Note: FP16 is enabled during inference via half=True parameter
    # Don't convert model to half() here as it breaks fuse() operation
    return model


def run_yolo11(
    detector,
    img,
    det_cat_id: int = 0,  # COCO class 0 = person
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
    imgsz: int = 640,  # Input image size for YOLO
):
    """
    Run YOLO11 detection on image.

    Args:
        detector: YOLO model instance
        img: Input image (BGR format, numpy array)
        det_cat_id: Detection category ID (0 = person in COCO)
        bbox_thr: Bounding box confidence threshold
        nms_thr: NMS IoU threshold (not directly used, YOLO handles internally)
        default_to_full_image: If no detections, return full image as bbox
        imgsz: Input image size for YOLO inference (smaller = faster, default: 640)

    Returns:
        boxes: numpy array of shape [N, 4] with format [x1, y1, x2, y2]
    """
    height, width = img.shape[:2]

    # Get device (stored during load or default to cuda)
    device = getattr(detector, '_device', 'cuda')

    # Check if using TensorRT engine (no half parameter needed - already baked in)
    model_name = str(getattr(detector, 'model_name', getattr(detector, 'ckpt_path', '')))
    is_engine = model_name.endswith('.engine')

    # Run YOLO inference with optimizations
    results = detector(
        img,
        conf=bbox_thr,
        classes=[det_cat_id],
        verbose=False,
        imgsz=imgsz,
        device=device,
        half=not is_engine,  # FP16 for PyTorch models, TensorRT already uses FP16
    )

    # Extract boxes
    boxes_list = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Get boxes in xyxy format
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            # Filter by confidence (should already be filtered, but double check)
            valid_mask = confs >= bbox_thr
            xyxy = xyxy[valid_mask]

            boxes_list.extend(xyxy)

    if len(boxes_list) == 0:
        if default_to_full_image:
            boxes = np.array([[0, 0, width, height]], dtype=np.float32)
        else:
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
    else:
        boxes = np.array(boxes_list, dtype=np.float32)

    # Sort boxes to keep a consistent output order
    if len(boxes) > 0:
        sorted_indices = np.lexsort(
            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
        )
        boxes = boxes[sorted_indices]

    print(f"          [DEBUG] YOLO11 detected {len(boxes)} person(s)")
    return boxes


def run_yolo_pose(
    detector,
    img,
    det_cat_id: int = 0,  # COCO class 0 = person
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
    imgsz: int = 640,
):
    """
    Run YOLO-Pose detection on image, returning both boxes and keypoints.

    Args:
        detector: YOLO-Pose model instance
        img: Input image (BGR format, numpy array)
        det_cat_id: Detection category ID (0 = person in COCO)
        bbox_thr: Bounding box confidence threshold
        nms_thr: NMS IoU threshold
        default_to_full_image: If no detections, return full image as bbox
        imgsz: Input image size for YOLO inference

    Returns:
        dict with keys:
            - boxes: numpy array [N, 4] with format [x1, y1, x2, y2]
            - keypoints: numpy array [N, 17, 3] with (x, y, conf) per keypoint
                COCO keypoint order: nose, left_eye, right_eye, left_ear, right_ear,
                left_shoulder, right_shoulder, left_elbow, right_elbow,
                left_wrist(9), right_wrist(10), left_hip, right_hip,
                left_knee, right_knee, left_ankle, right_ankle
    """
    height, width = img.shape[:2]

    # Get device
    device = getattr(detector, '_device', 'cuda')

    # Check if using TensorRT engine - try multiple attributes
    model_path = ""
    for attr in ['model_name', 'ckpt_path', 'model']:
        val = getattr(detector, attr, None)
        if val is not None:
            model_path = str(val)
            if '.engine' in model_path or '.onnx' in model_path:
                break
    is_engine = '.engine' in model_path or '.onnx' in model_path

    # Build inference kwargs
    inference_kwargs = {
        'conf': bbox_thr,
        'verbose': False,
        'imgsz': imgsz,
        'device': device,
    }

    # TensorRT engine doesn't need half parameter (already baked in)
    # Also don't filter by classes for TensorRT - it may not support it
    if not is_engine:
        inference_kwargs['half'] = True
        inference_kwargs['classes'] = [det_cat_id]

    # Run YOLO-Pose inference
    if is_engine:
        print(f"          [DEBUG] YOLO-Pose using TensorRT engine")
    results = detector(img, **inference_kwargs)

    boxes_list = []
    keypoints_list = []

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Get boxes
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            # Get class IDs (for TensorRT we need to filter manually)
            if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            else:
                # YOLO-Pose only detects person, so assume all are person
                cls_ids = np.zeros(len(xyxy), dtype=int)

            # Get keypoints if available
            if result.keypoints is not None:
                kpts = result.keypoints.data.cpu().numpy()  # [N, 17, 3]
            else:
                # No keypoints, create empty
                kpts = np.zeros((len(xyxy), 17, 3), dtype=np.float32)

            # Filter by confidence and class (person = 0)
            valid_mask = (confs >= bbox_thr) & (cls_ids == det_cat_id)
            xyxy = xyxy[valid_mask]
            kpts = kpts[valid_mask]

            boxes_list.extend(xyxy)
            keypoints_list.extend(kpts)

    if len(boxes_list) == 0:
        if default_to_full_image:
            boxes = np.array([[0, 0, width, height]], dtype=np.float32)
            keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        else:
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            keypoints = np.array([], dtype=np.float32).reshape(0, 17, 3)
    else:
        boxes = np.array(boxes_list, dtype=np.float32)
        keypoints = np.array(keypoints_list, dtype=np.float32)

    # Sort boxes and keypoints to keep consistent order
    if len(boxes) > 0:
        sorted_indices = np.lexsort(
            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
        )
        boxes = boxes[sorted_indices]
        keypoints = keypoints[sorted_indices]

    print(f"          [DEBUG] YOLO-Pose detected {len(boxes)} person(s)")

    # Return dict with boxes and keypoints
    return {
        "boxes": boxes,
        "keypoints": keypoints,
    }
