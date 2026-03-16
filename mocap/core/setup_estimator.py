import os
import torch

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from tools.build_detector import HumanDetector
from tools.build_fov_estimator import FOVEstimator

DEFAULT_IMAGE_SIZE = 512


def build_default_estimator(
    *,
    image_size=DEFAULT_IMAGE_SIZE,
    yolo_model_path="checkpoints/yolo_pose/yolo11m-pose.engine",
    fov_model_size="s",
    fov_resolution_level=0,
    fov_fixed_size=512,
    fov_fast_mode=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if image_size:
        os.environ["IMG_SIZE"] = str(image_size)

    os.environ["FOV_MODEL"] = str(fov_model_size)
    os.environ["FOV_LEVEL"] = str(fov_resolution_level)
    os.environ["FOV_SIZE"] = str(fov_fixed_size)
    os.environ["FOV_FAST"] = "1" if fov_fast_mode else "0"

    model, model_cfg = load_sam_3d_body(
        checkpoint_path="./checkpoints/sam-3d-body-dinov3/model.ckpt",
        device=device,
        mhr_path="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    )

    human_detector = HumanDetector(
        name="yolo_pose",
        device=device,
        model=yolo_model_path,
    )

    fov_estimator = FOVEstimator(name="moge2", device=device)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
