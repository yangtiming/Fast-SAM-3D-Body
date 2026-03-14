# Copyright (c) Meta Platforms, Inc. and affiliates.

import time
from typing import Any, Dict, Optional, Tuple
import os
import numpy as np
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.models.decoders.prompt_encoder import PositionEmbeddingRandom
from sam_3d_body.models.modules.mhr_utils import (
    euler_to_rotmat_XZY,
    fix_wrist_euler,
    rotation_angle_difference,
    rotmat_to_euler_XZY,
)
from sam_3d_body.utils import recursive_to
from sam_3d_body.utils.logging import get_pylogger

from ..backbones import create_backbone
from ..decoders import build_decoder, build_keypoint_sampler, PromptEncoder
from ..heads import build_head
from ..modules.camera_embed import CameraEncoder
from ..modules.transformer import FFN, MLP

from .base_model import BaseModel


logger = get_pylogger(__name__)


# IntermPred detailed timing control
_INTERM_DETAIL_TIMING = os.environ.get("INTERM_TIMING", "0") == "1"
_INTERM_TIMING_WARMUP = 3
_INTERM_TIMING_COUNT = [0]  # Use list to simulate mutable reference


def _sync_time():
    """Synchronize CUDA and return current time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# ============================================================
# GPU-accelerated hand batch preparation
# ============================================================
_USE_GPU_HAND_PREP = os.environ.get("GPU_HAND_PREP", "1") == "1"


def _get_affine_matrix_for_crop(center, scale, output_size):
    """
    Compute the affine transformation matrix from original image to crop (for grid_sample).

    Args:
        center: (2,) bbox center [cx, cy]
        scale: (2,) bbox size [w, h]
        output_size: (2,) output size [out_w, out_h]

    Returns:
        theta: (2, 3) affine transformation matrix (for F.affine_grid)
    """
    out_w, out_h = output_size
    cx, cy = center
    w, h = scale

    # grid_sample uses normalized coordinates [-1, 1]
    # We need to map output coordinates [0, out_w] x [0, out_h] to input coordinates [cx-w/2, cx+w/2] x [cy-h/2, cy+h/2]
    #
    # Output normalization: x_out_norm = 2 * x_out / out_w - 1
    # Target: x_in = cx + (x_out_norm) * w/2
    #         y_in = cy + (y_out_norm) * h/2
    #
    # Convert to input normalization (assuming input size is img_w, img_h):
    # x_in_norm = 2 * x_in / img_w - 1 = 2 * (cx + x_out_norm * w/2) / img_w - 1
    #           = (2*cx/img_w - 1) + x_out_norm * (w/img_w)
    #
    # But grid_sample's theta is: x_in_norm = theta[0,0]*x_out_norm + theta[0,1]*y_out_norm + theta[0,2]
    # So: theta[0,0] = w/img_w, theta[0,2] = 2*cx/img_w - 1
    #
    # Note: the scale here is relative to the full image, so it needs to be normalized by image size when called

    # Return transformation relative to bbox (caller needs to normalize by image size)
    return center, scale


def _apply_gpu_affine_crop(img_tensor, bboxes_xyxy, output_size, device='cuda'):
    """
    Batch affine crop on GPU.

    Args:
        img_tensor: (H, W, 3) or (B, 3, H, W) image tensor
        bboxes_xyxy: (N, 4) bbox coordinates [x1, y1, x2, y2]
        output_size: (out_h, out_w) output size
        device: target device

    Returns:
        cropped: (N, 3, out_h, out_w) cropped images
        affine_mats: (N, 2, 3) affine transformation matrices (for subsequent coordinate transforms)
    """
    # Ensure input is in correct format
    if img_tensor.dim() == 3:
        # (H, W, 3) -> (1, 3, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float()

    if not img_tensor.is_cuda:
        img_tensor = img_tensor.to(device)

    B, C, H, W = img_tensor.shape
    N = bboxes_xyxy.shape[0]
    out_h, out_w = output_size

    # Expand image to match the number of bboxes
    if B == 1 and N > 1:
        img_tensor = img_tensor.expand(N, -1, -1, -1)

    # Compute bbox center and size
    bboxes = torch.as_tensor(bboxes_xyxy, dtype=torch.float32, device=device)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1

    # Use the larger side as crop size (keep square)
    crop_size = torch.max(bw, bh)

    # Build affine transformation matrix (from output coordinates to input coordinates)
    # grid_sample requires: input_point = theta @ output_point
    # Output normalization: [-1, 1]
    # Input normalization: [-1, 1] (relative to input image size)
    #
    # Mapping: output (-1,-1) -> input (cx - crop_size/2, cy - crop_size/2)
    #          output (1, 1) -> input (cx + crop_size/2, cy + crop_size/2)
    #
    # Input normalization: x_in_norm = 2 * x_in / W - 1
    #            x_in = cx + x_out_norm * crop_size/2
    #            x_in_norm = 2 * (cx + x_out_norm * crop_size/2) / W - 1
    #                      = (2*cx/W - 1) + x_out_norm * (crop_size/W)
    #
    # theta[0,0] = crop_size / W
    # theta[0,2] = 2*cx/W - 1

    theta = torch.zeros(N, 2, 3, device=device)
    theta[:, 0, 0] = crop_size / W  # scale x
    theta[:, 1, 1] = crop_size / H  # scale y
    theta[:, 0, 2] = 2 * cx / W - 1  # translate x
    theta[:, 1, 2] = 2 * cy / H - 1  # translate y

    # Generate sampling grid and perform sampling
    grid = F.affine_grid(theta, [N, C, out_h, out_w], align_corners=False)
    cropped = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Compute affine matrix for coordinate transformation (from original image coordinates to crop coordinates)
    # crop_x = (x - (cx - crop_size/2)) * out_w / crop_size
    #        = (x - cx + crop_size/2) * out_w / crop_size
    #        = x * out_w/crop_size - cx * out_w/crop_size + out_w/2
    #
    # affine_mat @ [x, y, 1] = [crop_x, crop_y]
    affine_mats = torch.zeros(N, 2, 3, device=device)
    affine_mats[:, 0, 0] = out_w / crop_size  # scale x
    affine_mats[:, 1, 1] = out_h / crop_size  # scale y
    affine_mats[:, 0, 2] = -cx * out_w / crop_size + out_w / 2  # translate x
    affine_mats[:, 1, 2] = -cy * out_h / crop_size + out_h / 2  # translate y

    return cropped, affine_mats, crop_size


def _prepare_hand_batch_gpu(img_tensor, bbox_xyxy, cam_int, output_size=(512, 512), device='cuda'):
    """
    Prepare a single hand batch on GPU.

    Args:
        img_tensor: (3, H, W) image tensor (already on GPU)
        bbox_xyxy: (1, 4) bbox coordinates
        cam_int: (1, 3, 3) camera intrinsics
        output_size: (out_h, out_w) output size
        device: target device

    Returns:
        batch: dict containing all required data
    """
    out_h, out_w = output_size
    _, H, W = img_tensor.shape

    # Compute bbox center and size
    bbox = bbox_xyxy[0]
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1
    crop_size = max(bw, bh)

    # Build affine transformation matrix
    theta = torch.zeros(1, 2, 3, device=device)
    theta[0, 0, 0] = crop_size / W
    theta[0, 1, 1] = crop_size / H
    theta[0, 0, 2] = 2 * cx / W - 1
    theta[0, 1, 2] = 2 * cy / H - 1

    # GPU crop
    img_input = img_tensor.unsqueeze(0).float()  # (1, 3, H, W)
    grid = F.affine_grid(theta, [1, 3, out_h, out_w], align_corners=False)
    cropped = F.grid_sample(img_input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Normalize (ImageNet mean/std)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    cropped = (cropped / 255.0 - mean) / std

    # Compute affine transformation matrix (for coordinate transformation)
    affine_mat = torch.zeros(1, 2, 3, device=device)
    affine_mat[0, 0, 0] = out_w / crop_size
    affine_mat[0, 1, 1] = out_h / crop_size
    affine_mat[0, 0, 2] = -cx * out_w / crop_size + out_w / 2
    affine_mat[0, 1, 2] = -cy * out_h / crop_size + out_h / 2

    # Build batch
    batch = {
        'img': cropped.unsqueeze(0),  # (1, 1, 3, H, W)
        'img_size': torch.tensor([[out_w, out_h]], device=device).float().unsqueeze(0),
        'ori_img_size': torch.tensor([[W, H]], device=device).float().unsqueeze(0),
        'bbox_center': torch.tensor([[[cx, cy]]], device=device).float(),
        'bbox_scale': torch.tensor([[[crop_size, crop_size]]], device=device).float(),
        'bbox': bbox_xyxy.unsqueeze(0).float().to(device),
        'affine_trans': affine_mat.unsqueeze(0),
        'mask': torch.zeros(1, 1, 1, out_h, out_w, device=device).float(),
        'mask_score': torch.tensor([[0.0]], device=device).float(),
        'person_valid': torch.ones(1, 1, device=device),
        'cam_int': cam_int.to(device),
    }

    return batch


def _prepare_hand_batches_gpu(img, left_xyxy, right_xyxy, cam_int, output_size=(512, 512), padding=0.9, device='cuda'):
    """
    Prepare left and right hand batches simultaneously on GPU (~10x faster than CPU version).
    Supports multi-person batch processing.

    Args:
        img: (H, W, 3) numpy array or tensor
        left_xyxy: (N, 4) left hand bbox (original image coordinates, will be flipped)
        right_xyxy: (N, 4) right hand bbox
        cam_int: (1, 3, 3) camera intrinsics
        output_size: (out_h, out_w) output size
        padding: bbox expansion factor
        device: target device

    Returns:
        batch_lhand: left hand batch
        batch_rhand: right hand batch
        left_xyxy_flipped: flipped left hand bbox (numpy array)
    """
    out_h, out_w = output_size
    H, W = img.shape[:2]

    # 1. Upload image to GPU (once)
    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img).to(device)  # (H, W, 3)
    else:
        img_tensor = img.to(device)

    # Convert to (3, H, W) format
    if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)

    # 2. Flip image for left hand (GPU operation)
    img_flipped = torch.flip(img_tensor, dims=[2])  # Horizontal flip

    # 3. Compute flipped left hand bbox
    left_xyxy_t = torch.as_tensor(left_xyxy, dtype=torch.float32, device=device)
    left_xyxy_flipped = left_xyxy_t.clone()
    left_xyxy_flipped[:, 0] = W - left_xyxy_t[:, 2] - 1  # x1 = W - x2 - 1
    left_xyxy_flipped[:, 2] = W - left_xyxy_t[:, 0] - 1  # x2 = W - x1 - 1

    # Get number of persons
    N = left_xyxy_t.shape[0]

    # 4. Apply padding and compute final crop_size (matching CPU version's transform chain)
    # CPU version processing flow:
    #   1. GetBBoxCenterScale(padding=0.9): scale = (w, h) * 0.9
    #   2. fix_aspect_ratio(scale, 0.75): adjust to 0.75 aspect ratio
    #   3. fix_aspect_ratio(scale, 1.0): for 512x512 output, make it square
    def compute_crop_params(bbox_xyxy, padding_factor, aspect_ratio=0.75):
        x1, y1, x2, y2 = bbox_xyxy[:, 0], bbox_xyxy[:, 1], bbox_xyxy[:, 2], bbox_xyxy[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Step 1: Apply padding
        w = (x2 - x1) * padding_factor  # bbox_w * 0.9
        h = (y2 - y1) * padding_factor  # bbox_h * 0.9
        # Step 2: fix_aspect_ratio(aspect_ratio=0.75)
        scale_w = torch.where(w > h * aspect_ratio, w, h * aspect_ratio)
        scale_h = torch.where(w > h * aspect_ratio, w / aspect_ratio, h)
        # Step 3: fix_aspect_ratio(aspect_ratio=1.0) - make it square
        crop_size = torch.max(scale_w, scale_h)
        return cx, cy, crop_size

    # Left hand (flipped image)
    lx, ly, l_crop_size = compute_crop_params(left_xyxy_flipped, padding)  # Each is (N,)

    # Right hand (original image)
    right_xyxy_t = torch.as_tensor(right_xyxy, dtype=torch.float32, device=device)
    rx, ry, r_crop_size = compute_crop_params(right_xyxy_t, padding)  # Each is (N,)

    # 5. Directly build sampling grid (matching cv2.warpAffine behavior)
    # cv2.warpAffine inverse transform M^(-1) (dst -> src):
    #   src_x = dst_x * (crop_size / out_w) + (cx - crop_size / 2)
    #   src_y = dst_y * (crop_size / out_h) + (cy - crop_size / 2)
    #
    # When using align_corners=False, the relationship between normalized coord and pixel:
    #   norm = 2 * (pixel + 0.5) / N - 1
    #   pixel = (norm + 1) * N / 2 - 0.5

    # Create output pixel coordinate grid (out_h, out_w)
    dx = torch.arange(out_w, device=device, dtype=torch.float32)
    dy = torch.arange(out_h, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')  # (out_h, out_w) each

    # Expand to batch form (N, out_h, out_w)
    grid_x = grid_x.unsqueeze(0).expand(N, -1, -1)  # (N, out_h, out_w)
    grid_y = grid_y.unsqueeze(0).expand(N, -1, -1)  # (N, out_h, out_w)

    # Expand crop parameters to broadcastable shape (N, 1, 1)
    lx = lx.view(N, 1, 1)
    ly = ly.view(N, 1, 1)
    l_crop_size = l_crop_size.view(N, 1, 1)
    rx = rx.view(N, 1, 1)
    ry = ry.view(N, 1, 1)
    r_crop_size = r_crop_size.view(N, 1, 1)

    # Left hand: compute source pixel coordinates
    l_scale_inv = l_crop_size / out_w  # Inverse scale factor (N, 1, 1)
    l_src_x = grid_x * l_scale_inv + (lx - l_crop_size / 2)  # (N, out_h, out_w)
    l_src_y = grid_y * l_scale_inv + (ly - l_crop_size / 2)  # (N, out_h, out_w)

    # Convert to normalized coordinates (align_corners=False)
    l_norm_x = 2 * (l_src_x + 0.5) / W - 1
    l_norm_y = 2 * (l_src_y + 0.5) / H - 1
    grid_left = torch.stack([l_norm_x, l_norm_y], dim=-1)  # (N, out_h, out_w, 2)

    # Right hand: compute source pixel coordinates
    r_scale_inv = r_crop_size / out_w  # (N, 1, 1)
    r_src_x = grid_x * r_scale_inv + (rx - r_crop_size / 2)  # (N, out_h, out_w)
    r_src_y = grid_y * r_scale_inv + (ry - r_crop_size / 2)  # (N, out_h, out_w)

    # Convert to normalized coordinates (align_corners=False)
    r_norm_x = 2 * (r_src_x + 0.5) / W - 1
    r_norm_y = 2 * (r_src_y + 0.5) / H - 1
    grid_right = torch.stack([r_norm_x, r_norm_y], dim=-1)  # (N, out_h, out_w, 2)

    # 6. GPU crop - need to expand image to batch form
    # grid_sample requires input: (N, C, H, W), grid: (N, out_h, out_w, 2)
    img_flipped_batch = img_flipped.unsqueeze(0).expand(N, -1, -1, -1).float()  # (N, 3, H, W)
    img_tensor_batch = img_tensor.unsqueeze(0).expand(N, -1, -1, -1).float()  # (N, 3, H, W)

    cropped_left = F.grid_sample(
        img_flipped_batch, grid_left,
        mode='bilinear', padding_mode='zeros', align_corners=False
    )  # (N, 3, out_h, out_w)
    cropped_right = F.grid_sample(
        img_tensor_batch, grid_right,
        mode='bilinear', padding_mode='zeros', align_corners=False
    )  # (N, 3, out_h, out_w)

    # 7. Normalize: ToTensor() only divides by 255, no ImageNet mean/std normalization
    # clamp(0, 1): grid_sample bilinear floating-point precision may produce values >255,
    # after dividing by 255, values >1.0 would cause data_preprocess to incorrectly detect inputs.max()>1 and divide the entire batch by 255 again
    cropped_left = (cropped_left / 255.0).clamp(0, 1)
    cropped_right = (cropped_right / 255.0).clamp(0, 1)

    # 8. Compute affine transformation matrix (for coordinate transformation)
    # Squeeze back to (N,) for building affine
    lx_flat = lx.view(N)
    ly_flat = ly.view(N)
    l_crop_size_flat = l_crop_size.view(N)
    rx_flat = rx.view(N)
    ry_flat = ry.view(N)
    r_crop_size_flat = r_crop_size.view(N)

    affine_left = torch.zeros(N, 2, 3, device=device)
    affine_left[:, 0, 0] = out_w / l_crop_size_flat
    affine_left[:, 1, 1] = out_h / l_crop_size_flat
    affine_left[:, 0, 2] = -lx_flat * out_w / l_crop_size_flat + out_w / 2
    affine_left[:, 1, 2] = -ly_flat * out_h / l_crop_size_flat + out_h / 2

    affine_right = torch.zeros(N, 2, 3, device=device)
    affine_right[:, 0, 0] = out_w / r_crop_size_flat
    affine_right[:, 1, 1] = out_h / r_crop_size_flat
    affine_right[:, 0, 2] = -rx_flat * out_w / r_crop_size_flat + out_w / 2
    affine_right[:, 1, 2] = -ry_flat * out_h / r_crop_size_flat + out_h / 2

    # 9. Build batch - format: (1, N, ...) where 1 is batch dimension, N is number of persons
    cam_int_gpu = cam_int.to(device)

    # bbox_center and bbox_scale need to be constructed in (1, N, 2) format
    bbox_center_left = torch.stack([lx_flat, ly_flat], dim=-1).unsqueeze(0)  # (1, N, 2)
    bbox_scale_left = torch.stack([l_crop_size_flat, l_crop_size_flat], dim=-1).unsqueeze(0)  # (1, N, 2)
    bbox_center_right = torch.stack([rx_flat, ry_flat], dim=-1).unsqueeze(0)  # (1, N, 2)
    bbox_scale_right = torch.stack([r_crop_size_flat, r_crop_size_flat], dim=-1).unsqueeze(0)  # (1, N, 2)

    batch_lhand = {
        'img': cropped_left.unsqueeze(0),  # (1, N, 3, H, W)
        'img_size': torch.tensor([[out_w, out_h]], device=device).float().expand(N, -1).unsqueeze(0),  # (1, N, 2)
        'ori_img_size': torch.tensor([[W, H]], device=device).float().expand(N, -1).unsqueeze(0),  # (1, N, 2)
        'bbox_center': bbox_center_left,  # (1, N, 2)
        'bbox_scale': bbox_scale_left,  # (1, N, 2)
        'bbox': left_xyxy_flipped.unsqueeze(0),  # (1, N, 4)
        'affine_trans': affine_left.unsqueeze(0),  # (1, N, 2, 3)
        'mask': torch.zeros(1, N, 1, out_h, out_w, device=device).float(),
        'mask_score': torch.zeros(1, N, device=device).float(),
        'person_valid': torch.ones(1, N, device=device),
        'cam_int': cam_int_gpu.clone(),
    }

    batch_rhand = {
        'img': cropped_right.unsqueeze(0),  # (1, N, 3, H, W)
        'img_size': torch.tensor([[out_w, out_h]], device=device).float().expand(N, -1).unsqueeze(0),  # (1, N, 2)
        'ori_img_size': torch.tensor([[W, H]], device=device).float().expand(N, -1).unsqueeze(0),  # (1, N, 2)
        'bbox_center': bbox_center_right,  # (1, N, 2)
        'bbox_scale': bbox_scale_right,  # (1, N, 2)
        'bbox': right_xyxy_t.unsqueeze(0),  # (1, N, 4)
        'affine_trans': affine_right.unsqueeze(0),  # (1, N, 2, 3)
        'mask': torch.zeros(1, N, 1, out_h, out_w, device=device).float(),
        'mask_score': torch.zeros(1, N, device=device).float(),
        'person_valid': torch.ones(1, N, device=device),
        'cam_int': cam_int_gpu.clone(),
    }

    return batch_lhand, batch_rhand, left_xyxy_flipped.cpu().numpy()


# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "mhr70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
KEY_RIGHT_HAND = list(range(21, 42))
# fmt: on


class SAM3DBody(BaseModel):
    pelvis_idx = [9, 10]  # left_hip, right_hip

    def _initialze_model(self):
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # Create backbone feature extractor for human crops
        self.backbone = create_backbone(self.cfg.MODEL.BACKBONE.TYPE, self.cfg)

        # Create header for pose estimation output
        self.head_pose = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        self.head_pose.hand_pose_comps_ori = nn.Parameter(
            self.head_pose.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose.hand_pose_comps.data).float()
        )

        # Initialize pose token with learnable params
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        self.init_pose = nn.Embedding(1, self.head_pose.npose)

        # Define header for hand pose estimation
        self.head_pose_hand = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(
            self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose_hand.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        )
        self.init_pose_hand = nn.Embedding(1, self.head_pose_hand.npose)

        self.head_camera = build_head(self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE)
        self.init_camera = nn.Embedding(1, self.head_camera.ncam)
        nn.init.zeros_(self.init_camera.weight)

        self.head_camera_hand = build_head(
            self.cfg,
            self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=self.cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
        )
        self.init_camera_hand = nn.Embedding(1, self.head_camera_hand.ncam)
        nn.init.zeros_(self.init_camera_hand.weight)

        self.camera_type = "perspective"

        # Support conditioned information for decoder
        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        self.init_to_token_mhr = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )
        self.init_to_token_mhr_hand = nn.Linear(init_dim, self.cfg.MODEL.DECODER.DIM)
        self.prev_to_token_mhr_hand = nn.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM
        )

        # Create prompt encoder
        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,  # need to match backbone dims for PE
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
            )
            self.prompt_to_token = nn.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=(
                    KEY_BODY
                    if not self.cfg.MODEL.PROMPT_ENCODER.get("SAMPLE_HAND", False)
                    else KEY_RIGHT_HAND
                ),
            )
            # To keep track of prompting history
            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False

        # Create promptable decoder
        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )
        # shared config for the two decoders
        self.decoder_hand = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims
        )

        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

        # Manually convert the torso of the model to fp16.
        if self.cfg.TRAIN.USE_FP16:
            self.convert_to_fp16()
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )
        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
        )

        self.keypoint_embedding_idxs = list(range(70))
        self.keypoint_embedding = nn.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = nn.Embedding(
            len(self.keypoint_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = nn.Embedding(
                2, self.cfg.MODEL.DECODER.DIM
            )  # for two hands
            # decice if there is left or right hand inside the image
            self.hand_cls_embed = nn.Linear(self.cfg.MODEL.DECODER.DIM, 2)
            self.bbox_embed = MLP(
                self.cfg.MODEL.DECODER.DIM, self.cfg.MODEL.DECODER.DIM, 4, 3
            )

        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint_feat_linear = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )
        self.keypoint_feat_linear_hand = nn.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM
        )

        # Do all KPS
        self.keypoint3d_embedding_idxs = list(range(70))
        self.keypoint3d_embedding = nn.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM
        )

        # Assume always do full body for the hand decoder
        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = nn.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM
        )

        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )
        self.keypoint3d_posemb_linear_hand = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
        )

        # torch.compile IntermPred support
        self._interm_compiled = False
        self._compiled_interm_pred_body = None
        self._compiled_interm_pred_hand = None
        self._compiled_interm_pred_body_slim = None
        self._compiled_interm_pred_hand_slim = None
        self._use_slim_interm = False

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        num_person = batch["img"].shape[1]

        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, [0]]

            focal_length = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, num_person, -1, -1)
                .contiguous()
            )[:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, num_person, -1, -1)
                    .contiguous()
                )[:, [0, 1], [2, 2]]
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )  # [-1, 1]
            condition_info[:, 2] = condition_info[:, 2] / focal_length  # [-1, 1]
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def apply_compile(self, mode: str = "reduce-overhead", dtype: torch.dtype = None):
        """
        Apply torch.compile to Decoders in the model to accelerate inference.
        This method should be called after weights are loaded.

        Args:
            mode: torch.compile mode, options are "default", "reduce-overhead", "max-autotune"
            dtype: Layer precision, options are torch.float16, torch.bfloat16, None (keep original precision)
        """
        print(f"[SAM3DBody] Applying torch.compile to decoders with mode='{mode}', dtype={dtype}")

        # Warmup index cache to avoid CPU-to-GPU transfer during torch.compile trace
        from sam_3d_body.models.modules.mhr_utils import warmup_mhr_idx_cache
        device = next(self.parameters()).device
        warmup_mhr_idx_cache(device)
        print(f"[SAM3DBody] Warmed up MHR index cache on {device}")

        # Check if decoder layers compilation is disabled (for debugging)
        decoder_compile = os.environ.get("DECODER_COMPILE", "1") == "1"
        if decoder_compile:
            if hasattr(self.decoder, 'apply_compile'):
                self.decoder.apply_compile(mode=mode, dtype=dtype)
            if hasattr(self.decoder_hand, 'apply_compile'):
                self.decoder_hand.apply_compile(mode=mode, dtype=dtype)
        else:
            print(f"[SAM3DBody] DECODER_COMPILE=0, skipping decoder layers compilation")
            # Still set autocast dtype
            if dtype is not None:
                if hasattr(self.decoder, 'convert_layers_dtype'):
                    self.decoder.convert_layers_dtype(dtype)
                if hasattr(self.decoder_hand, 'convert_layers_dtype'):
                    self.decoder_hand.convert_layers_dtype(dtype)

        # Also apply torch.compile to MHRHead (IntermPred)
        if hasattr(self, 'head_pose') and hasattr(self.head_pose, 'apply_compile'):
            self.head_pose.apply_compile(mode=mode)
        if hasattr(self, 'head_pose_hand') and hasattr(self.head_pose_hand, 'apply_compile'):
            self.head_pose_hand.apply_compile(mode=mode)

        # Check if IntermPred compilation is disabled (for debugging, keeping decoder layers compiled)
        interm_compile = os.environ.get("INTERM_COMPILE", "1") == "1"
        if not interm_compile:
            print(f"[SAM3DBody] INTERM_COMPILE=0, skipping IntermPred compilation (decoder layers still compiled)")
            self._interm_compiled = False
            return

        # Compile full IntermPred core functions (head_pose + head_camera + projection + full_to_crop)
        # Try fullgraph=True for better CUDA Graph optimization
        interm_fullgraph = os.environ.get("INTERM_FULLGRAPH", "0") == "1"
        interm_slim = os.environ.get("INTERM_SLIM", "1") == "1"  # Slim version enabled by default
        print(f"[SAM3DBody] Compiling IntermPred core functions... (fullgraph={interm_fullgraph}, slim={interm_slim})")

        # Full version (for the last layer)
        # Use dynamic=True to support different batch_sizes (single/multi-person mode)
        self._compiled_interm_pred_body = torch.compile(
            self._interm_pred_body_core,
            mode=mode,
            fullgraph=interm_fullgraph,
            dynamic=True,
        )
        self._compiled_interm_pred_hand = torch.compile(
            self._interm_pred_hand_core,
            mode=mode,
            fullgraph=interm_fullgraph,
            dynamic=True,
        )

        # Slim version (for intermediate layers, skipping unnecessary computation)
        self._use_slim_interm = interm_slim
        if interm_slim:
            self._compiled_interm_pred_body_slim = torch.compile(
                self._interm_pred_body_core_slim,
                mode=mode,
                fullgraph=interm_fullgraph,
                dynamic=True,
            )
            self._compiled_interm_pred_hand_slim = torch.compile(
                self._interm_pred_hand_core_slim,
                mode=mode,
                fullgraph=interm_fullgraph,
                dynamic=True,
            )
        else:
            self._compiled_interm_pred_body_slim = None
            self._compiled_interm_pred_hand_slim = None

        self._interm_compiled = True
        print(f"[SAM3DBody] IntermPred core functions compiled successfully")

    def _interm_pred_body_core(
        self,
        pose_token: torch.Tensor,
        prev_pose: torch.Tensor,
        prev_camera: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_scale: torch.Tensor,
        ori_img_size: torch.Tensor,
        cam_int: torch.Tensor,
        affine_trans: torch.Tensor,
        img_size: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Core computation function for IntermPred Body (for torch.compile).
        Contains: head_pose + head_camera + perspective_projection + full_to_crop
        """
        batch_size = pose_token.shape[0]

        # === head_pose (MHRHead.forward) ===
        # Use _head_forward_core to get tensor output
        (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
         pred_shape, pred_scale, pred_hand, pred_face,
         verts, j3d, jcoords, mhr_model_params, joint_global_rots) = \
            self.head_pose._head_forward_core(pose_token, prev_pose)

        # === head_camera ===
        pred_cam = self.head_camera.proj(pose_token)
        if prev_camera is not None:
            pred_cam = pred_cam + prev_camera

        # === perspective_projection (inline) ===
        # Camera system difference
        s = -pred_cam[:, 0]
        tx = pred_cam[:, 1]
        ty = -pred_cam[:, 2]

        bs = bbox_scale * s * self.head_camera.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (ori_img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (ori_img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
            cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        j3d_cam = j3d + pred_cam_t.unsqueeze(1)

        # perspective projection
        j2d = j3d_cam[..., :2] / j3d_cam[..., 2:3]
        j2d = j2d * cam_int[:, :2, :2].diagonal(dim1=1, dim2=2).unsqueeze(1) + cam_int[:, :2, 2].unsqueeze(1)
        pred_keypoints_2d = j2d.reshape(batch_size, -1, 2)
        pred_keypoints_2d_depth = j3d_cam.reshape(batch_size, -1, 3)[:, :, 2]

        # === full_to_crop ===
        pred_kps_cropped = torch.cat([pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1)
        pred_kps_cropped = pred_kps_cropped @ affine_trans.to(pred_kps_cropped).mT
        pred_kps_cropped = pred_kps_cropped[..., :2] / img_size.unsqueeze(1) - 0.5

        return (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
                pred_shape, pred_scale, pred_hand, pred_face,
                verts, j3d, jcoords, mhr_model_params, joint_global_rots,
                pred_cam, pred_cam_t, focal_length, pred_keypoints_2d, pred_keypoints_2d_depth,
                pred_kps_cropped)

    def _interm_pred_body_core_slim(
        self,
        pose_token: torch.Tensor,
        prev_pose: torch.Tensor,
        prev_camera: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_scale: torch.Tensor,
        ori_img_size: torch.Tensor,
        cam_int: torch.Tensor,
        affine_trans: torch.Tensor,
        img_size: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Slim core computation function for IntermPred Body (for intermediate layers).
        Skips: joint_global_rots, verts/jcoords postprocess
        Only returns what intermediate layers need: pred_pose_raw, pred_cam, j3d, pred_keypoints_2d_cropped, pred_keypoints_2d_depth
        """
        batch_size = pose_token.shape[0]

        # === head_pose (slim version) ===
        global_rot_6d, pred_pose_cont, j3d = \
            self.head_pose._head_forward_core_slim(pose_token, prev_pose)

        # === head_camera ===
        pred_cam = self.head_camera.proj(pose_token)
        if prev_camera is not None:
            pred_cam = pred_cam + prev_camera

        # === perspective_projection (inline) ===
        s = -pred_cam[:, 0]
        tx = pred_cam[:, 1]
        ty = -pred_cam[:, 2]

        bs = bbox_scale * s * self.head_camera.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (ori_img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (ori_img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
            cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        j3d_cam = j3d + pred_cam_t.unsqueeze(1)

        # perspective projection
        j2d = j3d_cam[..., :2] / j3d_cam[..., 2:3]
        j2d = j2d * cam_int[:, :2, :2].diagonal(dim1=1, dim2=2).unsqueeze(1) + cam_int[:, :2, 2].unsqueeze(1)
        pred_keypoints_2d = j2d.reshape(batch_size, -1, 2)
        pred_keypoints_2d_depth = j3d_cam.reshape(batch_size, -1, 3)[:, :, 2]

        # === full_to_crop ===
        pred_kps_cropped = torch.cat([pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1)
        pred_kps_cropped = pred_kps_cropped @ affine_trans.to(pred_kps_cropped).mT
        pred_kps_cropped = pred_kps_cropped[..., :2] / img_size.unsqueeze(1) - 0.5

        # Return the minimal output set needed by intermediate layers
        return (global_rot_6d, pred_pose_cont, pred_cam, j3d,
                pred_kps_cropped, pred_keypoints_2d_depth)

    def _interm_pred_hand_core(
        self,
        pose_token: torch.Tensor,
        prev_pose: torch.Tensor,
        prev_camera: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_scale: torch.Tensor,
        ori_img_size: torch.Tensor,
        cam_int: torch.Tensor,
        affine_trans: torch.Tensor,
        img_size: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Core computation function for IntermPred Hand (for torch.compile).
        Contains: head_pose_hand + head_camera_hand + perspective_projection + full_to_crop
        """
        batch_size = pose_token.shape[0]

        # === head_pose_hand (MHRHead.forward) ===
        (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
         pred_shape, pred_scale, pred_hand, pred_face,
         verts, j3d, jcoords, mhr_model_params, joint_global_rots) = \
            self.head_pose_hand._head_forward_core(pose_token, prev_pose)

        # === head_camera_hand ===
        pred_cam = self.head_camera_hand.proj(pose_token)
        if prev_camera is not None:
            pred_cam = pred_cam + prev_camera

        # === perspective_projection (inline) ===
        s = -pred_cam[:, 0]
        tx = pred_cam[:, 1]
        ty = -pred_cam[:, 2]

        bs = bbox_scale * s * self.head_camera_hand.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (ori_img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (ori_img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
            cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        j3d_cam = j3d + pred_cam_t.unsqueeze(1)

        # perspective projection
        j2d = j3d_cam[..., :2] / j3d_cam[..., 2:3]
        j2d = j2d * cam_int[:, :2, :2].diagonal(dim1=1, dim2=2).unsqueeze(1) + cam_int[:, :2, 2].unsqueeze(1)
        pred_keypoints_2d = j2d.reshape(batch_size, -1, 2)
        pred_keypoints_2d_depth = j3d_cam.reshape(batch_size, -1, 3)[:, :, 2]

        # === full_to_crop ===
        pred_kps_cropped = torch.cat([pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1)
        pred_kps_cropped = pred_kps_cropped @ affine_trans.to(pred_kps_cropped).mT
        pred_kps_cropped = pred_kps_cropped[..., :2] / img_size.unsqueeze(1) - 0.5

        return (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
                pred_shape, pred_scale, pred_hand, pred_face,
                verts, j3d, jcoords, mhr_model_params, joint_global_rots,
                pred_cam, pred_cam_t, focal_length, pred_keypoints_2d, pred_keypoints_2d_depth,
                pred_kps_cropped)

    def _interm_pred_hand_core_slim(
        self,
        pose_token: torch.Tensor,
        prev_pose: torch.Tensor,
        prev_camera: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_scale: torch.Tensor,
        ori_img_size: torch.Tensor,
        cam_int: torch.Tensor,
        affine_trans: torch.Tensor,
        img_size: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        """
        Slim core computation function for IntermPred Hand (for intermediate layers).
        Skips: joint_global_rots, verts/jcoords postprocess
        Only returns what intermediate layers need: pred_pose_raw, pred_cam, j3d, pred_keypoints_2d_cropped, pred_keypoints_2d_depth
        """
        batch_size = pose_token.shape[0]

        # === head_pose_hand (slim version) ===
        global_rot_6d, pred_pose_cont, j3d = \
            self.head_pose_hand._head_forward_core_slim(pose_token, prev_pose)

        # === head_camera_hand ===
        pred_cam = self.head_camera_hand.proj(pose_token)
        if prev_camera is not None:
            pred_cam = pred_cam + prev_camera

        # === perspective_projection (inline) ===
        s = -pred_cam[:, 0]
        tx = pred_cam[:, 1]
        ty = -pred_cam[:, 2]

        bs = bbox_scale * s * self.head_camera_hand.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (ori_img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (ori_img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
            cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        j3d_cam = j3d + pred_cam_t.unsqueeze(1)

        # perspective projection
        j2d = j3d_cam[..., :2] / j3d_cam[..., 2:3]
        j2d = j2d * cam_int[:, :2, :2].diagonal(dim1=1, dim2=2).unsqueeze(1) + cam_int[:, :2, 2].unsqueeze(1)
        pred_keypoints_2d = j2d.reshape(batch_size, -1, 2)
        pred_keypoints_2d_depth = j3d_cam.reshape(batch_size, -1, 3)[:, :, 2]

        # === full_to_crop ===
        pred_kps_cropped = torch.cat([pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1)
        pred_kps_cropped = pred_kps_cropped @ affine_trans.to(pred_kps_cropped).mT
        pred_kps_cropped = pred_kps_cropped[..., :2] / img_size.unsqueeze(1) - 0.5

        # Return the minimal output set needed by intermediate layers
        return (global_rot_6d, pred_pose_cont, pred_cam, j3d,
                pred_kps_cropped, pred_keypoints_2d_depth)

    def convert_decoder_dtype(self, dtype: torch.dtype):
        """
        Set Decoder autocast precision (without compiling).

        Args:
            dtype: Target precision, e.g. torch.float16, torch.bfloat16
        """
        print(f"[SAM3DBody] Setting decoder autocast dtype to {dtype}")
        if hasattr(self.decoder, 'convert_layers_dtype'):
            self.decoder.convert_layers_dtype(dtype)
        if hasattr(self.decoder_hand, 'convert_layers_dtype'):
            self.decoder_hand.convert_layers_dtype(dtype)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
        override_interm_interval: Optional[int] = None,
        override_interm_layers: Optional[set] = None,
    ):
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate

            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_mhr(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )  # (1, C, H, W)

            image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])

            # DEBUG: Check for NaN after ray_cond_emb
            import os
            if os.environ.get('DEBUG_NAN', '0') == '1':
                if torch.isnan(image_embeddings).any():
                    print(f"          [DEBUG forward_decoder] NaN after ray_cond_emb! shape={image_embeddings.shape}")

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280

            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024

        # Optimization: pre-compute batch parameters needed by camera_project (avoid repeated computation in IntermPred)
        _cached_bbox_center = self._flatten_person(batch["bbox_center"])[self.body_batch_idx]
        _cached_bbox_scale = self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0]
        _cached_ori_img_size = self._flatten_person(batch["ori_img_size"])[self.body_batch_idx]
        _cached_cam_int = self._flatten_person(
            batch["cam_int"]
            .unsqueeze(1)
            .expand(-1, batch["img"].shape[1], -1, -1)
            .contiguous()
        )[self.body_batch_idx]
        _cached_affine_trans = self._flatten_person(batch["affine_trans"])[self.body_batch_idx]
        _cached_img_size = self._flatten_person(batch["img_size"])[self.body_batch_idx]
        _use_intrin_center = self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False)

        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            # Timing control
            _INTERM_TIMING_COUNT[0] += 1
            do_timing = _INTERM_DETAIL_TIMING and _INTERM_TIMING_COUNT[0] > _INTERM_TIMING_WARMUP

            # Get the pose token
            pose_token = tokens[:, 0]
            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # ========== Use compiled version (if available) ==========
            use_compiled_interm = (
                self._interm_compiled
                and self._compiled_interm_pred_body is not None
                and not do_timing
            )

            if use_compiled_interm:
                # Determine if this is the last layer
                is_last_layer = (layer_idx == len(self.decoder.layers) - 1)

                # Intermediate layers use slim version (skip unnecessary computation)
                if not is_last_layer and self._use_slim_interm and self._compiled_interm_pred_body_slim is not None:
                    # Use slim version: only return output needed by intermediate layers
                    (global_rot_6d, pred_pose_cont, pred_cam, j3d,
                     pred_kps_cropped, pred_keypoints_2d_depth) = self._compiled_interm_pred_body_slim(
                        pose_token, prev_pose, prev_camera,
                        _cached_bbox_center, _cached_bbox_scale, _cached_ori_img_size,
                        _cached_cam_int, _cached_affine_trans, _cached_img_size,
                        _use_intrin_center,
                    )
                    # Slim intermediate layer output (only includes fields needed by keypoint_token_update)
                    pose_output = {
                        "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1),
                        "pred_cam": pred_cam,
                        "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
                        "pred_keypoints_2d_cropped": pred_kps_cropped,
                        "pred_keypoints_2d_depth": pred_keypoints_2d_depth,
                        # The following fields are not needed by intermediate layers, set to None
                        "pred_pose_rotmat": None,
                        "global_rot": None,
                        "body_pose": None,
                        "shape": None,
                        "scale": None,
                        "hand": None,
                        "face": None,
                        "pred_vertices": None,
                        "pred_joint_coords": None,
                        "faces": None,
                        "joint_global_rots": None,
                        "mhr_model_params": None,
                        "pred_keypoints_2d": None,
                        "pred_cam_t": None,
                        "focal_length": None,
                    }
                    return pose_output

                # Call the compiled full IntermPred core function (last layer or slim not enabled)
                (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
                 pred_shape, pred_scale, pred_hand, pred_face,
                 verts, j3d, jcoords, mhr_model_params, joint_global_rots,
                 pred_cam, pred_cam_t, focal_length, pred_keypoints_2d, pred_keypoints_2d_depth,
                 pred_kps_cropped) = self._compiled_interm_pred_body(
                    pose_token, prev_pose, prev_camera,
                    _cached_bbox_center, _cached_bbox_scale, _cached_ori_img_size,
                    _cached_cam_int, _cached_affine_trans, _cached_img_size,
                    _use_intrin_center,
                )

                if is_last_layer:
                    # Last layer: clone all returned tensors (avoid buffer reuse issues)
                    pose_output = {
                        "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1).clone(),
                        "pred_pose_rotmat": None,
                        "global_rot": global_rot_euler.clone(),
                        "body_pose": pred_pose_euler.clone(),
                        "shape": pred_shape.clone(),
                        "scale": pred_scale.clone(),
                        "hand": pred_hand.clone(),
                        "face": pred_face.clone(),
                        "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3).clone(),
                        "pred_vertices": verts.reshape(batch_size, -1, 3).clone() if verts is not None else None,
                        "pred_joint_coords": jcoords.reshape(batch_size, -1, 3).clone() if jcoords is not None else None,
                        "faces": self.head_pose._get_faces_numpy(),
                        "joint_global_rots": joint_global_rots.clone(),
                        "mhr_model_params": mhr_model_params.clone(),
                        "pred_cam": pred_cam.clone(),
                        "pred_keypoints_2d": pred_keypoints_2d.clone(),
                        "pred_cam_t": pred_cam_t.clone(),
                        "focal_length": focal_length.clone(),
                        "pred_keypoints_2d_depth": pred_keypoints_2d_depth.clone(),
                        "pred_keypoints_2d_cropped": pred_kps_cropped.clone(),
                    }
                else:
                    # Intermediate layer (slim not enabled): no clone (keypoint_token_update will clone what it needs)
                    pose_output = {
                        "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1),
                        "pred_pose_rotmat": None,
                        "global_rot": global_rot_euler,
                        "body_pose": pred_pose_euler,
                        "shape": pred_shape,
                        "scale": pred_scale,
                        "hand": pred_hand,
                        "face": pred_face,
                        "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
                        "pred_vertices": verts.reshape(batch_size, -1, 3) if verts is not None else None,
                        "pred_joint_coords": jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None,
                        "faces": self.head_pose._get_faces_numpy(),
                        "joint_global_rots": joint_global_rots,
                        "mhr_model_params": mhr_model_params,
                        "pred_cam": pred_cam,
                        "pred_keypoints_2d": pred_keypoints_2d,
                        "pred_cam_t": pred_cam_t,
                        "focal_length": focal_length,
                        "pred_keypoints_2d_depth": pred_keypoints_2d_depth,
                        "pred_keypoints_2d_cropped": pred_kps_cropped,
                    }
                return pose_output

            # ========== Original path (with timing or uncompiled) ==========
            if do_timing:
                t0 = _sync_time()

            # Get pose outputs (head_pose: MHRHead.forward)
            pose_output = self.head_pose(pose_token, prev_pose)

            if do_timing:
                t_head_pose = _sync_time()

            # Get Camera Translation (head_camera)
            if hasattr(self, "head_camera"):
                pred_cam = self.head_camera(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam

            if do_timing:
                t_head_camera = _sync_time()

            # Run camera projection (using pre-cached batch parameters)
            cam_out = self.head_camera.perspective_projection(
                pose_output["pred_keypoints_3d"],
                pose_output["pred_cam"],
                _cached_bbox_center,
                _cached_bbox_scale,
                _cached_ori_img_size,
                _cached_cam_int,
                use_intrin_center=_use_intrin_center,
            )
            pose_output.update(cam_out)

            if do_timing:
                t_camera_proj = _sync_time()

            # Get 2D KPS in crop (using pre-cached parameters)
            pred_kps = pose_output["pred_keypoints_2d"]
            pred_kps_cropped = torch.cat([pred_kps, torch.ones_like(pred_kps[:, :, [-1]])], dim=-1)
            pred_kps_cropped = pred_kps_cropped @ _cached_affine_trans.to(pred_kps_cropped).mT
            pred_kps_cropped = pred_kps_cropped[..., :2] / _cached_img_size.unsqueeze(1) - 0.5
            pose_output["pred_keypoints_2d_cropped"] = pred_kps_cropped

            if do_timing:
                t_full_to_crop = _sync_time()
                total = (t_full_to_crop - t0) * 1000
                print(f"[IntermPred body L{layer_idx}] "
                      f"head_pose: {(t_head_pose-t0)*1000:.2f}ms | "
                      f"head_camera: {(t_head_camera-t_head_pose)*1000:.2f}ms | "
                      f"camera_proj: {(t_camera_proj-t_head_camera)*1000:.2f}ms | "
                      f"full_to_crop: {(t_full_to_crop-t_camera_proj)*1000:.2f}ms | "
                      f"TOTAL: {total:.2f}ms")

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn

        # Combine the 2D and 3D functions
        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        # DEBUG: Check inputs to decoder
        import os
        _debug_nan = os.environ.get('DEBUG_NAN', '0') == '1'
        if _debug_nan:
            if torch.isnan(token_embeddings).any():
                print(f"          [DEBUG forward_decoder] NaN in token_embeddings before decoder! shape={token_embeddings.shape}")
            if torch.isnan(image_embeddings).any():
                print(f"          [DEBUG forward_decoder] NaN in image_embeddings before decoder! shape={image_embeddings.shape}")

        decoder_output = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
            decoder_name="body_decoder",
            override_interm_interval=override_interm_interval,
            override_interm_layers=override_interm_layers,
        )

        # Handle the case when DO_INTERM_PREDS=False
        if isinstance(decoder_output, tuple):
            pose_token, pose_output = decoder_output
        else:
            # When DO_INTERM_PREDS=False, decoder only returns pose_token
            pose_token = decoder_output
            # Need to manually call token_to_pose_output_fn to generate pose_output
            pose_output = token_to_pose_output_fn(pose_token, prev_pose_output=None, layer_idx=-1)

        # DEBUG: Check decoder output
        if _debug_nan:
            if torch.isnan(pose_token).any():
                print(f"          [DEBUG forward_decoder] NaN in pose_token after decoder! shape={pose_token.shape}")
            # Check pose_output if it's a dict
            if isinstance(pose_output, dict):
                for k, v in pose_output.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"          [DEBUG forward_decoder] NaN in pose_output['{k}'] after decoder! shape={v.shape}")
                        break  # Only print first NaN field

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    def forward_decoder_hand(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        """
        Args:
            image_embeddings: image features from the backbone, shape (B, C, H, W)
            init_estimate: initial estimate to be refined on, shape (B, 1, C)
            keypoints: optional prompt input, shape (B, N, 3),
                3 for coordinates (x,y) + label.
                (x, y) should be normalized to range [0, 1].
                label==-1 indicates incorrect points,
                label==-2 indicates invalid points
            prev_estimate: optional prompt input, shape (B, 1, C),
                previous estimate for pose refinement.
            condition_info: optional condition information that is concatenated with
                the input tokens, shape (B, c)
        """
        batch_size = image_embeddings.shape[0]

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose_hand.weight.expand(batch_size, -1).unsqueeze(
                dim=1
            )
            if hasattr(self, "init_camera_hand"):
                init_camera = self.init_camera_hand.weight.expand(
                    batch_size, -1
                ).unsqueeze(dim=1)

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera_hand")
                else torch.cat([init_pose, init_camera], dim=-1)
            )  # This is basically pose & camera translation at the end. B x 1 x (404 + 3)

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )  # B x 1 x 410 (this is with the CLIFF condition)
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr_hand(init_input).view(
            batch_size, 1, -1
        )  # B x 1 x 1024 (linear layered)

        num_pose_token = token_embeddings.shape[1]

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                # Use initial embedding if no previous embedding
                prev_estimate = init_estimate
            # Previous estimate w/o the CLIFF condition.
            prev_embeddings = self.prev_to_token_mhr_hand(prev_estimate).view(
                batch_size, 1, -1
            )  # 407 -> B x 1 x 1024; linear layer-ed

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.hand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                # ViT backbone assumes a different aspect ratio as input size
                image_augment = self.hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]
            else:
                image_augment = self.hand_pe_layer(
                    image_embeddings.shape[-2:]
                ).unsqueeze(
                    0
                )  # (1, C, H, W)

            image_embeddings = self.ray_cond_emb_hand(
                image_embeddings, batch["ray_cond_hand"]
            )

            # DEBUG: Check for NaN after ray_cond_emb_hand
            import os
            if os.environ.get('DEBUG_NAN', '0') == '1':
                if torch.isnan(image_embeddings).any():
                    print(f"          [DEBUG forward_decoder_hand] NaN after ray_cond_emb_hand! shape={image_embeddings.shape}")

            # To start, keypoints is all [0, 0, -2]. The points get sent into self.pe_layer._pe_encoding,
            # the labels determine the embedding weight (special one for -2, -1, then each of joint.)
            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )  # B x 1 x 1280
            prompt_embeddings = self.prompt_to_token(
                prompt_embeddings
            )  # Linear layered: B x 1 x 1024

            # Concatenate pose tokens and prompt embeddings as decoder input
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                # Put in a token for each hand
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 5 + 70 x 1024

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            # Put in a token for each keypoint
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding_hand.weight[None, :, :].repeat(
                        batch_size, 1, 1
                    ),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024
            # No positional embeddings
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )  # B x 3 + 70 x 1024

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                # Put in a token for each keypoint
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding_hand.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024
                # No positional embeddings
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )  # B x 3 + 70 + 70 x 1024

        # Optimization: pre-compute batch parameters needed by camera_project_hand (avoid repeated computation in IntermPred)
        _cached_hand_bbox_center = self._flatten_person(batch["bbox_center"])[self.hand_batch_idx]
        _cached_hand_bbox_scale = self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0]
        _cached_hand_ori_img_size = self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx]
        _cached_hand_cam_int = self._flatten_person(
            batch["cam_int"].unsqueeze(1).expand(-1, batch["img"].shape[1], -1, -1).contiguous()
        )[self.hand_batch_idx]
        _cached_hand_affine_trans = self._flatten_person(batch["affine_trans"])[self.hand_batch_idx]
        _cached_hand_img_size = self._flatten_person(batch["img_size"])[self.hand_batch_idx]
        _use_intrin_center_hand = self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False)

        # We're doing intermediate model predictions
        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            # Timing control
            _INTERM_TIMING_COUNT[0] += 1
            do_timing = _INTERM_DETAIL_TIMING and _INTERM_TIMING_COUNT[0] > _INTERM_TIMING_WARMUP

            # Get the pose token
            pose_token = tokens[:, 0]
            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            # ========== Use compiled version (if available) ==========
            use_compiled_interm = (
                self._interm_compiled
                and self._compiled_interm_pred_hand is not None
                and not do_timing
            )

            if use_compiled_interm:
                # Determine if this is the last layer
                is_last_layer = (layer_idx == len(self.decoder_hand.layers) - 1)

                # Intermediate layers use slim version (skip unnecessary computation)
                if not is_last_layer and self._use_slim_interm and self._compiled_interm_pred_hand_slim is not None:
                    # Use slim version: only return output needed by intermediate layers
                    (global_rot_6d, pred_pose_cont, pred_cam, j3d,
                     pred_kps_cropped, pred_keypoints_2d_depth) = self._compiled_interm_pred_hand_slim(
                        pose_token, prev_pose, prev_camera,
                        _cached_hand_bbox_center, _cached_hand_bbox_scale, _cached_hand_ori_img_size,
                        _cached_hand_cam_int, _cached_hand_affine_trans, _cached_hand_img_size,
                        _use_intrin_center_hand,
                    )
                    # Slim intermediate layer output (only includes fields needed by keypoint_token_update)
                    pose_output = {
                        "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1),
                        "pred_cam": pred_cam,
                        "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
                        "pred_keypoints_2d_cropped": pred_kps_cropped,
                        "pred_keypoints_2d_depth": pred_keypoints_2d_depth,
                        # The following fields are not needed by intermediate layers, set to None
                        "pred_pose_rotmat": None,
                        "global_rot": None,
                        "body_pose": None,
                        "shape": None,
                        "scale": None,
                        "hand": None,
                        "face": None,
                        "pred_vertices": None,
                        "pred_joint_coords": None,
                        "faces": None,
                        "joint_global_rots": None,
                        "mhr_model_params": None,
                        "pred_keypoints_2d": None,
                        "pred_cam_t": None,
                        "focal_length": None,
                    }
                    return pose_output

                # Call the compiled full IntermPred core function (last layer or slim not enabled)
                (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
                 pred_shape, pred_scale, pred_hand, pred_face,
                 verts, j3d, jcoords, mhr_model_params, joint_global_rots,
                 pred_cam, pred_cam_t, focal_length, pred_keypoints_2d, pred_keypoints_2d_depth,
                 pred_kps_cropped) = self._compiled_interm_pred_hand(
                    pose_token, prev_pose, prev_camera,
                    _cached_hand_bbox_center, _cached_hand_bbox_scale, _cached_hand_ori_img_size,
                    _cached_hand_cam_int, _cached_hand_affine_trans, _cached_hand_img_size,
                    _use_intrin_center_hand,
                )

                if is_last_layer:
                    # Last layer: clone all returned tensors (avoid buffer reuse issues)
                    pose_output = {
                        "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1).clone(),
                        "pred_pose_rotmat": None,
                        "global_rot": global_rot_euler.clone(),
                        "body_pose": pred_pose_euler.clone(),
                        "shape": pred_shape.clone(),
                        "scale": pred_scale.clone(),
                        "hand": pred_hand.clone(),
                        "face": pred_face.clone(),
                        "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3).clone(),
                        "pred_vertices": verts.reshape(batch_size, -1, 3).clone() if verts is not None else None,
                        "pred_joint_coords": jcoords.reshape(batch_size, -1, 3).clone() if jcoords is not None else None,
                        "faces": self.head_pose_hand._get_faces_numpy(),
                        "joint_global_rots": joint_global_rots.clone(),
                        "mhr_model_params": mhr_model_params.clone(),
                        "pred_cam": pred_cam.clone(),
                        "pred_keypoints_2d": pred_keypoints_2d.clone(),
                        "pred_cam_t": pred_cam_t.clone(),
                        "focal_length": focal_length.clone(),
                        "pred_keypoints_2d_depth": pred_keypoints_2d_depth.clone(),
                        "pred_keypoints_2d_cropped": pred_kps_cropped.clone(),
                    }
                else:
                    # Intermediate layer (slim not enabled): no clone (keypoint_token_update will clone what it needs)
                    pose_output = {
                        "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1),
                        "pred_pose_rotmat": None,
                        "global_rot": global_rot_euler,
                        "body_pose": pred_pose_euler,
                        "shape": pred_shape,
                        "scale": pred_scale,
                        "hand": pred_hand,
                        "face": pred_face,
                        "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
                        "pred_vertices": verts.reshape(batch_size, -1, 3) if verts is not None else None,
                        "pred_joint_coords": jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None,
                        "faces": self.head_pose_hand._get_faces_numpy(),
                        "joint_global_rots": joint_global_rots,
                        "mhr_model_params": mhr_model_params,
                        "pred_cam": pred_cam,
                        "pred_keypoints_2d": pred_keypoints_2d,
                        "pred_cam_t": pred_cam_t,
                        "focal_length": focal_length,
                        "pred_keypoints_2d_depth": pred_keypoints_2d_depth,
                        "pred_keypoints_2d_cropped": pred_kps_cropped,
                    }
                return pose_output

            # ========== Original path (with timing or uncompiled) ==========
            if do_timing:
                t0 = _sync_time()

            # Get pose outputs (head_pose_hand: MHRHead.forward)
            pose_output = self.head_pose_hand(pose_token, prev_pose)

            if do_timing:
                t_head_pose = _sync_time()

            # Get Camera Translation (head_camera_hand)
            if hasattr(self, "head_camera_hand"):
                pred_cam = self.head_camera_hand(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam

            if do_timing:
                t_head_camera = _sync_time()

            # Run camera projection (using pre-cached batch parameters)
            cam_out = self.head_camera_hand.perspective_projection(
                pose_output["pred_keypoints_3d"],
                pose_output["pred_cam"],
                _cached_hand_bbox_center,
                _cached_hand_bbox_scale,
                _cached_hand_ori_img_size,
                _cached_hand_cam_int,
                use_intrin_center=_use_intrin_center_hand,
            )
            pose_output.update(cam_out)

            if do_timing:
                t_camera_proj = _sync_time()

            # Get 2D KPS in crop (using pre-cached parameters)
            pred_kps = pose_output["pred_keypoints_2d"]
            pred_kps_cropped = torch.cat([pred_kps, torch.ones_like(pred_kps[:, :, [-1]])], dim=-1)
            pred_kps_cropped = pred_kps_cropped @ _cached_hand_affine_trans.to(pred_kps_cropped).mT
            pred_kps_cropped = pred_kps_cropped[..., :2] / _cached_hand_img_size.unsqueeze(1) - 0.5
            pose_output["pred_keypoints_2d_cropped"] = pred_kps_cropped

            if do_timing:
                t_full_to_crop = _sync_time()
                total = (t_full_to_crop - t0) * 1000
                print(f"[IntermPred hand L{layer_idx}] "
                      f"head_pose: {(t_head_pose-t0)*1000:.2f}ms | "
                      f"head_camera: {(t_head_camera-t_head_pose)*1000:.2f}ms | "
                      f"camera_proj: {(t_camera_proj-t_head_camera)*1000:.2f}ms | "
                      f"full_to_crop: {(t_full_to_crop-t_camera_proj)*1000:.2f}ms | "
                      f"TOTAL: {total:.2f}ms")

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn_hand

        # Now for 3D
        kp3d_token_update_fn = self.keypoint3d_token_update_fn_hand

        # Combine the 2D and 3D functions
        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        # DEBUG: Check inputs to hand decoder
        import os
        _debug_nan_hand = os.environ.get('DEBUG_NAN', '0') == '1'
        if _debug_nan_hand:
            if torch.isnan(token_embeddings).any():
                print(f"          [DEBUG forward_decoder_hand] NaN in token_embeddings before decoder! shape={token_embeddings.shape}")
            if torch.isnan(image_embeddings).any():
                print(f"          [DEBUG forward_decoder_hand] NaN in image_embeddings before decoder! shape={image_embeddings.shape}")

        decoder_output = self.decoder_hand(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
            decoder_name="hand_decoder",
        )

        # Handle the case when DO_INTERM_PREDS=False
        if isinstance(decoder_output, tuple):
            pose_token, pose_output = decoder_output
        else:
            # When DO_INTERM_PREDS=False, decoder only returns pose_token
            pose_token = decoder_output
            # Need to manually call token_to_pose_output_fn to generate pose_output
            pose_output = token_to_pose_output_fn(pose_token, prev_pose_output=None, layer_idx=-1)

        # DEBUG: Check hand decoder output
        if _debug_nan_hand:
            if torch.isnan(pose_token).any():
                print(f"          [DEBUG forward_decoder_hand] NaN in pose_token after decoder! shape={pose_token.shape}")
            if isinstance(pose_output, dict):
                for k, v in pose_output.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"          [DEBUG forward_decoder_hand] NaN in pose_output['{k}'] after decoder! shape={v.shape}")
                        break  # Only print first NaN field

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    def _forward_decoders_combined(
        self,
        body_embeddings: torch.Tensor,
        body_keypoints: torch.Tensor,
        body_condition: torch.Tensor,
        hand_embeddings: torch.Tensor,
        hand_keypoints: torch.Tensor,
        hand_condition: torch.Tensor,
        batch,
    ):
        """
        Combined forward pass for both body and hand decoders.
        This allows torch.compile to optimize across both decoder calls.
        """
        import os
        _DEBUG_NAN_DECODERS = os.environ.get('DEBUG_NAN', '1') == '1'

        # DEBUG: Check inputs for NaN
        if _DEBUG_NAN_DECODERS:
            print(f"          [DEBUG _forward_decoders_combined] body_embeddings shape={body_embeddings.shape}, has_nan={torch.isnan(body_embeddings).any().item()}")
            print(f"          [DEBUG _forward_decoders_combined] body_condition shape={body_condition.shape}, has_nan={torch.isnan(body_condition).any().item()}")
            print(f"          [DEBUG _forward_decoders_combined] hand_embeddings shape={hand_embeddings.shape}, has_nan={torch.isnan(hand_embeddings).any().item()}")
            print(f"          [DEBUG _forward_decoders_combined] hand_condition shape={hand_condition.shape}, has_nan={torch.isnan(hand_condition).any().item()}")

        # Mark step begin to avoid CUDA graph tensor reuse conflicts
        torch.compiler.cudagraph_mark_step_begin()

        # Run body decoder
        tokens_output_body, pose_output_body = self.forward_decoder(
            body_embeddings,
            init_estimate=None,
            keypoints=body_keypoints,
            prev_estimate=None,
            condition_info=body_condition,
            batch=batch,
        )

        # DEBUG: Check body decoder output for NaN
        if _DEBUG_NAN_DECODERS:
            if isinstance(pose_output_body, list):
                pose_out = pose_output_body[-1] if pose_output_body else None
            else:
                pose_out = pose_output_body
            if pose_out is not None:
                for k, v in pose_out.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"          [DEBUG _forward_decoders_combined] BODY decoder output '{k}': contains NaN! shape={v.shape}")

        # Mark step begin before hand decoder to avoid CUDA graph tensor reuse conflicts
        torch.compiler.cudagraph_mark_step_begin()

        # Run hand decoder
        tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
            hand_embeddings,
            init_estimate=None,
            keypoints=hand_keypoints,
            prev_estimate=None,
            condition_info=hand_condition,
            batch=batch,
        )

        # DEBUG: Check hand decoder output for NaN
        if _DEBUG_NAN_DECODERS:
            if isinstance(pose_output_hand, list):
                pose_out = pose_output_hand[-1] if pose_output_hand else None
            else:
                pose_out = pose_output_hand
            if pose_out is not None:
                for k, v in pose_out.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"          [DEBUG _forward_decoders_combined] HAND decoder output '{k}': contains NaN! shape={v.shape}")

        return tokens_output_body, pose_output_body, tokens_output_hand, pose_output_hand

    @torch.no_grad()
    def _get_keypoint_prompt(self, batch, pred_keypoints_2d, force_dummy=False):
        if self.camera_type == "perspective":
            pred_keypoints_2d = self._full_to_crop(batch, pred_keypoints_2d)

        gt_keypoints_2d = self._flatten_person(batch["keypoints_2d"]).clone()

        keypoint_prompt = self.keypoint_prompt_sampler.sample(
            gt_keypoints_2d,
            pred_keypoints_2d,
            is_train=self.training,
            force_dummy=force_dummy,
        )
        return keypoint_prompt

    def _get_mask_prompt(self, batch, image_embeddings):
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
        ]:
            # ViT backbone assumes a different aspect ratio as input size
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            # for x2 resolution
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _one_prompt_iter(self, batch, output, prev_prompt, full_output):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]

        if "mhr" in output and output["mhr"] is not None:
            pose_output = output["mhr"]  # body-only output
            # Use previous estimate as initialization
            prev_estimate = torch.cat(
                [
                    pose_output["pred_pose_raw"].detach(),  # (B, 6)
                    pose_output["shape"].detach(),
                    pose_output["scale"].detach(),
                    pose_output["hand"].detach(),
                    pose_output["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                prev_estimate = torch.cat(
                    [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                    dim=-1,
                )
            prev_shape = prev_estimate.shape[1:]

            pred_keypoints_2d = output["mhr"]["pred_keypoints_2d"].detach().clone()
            kpt_shape = pred_keypoints_2d.shape[1:]

        if "mhr_hand" in output and output["mhr_hand"] is not None:
            pose_output_hand = output["mhr_hand"]
            # Use previous estimate as initialization
            prev_estimate_hand = torch.cat(
                [
                    pose_output_hand["pred_pose_raw"].detach(),  # (B, 6)
                    pose_output_hand["shape"].detach(),
                    pose_output_hand["scale"].detach(),
                    pose_output_hand["hand"].detach(),
                    pose_output_hand["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera_hand"):
                prev_estimate_hand = torch.cat(
                    [
                        prev_estimate_hand,
                        pose_output_hand["pred_cam"].detach().unsqueeze(1),
                    ],
                    dim=-1,
                )
            prev_shape = prev_estimate_hand.shape[1:]

            pred_keypoints_2d_hand = (
                output["mhr_hand"]["pred_keypoints_2d"].detach().clone()
            )
            kpt_shape = pred_keypoints_2d_hand.shape[1:]

        all_prev_estimate = torch.zeros(
            (image_embeddings.shape[0], *prev_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_prev_estimate[self.body_batch_idx] = prev_estimate
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_prev_estimate[self.hand_batch_idx] = prev_estimate_hand

        # Get keypoint prompts
        all_pred_keypoints_2d = torch.zeros(
            (image_embeddings.shape[0], *kpt_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_pred_keypoints_2d[self.body_batch_idx] = pred_keypoints_2d
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_pred_keypoints_2d[self.hand_batch_idx] = pred_keypoints_2d_hand

        keypoint_prompt = self._get_keypoint_prompt(batch, all_pred_keypoints_2d)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt  # [B, 1, 3]

        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            # Mark step begin to avoid CUDA graph tensor reuse conflicts
            torch.compiler.cudagraph_mark_step_begin()
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,  # not recurring previous estimate
                keypoints=cur_keypoint_prompt[self.body_batch_idx],
                prev_estimate=all_prev_estimate[self.body_batch_idx],
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
                full_output=None,
            )
            pose_output = pose_output[-1]

        # Update prediction output
        output.update(
            {
                "mhr": pose_output,
                "mhr_hand": pose_output_hand,
            }
        )

        return output, keypoint_prompt

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5. 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera"):
            head_camera = self.head_camera
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.body_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def camera_project_hand(self, pose_output: Dict, batch: Dict) -> Dict:
        """
        Project 3D keypoints to 2D using the camera parameters.
        Args:
            pose_output (Dict): Dictionary containing the pose output.
            batch (Dict): Dictionary containing the batch data.
        Returns:
            Dict: Dictionary containing the projected 2D keypoints.
        """
        if hasattr(self, "head_camera_hand"):
            head_camera = self.head_camera_hand
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.hand_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.hand_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(B, N, 1, 1, 1)
            .cuda()
        )  # B x N x H x W x 2
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )

        # Subtract out center & normalize to be rays
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(
            batch["img"].dtype
        )  # This is B x num_person x 2 x H x W

    def forward_pose_branch(self, batch: Dict) -> Dict:
        """Run a forward pass for the crop-image (pose) branch."""
        forward_total_start = time.time()
        batch_size, num_person = batch["img"].shape[:2]

        # Forward backbone encoder
        t0 = time.time()
        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                ]
            ),
        )
        print(f"          [forward_pose_branch] data_preprocess: {time.time() - t0:.4f}s")

        # Debug: save preprocessed images (before entering backbone)
        _DEBUG_BACKBONE_INPUT = os.environ.get("DEBUG_BACKBONE_INPUT", "0") == "1"
        if _DEBUG_BACKBONE_INPUT:
            import cv2
            suffix = "gpu" if os.environ.get("GPU_HAND_PREP", "1") == "1" else "cpu"
            # x shape: (B, 3, H, W), already normalized by data_preprocess
            for i in range(min(x.shape[0], 3)):  # Save at most 3
                img_t = x[i].clone()  # (3, H, W)
                # Denormalize: x_orig = x * std + mean
                image_mean = self.image_mean.squeeze()  # (3,)
                image_std = self.image_std.squeeze()  # (3,)
                img_t = img_t * image_std.view(3, 1, 1) + image_mean.view(3, 1, 1)
                # Convert to 0-255 range
                img_t = (img_t * 255).clamp(0, 255).byte()
                img_np = img_t.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"./output/debug_backbone_input_{i}_{suffix}.jpg", img_np)
                print(f"          [DEBUG] Saved backbone input {i}: ./output/debug_backbone_input_{i}_{suffix}.jpg")
                print(f"          [DEBUG] backbone_input_{i} stats: min={x[i].min().item():.4f}, max={x[i].max().item():.4f}, mean={x[i].mean().item():.4f}")

        # Optionally get ray conditioining
        t0 = time.time()
        ray_cond = self.get_ray_condition(batch)  # This is B x num_person x 2 x H x W
        ray_cond = self._flatten_person(ray_cond)
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
        ]:
            ray_cond = ray_cond[:, :, :, 32:-32]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            ray_cond = ray_cond[:, :, :, 64:-64]

        if len(self.body_batch_idx):
            batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
        if len(self.hand_batch_idx):
            batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
        ray_cond = None
        print(f"          [forward_pose_branch] ray_condition: {time.time() - t0:.4f}s")

        t0 = time.time()
        torch.cuda.synchronize()

        print(f"          [DEBUG] FOV input image size: {x.shape[2]}x{x.shape[3]} (H x W)")
        print(f"          [DEBUG] Model: SAM3DBody-Backbone ({self.cfg.MODEL.BACKBONE.TYPE}), input_dtype: {x.dtype}, compute_dtype: {self.backbone_dtype}")
        print(f"          [DEBUG] Backbone input shape: {x.shape} (B, C, H, W)")
        image_embeddings = self.backbone(
            x.type(self.backbone_dtype), extra_embed=ray_cond
        )  # (B, C, H, W)
        torch.cuda.synchronize()
        print(f"          [forward_pose_branch] backbone: {time.time() - t0:.4f}s")

        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)
        # image_embeddings: (B, C, H, W) -> flatten to (B, H*W, C) for decoder
        B, C, H, W = image_embeddings.shape
        print(f"          [DEBUG] Backbone output shape: {image_embeddings.shape} (B, C, H, W) -> {H*W} image tokens")

        # Mask condition if available
        t0 = time.time()
        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            # v1: non-iterative mask conditioning
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError
        print(f"          [forward_pose_branch] mask_condition: {time.time() - t0:.4f}s")

        # Prepare input for promptable decoder
        t0 = time.time()
        condition_info = self._get_decoder_condition(batch)
        print(f"          [forward_pose_branch] decoder_condition: {time.time() - t0:.4f}s")

        # Initial estimate with a dummy prompt
        keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
        keypoints_prompt[:, :, -1] = -2

        # Forward promptable decoder to get updated pose tokens and regression output
        pose_output, pose_output_hand = None, None
        tokens_output, tokens_output_hand = None, None

        # Check if both decoders are needed
        need_body = len(self.body_batch_idx) > 0
        need_hand = len(self.hand_batch_idx) > 0

        if need_body and need_hand:
            # ============================================================
            # COMBINED DECODER EXECUTION - both body and hand in one call
            # ============================================================
            t0 = time.time()
            torch.cuda.synchronize()

            # Run combined forward (can be torch.compiled for optimization)
            tokens_output, pose_output, tokens_output_hand, pose_output_hand = \
                self._forward_decoders_combined(
                    image_embeddings[self.body_batch_idx],
                    keypoints_prompt[self.body_batch_idx],
                    condition_info[self.body_batch_idx],
                    image_embeddings[self.hand_batch_idx],
                    keypoints_prompt[self.hand_batch_idx],
                    condition_info[self.hand_batch_idx],
                    batch,
                )

            torch.cuda.synchronize()
            print(f"          [forward_pose_branch] forward_decoders_combined (body+hand): {time.time() - t0:.4f}s")

            # Handle list outputs
            if isinstance(pose_output, list):
                pose_output = pose_output[-1]
            if isinstance(pose_output_hand, list):
                pose_output_hand = pose_output_hand[-1]

        elif need_body:
            # Only body decoder needed
            t0 = time.time()
            torch.cuda.synchronize()
            # Mark step begin to avoid CUDA graph tensor reuse conflicts
            torch.compiler.cudagraph_mark_step_begin()
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            torch.cuda.synchronize()
            print(f"          [forward_pose_branch] forward_decoder_body: {time.time() - t0:.4f}s")
            # When DO_INTERM_PREDS=True, pose_output is a list, take the last one
            # When DO_INTERM_PREDS=False, pose_output is already a dict
            if isinstance(pose_output, list):
                pose_output = pose_output[-1]

        elif need_hand:
            # Only hand decoder needed
            t0 = time.time()
            torch.cuda.synchronize()
            # Mark step begin to avoid CUDA graph tensor reuse conflicts
            torch.compiler.cudagraph_mark_step_begin()
            tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
                image_embeddings[self.hand_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.hand_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
            )
            torch.cuda.synchronize()
            print(f"          [forward_pose_branch] forward_decoder_hand: {time.time() - t0:.4f}s")
            # When DO_INTERM_PREDS=True, pose_output_hand is a list, take the last one
            # When DO_INTERM_PREDS=False, pose_output_hand is already a dict
            if isinstance(pose_output_hand, list):
                pose_output_hand = pose_output_hand[-1]

        output = {
            # "pose_token": pose_token,
            "mhr": pose_output,  # mhr prediction output
            "mhr_hand": pose_output_hand,  # mhr prediction output
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }
        print(f"          [forward_pose_branch] TOTAL: {time.time() - forward_total_start:.4f}s")

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            if len(self.body_batch_idx):
                output_hand_box_tokens = tokens_output
                hand_coords = self.bbox_embed(
                    output_hand_box_tokens
                ).sigmoid()  # x1, y1, w, h for body samples, 0 ~ 1
                hand_logits = self.hand_cls_embed(output_hand_box_tokens)

                output["mhr"]["hand_box"] = hand_coords
                output["mhr"]["hand_logits"] = hand_logits

            if len(self.hand_batch_idx):
                output_hand_box_tokens_hand_batch = tokens_output_hand

                hand_coords_hand_batch = self.bbox_embed(
                    output_hand_box_tokens_hand_batch
                ).sigmoid()  # x1, y1, w, h for hand samples
                hand_logits_hand_batch = self.hand_cls_embed(
                    output_hand_box_tokens_hand_batch
                )

                output["mhr_hand"]["hand_box"] = hand_coords_hand_batch
                output["mhr_hand"]["hand_logits"] = hand_logits_hand_batch

        return output

    def forward_step(
        self, batch: Dict, decoder_type: str = "body"
    ) -> Tuple[Dict, Dict]:
        batch_size, num_person = batch["img"].shape[:2]

        if decoder_type == "body":
            self.hand_batch_idx = []
            self.body_batch_idx = list(range(batch_size * num_person))
        elif decoder_type == "hand":
            self.hand_batch_idx = list(range(batch_size * num_person))
            self.body_batch_idx = []
        else:
            ValueError("Invalid decoder type: ", decoder_type)

        # Crop-image (pose) branch
        pose_output = self.forward_pose_branch(batch)

        return pose_output

    def forward_step_merged(
        self, batch: Dict, body_batch_idx: list, hand_batch_idx: list
    ) -> Dict:
        """Run forward pass with explicit body/hand batch indices.

        This is used for merged batch inference where body and hand crops
        are combined into a single batch for shared backbone execution.

        Args:
            batch: Combined batch with shape [1, N, ...] where N = body + hands
            body_batch_idx: Indices for body crops (e.g., [0] for first person)
            hand_batch_idx: Indices for hand crops (e.g., [1, 2] for left/right hands)

        Returns:
            Output dict with both 'mhr' (body) and 'mhr_hand' (hand) outputs
        """
        # Set the batch indices for routing in forward_pose_branch
        self.body_batch_idx = body_batch_idx
        self.hand_batch_idx = hand_batch_idx

        # Run forward with both body and hand decoders
        pose_output = self.forward_pose_branch(batch)

        return pose_output

    def run_inference(
        self,
        img,
        batch: Dict,
        inference_type: str = "full",
        transform_hand: Any = None,
        thresh_wrist_angle=1.4,
        hand_box_source: str = "body_decoder",  # "body_decoder" or "yolo_pose"
        yolo_pose_keypoints: Optional[np.ndarray] = None,  # YOLO-Pose keypoints [N, 17, 3]
        yolo_pose_body_boxes: Optional[np.ndarray] = None,  # YOLO-Pose body boxes [N, 4]
        parallel_decoders: bool = True,  # Enable merged batch execution for shared backbone
    ):
        """
        Run 3DB inference (optionally with hand detector).

        inference_type:
            - full: full-body inference with both body and hand decoders
            - body: inference with body decoder only (still full-body output)
            - hand: inference with hand decoder only (only hand output)

        hand_box_source:
            - body_decoder: use hand boxes from body decoder output (default)
            - yolo_pose: use hand boxes computed from YOLO-Pose wrist keypoints

        yolo_pose_keypoints:
            - YOLO-Pose keypoints array with shape [N, 17, 3] (x, y, conf)
            - Required when hand_box_source="yolo_pose"

        yolo_pose_body_boxes:
            - YOLO-Pose body boxes array with shape [N, 4] (x1, y1, x2, y2)
            - Required when hand_box_source="yolo_pose" for computing hand box size
        """
        run_inference_start = time.time()
        print("        [run_inference] Starting...")

        # DEBUG flag for NaN tracing
        _DEBUG_NAN = os.environ.get('DEBUG_NAN', '1') == '1'

        height, width = img.shape[:2]
        cam_int = batch["cam_int"].clone()

        if inference_type == "body":
            t0 = time.time()
            torch.cuda.synchronize()
            pose_output = self.forward_step(batch, decoder_type="body")
            torch.cuda.synchronize()
            print(f"          [run_inference] forward_step_body: {time.time() - t0:.4f}s")
            print(f"        [run_inference] TOTAL: {time.time() - run_inference_start:.4f}s")
            return pose_output
        elif inference_type == "hand":
            t0 = time.time()
            torch.cuda.synchronize()
            pose_output = self.forward_step(batch, decoder_type="hand")
            torch.cuda.synchronize()
            print(f"          [run_inference] forward_step_hand: {time.time() - t0:.4f}s")
            print(f"        [run_inference] TOTAL: {time.time() - run_inference_start:.4f}s")
            return pose_output
        elif not inference_type == "full":
            ValueError("Invalid inference type: ", inference_type)

        # Check if we can use parallel execution (merged backbone)
        # Parallel mode requires: yolo_pose hand boxes (no dependency on body decoder output)
        # Can be disabled via environment variable PARALLEL_DECODERS=0
        env_parallel = os.environ.get('PARALLEL_DECODERS', '1') != '0'
        use_parallel = (
            parallel_decoders
            and env_parallel
            and hand_box_source == "yolo_pose"
            and yolo_pose_keypoints is not None
        )

        if use_parallel:
            # ============================================================
            # MERGED BATCH EXECUTION PATH (Shared Backbone)
            # Combine body + hand crops into single batch, run backbone once
            # ============================================================
            print("          [run_inference] Using MERGED BATCH execution (shared backbone)")
            t0 = time.time()

            # Step 0: Get hand boxes from YOLO-Pose FIRST (before body decoder)
            left_xyxy, right_xyxy = self._get_hand_box_from_yolo_pose(
                yolo_pose_keypoints, yolo_pose_body_boxes, batch
            )

            # Prepare hand batches using GPU-accelerated version (~10x faster than CPU version)
            _DEBUG_HAND_PREP = os.environ.get("DEBUG_HAND_PREP", "0") == "1"

            if _USE_GPU_HAND_PREP:
                # GPU version: perform flip + crop + normalize on GPU
                output_size = (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0])  # (H, W)
                batch_lhand, batch_rhand, left_xyxy_flipped = _prepare_hand_batches_gpu(
                    img, left_xyxy, right_xyxy, cam_int,
                    output_size=output_size,
                    padding=0.9,  # Consistent with GetBBoxCenterScale(padding=0.9)
                    device='cuda'
                )
            else:
                # CPU version (fallback)
                flipped_img = img[:, ::-1]
                tmp = left_xyxy.copy()
                left_xyxy_flipped = left_xyxy.copy()
                left_xyxy_flipped[:, 0] = width - tmp[:, 2] - 1
                left_xyxy_flipped[:, 2] = width - tmp[:, 0] - 1

                batch_lhand = prepare_batch(
                    flipped_img, transform_hand, left_xyxy_flipped, cam_int=cam_int.clone()
                )
                batch_lhand = recursive_to(batch_lhand, "cuda")

                batch_rhand = prepare_batch(
                    img, transform_hand, right_xyxy, cam_int=cam_int.clone()
                )
                batch_rhand = recursive_to(batch_rhand, "cuda")

            # Debug: save cropped images for comparison
            if _DEBUG_HAND_PREP:
                import cv2
                suffix = "gpu" if _USE_GPU_HAND_PREP else "cpu"

                def save_debug_img(batch, name):
                    img_t = batch['img'][0, 0]  # (3, H, W)
                    # Print detailed tensor statistics
                    print(f"          [DEBUG] {name} img stats: min={img_t.min().item():.4f}, max={img_t.max().item():.4f}, mean={img_t.mean().item():.4f}")
                    print(f"          [DEBUG] {name} img shape: {img_t.shape}")
                    print(f"          [DEBUG] {name} bbox_center: {batch['bbox_center'].squeeze().tolist()}")
                    print(f"          [DEBUG] {name} bbox_scale: {batch['bbox_scale'].squeeze().tolist()}")
                    print(f"          [DEBUG] {name} affine_trans:\n{batch['affine_trans'].squeeze().cpu().numpy()}")
                    # Print image corner pixel values for comparison
                    print(f"          [DEBUG] {name} img[0,:3,:3]:\n{img_t[0,:3,:3].cpu().numpy()}")
                    # Save image
                    img_save = (img_t * 255).clamp(0, 255).byte()
                    img_np = img_save.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"./output/debug_{name}_{suffix}.jpg", img_np)
                    print(f"          [DEBUG] Saved ./output/debug_{name}_{suffix}.jpg")

                save_debug_img(batch_lhand, "lhand")
                save_debug_img(batch_rhand, "rhand")

            # Merge left and right hand batches: [1, 2, ...]
            batch_hands = self._merge_hand_batches(batch_lhand, batch_rhand)
            print(f"          [run_inference] prepare_hand_batches: {time.time() - t0:.4f}s")

            # Merge body batch with hand batches: [1, 3*N, ...]
            # Combined batch order:
            #   - indices 0 to N-1: body crops
            #   - indices N to 2*N-1: left hand crops
            #   - indices 2*N to 3*N-1: right hand crops
            t0 = time.time()
            combined_batch = self._merge_body_hand_batches(batch, batch_hands)
            self._initialize_batch(combined_batch)
            print(f"          [run_inference] merge_body_hand_batches: {time.time() - t0:.4f}s")

            # Get number of persons from batch
            num_persons = batch["img"].shape[1]

            # Run single forward pass with shared backbone
            # body_batch_idx routes body crops to body decoder
            # hand_batch_idx routes hand crops to hand decoder
            t_merged = time.time()
            torch.cuda.synchronize()
            combined_output = self.forward_step_merged(
                combined_batch,
                body_batch_idx=list(range(num_persons)),
                hand_batch_idx=list(range(num_persons, 3 * num_persons))
            )
            torch.cuda.synchronize()
            print(f"          [run_inference] merged_forward (backbone + body_decoder + hand_decoder): {time.time() - t_merged:.4f}s")

            # Extract body output from combined output (include image_embeddings and condition_info for post-processing)
            # Note: image_embeddings has shape [3*N, C, H, W], we only need the body part [0:N]
            # condition_info also needs to be sliced to body only
            pose_output = {
                "mhr": combined_output["mhr"],
                "image_embeddings": combined_output["image_embeddings"][0:num_persons],  # Only body embeddings
                "condition_info": combined_output["condition_info"][0:num_persons],  # Only body condition
            }

            # Copy ray_cond from combined_batch back to original batch for post-processing
            # forward_pose_branch sets ray_cond on combined_batch, but run_keypoint_prompt uses the original batch
            if "ray_cond" in combined_batch:
                batch["ray_cond"] = combined_batch["ray_cond"]

            # Extract hand output from combined output
            merged_output = {"mhr_hand": combined_output["mhr_hand"]}

            # DEBUG: Check for NaN in outputs after merged forward
            _DEBUG_NAN = True
            if _DEBUG_NAN:
                def check_nan(tensor, name):
                    if tensor is not None and torch.isnan(tensor).any():
                        print(f"          [DEBUG NaN] {name}: contains NaN!")
                        return True
                    return False

                print("          [DEBUG NaN] Checking body output (pose_output['mhr'])...")
                for k, v in pose_output["mhr"].items():
                    if isinstance(v, torch.Tensor):
                        check_nan(v, f"pose_output['mhr']['{k}']")

                print("          [DEBUG NaN] Checking hand output (merged_output['mhr_hand'])...")
                if merged_output["mhr_hand"] is not None:
                    for k, v in merged_output["mhr_hand"].items():
                        if isinstance(v, torch.Tensor):
                            check_nan(v, f"merged_output['mhr_hand']['{k}']")

            # Compute ori_local_wrist_rotmat from body decoder output (needed for post-processing)
            ori_local_wrist_rotmat = euler_to_rotmat_XZY(
                pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
                    1, (2, 3)
                )
            )

            # Update left_xyxy for unflipping (use the flipped version)
            left_xyxy = left_xyxy_flipped

        else:
            # ============================================================
            # SEQUENTIAL EXECUTION PATH (original behavior)
            # ============================================================
            # Step 1. For full-body inference, we first inference with the body decoder.
            t0 = time.time()
            torch.cuda.synchronize()
            pose_output = self.forward_step(batch, decoder_type="body")
            torch.cuda.synchronize()
            print(f"          [run_inference] step1_body_decoder: {time.time() - t0:.4f}s")

            # Get hand boxes - either from body decoder or YOLO-Pose
            if hand_box_source == "yolo_pose" and yolo_pose_keypoints is not None:
                print("          [run_inference] Using YOLO-Pose hand boxes (replacing body decoder)")
                left_xyxy, right_xyxy = self._get_hand_box_from_yolo_pose(
                    yolo_pose_keypoints, yolo_pose_body_boxes, batch
                )
            else:
                if hand_box_source == "yolo_pose":
                    print("          [run_inference] WARNING: hand_box_source='yolo_pose' but no keypoints provided, falling back to body_decoder")
                left_xyxy, right_xyxy = self._get_hand_box(pose_output, batch)

            ori_local_wrist_rotmat = euler_to_rotmat_XZY(
                pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
                    1, (2, 3)
                )
            )

            # Step 2. Re-run with both hands (merged batch for efficiency)
            t0 = time.time()

            ## Prepare left hand batch (flip image & box)
            flipped_img = img[:, ::-1]
            tmp = left_xyxy.copy()
            left_xyxy[:, 0] = width - tmp[:, 2] - 1
            left_xyxy[:, 2] = width - tmp[:, 0] - 1

            batch_lhand = prepare_batch(
                flipped_img, transform_hand, left_xyxy, cam_int=cam_int.clone()
            )
            batch_lhand = recursive_to(batch_lhand, "cuda")

            ## Prepare right hand batch
            batch_rhand = prepare_batch(
                img, transform_hand, right_xyxy, cam_int=cam_int.clone()
            )
            batch_rhand = recursive_to(batch_rhand, "cuda")

            ## Merge both hand batches and run single forward pass
            batch_hands = self._merge_hand_batches(batch_lhand, batch_rhand)
            self._initialize_batch(batch_hands)  # Re-initialize for merged batch [1, 2, ...]
            torch.cuda.synchronize()
            merged_output = self.forward_step(batch_hands, decoder_type="hand")
            torch.cuda.synchronize()
            print(f"          [run_inference] step2_hands_merged_decoder: {time.time() - t0:.4f}s")

        ## Split output back to left and right
        # Get number of persons for correct splitting
        num_persons = batch["img"].shape[1]
        lhand_output, rhand_output = self._split_hand_outputs(merged_output, batch_size=num_persons)

        # Unflip left hand output
        ## Flip scale
        ### Get MHR values (use cache to avoid .item() synchronization)
        if not hasattr(self, '_cached_scale_params'):
            self._cached_scale_params = {
                'r_hands_mean': self.head_pose.scale_mean[8].item(),
                'l_hands_mean': self.head_pose.scale_mean[9].item(),
                'r_hands_std': self.head_pose.scale_comps[8, 8].item(),
                'l_hands_std': self.head_pose.scale_comps[9, 9].item(),
            }
        scale_r_hands_mean = self._cached_scale_params['r_hands_mean']
        scale_l_hands_mean = self._cached_scale_params['l_hands_mean']
        scale_r_hands_std = self._cached_scale_params['r_hands_std']
        scale_l_hands_std = self._cached_scale_params['l_hands_std']
        ### Apply
        lhand_output["mhr_hand"]["scale"][:, 9] = (
            (
                scale_r_hands_mean
                + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8]
            )
            - scale_l_hands_mean
        ) / scale_l_hands_std
        ## Get the right hand global rotation, flip it, put it in as left.
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = lhand_output["mhr_hand"][
            "joint_global_rots"
        ][:, 42].clone()
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1
        ### Flip hand pose
        lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][
            :, 54:
        ]
        ### Unflip box
        batch_lhand["bbox_center"][:, :, 0] = (
            width - batch_lhand["bbox_center"][:, :, 0] - 1
        )

        # Step 3. replace hand pose estimation from the body decoder.
        # Restore batch parameters for body batch (undoing the merged hand batch initialization)
        self._initialize_batch(batch)

        # Synchronize first to ensure all previous operations are complete
        torch.cuda.synchronize()
        t0 = time.time()
        _ik_timing = {}
        _ik_detail = {}  # Finer-grained timing

        ## CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
        joint_rotations = pose_output["mhr"]["joint_global_rots"]
        ### Get lowarm
        # Optimization: use pre-cached indices (avoid creating new tensors each time)
        if not hasattr(self, '_lowarm_joint_idxs'):
            self._lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()
            self._wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()
        lowarm_joint_idxs = self._lowarm_joint_idxs
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3
        ### Get zero-wrist pose
        wrist_twist_joint_idxs = self._wrist_twist_joint_idxs  # Use cached indices
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )
        ### Get globals from left & right
        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )
        ### Get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )  # B x 2 x 3 x3
        angle_difference_valid_mask = angle_difference < thresh_wrist_angle

        ## CRITERIA 2: hand box size
        hand_box_size_thresh = 64
        hand_box_size_valid_mask = torch.stack(
            [
                (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
                (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
            ],
            dim=1,
        )

        ## CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
        hand_kps2d_thresh = 0.5
        hand_kps2d_valid_mask = torch.stack(
            [
                lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
                rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
            ],
            dim=1,
        )

        ## CRITERIA 4: 2D wrist distance.
        hand_wrist_kps2d_thresh = 0.25
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand
        body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_wrist_idx]
        ].clone()
        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
        hand_wrist_kps2d_valid_mask = torch.stack(
            [
                left_kps_dist < hand_wrist_kps2d_thresh,
                right_kps_dist < hand_wrist_kps2d_thresh,
            ],
            dim=1,
        )
        ## Left-right
        hand_valid_mask = (
            angle_difference_valid_mask
            & hand_box_size_valid_mask
            & hand_kps2d_valid_mask
            & hand_wrist_kps2d_valid_mask
        )

        # Keypoint prompting with the body decoder.
        # We use the wrist location from the hand decoder and the elbow location
        # from the body decoder as prompts to get an updated body pose estimation.
        batch_size, num_person = batch["img"].shape[:2]
        self.hand_batch_idx = []
        self.body_batch_idx = list(range(batch_size * num_person))

        ## Get right & left wrist keypoints from crops; full image. Each are B x 1 x 2
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand

        # Next, get them to crop-normalized space.
        right_kps_crop = self._full_to_crop(batch, right_kps_full)
        left_kps_crop = self._full_to_crop(batch, left_kps_full)

        # Get right & left elbow keypoints from crops; full image. Each are B x 1 x 2
        kps_right_elbow_idx = 8
        kps_left_elbow_idx = 7
        right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_elbow_idx]
        ].clone()
        left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_elbow_idx]
        ].clone()

        # Next, get them to crop-normalized space.
        right_kps_elbow_crop = self._full_to_crop(batch, right_kps_elbow_full)
        left_kps_elbow_crop = self._full_to_crop(batch, left_kps_elbow_full)

        # Assemble them into keypoint prompts
        keypoint_prompt = torch.cat(
            [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop],
            dim=1,
        )
        keypoint_prompt = torch.cat(
            [keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1
        )
        keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
        keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
        keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
        keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

        if keypoint_prompt.shape[0] > 1:
            # Replace invalid keypoints to dummy prompts
            invalid_prompt = (
                (keypoint_prompt[..., 0] < -0.5)
                | (keypoint_prompt[..., 0] > 0.5)
                | (keypoint_prompt[..., 1] < -0.5)
                | (keypoint_prompt[..., 1] > 0.5)
                | (~hand_valid_mask[..., [1, 0, 1, 0]])
            ).unsqueeze(-1)
            dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
            dummy_prompt[:, :, -1] = -2
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )  # [-0.5, 0.5] --> [0, 1]
            keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
        else:
            # Only keep valid keypoints
            valid_keypoint = (
                torch.all(
                    (keypoint_prompt[:, :, :2] > -0.5)
                    & (keypoint_prompt[:, :, :2] < 0.5),
                    dim=2,
                )
                & hand_valid_mask[..., [1, 0, 1, 0]]
            ).squeeze()
            keypoint_prompt = keypoint_prompt[:, valid_keypoint]
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )  # [-0.5, 0.5] --> [0, 1]

        torch.cuda.synchronize()
        _ik_detail['criteria_prep'] = (time.time() - t0) * 1000

        t_kp = time.time()
        # Environment variable SKIP_KEYPOINT_PROMPT=1 can skip keypoint prompting (saves ~30ms)
        skip_kp_prompt = os.environ.get('SKIP_KEYPOINT_PROMPT', '0') == '1'
        if not skip_kp_prompt and keypoint_prompt.numel() != 0:
            pose_output, _ = self.run_keypoint_prompt(
                batch, pose_output, keypoint_prompt
            )
        torch.cuda.synchronize()
        _ik_detail['run_keypoint_prompt'] = (time.time() - t_kp) * 1000

        ##############################################################################

        t_drop = time.time()
        # Drop in hand pose
        left_hand_pose_params = lhand_output["mhr_hand"]["hand"][:, :54]
        right_hand_pose_params = rhand_output["mhr_hand"]["hand"][:, 54:]
        updated_hand_pose = torch.cat(
            [left_hand_pose_params, right_hand_pose_params], dim=1
        )

        # Drop in hand scales
        updated_scale = pose_output["mhr"]["scale"].clone()
        updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
        updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
        updated_scale[:, 18:] = (
            lhand_output["mhr_hand"]["scale"][:, 18:]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
        ) / 2

        # Update hand shape
        updated_shape = pose_output["mhr"]["shape"].clone()
        updated_shape[:, 40:] = (
            lhand_output["mhr_hand"]["shape"][:, 40:]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
        ) / 2

        torch.cuda.synchronize()
        _ik_detail['drop_in_hand'] = (time.time() - t_drop) * 1000

        ############################ Doing IK ############################

        t_fk = time.time()
        # First, forward just FK
        joint_rotations = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
            expr_params=pose_output["mhr"]["face"],
            return_joint_rotations=True,
        )[1]
        torch.cuda.synchronize()
        _ik_detail['mhr_fk'] = (time.time() - t_fk) * 1000

        t_ik_calc = time.time()
        # Get lowarm - use cached indices
        lowarm_joint_rotations = joint_rotations[:, self._lowarm_joint_idxs]  # B x 2 x 3 x 3

        # Get zero-wrist pose - use cached indices
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[self._wrist_twist_joint_idxs]
        )

        # Get globals from left & right
        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )

        # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        wrist_xzy = fix_wrist_euler(
            rotmat_to_euler_XZY(fused_local_wrist_rotmat)
        )

        # Put it in.
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )  # B x 2 x 3 x3
        valid_angle = angle_difference < thresh_wrist_angle
        valid_angle = valid_angle & hand_valid_mask
        valid_angle = valid_angle.unsqueeze(-1)

        body_pose = pose_output["mhr"]["body_pose"][
            :, [41, 43, 42, 31, 33, 32]
        ].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
            updated_body_pose.flatten(1, 2)
        )

        hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
        pose_output["mhr"]["hand"] = torch.where(
            valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
        ).flatten(1, 2)

        hand_scale = torch.stack(
            [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
            dim=1,
        )
        updated_hand_scale = torch.stack(
            [updated_scale[:, 9], updated_scale[:, 8]], dim=1
        )
        masked_hand_scale = torch.where(
            valid_angle.squeeze(-1), updated_hand_scale, hand_scale
        )
        pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
        pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

        # Replace shared shape and scale
        pose_output["mhr"]["scale"][:, 18:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["scale"][:, 18:],
        )
        pose_output["mhr"]["shape"][:, 40:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["shape"][:, 40:],
        )
        torch.cuda.synchronize()
        _ik_detail['ik_calc'] = (time.time() - t_ik_calc) * 1000

        ########################################################
        _ik_timing['criteria_fusion'] = (time.time() - t0) * 1000

        # Re-run forward
        t_mhr = time.time()

        # DEBUG: Check for NaN in pose_output["mhr"] before mhr_forward
        if _DEBUG_NAN:
            print("          [DEBUG NaN] Checking pose_output['mhr'] before mhr_forward...")
            for k in ["global_rot", "body_pose", "hand", "scale", "shape", "face"]:
                v = pose_output["mhr"].get(k)
                if v is not None and isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    print(f"          [DEBUG NaN] pose_output['mhr']['{k}']: contains NaN! shape={v.shape}")

        with torch.no_grad():
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
                self.head_pose.mhr_forward(
                    global_trans=pose_output["mhr"]["global_rot"] * 0,
                    global_rot=pose_output["mhr"]["global_rot"],
                    body_pose_params=pose_output["mhr"]["body_pose"],
                    hand_pose_params=pose_output["mhr"]["hand"],
                    scale_params=pose_output["mhr"]["scale"],
                    shape_params=pose_output["mhr"]["shape"],
                    expr_params=pose_output["mhr"]["face"],
                    return_keypoints=True,
                    return_joint_coords=True,
                    return_model_params=True,
                    return_joint_rotations=True,
                )
            )
            j3d = j3d[:, :70]  # 308 --> 70 keypoints
            verts[..., [1, 2]] *= -1  # Camera system difference
            j3d[..., [1, 2]] *= -1  # Camera system difference
            jcoords[..., [1, 2]] *= -1

            # DEBUG: Check for NaN in mhr_forward outputs
            if _DEBUG_NAN:
                if torch.isnan(verts).any():
                    print(f"          [DEBUG NaN] verts after mhr_forward: contains NaN! shape={verts.shape}")
                if torch.isnan(j3d).any():
                    print(f"          [DEBUG NaN] j3d after mhr_forward: contains NaN! shape={j3d.shape}")

            pose_output["mhr"]["pred_keypoints_3d"] = j3d
            pose_output["mhr"]["pred_vertices"] = verts
            pose_output["mhr"]["pred_joint_coords"] = jcoords
            pose_output["mhr"]["pred_pose_raw"][
                ...
            ] = 0  # pred_pose_raw is not valid anymore
            pose_output["mhr"]["mhr_model_params"] = mhr_model_params
        torch.cuda.synchronize()
        _ik_timing['mhr_rerun'] = (time.time() - t_mhr) * 1000

        ########################################################
        # Project to 2D
        t_proj = time.time()
        pred_keypoints_3d_proj = (
            pose_output["mhr"]["pred_keypoints_3d"]
            + pose_output["mhr"]["pred_cam_t"][:, None, :]
        )
        pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][
            :, None, None
        ]
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[
                None, None, :
            ]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]
        torch.cuda.synchronize()
        _ik_timing['projection'] = (time.time() - t_proj) * 1000

        # Print IK timing
        print(f"  ┌─────────────────────────────────────────────────────────────")
        print(f"  | [postprocess_ik breakdown]:")
        print(f"  |   - criteria_fusion: {_ik_timing['criteria_fusion']:.3f} ms  (hand fusion condition computation)")
        print(f"  |     - criteria_prep:       {_ik_detail.get('criteria_prep', 0):.3f} ms  (condition preparation)")
        print(f"  |     - run_keypoint_prompt: {_ik_detail.get('run_keypoint_prompt', 0):.3f} ms  <- decoder forward!")
        print(f"  |     - drop_in_hand:        {_ik_detail.get('drop_in_hand', 0):.3f} ms  (hand parameters)")
        print(f"  |     - mhr_fk:              {_ik_detail.get('mhr_fk', 0):.3f} ms  (FK computation)")
        print(f"  |     - ik_calc:             {_ik_detail.get('ik_calc', 0):.3f} ms  (IK fusion)")
        print(f"  |   - mhr_rerun:       {_ik_timing['mhr_rerun']:.3f} ms  <- MHR re-inference")
        print(f"  |   - projection:      {_ik_timing['projection']:.3f} ms  (2D projection)")
        total_ik = sum(_ik_timing.values())
        print(f"  |   total:             {total_ik:.3f} ms")
        print(f"  └─────────────────────────────────────────────────────────────")
        print(f"          [run_inference] step3_postprocess_ik: {time.time() - t0:.4f}s")
        print(f"        [run_inference] TOTAL: {time.time() - run_inference_start:.4f}s")

        return pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output

    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        import os
        # Environment variables control IntermPred configuration for keypoint prompting
        # KEYPOINT_PROMPT_INTERM_LAYERS=0,1,2,3 specifies layer list (higher priority)
        # KEYPOINT_PROMPT_INTERM_INTERVAL=999 interval mode, default skips all intermediate predictions
        env_kp_layers = os.environ.get('KEYPOINT_PROMPT_INTERM_LAYERS')
        env_kp_interval = os.environ.get('KEYPOINT_PROMPT_INTERM_INTERVAL', '999')

        if env_kp_layers:
            # Parse comma-separated layer index list
            kp_interm_layers = set(int(x.strip()) for x in env_kp_layers.split(',') if x.strip())
            kp_interm_interval = None
        else:
            kp_interm_layers = None
            kp_interm_interval = int(env_kp_interval)

        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["mhr"]  # body-only output
        # Use previous estimate as initialization
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),  # (B, 6)
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                dim=-1,
            )

        # Mark step begin to avoid CUDA graph tensor reuse conflicts
        torch.compiler.cudagraph_mark_step_begin()
        tokens_output, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,  # not recurring previous estimate
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
            override_interm_interval=kp_interm_interval,  # Override IntermPred interval
            override_interm_layers=kp_interm_layers,  # Override IntermPred layer list
        )
        pose_output = pose_output[-1]

        output.update({"mhr": pose_output})
        return output, keypoint_prompt

    def _get_hand_box(self, pose_output, batch):
        """Get hand bbox from the hand detector"""
        pred_left_hand_box = (
            pose_output["mhr"]["hand_box"][:, 0].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )
        pred_right_hand_box = (
            pose_output["mhr"]["hand_box"][:, 1].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )

        # Change boxes into squares
        batch["left_center"] = pred_left_hand_box[:, :2]
        batch["left_scale"] = (
            pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )
        batch["right_center"] = pred_right_hand_box[:, :2]
        batch["right_scale"] = (
            pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )

        # Crop to full. batch["affine_trans"] is full-to-crop, right application
        batch["left_scale"] = (
            batch["left_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["right_scale"] = (
            batch["right_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["left_center"] = (
            batch["left_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        batch["right_center"] = (
            batch["right_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]

        left_xyxy = np.concatenate(
            [
                (
                    batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )
        right_xyxy = np.concatenate(
            [
                (
                    batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )

        return left_xyxy, right_xyxy

    def _get_hand_box_from_yolo_pose(
        self,
        yolo_pose_keypoints: np.ndarray,
        yolo_pose_body_boxes: np.ndarray,
        batch: Dict,
        hand_box_scale: float = 3.0,
        min_wrist_conf: float = 0.3,
    ):
        """
        Get hand bbox from YOLO-Pose wrist keypoints.

        Args:
            yolo_pose_keypoints: YOLO-Pose keypoints [N, 17, 3] (x, y, conf)
                - COCO format: idx 9 = left_wrist, idx 10 = right_wrist
            yolo_pose_body_boxes: YOLO-Pose body boxes [N, 4] (x1, y1, x2, y2)
            batch: batch dict containing affine_trans
            hand_box_scale: scale factor for hand box size (default: 2.5)
            min_wrist_conf: minimum confidence threshold for wrist detection

        Returns:
            left_xyxy: [N, 4] left hand boxes in original image coordinates
            right_xyxy: [N, 4] right hand boxes in original image coordinates

        Note:
            COCO keypoint mapping:
            - left_wrist (idx 9) corresponds to person's LEFT hand
            - right_wrist (idx 10) corresponds to person's RIGHT hand
        """
        n_persons = yolo_pose_keypoints.shape[0]

        # COCO keypoint indices
        LEFT_WRIST_IDX = 9
        RIGHT_WRIST_IDX = 10

        left_centers = []
        left_scales = []
        right_centers = []
        right_scales = []

        for i in range(n_persons):
            kpts = yolo_pose_keypoints[i]  # [17, 3]
            body_box = yolo_pose_body_boxes[i]  # [4]

            # Compute hand box size based on body box
            body_width = body_box[2] - body_box[0]
            body_height = body_box[3] - body_box[1]
            hand_size = (body_width + body_height) / 2 / hand_box_scale

            # Left wrist -> Left hand box
            left_wrist = kpts[LEFT_WRIST_IDX]
            if left_wrist[2] > min_wrist_conf:
                lx, ly = left_wrist[0], left_wrist[1]
                left_centers.append([lx, ly])
                left_scales.append([hand_size, hand_size])
            else:
                # Use body center as fallback
                cx = (body_box[0] + body_box[2]) / 2
                cy = (body_box[1] + body_box[3]) / 2
                left_centers.append([cx, cy])
                left_scales.append([hand_size, hand_size])
                print(f"          [YOLO-Pose] WARNING: Person {i} left_wrist low conf ({left_wrist[2]:.2f}), using body center")

            # Right wrist -> Right hand box
            right_wrist = kpts[RIGHT_WRIST_IDX]
            if right_wrist[2] > min_wrist_conf:
                rx, ry = right_wrist[0], right_wrist[1]
                right_centers.append([rx, ry])
                right_scales.append([hand_size, hand_size])
            else:
                # Use body center as fallback
                cx = (body_box[0] + body_box[2]) / 2
                cy = (body_box[1] + body_box[3]) / 2
                right_centers.append([cx, cy])
                right_scales.append([hand_size, hand_size])
                print(f"          [YOLO-Pose] WARNING: Person {i} right_wrist low conf ({right_wrist[2]:.2f}), using body center")

        # Convert to numpy arrays
        batch["left_center"] = np.array(left_centers)
        batch["left_scale"] = np.array(left_scales)
        batch["right_center"] = np.array(right_centers)
        batch["right_scale"] = np.array(right_scales)

        # Log computed values
        for i in range(n_persons):
            print(f"          [YOLO-Pose] Person {i}:")
            print(f"            left_center: {batch['left_center'][i]}, left_scale: {batch['left_scale'][i]}")
            print(f"            right_center: {batch['right_center'][i]}, right_scale: {batch['right_scale'][i]}")

        # Compute xyxy boxes
        left_xyxy = np.concatenate(
            [
                (batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2).reshape(-1, 1),
                (batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2).reshape(-1, 1),
                (batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2).reshape(-1, 1),
                (batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2).reshape(-1, 1),
            ],
            axis=1,
        )
        right_xyxy = np.concatenate(
            [
                (batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2).reshape(-1, 1),
                (batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2).reshape(-1, 1),
                (batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2).reshape(-1, 1),
                (batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2).reshape(-1, 1),
            ],
            axis=1,
        )

        print(f"          [YOLO-Pose] left_xyxy: {left_xyxy}")
        print(f"          [YOLO-Pose] right_xyxy: {right_xyxy}")

        return left_xyxy, right_xyxy

    def _merge_hand_batches(self, batch_lhand: Dict, batch_rhand: Dict) -> Dict:
        """Merge left and right hand batches into a single batch for efficient inference.

        Batch structure is [batch_size, num_person, ...], so we concatenate along dim=1.

        Args:
            batch_lhand: Left hand batch dict, shape [1, 1, ...]
            batch_rhand: Right hand batch dict, shape [1, 1, ...]

        Returns:
            Merged batch dict with shape [1, 2, ...] (2 persons = left + right hand)
        """
        merged_batch = {}

        # Keys that need to be concatenated along person dimension (dim=1)
        # Batch structure: [batch_size, num_person, ...]
        concat_keys = [
            "img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
            "bbox", "affine_trans", "mask", "mask_score", "person_valid"
        ]

        for key in concat_keys:
            if key in batch_lhand and key in batch_rhand:
                merged_batch[key] = torch.cat([batch_lhand[key], batch_rhand[key]], dim=1)

        # cam_int is [batch_size, 3, 3], just take one (they should be the same)
        if "cam_int" in batch_lhand:
            merged_batch["cam_int"] = batch_lhand["cam_int"]

        # Handle img_ori specially (it's a list)
        if "img_ori" in batch_lhand and "img_ori" in batch_rhand:
            merged_batch["img_ori"] = batch_lhand["img_ori"] + batch_rhand["img_ori"]

        return merged_batch

    def _merge_body_hand_batches(self, batch_body: Dict, batch_hands: Dict) -> Dict:
        """Merge body batch with hand batches for shared backbone inference.

        This combines body crop + hand crops into a single batch so backbone runs once.
        The batch structure is [batch_size, num_person, ...].

        Args:
            batch_body: Body batch dict, shape [1, 1, ...] (1 body crop)
            batch_hands: Merged hand batch dict, shape [1, 2, ...] (left + right hand)

        Returns:
            Combined batch dict with shape [1, 3, ...] (body + left_hand + right_hand)
            - Person 0: body crop
            - Person 1: left hand crop
            - Person 2: right hand crop
        """
        combined_batch = {}

        # Keys that need to be concatenated along person dimension (dim=1)
        concat_keys = [
            "img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
            "bbox", "affine_trans", "mask", "mask_score", "person_valid"
        ]

        for key in concat_keys:
            if key in batch_body and key in batch_hands:
                combined_batch[key] = torch.cat([batch_body[key], batch_hands[key]], dim=1)

        # cam_int is [batch_size, 3, 3], just take from body batch
        if "cam_int" in batch_body:
            combined_batch["cam_int"] = batch_body["cam_int"]

        # Handle img_ori specially (it's a list)
        if "img_ori" in batch_body and "img_ori" in batch_hands:
            combined_batch["img_ori"] = batch_body["img_ori"] + batch_hands["img_ori"]

        return combined_batch

    def _split_hand_outputs(self, merged_output: Dict, batch_size: int = 1) -> Tuple[Dict, Dict]:
        """Split merged hand output back into left and right hand outputs.

        Args:
            merged_output: Output from merged batch inference
            batch_size: Original batch size (before merging)

        Returns:
            Tuple of (lhand_output, rhand_output)
        """
        lhand_output = {"mhr_hand": {}}
        rhand_output = {"mhr_hand": {}}

        if "mhr_hand" in merged_output and merged_output["mhr_hand"] is not None:
            for key, value in merged_output["mhr_hand"].items():
                if isinstance(value, torch.Tensor):
                    # Split along batch dimension
                    lhand_output["mhr_hand"][key] = value[:batch_size]
                    rhand_output["mhr_hand"][key] = value[batch_size:]
                else:
                    # For non-tensor values, just copy
                    lhand_output["mhr_hand"][key] = value
                    rhand_output["mhr_hand"][key] = value

        return lhand_output, rhand_output

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()  # These are -0.5 ~ 0.5
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs_hand
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs_hand
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also maybe update token_embeddings with the grid sampled 2D feature.
        # Remember that pred_keypoints_2d_cropped are -0.5 ~ 0.5. We want -1 ~ 1
        # Sample points...
        ## Get sampling points
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            # Need to go from 256 x 256 coords to 256 x 192 (HW) because image_embeddings is 16x12
            # Aka, for x, what was normally -1 ~ 1 for 256 should be -16/12 ~ 16/12 (since to sample at original 256, need to overflow)
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        # Version 2 is projecting & bilinear sampling
        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],  # -1 ~ 1, xy
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )  # B x kps x C
        # Zero out invalid locations...
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        # This is ADDING
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm (just in case idxs shift)
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs_hand]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx
