#!/usr/bin/env python
"""
Stage 1 Single Person for 3DPW: GT bbox -> Crop Image -> Inference -> Save Outputs

This mode crops the image to GT bounding box (computed from poses2d).

Run with conda environment:
    /home/jiawei/miniforge3/envs/fast_sam_3d_body/bin/python eval_ECCV/stage1_single_3dpw.py \
        --data_dir eval_ECCV/dataset/3DPW \
        --output_dir eval_ECCV/outputs_3dpw \
        --split test \
        --max_samples 100
"""

# Set optimization environment variables (must be set before importing other modules)
import os
os.environ.setdefault("GPU_HAND_PREP", "1")
os.environ.setdefault("LAYER_DTYPE", "fp32")
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("FOV_TRT", "1")
os.environ.setdefault("FOV_FAST", "1")
os.environ.setdefault("FOV_MODEL", "s")
os.environ.setdefault("FOV_LEVEL", "0")
os.environ.setdefault("USE_COMPILE_BACKBONE", "1")
os.environ.setdefault("USE_COMPILE", "1")
os.environ.setdefault("DECODER_COMPILE", "1")
os.environ.setdefault("MHR_USE_CUDA_GRAPH", "0")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")
os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")
os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")

SKIP_GT_CAM = os.environ.get("SKIP_GT_CAM", "0") == "1"
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
os.environ.setdefault("IMG_SIZE", "512")

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Fast_sam-3d-body/
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body


def bbox_from_poses2d(poses2d, padding=0.2):
    """
    Compute bounding box from 2D poses (COCO format).
    poses2d: (3, 18) - [x, y, confidence] for 18 keypoints
    Returns: [x1, y1, x2, y2]
    """
    x = poses2d[0]
    y = poses2d[1]
    conf = poses2d[2]

    # Filter by confidence
    valid = conf > 0.1
    if valid.sum() < 4:
        return None

    x_valid = x[valid]
    y_valid = y[valid]

    x1, x2 = x_valid.min(), x_valid.max()
    y1, y2 = y_valid.min(), y_valid.max()

    # Add padding
    w, h = x2 - x1, y2 - y1
    x1 -= w * padding
    y1 -= h * padding
    x2 += w * padding
    y2 += h * padding

    return np.array([x1, y1, x2, y2])


def load_3dpw_annotations(data_dir, split='test', sequence_filter=None, frame_step=1):
    """
    Load annotations from 3DPW dataset.
    Returns list of annotations, each containing:
    - image_path, bbox, cam_int, gt_poses, gt_betas, gt_trans, gt_joints_3d, gender
    """
    data_dir = Path(data_dir)
    seq_dir = data_dir / 'sequenceFiles' / split

    annotations = []

    for pkl_file in sorted(seq_dir.glob("*.pkl")):
        seq_name = pkl_file.stem

        if sequence_filter and seq_name != sequence_filter:
            continue

        with open(pkl_file, 'rb') as f:
            seq = pickle.load(f, encoding='latin1')

        num_persons = len(seq['poses'])
        num_frames = len(seq['poses'][0])
        cam_int = seq['cam_intrinsics']

        for frame_idx in range(0, num_frames, frame_step):
            for person_idx in range(num_persons):
                # Check if campose is valid for this person/frame
                if not seq['campose_valid'][person_idx][frame_idx]:
                    continue

                # Get poses2d and compute bbox
                poses2d = seq['poses2d'][person_idx][frame_idx]  # (3, 18)
                bbox = bbox_from_poses2d(poses2d)
                if bbox is None:
                    continue

                # Image path
                img_path = f"{seq_name}/image_{frame_idx:05d}.jpg"

                # GT SMPL parameters
                gt_poses = seq['poses'][person_idx][frame_idx]  # (72,)
                gt_betas = seq['betas'][person_idx][:10]  # (10,) - only first 10
                gt_trans = seq['trans'][person_idx][frame_idx]  # (3,)
                gt_joints_3d = seq['jointPositions'][person_idx][frame_idx].reshape(24, 3)  # (24, 3)
                gender = str(seq['genders'][person_idx])

                annotations.append({
                    'image_path': img_path,
                    'bbox': bbox,
                    'cam_int': None if SKIP_GT_CAM else cam_int.copy(),
                    'gt_poses': gt_poses,
                    'gt_betas': gt_betas,
                    'gt_trans': gt_trans,
                    'gt_joints_3d': gt_joints_3d,
                    'gender': gender,
                    'sequence': seq_name,
                    'frame_idx': frame_idx,
                    'person_idx': person_idx,
                })

    return annotations


def crop_image_with_context(img, bbox, context_factor=1.5):
    """
    Crop image around bbox with context.
    Returns cropped image, new bbox in cropped coords, and transform info.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clamp bbox to image bounds first
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Expand bbox with context
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    new_w = bw * context_factor
    new_h = bh * context_factor

    # New crop region (with padding for out-of-bounds)
    crop_x1 = int(max(0, cx - new_w / 2))
    crop_y1 = int(max(0, cy - new_h / 2))
    crop_x2 = int(min(w, cx + new_w / 2))
    crop_y2 = int(min(h, cy + new_h / 2))

    # Crop
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Transform bbox to cropped coords
    new_bbox = np.array([
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1
    ])

    transform_info = {
        'crop_x1': crop_x1,
        'crop_y1': crop_y1,
        'crop_w': crop_x2 - crop_x1,
        'crop_h': crop_y2 - crop_y1,
    }

    return cropped, new_bbox, transform_info


def adjust_cam_int_for_crop(cam_int, transform_info):
    """Adjust camera intrinsics for cropped image."""
    if cam_int is None:
        return None
    new_cam_int = cam_int.copy()
    # Shift principal point
    new_cam_int[0, 2] -= transform_info['crop_x1']  # cx
    new_cam_int[1, 2] -= transform_info['crop_y1']  # cy
    return new_cam_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to 3DPW dataset root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--sequence", default=None, help="Filter by sequence name")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--frame_step", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--model", default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--detector", default="yolo_pose")
    parser.add_argument("--detector_model", default="checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--hand_box_source", default="yolo_pose")
    parser.add_argument("--context_factor", type=float, default=1.5,
                        help="Factor to expand bbox for cropping (1.5 = 50% padding)")
    parser.add_argument("--warmup_pass", action="store_true",
                        help="Run a full pass over the dataset for warmup, then run a second pass to record timing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "imageFiles"

    print("Loading 3DPW annotations...")
    annotations = load_3dpw_annotations(
        data_dir,
        split=args.split,
        sequence_filter=args.sequence,
        frame_step=args.frame_step
    )
    print(f"Loaded {len(annotations)} annotations")

    if args.max_samples > 0:
        annotations = annotations[:args.max_samples]

    print("Loading SAM 3D Body model...")
    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model
    )

    # ============ WARMUP PASS ============
    if args.warmup_pass:
        print("\n" + "=" * 60)
        print("[WARMUP] Pass 1 - Warmup run (results not saved)")
        print("=" * 60)
        warmup_times = []
        for ann_idx, gt in enumerate(annotations):
            full_path = img_dir / gt['image_path']
            if not full_path.exists():
                continue
            img = cv2.imread(str(full_path))
            if img is None:
                continue
            bbox = gt['bbox']
            cropped_img, new_bbox, transform_info = crop_image_with_context(
                img, bbox, args.context_factor
            )
            cam_int_np = gt.get('cam_int', None)
            if cam_int_np is not None:
                adjusted_cam_int = adjust_cam_int_for_crop(cam_int_np, transform_info)
                cam_int = torch.from_numpy(adjusted_cam_int).float().unsqueeze(0)
            else:
                cam_int = None
            temp_crop_path = output_dir / "temp_crop.jpg"
            cv2.imwrite(str(temp_crop_path), cropped_img)
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                preds = estimator.process_one_image(
                    str(temp_crop_path),
                    bboxes=new_bbox.reshape(1, 4),
                    cam_int=cam_int,
                    hand_box_source=args.hand_box_source
                )
                torch.cuda.synchronize()
                warmup_times.append((time.time() - t0) * 1000)
            except Exception:
                continue
        if warmup_times:
            avg_ms = np.mean(warmup_times)
            print(f"[WARMUP] Done. {len(warmup_times)} samples, "
                  f"Avg: {avg_ms:.1f} ms, FPS: {1000.0/avg_ms:.2f}")
        # Clean up warmup data and free GPU memory
        del warmup_times
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("\n" + "=" * 60)
        print("[EVAL] Pass 2 - Formal evaluation (recording timing)")
        print("=" * 60)

    print(f"Running single-person inference on {len(annotations)} samples...")
    print(f"Context factor: {args.context_factor}")
    sample_idx = 0
    inference_times = []

    for ann_idx, gt in enumerate(annotations):
        img_path = gt['image_path']
        full_path = img_dir / img_path
        if not full_path.exists():
            print(f"Image not found: {full_path}")
            continue

        # Load image
        img = cv2.imread(str(full_path))
        if img is None:
            continue

        # Crop image around GT bbox
        bbox = gt['bbox']
        cropped_img, new_bbox, transform_info = crop_image_with_context(
            img, bbox, args.context_factor
        )

        # Adjust camera intrinsics for crop
        cam_int_np = gt.get('cam_int', None)
        if cam_int_np is not None:
            adjusted_cam_int = adjust_cam_int_for_crop(cam_int_np, transform_info)
            cam_int = torch.from_numpy(adjusted_cam_int).float().unsqueeze(0)
        else:
            cam_int = None

        # Save cropped image temporarily
        temp_crop_path = output_dir / "temp_crop.jpg"
        cv2.imwrite(str(temp_crop_path), cropped_img)

        try:
            # Run inference on cropped image
            start_time = time.time()
            preds = estimator.process_one_image(
                str(temp_crop_path),
                bboxes=new_bbox.reshape(1, 4),
                cam_int=cam_int,
                hand_box_source=args.hand_box_source
            )
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)
        except Exception as e:
            print(f"Error on sample {ann_idx}: {e}")
            continue

        if not preds:
            continue

        pred = preds[0]
        save_dict = {
            # Predictions (MHR format)
            'pred_vertices': pred['pred_vertices'],
            'pred_cam_t': pred['pred_cam_t'],
            'pred_keypoints_3d': pred['pred_keypoints_3d'],
            # GT (SMPL format)
            'gt_poses': gt['gt_poses'],
            'gt_betas': gt['gt_betas'],
            'gt_trans': gt['gt_trans'],
            'gt_joints_3d': gt['gt_joints_3d'],
            'gt_gender': gt['gender'],
            # Metadata
            'image_path': img_path,
            'bbox': bbox,
            'sequence': gt['sequence'],
            'frame_idx': gt['frame_idx'],
            'person_idx': gt['person_idx'],
        }
        if cam_int_np is not None:
            save_dict['cam_int'] = cam_int_np
        np.savez(samples_dir / f"sample_{sample_idx:05d}.npz", **save_dict)
        sample_idx += 1

        if (ann_idx + 1) % 50 == 0:
            print(f"Progress: {ann_idx+1}/{len(annotations)}, {sample_idx} samples saved")

    # Cleanup
    temp_crop_path = output_dir / "temp_crop.jpg"
    if temp_crop_path.exists():
        temp_crop_path.unlink()

    # Save timing stats
    if inference_times:
        timing_stats = {
            'avg_inference_time_ms': float(np.mean(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'fps': float(1000.0 / np.mean(inference_times)),
            'fps_std': float(np.std(1000.0 / np.array(inference_times))),
            'total_samples': sample_idx,
        }
        with open(output_dir / "timing_stats.json", 'w') as f:
            json.dump(timing_stats, f, indent=2)
        print(f"\nTiming: {timing_stats['avg_inference_time_ms']:.1f} ms/sample, FPS: {timing_stats['fps']:.2f}")

    print(f"\nDone! Saved {sample_idx} samples to {samples_dir}")


if __name__ == "__main__":
    main()
