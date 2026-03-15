#!/usr/bin/env python
"""
Stage 1 Single Person for EMDB: GT bbox -> Crop Image -> Inference -> Save Outputs

EMDB dataset contains monocular video sequences with SMPL GT annotations.
Each sequence has a pickle file with SMPL params, camera intrinsics, bboxes, and 2D keypoints.

Run with conda environment:
    /home/jiawei/miniforge3/envs/fast_sam_3d_body/bin/python eval_ECCV/stage1_single_EMDB.py \
        --data_dir eval_ECCV/dataset/EMDB/data \
        --output_dir eval_ECCV/outputs_EMDB \
        --subset all \
        --max_samples -1
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
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")

SKIP_GT_CAM = os.environ.get("SKIP_GT_CAM", "0") == "1"
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


def load_emdb_annotations(data_dir, subset='all', sequence_filter=None, frame_step=1):
    """
    Load annotations from EMDB dataset.

    Args:
        data_dir: Path to EMDB/data/ containing P0/, P1/, ... P9/
        subset: 'all', 'emdb1', or 'emdb2'
        sequence_filter: If set, only process this sequence name (e.g. 'P0_00_mvs_a')
        frame_step: Process every N-th frame

    Returns list of annotations, each containing:
    - image_path, bbox, cam_int, gt_poses, gt_betas, gt_trans, gender, sequence, frame_idx
    """
    data_dir = Path(data_dir)
    annotations = []

    # Iterate over all subject directories (P0, P1, ..., P9)
    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('P'):
            continue

        # Iterate over sequences within each subject
        for seq_dir in sorted(subject_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            # Find the pickle file
            pkl_files = list(seq_dir.glob("*_data.pkl"))
            if not pkl_files:
                continue
            pkl_file = pkl_files[0]

            with open(pkl_file, 'rb') as f:
                seq_data = pickle.load(f)

            seq_name = seq_data['name']

            # Filter by sequence name
            if sequence_filter and seq_name != sequence_filter:
                continue

            # Filter by subset
            if subset == 'emdb1' and not seq_data.get('emdb1', False):
                continue
            if subset == 'emdb2' and not seq_data.get('emdb2', False):
                continue

            n_frames = seq_data['n_frames']
            good_mask = seq_data['good_frames_mask']  # (N,)
            cam_int = seq_data['camera']['intrinsics']  # (3, 3)
            gender = seq_data['gender']
            bboxes = seq_data['bboxes']['bboxes']  # (N, 4)
            invalid_bbox_idxs = set(seq_data['bboxes']['invalid_idxs'].tolist())

            # SMPL parameters
            poses_root = seq_data['smpl']['poses_root']  # (N, 3)
            poses_body = seq_data['smpl']['poses_body']  # (N, 69)
            betas = seq_data['smpl']['betas']  # (10,)
            trans = seq_data['smpl']['trans']  # (N, 3)

            for frame_idx in range(0, n_frames, frame_step):
                # Skip bad frames
                if not good_mask[frame_idx]:
                    continue

                # Skip frames with invalid bboxes
                if frame_idx in invalid_bbox_idxs:
                    continue

                bbox = bboxes[frame_idx]  # (4,)

                # Construct full 72-dim SMPL pose: root (3) + body (69)
                gt_poses = np.concatenate([
                    poses_root[frame_idx],  # (3,)
                    poses_body[frame_idx],  # (69,)
                ])  # (72,)

                # Image path: {subject}/{sequence}/images/{frame:05d}.jpg
                img_path = seq_dir / "images" / f"{frame_idx:05d}.jpg"

                annotations.append({
                    'image_path': str(img_path),
                    'bbox': bbox.copy(),
                    'cam_int': None if SKIP_GT_CAM else cam_int.copy(),
                    'gt_poses': gt_poses,
                    'gt_betas': betas.copy(),
                    'gt_trans': trans[frame_idx].copy(),
                    'gender': gender,
                    'sequence': seq_name,
                    'frame_idx': frame_idx,
                    'emdb1': seq_data.get('emdb1', False),
                    'emdb2': seq_data.get('emdb2', False),
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
    new_cam_int[0, 2] -= transform_info['crop_x1']  # cx
    new_cam_int[1, 2] -= transform_info['crop_y1']  # cy
    return new_cam_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to EMDB/data/ directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subset", default="all", choices=["all", "emdb1", "emdb2"],
                        help="Which subset to evaluate: all, emdb1, or emdb2")
    parser.add_argument("--sequence", default=None, help="Filter by sequence name (e.g. P0_00_mvs_a)")
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

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EMDB annotations (subset={args.subset})...")
    annotations = load_emdb_annotations(
        args.data_dir,
        subset=args.subset,
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
            img_path = gt['image_path']
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
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
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Load image
        img = cv2.imread(img_path)
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
            'gt_gender': gt['gender'],
            # Metadata
            'image_path': img_path,
            'bbox': bbox,
            'cam_int': cam_int_np,
            'sequence': gt['sequence'],
            'frame_idx': gt['frame_idx'],
            'emdb1': gt['emdb1'],
            'emdb2': gt['emdb2'],
        }
        np.savez(samples_dir / f"sample_{sample_idx:05d}.npz", **save_dict)
        sample_idx += 1

        if (ann_idx + 1) % 50 == 0:
            avg_t = np.mean(inference_times[-50:])
            print(f"Progress: {ann_idx+1}/{len(annotations)}, {sample_idx} saved, {avg_t:.0f} ms/sample")

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
