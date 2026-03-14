#!/usr/bin/env python
"""
Stage 1 Single Person for RICH dataset: Load images -> Inference -> Save Outputs

RICH dataset structure:
- Images: ps/project/multi-ioi/rich_release/test/{scene}/cam_{xx}/{frame}_{cam}.bmp
- GT SMPL-X params: test/{scene}/{frame}/{cam_id}.pkl

Run with conda environment:
    /home/jiawei/miniforge3/envs/sam_3d_body/bin/python eval_ECCV/stage1_single_RICH.py \
        --data_dir eval_ECCV/dataset/RICH \
        --output_dir eval_ECCV/outputs_RICH \
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

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Fast_sam-3d-body/
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body


def load_rich_annotations(data_dir, scenes=None, cameras=None, frame_step=1):
    """
    Load RICH dataset annotations.

    RICH structure:
    - Images: ps/project/multi-ioi/rich_release/test/{scene}/cam_{xx}/{frame}_{cam}.bmp
    - GT: test/{scene}/{frame}/{person_id}.pkl (one pkl per frame, person_id is typically 010)

    Args:
        data_dir: Root directory of RICH dataset
        scenes: List of scene names to process (None = all)
        cameras: List of camera IDs to use (None = all)
        frame_step: Process every N-th frame

    Returns:
        List of annotation dicts
    """
    data_dir = Path(data_dir)
    gt_dir = data_dir / "test"
    img_base_dir = data_dir / "ps" / "project" / "multi-ioi" / "rich_release" / "test"

    annotations = []

    # Get all scenes
    available_scenes = sorted([d.name for d in gt_dir.iterdir() if d.is_dir()])
    if scenes:
        available_scenes = [s for s in available_scenes if s in scenes]

    print(f"Processing {len(available_scenes)} scenes: {available_scenes[:5]}...")

    for scene in available_scenes:
        scene_gt_dir = gt_dir / scene
        scene_img_dir = img_base_dir / scene

        if not scene_img_dir.exists():
            print(f"Image dir not found: {scene_img_dir}")
            continue

        # Get available cameras
        available_cams = sorted([d.name for d in scene_img_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")])
        if cameras:
            available_cams = [c for c in available_cams if c in cameras]

        # Get frames from GT directory
        frame_dirs = sorted([d.name for d in scene_gt_dir.iterdir() if d.is_dir()])

        for frame_idx, frame_id in enumerate(frame_dirs):
            if frame_idx % frame_step != 0:
                continue

            frame_gt_dir = scene_gt_dir / frame_id

            # Find GT pkl file (any .pkl file in the frame dir, typically named by person_id like 010.pkl)
            pkl_files = list(frame_gt_dir.glob("*.pkl"))
            if not pkl_files:
                continue
            pkl_file = pkl_files[0]  # Take first pkl file

            # Load GT SMPL-X params once per frame
            with open(pkl_file, 'rb') as f:
                smplx_params = pickle.load(f)

            for cam_dir in available_cams:
                cam_id = cam_dir.replace("cam_", "")  # e.g., "01" from "cam_01"

                # Find corresponding image
                # Image naming: {frame}_{cam}.bmp, e.g., 00005_01.bmp
                img_name = f"{frame_id}_{cam_id}.bmp"
                img_path = scene_img_dir / cam_dir / img_name

                if not img_path.exists():
                    continue

                annotations.append({
                    'image_path': str(img_path),
                    'scene': scene,
                    'frame_id': frame_id,
                    'cam_id': cam_id,
                    'smplx_params': smplx_params,
                })

    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="RICH dataset root directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenes", nargs="+", default=None, help="Specific scenes to process")
    parser.add_argument("--cameras", nargs="+", default=None, help="Specific cameras to use (e.g., cam_01)")
    parser.add_argument("--frame_step", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--model", default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--detector", default="yolo_pose")
    parser.add_argument("--detector_model", default="checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--hand_box_source", default="yolo_pose")
    parser.add_argument("--warmup_pass", action="store_true",
                        help="Run a full pass over the dataset for warmup, then run a second pass to record timing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RICH annotations...")
    annotations = load_rich_annotations(
        args.data_dir,
        scenes=args.scenes,
        cameras=args.cameras,
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
        import torch
        print("\n" + "=" * 60)
        print("[WARMUP] Pass 1 - Warmup run (results not saved)")
        print("=" * 60)
        warmup_times = []
        for ann_idx, gt in enumerate(annotations):
            img_path = gt['image_path']
            img = cv2.imread(img_path)
            if img is None:
                continue
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                preds = estimator.process_one_image(img_path, hand_box_source=args.hand_box_source)
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

    print(f"Running inference on {len(annotations)} samples...")
    sample_idx = 0
    inference_times = []

    for ann_idx, gt in enumerate(annotations):
        img_path = gt['image_path']

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        try:
            # Run inference (full image, no bbox)
            start_time = time.time()
            preds = estimator.process_one_image(img_path, hand_box_source=args.hand_box_source)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)
        except Exception as e:
            print(f"Error on sample {ann_idx}: {e}")
            continue

        if not preds:
            print(f"No prediction for {img_path}")
            continue

        # Take first prediction (assuming single person)
        pred = preds[0]

        # Extract SMPL-X params for saving
        smplx_params = gt['smplx_params']

        save_dict = {
            'pred_vertices': pred['pred_vertices'],
            'pred_cam_t': pred['pred_cam_t'],
            'pred_keypoints_3d': pred['pred_keypoints_3d'],
            # GT SMPL-X params
            'gt_betas': np.array(smplx_params['betas']),
            'gt_body_pose': np.array(smplx_params['body_pose']),
            'gt_global_orient': np.array(smplx_params['global_orient']),
            'gt_transl': np.array(smplx_params.get('transl', np.zeros((1, 3)))),
            # Metadata
            'image_path': img_path,
            'scene': gt['scene'],
            'frame_id': gt['frame_id'],
            'cam_id': gt['cam_id'],
        }

        np.savez(samples_dir / f"sample_{sample_idx:05d}.npz", **save_dict)
        sample_idx += 1

        if (ann_idx + 1) % 10 == 0:
            print(f"Progress: {ann_idx+1}/{len(annotations)}, {sample_idx} samples saved")

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
