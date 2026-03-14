# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Body Demo Script - Export 3D Mesh (PLY) files
Usage: python demo_human.py --image_path <image_path> --output_dir <output_directory>
 FOV_MODEL=s  FOV_LEVEL=0 MHR_NO_CORRECTIVES=1 python demo_human.py --image_path ./notebook/images/dancing.jpg --output_dir ./my_output --no-visualize --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine

--detector yolo --detector_model ./checkpoints/yolo/yolo11n.pt

"""

import argparse
import json
import pickle
import sys
import os
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from notebook.utils import (
    setup_sam_3d_body, setup_visualizer,
    visualize_2d_results, visualize_3d_mesh, save_mesh_results,
    process_image_with_mask
)
from tools.vis_utils import visualize_sample_together


# ============ Quantitative Evaluation Tools ============
def procrustes_align(S1, S2):
    """Procrustes alignment: align S1 to S2."""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1**2).sum()
    U, s, Vt = np.linalg.svd(X1.T @ X2)
    Z = np.eye(3)
    Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = s.sum() / var1
    return scale * (S1 @ R.T) + (mu2 - scale * (R @ mu1))


def compute_mpjpe(pred, gt):
    """Mean Per Joint Position Error (mm)."""
    return float(np.sqrt(((pred - gt)**2).sum(1)).mean() * 1000)


def compute_pa_mpjpe(pred, gt):
    """Procrustes-Aligned MPJPE (mm)."""
    return compute_mpjpe(procrustes_align(pred, gt), gt)


def compute_pve(pred_verts, gt_verts):
    """Per Vertex Error (mm)."""
    return float(np.sqrt(((pred_verts - gt_verts)**2).sum(1)).mean() * 1000)


def load_gt_smplx_params(gt_path):
    """Load GT SMPL-X parameters from pkl or npz file."""
    if gt_path.endswith('.pkl'):
        with open(gt_path, 'rb') as f:
            params = pickle.load(f)
    elif gt_path.endswith('.npz'):
        params = dict(np.load(gt_path, allow_pickle=True))
    else:
        raise ValueError(f"Unsupported GT file format: {gt_path}")
    return params


def compute_metrics_with_conversion(pred_outputs, gt_smplx_params, converter, smpl_model, device):
    """
    Compute MPJPE, PA-MPJPE, PVE using MHR -> SMPL conversion.

    Args:
        pred_outputs: List of prediction dicts from estimator
        gt_smplx_params: GT SMPL-X parameters dict
        converter: MHR->SMPL Conversion object
        smpl_model: SMPL model
        device: torch device

    Returns:
        Dict of per-person metrics
    """
    all_metrics = []

    for pid, pred in enumerate(pred_outputs):
        # Prepare pred MHR vertices: flip Y,Z, add cam_t, scale
        pred_verts_raw = pred['pred_vertices'].copy()
        pred_verts_raw[:, [1, 2]] *= -1
        pred_cam_t = pred['pred_cam_t'].copy()
        pred_cam_t[[1, 2]] *= -1
        pred_verts_mhr = (pred_verts_raw + pred_cam_t[None, :]) * 100.0

        # MHR -> SMPL conversion
        pred_verts_tensor = torch.tensor(pred_verts_mhr).float().unsqueeze(0)
        pred_result = converter.convert_mhr2smpl(
            mhr_vertices=pred_verts_tensor,
            return_smpl_parameters=True,
            return_smpl_vertices=True,
            batch_size=1,
        )
        pred_params = pred_result.result_parameters

        # Compute pred joints with global_orient=0
        with torch.no_grad():
            pred_output = smpl_model(
                betas=pred_params['betas'].float().to(device),
                body_pose=pred_params['body_pose'].float().to(device),
                global_orient=torch.zeros(1, 3).float().to(device),
            )
            pred_joints = pred_output.joints[:, :24].cpu().numpy()[0]
            pred_verts_canonical = pred_output.vertices.cpu().numpy()[0]

        # Prepare GT: SMPL-X body_pose -> SMPL body_pose
        gt_betas = np.array(gt_smplx_params['betas']).squeeze()
        gt_body_pose_smplx = np.array(gt_smplx_params['body_pose']).squeeze()
        gt_global_orient = np.array(gt_smplx_params['global_orient']).squeeze()
        gt_body_pose_smpl = np.zeros(69)
        gt_body_pose_smpl[:len(gt_body_pose_smplx)] = gt_body_pose_smplx

        # Compute GT joints with global_orient=0
        with torch.no_grad():
            gt_output = smpl_model(
                betas=torch.tensor(gt_betas).float().unsqueeze(0).to(device),
                body_pose=torch.tensor(gt_body_pose_smpl).float().unsqueeze(0).to(device),
                global_orient=torch.zeros(1, 3).float().to(device),
            )
            gt_joints = gt_output.joints[:, :24].cpu().numpy()[0]
            gt_verts = gt_output.vertices.cpu().numpy()[0]

        # Root-relative joints
        pred_j24 = pred_joints - pred_joints[0:1]
        gt_j24 = gt_joints - gt_joints[0:1]

        # Compute metrics
        m = compute_mpjpe(pred_j24, gt_j24)
        pa = compute_pa_mpjpe(pred_j24, gt_j24)
        pv = compute_pve(
            pred_verts_canonical - pred_verts_canonical.mean(0),
            gt_verts - gt_verts.mean(0)
        )

        all_metrics.append({
            'person_id': pid,
            'MPJPE_mm': round(m, 2),
            'PA-MPJPE_mm': round(pa, 2),
            'PVE_mm': round(pv, 2),
        })

    return all_metrics


# ============ Timing Tools ============
class TimingStats:
    """Collect and summarize timing statistics"""
    def __init__(self):
        self.timings = {}

    def add(self, name, duration):
        self.timings[name] = duration

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)

        total = 0
        for name, duration in self.timings.items():
            print(f"  {name}: {duration:.4f}s")
            if not name.startswith("  "):  # Only accumulate top-level timings
                total += duration

        print("-" * 70)
        print(f"  TOTAL: {total:.4f}s")
        print("=" * 70)


timing_stats = TimingStats()


def main(args):
    pipeline_total_start = time.time()

    print("=" * 60)
    print("SAM 3D Body Demo - 3D Mesh Export")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Model: {args.model}")
    print(f"Detector: {args.detector}" + (f" ({args.detector_model})" if args.detector in ["yolo", "yolo_pose"] else ""))
    print(f"Hand Box Source: {args.hand_box_source}")
    print(f"Local Checkpoint: {'✓ (' + args.local_checkpoint + ')' if args.local_checkpoint else '✗ (using HuggingFace)'}")

    # ============================================================
    # 1. Load model
    # ============================================================
    print("\n" + "-" * 60)
    print("[1/5] Loading SAM 3D Body model...")
    print("-" * 60)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_load_start = time.time()

    t0 = time.time()
    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,  # Local checkpoint
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  [model_loading] setup_sam_3d_body: {time.time() - t0:.4f}s")
    timing_stats.add("  [model_loading] setup_sam_3d_body", time.time() - t0)

    t0 = time.time()
    visualizer = setup_visualizer()
    print(f"  [model_loading] setup_visualizer: {time.time() - t0:.4f}s")
    timing_stats.add("  [model_loading] setup_visualizer", time.time() - t0)

    model_load_total = time.time() - model_load_start
    print(f"  [model_loading] TOTAL: {model_load_total:.4f}s")
    timing_stats.add("[model_loading] TOTAL", model_load_total)
    print("Model loading complete!")

    # ============================================================
    # 2. Read and process image
    # ============================================================
    print("\n" + "-" * 60)
    print(f"[2/5] Processing image: {args.image_path}")
    print("-" * 60)

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found - {args.image_path}")
        return

    process_start = time.time()

    t0 = time.time()
    img_cv2 = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    print(f"  [process_image] read_image: {time.time() - t0:.4f}s")
    timing_stats.add("  [process_image] read_image", time.time() - t0)

    # Run inference
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outputs = estimator.process_one_image(
        args.image_path,
        hand_box_source=args.hand_box_source,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.time() - t0
    print(f"  [process_image] inference (process_one_image): {inference_time:.4f}s")
    timing_stats.add("  [process_image] inference", inference_time)

    process_total = time.time() - process_start
    print(f"  [process_image] TOTAL: {process_total:.4f}s")
    timing_stats.add("[process_image] TOTAL", process_total)

    print(f"Detected {len(outputs)} person(s)")

    if not outputs:
        print("No person detected, exiting.")
        return

    print(f"Output fields: {list(outputs[0].keys())}")

    # ============================================================
    # 2.5. Visualize Hand Boxes
    # ============================================================
    print("\n" + "-" * 60)
    print("[2.5/5] Visualizing Hand Boxes...")
    print("-" * 60)

    hand_box_img = img_cv2.copy()
    for i, person in enumerate(outputs):
        # Left hand (blue)
        if "lhand_bbox" in person:
            bbox = person["lhand_bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(hand_box_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(hand_box_img, f"L{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            print(f"  Person {i} left hand bbox: [{x1}, {y1}, {x2}, {y2}]")

        # Right hand (red)
        if "rhand_bbox" in person:
            bbox = person["rhand_bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(hand_box_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(hand_box_img, f"R{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"  Person {i} right hand bbox: [{x1}, {y1}, {x2}, {y2}]")

    # Save hand box visualization
    os.makedirs(args.output_dir, exist_ok=True)
    hand_box_path = os.path.join(args.output_dir, "hand_boxes.jpg")
    cv2.imwrite(hand_box_path, hand_box_img)
    print(f"  Saved Hand Box visualization: {hand_box_path}")

    # Generate concat image: [original, skeleton, mesh, side view]
    print("  Generating concat visualization...")
    concat_img = visualize_sample_together(img_cv2, outputs, estimator.faces)

    # Add hand box to concat image: [original, hand_box, skeleton, mesh, side view]
    # Need to split the concat_img into parts
    h, w = img_cv2.shape[:2]
    img_orig = concat_img[:, :w]
    img_keypoints = concat_img[:, w:2*w]
    img_mesh = concat_img[:, 2*w:3*w]
    img_side = concat_img[:, 3*w:]

    # Re-concat: [original, hand_box, skeleton, mesh, side view]
    concat_with_hands = np.concatenate([img_orig, hand_box_img, img_keypoints, img_mesh, img_side], axis=1)
    concat_path = os.path.join(args.output_dir, "concat_all.jpg")
    cv2.imwrite(concat_path, concat_with_hands.astype(np.uint8))
    print(f"  Saved Concat visualization: {concat_path}")
    print(f"  Layout: [Original | Hand Box | Skeleton | Mesh | Side View]")

    # ============================================================
    # 3. Visualize 2D keypoints
    # ============================================================
    if args.visualize:
        print("\n" + "-" * 60)
        print("[3/5] Visualizing 2D keypoints...")
        print("-" * 60)

        viz_2d_start = time.time()

        t0 = time.time()
        vis_results = visualize_2d_results(img_cv2, outputs, visualizer)
        print(f"  [visualize_2d] render_keypoints: {time.time() - t0:.4f}s")
        timing_stats.add("  [visualize_2d] render_keypoints", time.time() - t0)

        for i, vis_img in enumerate(vis_results):
            vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_rgb)
            plt.title(f'Person {i} - 2D Keypoints & BBox')
            plt.axis('off')
            if args.save_viz:
                t0 = time.time()
                viz_path = os.path.join(args.output_dir, f"keypoints_person_{i}.png")
                plt.savefig(viz_path, bbox_inches='tight', dpi=150)
                print(f"  [visualize_2d] save_keypoints_{i}: {time.time() - t0:.4f}s")
                print(f"  Saved keypoint visualization: {viz_path}")
            plt.show()

        viz_2d_total = time.time() - viz_2d_start
        print(f"  [visualize_2d] TOTAL: {viz_2d_total:.4f}s")
        timing_stats.add("[visualize_2d] TOTAL", viz_2d_total)

    # ============================================================
    # 4. Visualize 3D Mesh
    # ============================================================
    if args.visualize:
        print("\n" + "-" * 60)
        print("[4/5] Visualizing 3D Mesh...")
        print("-" * 60)

        viz_3d_start = time.time()

        t0 = time.time()
        mesh_results = visualize_3d_mesh(img_cv2, outputs, estimator.faces)
        print(f"  [visualize_3d] render_mesh: {time.time() - t0:.4f}s")
        timing_stats.add("  [visualize_3d] render_mesh", time.time() - t0)

        for i, combined_img in enumerate(mesh_results):
            combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(20, 5))
            plt.imshow(combined_rgb)
            plt.title(f'Person {i}: Original | Mesh Overlay | Front View | Side View')
            plt.axis('off')
            if args.save_viz:
                t0 = time.time()
                viz_path = os.path.join(args.output_dir, f"mesh_viz_person_{i}.png")
                plt.savefig(viz_path, bbox_inches='tight', dpi=150)
                print(f"  [visualize_3d] save_mesh_viz_{i}: {time.time() - t0:.4f}s")
                print(f"  Saved Mesh visualization: {viz_path}")
            plt.show()

        viz_3d_total = time.time() - viz_3d_start
        print(f"  [visualize_3d] TOTAL: {viz_3d_total:.4f}s")
        timing_stats.add("[visualize_3d] TOTAL", viz_3d_total)

    # ============================================================
    # 5. Save 3D Mesh files (PLY)
    # ============================================================
    print("\n" + "-" * 60)
    print(f"[5/5] Saving 3D Mesh files to: {args.output_dir}")
    print("-" * 60)

    save_start = time.time()

    # Get image name
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]

    # Create output directory
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  [save_results] create_output_dir: {time.time() - t0:.4f}s")

    # Save all results (PLY mesh, overlay images, bbox images)
    t0 = time.time()
    ply_files = save_mesh_results(
        img_cv2,
        outputs,
        estimator.faces,
        args.output_dir,
        image_name
    )
    print(f"  [save_results] save_mesh_results: {time.time() - t0:.4f}s")
    timing_stats.add("  [save_results] save_mesh_results", time.time() - t0)

    save_total = time.time() - save_start
    print(f"  [save_results] TOTAL: {save_total:.4f}s")
    timing_stats.add("[save_results] TOTAL", save_total)

    print("\n" + "=" * 60)
    print(f"Done! Results saved to: {args.output_dir}")
    print("=" * 60)
    print(f"  - Number of PLY 3D Mesh files: {len(ply_files)}")
    for ply_file in ply_files:
        print(f"    - {ply_file}")
    print("\nTip: PLY files can be opened with MeshLab, Blender, or other 3D software")

    # ============================================================
    # 6. (Optional) Compute quantitative evaluation metrics (MPJPE, PA-MPJPE, PVE)
    # ============================================================
    if args.gt_path and os.path.exists(args.gt_path):
        print("\n" + "-" * 60)
        print(f"[6/7] Computing quantitative evaluation metrics...")
        print("-" * 60)

        metrics_start = time.time()

        try:
            # Add MHR conversion tools to path
            mhr_conv_dir = args.mhr_conversion_dir
            if mhr_conv_dir and os.path.exists(mhr_conv_dir):
                sys.path.insert(0, mhr_conv_dir)

            from mhr.mhr import MHR
            import smplx
            from conversion import Conversion

            # Load models for conversion
            t0 = time.time()
            print(f"  Loading MHR model...")
            mhr_model = MHR.from_files(lod=1, device=device)
            print(f"  Loading SMPL model: {args.smpl_model_path}")
            smpl_model = smplx.SMPL(model_path=args.smpl_model_path, gender='neutral').to(device)
            print(f"  Creating MHR->SMPL converter...")
            converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method='pytorch', batch_size=1)
            print(f"  [metrics] Model loading: {time.time() - t0:.4f}s")

            # Load GT
            t0 = time.time()
            print(f"  Loading GT: {args.gt_path}")
            gt_smplx_params = load_gt_smplx_params(args.gt_path)
            print(f"  [metrics] GT loading: {time.time() - t0:.4f}s")

            # Compute metrics
            t0 = time.time()
            metrics = compute_metrics_with_conversion(outputs, gt_smplx_params, converter, smpl_model, device)
            print(f"  [metrics] Metric computation: {time.time() - t0:.4f}s")

            # Print metrics
            print("\n  " + "=" * 50)
            print("  Quantitative Evaluation Metrics (MHR -> SMPL Space, 24 joints)")
            print("  " + "=" * 50)
            for m in metrics:
                print(f"  Person {m['person_id']}: "
                      f"MPJPE={m['MPJPE_mm']:.1f}mm  "
                      f"PA-MPJPE={m['PA-MPJPE_mm']:.1f}mm  "
                      f"PVE={m['PVE_mm']:.1f}mm")

            if len(metrics) > 1:
                avg_mpjpe = np.mean([m['MPJPE_mm'] for m in metrics])
                avg_pa = np.mean([m['PA-MPJPE_mm'] for m in metrics])
                avg_pve = np.mean([m['PVE_mm'] for m in metrics])
                print(f"  Average:   "
                      f"MPJPE={avg_mpjpe:.1f}mm  "
                      f"PA-MPJPE={avg_pa:.1f}mm  "
                      f"PVE={avg_pve:.1f}mm")
            print("  " + "=" * 50)

            # Save metrics JSON
            metrics_result = {
                'image_path': args.image_path,
                'gt_path': args.gt_path,
                'num_persons': len(outputs),
                'per_person_metrics': metrics,
                'inference_time_s': inference_time,
            }
            if len(metrics) > 0:
                metrics_result['avg_MPJPE_mm'] = round(float(np.mean([m['MPJPE_mm'] for m in metrics])), 2)
                metrics_result['avg_PA-MPJPE_mm'] = round(float(np.mean([m['PA-MPJPE_mm'] for m in metrics])), 2)
                metrics_result['avg_PVE_mm'] = round(float(np.mean([m['PVE_mm'] for m in metrics])), 2)

            metrics_json_path = os.path.join(args.output_dir, f"{image_name}_metrics.json")
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_result, f, indent=2)
            print(f"\n  Metrics saved: {metrics_json_path}")

        except ImportError as e:
            print(f"  [WARNING] Cannot import evaluation dependencies: {e}")
            print(f"  Please ensure --mhr_conversion_dir points to the correct mhr_smpl_conversion directory")
            print(f"  and that mhr, smplx, and other required packages are installed")
        except Exception as e:
            print(f"  [ERROR] Evaluation metric computation failed: {e}")
            import traceback
            traceback.print_exc()

        metrics_total = time.time() - metrics_start
        timing_stats.add("[metrics] TOTAL", metrics_total)

    # ============================================================
    # 7. (Optional) Mask-based inference
    # ============================================================
    if args.mask_path and os.path.exists(args.mask_path):
        print("\n" + "-" * 60)
        print(f"[7/7] Running mask-based inference: {args.mask_path}")
        print("-" * 60)

        mask_start = time.time()

        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mask_outputs = process_image_with_mask(estimator, args.image_path, args.mask_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  [mask_inference] process_image_with_mask: {time.time() - t0:.4f}s")
        timing_stats.add("  [mask_inference] process_image_with_mask", time.time() - t0)

        if mask_outputs:
            mask_output_dir = os.path.join(args.output_dir, "mask_based")
            os.makedirs(mask_output_dir, exist_ok=True)

            t0 = time.time()
            mask_ply_files = save_mesh_results(
                img_cv2,
                mask_outputs,
                estimator.faces,
                mask_output_dir,
                f"mask_{image_name}"
            )
            print(f"  [mask_inference] save_mask_results: {time.time() - t0:.4f}s")
            print(f"  Mask-based results saved to: {mask_output_dir} ({len(mask_ply_files)} files)")
        else:
            print("  Mask-based inference detected no person")

        mask_total = time.time() - mask_start
        print(f"  [mask_inference] TOTAL: {mask_total:.4f}s")
        timing_stats.add("[mask_inference] TOTAL", mask_total)

    # ============================================================
    # Print timing summary
    # ============================================================
    pipeline_total = time.time() - pipeline_total_start
    timing_stats.add("[pipeline] TOTAL", pipeline_total)
    timing_stats.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Export 3D Mesh (PLY) files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (defaults to YOLO11)
  python demo_human.py --image_path ./notebook/images/dancing.jpg

  # Specify output directory
  python demo_human.py --image_path ./notebook/images/dancing.jpg --output_dir ./my_output

  # Use ViT-H model
  python demo_human.py --image_path ./notebook/images/dancing.jpg --model facebook/sam-3d-body-vith

  # Use different YOLO model variants (n/s/m/l/x)
  python demo_human.py --image_path ./notebook/images/dancing.jpg --detector_model ./checkpoints/yolo/yolo11m.pt

  # Use ViTDet (Detectron2) detector
  python demo_human.py --image_path ./notebook/images/dancing.jpg --detector vitdet

  # No visualization display, only save files
  python demo_human.py --image_path ./notebook/images/dancing.jpg --no-visualize

  # With quantitative evaluation metrics (requires GT SMPL-X parameters)
  python demo_human.py --image_path ./image.bmp --gt_path ./gt/010.pkl \
      --mhr_conversion_dir ./path/to/mhr_smpl_conversion
        """
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default='./notebook/images/dancing.jpg',
        help="Input image path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam-3d-body-dinov3",
        choices=["facebook/sam-3d-body-dinov3", "facebook/sam-3d-body-vith"],
        help="Model selection (default: facebook/sam-3d-body-dinov3)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=["vitdet", "yolo", "yolo_pose"],
        help="Person detector: vitdet (Detectron2), yolo (YOLO11), yolo_pose (YOLO11-Pose with keypoints) (default: yolo)"
    )
    parser.add_argument(
        "--hand_box_source",
        type=str,
        default="body_decoder",
        choices=["body_decoder", "yolo_pose"],
        help="Hand box source: body_decoder (from body decoder output), yolo_pose (computed from YOLO-Pose wrist positions) (default: body_decoder)"
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="./checkpoints/yolo/yolo11n.pt",
        help="YOLO model path (default: ./checkpoints/yolo/yolo11n.pt)"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="(Optional) Mask image path for mask-based inference"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Display visualization results (default: True)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Do not display visualization results"
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        default=True,
        help="Save visualization images (default: True)"
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default="./checkpoints/sam-3d-body-dinov3",
        help="Local checkpoint directory path (containing model.ckpt and model_config.yaml), used to override HuggingFace config"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="(Optional) GT SMPL-X parameter file path (.pkl or .npz) for computing MPJPE/PA-MPJPE/PVE"
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="./data/SMPL_NEUTRAL.pkl",  # NOTE: Configure this path to point to your SMPL_NEUTRAL.pkl file
        help="Path to SMPL_NEUTRAL.pkl model file"
    )
    parser.add_argument(
        "--mhr_conversion_dir",
        type=str,
        default="./tools/mhr_smpl_conversion",  # NOTE: Configure this path to point to your mhr_smpl_conversion directory
        help="Path to mhr_smpl_conversion tool directory (containing conversion.py, mhr/, etc.)"
    )

    args = parser.parse_args()
    main(args)
