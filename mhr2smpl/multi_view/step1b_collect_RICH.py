#!/usr/bin/env python
"""
Step 1 (Multi-View): Collect training pairs from RICH dataset.

RICH has multi-camera data with SMPL GT format:
  - cam_id, frame_id, scene: for cross-view matching
  - gt_body_pose [1, 63], gt_betas [1, 10], gt_global_orient [1, 3]: SMPL GT
  - pred_vertices [18439, 3]: MHR mesh prediction

Groups samples by (scene, frame_id), yielding multi-view pairs.

Usage (from mhr_smpl_conversion dir with pixi):
  python step1_collect_RICH.py \
    --input_dir /path/to/outputs_RICH/samples \
    --output_path ./data/mv_pairs_RICH.npz \
    --smpl_model ./data/SMPL_NEUTRAL.pkl
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from tqdm import tqdm

import smplx
from conversion import Conversion
from mhr.mhr import MHR


def compute_sample_indices(smpl_model, num_target=1500):
    """Compute vertex subsampling indices (uniform + ankle/foot dense)."""
    n_total_verts = 6890
    lbs_w = smpl_model.lbs_weights.detach().cpu().numpy()
    ankle_foot_score = lbs_w[:, [7, 8, 10, 11]].sum(axis=1)
    ankle_foot_verts = np.where(ankle_foot_score > 0.3)[0]

    num_extra = min(len(ankle_foot_verts), 300)
    num_base = num_target - num_extra
    base_idx = np.linspace(0, n_total_verts - 1, num_base, dtype=int)
    np.random.seed(42)
    extra_idx = np.random.choice(ankle_foot_verts, num_extra, replace=False)

    sample_idx = np.unique(np.concatenate([base_idx, extra_idx]))
    if len(sample_idx) < num_target:
        remaining = np.setdiff1d(np.arange(n_total_verts), sample_idx)
        fill = np.random.choice(remaining, num_target - len(sample_idx), replace=False)
        sample_idx = np.sort(np.concatenate([sample_idx, fill]))
    elif len(sample_idx) > num_target:
        ankle_set = set(ankle_foot_verts)
        is_ankle = np.array([v in ankle_set for v in sample_idx])
        non_ankle_idx = np.where(~is_ankle)[0]
        n_remove = len(sample_idx) - num_target
        remove_pos = np.random.choice(non_ankle_idx, min(n_remove, len(non_ankle_idx)), replace=False)
        sample_idx = np.delete(sample_idx, remove_pos)[:num_target]

    print(f"  Sample indices: {len(sample_idx)} verts "
          f"(ankle/foot: {np.isin(sample_idx, ankle_foot_verts).sum()})")
    return sample_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="One or more RICH samples dirs (sample_*.npz)")
    parser.add_argument("--smpl_model", default="./data/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output_path", required=True,
                        help="Output npz path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_views", type=int, default=4,
                        help="Max views to store per frame (pad/truncate)")
    parser.add_argument("--num_sampled_verts", type=int, default=1500)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max multi-view frames to use (-1=all)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Group samples by (scene, frame_id) =====
    print("[Phase 0] Grouping RICH samples by (scene, frame)...")
    sample_files = []
    for d in args.input_dirs:
        found = sorted(Path(d).glob("sample_*.npz"))
        print(f"  {d}: {len(found)} files")
        sample_files.extend(found)
    print(f"  Total sample files: {len(sample_files)}")

    groups = defaultdict(list)  # (scene, frame_id) -> list of file paths
    for f in sample_files:
        d = np.load(f, allow_pickle=True)
        scene = str(d['scene'])
        frame = str(d['frame_id'])
        groups[(scene, frame)].append(f)

    # Only keep frames with >= 2 views
    multi_view_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"  Multi-view frames (>=2 cams): {len(multi_view_groups)}")

    if args.max_samples > 0 and len(multi_view_groups) > args.max_samples:
        keys = sorted(multi_view_groups.keys())[:args.max_samples]
        multi_view_groups = {k: multi_view_groups[k] for k in keys}
        print(f"  Truncated to {args.max_samples} frames (--max_samples)")
    view_counts = [len(v) for v in multi_view_groups.values()]
    print(f"  Views per frame: min={min(view_counts)}, max={max(view_counts)}, "
          f"mean={np.mean(view_counts):.1f}")

    # ===== Load models =====
    print("\n[Phase 1] Loading models...")
    mhr = MHR.from_files(lod=1, device=device)
    smpl_model = smplx.SMPL(model_path=args.smpl_model, gender='neutral').to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad_(False)
    converter = Conversion(mhr_model=mhr, smpl_model=smpl_model,
                           method='pytorch', batch_size=args.batch_size)

    # ===== Compute sample indices =====
    print("\n[Phase 2] Computing vertex sample indices...")
    sample_idx = compute_sample_indices(smpl_model, args.num_sampled_verts)
    V_sub = len(sample_idx)

    J_reg_np = smpl_model.J_regressor.detach().cpu().numpy()

    # ===== Process all multi-view frames =====
    V_max = args.max_views
    num_frames = len(multi_view_groups)
    print(f"\n[Phase 3] Processing {num_frames} multi-view frames (max {V_max} views)...")

    all_verts = np.zeros((num_frames, V_max, V_sub, 3), dtype=np.float32)
    all_mask = np.zeros((num_frames, V_max), dtype=bool)
    all_joints_world = np.zeros((num_frames, 24, 3), dtype=np.float32)
    all_gt_bp = []
    all_gt_betas = []
    all_gt_go = []

    sorted_groups = sorted(multi_view_groups.items())

    for fi, ((scene, frame_id), view_files) in enumerate(tqdm(sorted_groups, desc="Frames")):
        gt_loaded = False

        for vi, vf in enumerate(view_files[:V_max]):
            d = np.load(vf, allow_pickle=True)

            # MHR preprocessing (same as single-view step1)
            pred_verts_raw = d['pred_vertices'].copy()
            pred_verts_raw[:, [1, 2]] *= -1
            pred_cam_t = d['pred_cam_t'].copy()
            pred_cam_t[[1, 2]] *= -1
            mhr_verts_cm = (pred_verts_raw + pred_cam_t[None, :]) * 100.0

            # Barycentric mapping
            verts_tensor = torch.tensor(mhr_verts_cm).float().unsqueeze(0).to(device)
            with torch.no_grad():
                smpl_verts = converter._compute_target_vertices(
                    verts_tensor, direction="mhr2smpl"
                )[0].cpu().numpy()  # [6890, 3]

            # World-space joints from first view
            if vi == 0:
                all_joints_world[fi] = J_reg_np @ smpl_verts

            # Subsample + centroid-center
            verts_sub = smpl_verts[sample_idx]
            centroid = verts_sub.mean(axis=0, keepdims=True)
            verts_centered = verts_sub - centroid

            all_verts[fi, vi] = verts_centered
            all_mask[fi, vi] = True

            # GT (same across views, load once)
            if not gt_loaded:
                all_gt_bp.append(d['gt_body_pose'].flatten()[:63])      # [63]
                all_gt_betas.append(d['gt_betas'].flatten()[:10])       # [10]
                all_gt_go.append(d['gt_global_orient'].flatten()[:3])   # [3]
                gt_loaded = True

    all_gt_bp = np.stack(all_gt_bp).astype(np.float32)        # [N, 63]
    all_gt_betas = np.stack(all_gt_betas).astype(np.float32)  # [N, 10]
    all_gt_go = np.stack(all_gt_go).astype(np.float32)        # [N, 3]

    # Pad gt_body_pose to 69 dims (23 joints)
    gt_bp_69 = np.zeros((num_frames, 69), dtype=np.float32)
    gt_bp_69[:, :63] = all_gt_bp

    print(f"\n  Processed: {num_frames} frames")
    print(f"  View availability: {all_mask.sum(axis=0)}")

    # ===== Phase 4: Canonical joints =====
    print(f"\n[Phase 4] Computing canonical joints...")
    canon_joints_list = []
    for i in tqdm(range(0, num_frames, args.batch_size), desc="Canonical joints"):
        end_i = min(i + args.batch_size, num_frames)
        bs = end_i - i
        body_pose = torch.tensor(gt_bp_69[i:end_i]).float().to(device)
        betas = torch.tensor(all_gt_betas[i:end_i]).float().to(device)
        go = torch.zeros(bs, 3).float().to(device)
        with torch.no_grad():
            output = smpl_model(betas=betas, body_pose=body_pose, global_orient=go)
        j = output.joints[:, :24].cpu().numpy()
        j = j - j[:, 0:1]
        canon_joints_list.append(j)
    canon_joints = np.concatenate(canon_joints_list, axis=0)

    # ===== Phase 5: Save =====
    print(f"\n[Phase 5] Saving...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'smpl_target_verts_sampled': all_verts,
        'view_mask': all_mask,
        'num_views_total': np.array([V_max]),
        'smpl_vert_sample_indices': sample_idx,
        'smpl_joints_canonical': canon_joints.astype(np.float32),
        'smpl_joints_world': all_joints_world.astype(np.float32),
        'gt_body_pose': gt_bp_69,
        'gt_betas': all_gt_betas,
        'gt_global_orient': all_gt_go,
        'supervision_mode': np.array([0]),  # GT
    }

    np.savez(str(output_path), **save_dict)
    file_size_mb = os.path.getsize(str(output_path)) / 1024 / 1024
    print(f"\n  Saved: {output_path} ({file_size_mb:.1f} MB)")
    for k, v in save_dict.items():
        print(f"    {k:30s} {str(v.shape):20s} dtype={v.dtype}")
    print("\nDone!")


if __name__ == "__main__":
    main()
