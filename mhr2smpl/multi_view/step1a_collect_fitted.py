#!/usr/bin/env python
"""
Step 1: Collect training pairs for MHR→SMPL network (v4 design).

Pipeline:
  1. Load MHR vertices from Stage 1 sample_*.npz files
  2. Barycentric surface mapping: MHR (18439 verts, cm) → SMPL topology (6890 verts, meters)
  3. Subsample + centroid-center → network input
  4. Supervision targets (GT or fitted via convert_mhr2smpl):
     - canonical joints (go=0, pelvis-centered) → FK loss target
     - body_pose + betas → param supervision target

--use_fitted: run convert_mhr2smpl optimization to get fitted params as supervision
              (slower but more realistic — matches real inference domain)

Usage (from mhr_smpl_conversion directory with pixi):
  cd /home/jiawei/timingyang/sam-3d-body/eval_ECCV/MHR/tools/mhr_smpl_conversion
  /home/jiawei/.pixi/bin/pixi run --manifest-path /home/jiawei/timingyang/sam-3d-body/eval_ECCV/MHR/pyproject.toml python \
    /home/jiawei/timingyang/CLEAN/Fast_sam-3d-body_mhr2smpl/mhr2smpl/step1_collect_from_stage2.py \
    --input_dir /path/to/outputs_EMDB/samples \
    --output_path /path/to/mhr2smpl/data/pairs_stage2.npz \
    --use_fitted
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # run from mhr_smpl_conversion dir

import numpy as np
import torch
from tqdm import tqdm

import smplx
from conversion import Conversion
from mhr.mhr import MHR


def _viz_sampling(smpl_model, smpl_target_verts_np, sample_idx,
                  ankle_foot_verts, num_samples, args, device):
    """Visualize sampling distribution on T-pose and real samples."""
    import cv2
    print(f"\n[Phase 5b] Visualizing sampling distribution...")

    def _draw(verts_all, verts_sampled, ankle_mask, title, w=600, h=800):
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        scale, cx, cy = 350, w // 2, h // 2 + 80
        def to2d(v):
            return np.stack([v[:, 0] * scale + cx, -v[:, 1] * scale + cy], axis=1).astype(int)
        p = to2d(verts_all)
        for i in range(len(p)):
            cv2.circle(img, tuple(p[i]), 1, (220, 220, 220), -1)
        ps = to2d(verts_sampled[~ankle_mask])
        for i in range(len(ps)):
            cv2.circle(img, tuple(ps[i]), 2, (255, 150, 50), -1)
        pa = to2d(verts_sampled[ankle_mask])
        for i in range(len(pa)):
            cv2.circle(img, tuple(pa[i]), 3, (0, 0, 255), -1)
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        n_ankle = ankle_mask.sum()
        cv2.putText(img, f"Total:{len(verts_sampled)} Ankle(red):{n_ankle} Other(blue):{len(verts_sampled)-n_ankle}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        return img

    with torch.no_grad():
        tpose_verts = smpl_model(betas=torch.zeros(1, 10).to(device),
                                 body_pose=torch.zeros(1, 69).to(device),
                                 global_orient=torch.zeros(1, 3).to(device)).vertices[0].cpu().numpy()

    ankle_mask = np.isin(sample_idx, ankle_foot_verts)
    viz_dir = Path(args.output_path).parent / "viz_sampling"
    viz_dir.mkdir(parents=True, exist_ok=True)

    img_tpose = _draw(tpose_verts, tpose_verts[sample_idx], ankle_mask, "T-pose sampling")
    cv2.imwrite(str(viz_dir / "sampling_tpose.jpg"), img_tpose)

    vis_indices = np.linspace(0, num_samples - 1, min(5, num_samples), dtype=int)
    for vi in vis_indices:
        all_v = smpl_target_verts_np[vi]
        img = _draw(all_v, all_v[sample_idx], ankle_mask, f"Sample {vi}")
        cv2.imwrite(str(viz_dir / f"sampling_sample_{vi:05d}.jpg"), img)

    print(f"  Saved {1 + len(vis_indices)} visualizations to {viz_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Stage 1 samples dir (sample_*.npz)")
    parser.add_argument("--smpl_model", default="./data/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output_path", required=True,
                        help="Output npz path for training pairs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_sampled_verts", type=int, default=1500,
                        help="Number of sampled SMPL vertices (out of 6890), with dense ankle/foot")
    parser.add_argument("--use_fitted", action="store_true",
                        help="Use convert_mhr2smpl fitted params as supervision (slower but more realistic)")
    parser.add_argument("--max_views", type=int, default=4,
                        help="Max views (for multi-view format compatibility; single-view fills slot 0)")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip sampling distribution visualization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sample_files = sorted(input_dir.glob("sample_*.npz"))
    if args.max_samples > 0:
        sample_files = sample_files[:args.max_samples]
    print(f"Found {len(sample_files)} sample files")
    print(f"Supervision mode: {'FITTED (convert_mhr2smpl)' if args.use_fitted else 'GT'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Load models =====
    print("Loading MHR model...")
    mhr = MHR.from_files(lod=1, device=device)
    print("Loading SMPL model...")
    smpl_model = smplx.SMPL(model_path=args.smpl_model, gender='neutral').to(device)
    print("Creating converter...")
    converter = Conversion(mhr_model=mhr, smpl_model=smpl_model,
                           method='pytorch', batch_size=args.batch_size)

    # ===== Phase 1: Load all samples =====
    print("\n[Phase 1] Loading samples...")
    all_pred_verts = []      # MHR vertices (preprocessed, cm)
    all_gt_params = []       # GT SMPL params (always loaded for reference)

    for f in tqdm(sample_files, desc="Loading"):
        data = np.load(f, allow_pickle=True)

        # === Same preprocessing as Stage 2 (stage2_convert_eval_smpl_EMDB.py lines 175-179) ===
        pred_verts_raw = data['pred_vertices'].copy()
        pred_verts_raw[:, [1, 2]] *= -1                    # YZ-flip
        pred_cam_t = data['pred_cam_t'].copy()
        pred_cam_t[[1, 2]] *= -1                           # YZ-flip cam_t
        pred_verts_mhr = (pred_verts_raw + pred_cam_t[None, :]) * 100.0  # translate + ×100 (cm)

        all_pred_verts.append(pred_verts_mhr)

        all_gt_params.append({
            'poses': data['gt_poses'],       # [72] full pose (go + body_pose)
            'betas': data['gt_betas'],       # [10]
        })

    num_samples = len(all_pred_verts)
    print(f"  Loaded {num_samples} samples")

    # ===== Phase 2: Barycentric mapping MHR → SMPL topology =====
    print(f"\n[Phase 2] Barycentric mapping MHR→SMPL on {num_samples} samples...")
    pred_verts_tensor = torch.tensor(np.stack(all_pred_verts)).float().to(device)

    # _compute_target_vertices: MHR [N, 18439, 3] cm → SMPL [N, 6890, 3] meters
    with torch.no_grad():
        smpl_target_verts = converter._compute_target_vertices(
            pred_verts_tensor, direction="mhr2smpl"
        )  # [N, 6890, 3] meters
    smpl_target_verts_np = smpl_target_verts.cpu().numpy()
    print(f"  SMPL target verts: {smpl_target_verts_np.shape}, "
          f"range=[{smpl_target_verts_np.min():.4f}, {smpl_target_verts_np.max():.4f}] m")

    # World-space joints from J_regressor applied to barycentric-mapped verts
    J_reg_np = smpl_model.J_regressor.detach().cpu().numpy()  # [24, 6890]
    smpl_joints_world = np.einsum('jv,nvd->njd', J_reg_np, smpl_target_verts_np)  # [N, 24, 3]
    print(f"  World-space joints (J_reg @ bary_verts): {smpl_joints_world.shape}, "
          f"range=[{smpl_joints_world.min():.4f}, {smpl_joints_world.max():.4f}] m")

    # ===== Phase 2b (optional): Run convert_mhr2smpl for fitted params =====
    if args.use_fitted:
        print(f"\n[Phase 2b] Running convert_mhr2smpl optimization on {num_samples} samples...")
        print("  (This is slow — ~35s/sample on CPU, faster on GPU with batching)")
        fitted_result = converter.convert_mhr2smpl(
            mhr_vertices=pred_verts_tensor,
            return_smpl_parameters=True,
            return_smpl_vertices=False,
            batch_size=args.batch_size,
        )
        fitted_params = fitted_result.result_parameters
        # Extract fitted body_pose and betas
        sup_body_pose = fitted_params['body_pose'].detach().cpu().numpy() if torch.is_tensor(fitted_params['body_pose']) else np.array(fitted_params['body_pose'])
        sup_betas = fitted_params['betas'].detach().cpu().numpy() if torch.is_tensor(fitted_params['betas']) else np.array(fitted_params['betas'])
        sup_label = "fitted"
        print(f"  Fitted body_pose: {sup_body_pose.shape}, range=[{sup_body_pose.min():.3f}, {sup_body_pose.max():.3f}]")
        print(f"  Fitted betas: {sup_betas.shape}, range=[{sup_betas.min():.3f}, {sup_betas.max():.3f}]")
        # global_orient: use fitted if available, else fall back to GT
        if 'global_orient' in fitted_params and fitted_params['global_orient'] is not None:
            sup_go = fitted_params['global_orient'].detach().cpu().numpy() if torch.is_tensor(fitted_params['global_orient']) else np.array(fitted_params['global_orient'])
            print(f"  Fitted global_orient: {sup_go.shape}")
        else:
            sup_go = np.stack([g['poses'][:3] for g in all_gt_params])
            print("  Note: no fitted global_orient found, using GT go as fallback")
    else:
        # Use GT params
        sup_body_pose = np.stack([g['poses'][3:] for g in all_gt_params])  # [N, 69]
        sup_betas = np.stack([g['betas'] for g in all_gt_params])          # [N, 10]
        sup_go = np.stack([g['poses'][:3] for g in all_gt_params])         # [N, 3]
        sup_label = "GT"

    # ===== Phase 3: Compute canonical joints from supervision params =====
    print(f"\n[Phase 3] Computing {sup_label} SMPL canonical joints (go=0, pelvis-centered)...")
    canon_joints_list = []
    for i in tqdm(range(0, num_samples, args.batch_size), desc=f"{sup_label} canonical joints"):
        end_i = min(i + args.batch_size, num_samples)
        bs = end_i - i

        body_pose = torch.tensor(sup_body_pose[i:end_i]).float().to(device)
        betas = torch.tensor(sup_betas[i:end_i]).float().to(device)
        go = torch.zeros(bs, 3).float().to(device)

        with torch.no_grad():
            output = smpl_model(betas=betas, body_pose=body_pose, global_orient=go)
        j = output.joints[:, :24].cpu().numpy()  # [B, 24, 3] meters
        j = j - j[:, 0:1]  # pelvis-center
        canon_joints_list.append(j)

    canon_joints = np.concatenate(canon_joints_list, axis=0)  # [N, 24, 3] meters
    print(f"  {sup_label} canonical joints: {canon_joints.shape}, "
          f"range=[{canon_joints.min():.4f}, {canon_joints.max():.4f}] m")

    # ===== Phase 4: Subsample + centroid-center SMPL target verts =====
    # Use LBS weights for dense sampling of ankle/foot regions
    print(f"\n[Phase 4] Subsampling {args.num_sampled_verts} / 6890 SMPL verts "
          f"(dense ankle/foot) + centroid-center...")
    n_total_verts = smpl_target_verts_np.shape[1]  # 6890
    num_target = args.num_sampled_verts  # 1000

    # --- Find ankle/foot region vertices (LBS weights) ---
    lbs_w = smpl_model.lbs_weights.detach().cpu().numpy()  # [6890, 24]
    # Joint 7=L_Ankle, 8=R_Ankle, 10=L_Foot, 11=R_Foot
    ankle_foot_score = lbs_w[:, [7, 8, 10, 11]].sum(axis=1)  # [6890]
    ankle_foot_verts = np.where(ankle_foot_score > 0.3)[0]
    print(f"  Ankle/foot vertices (LBS>0.3): {len(ankle_foot_verts)} / {n_total_verts}")

    # --- Base uniform sampling + dense ankle sampling ---
    num_extra = min(len(ankle_foot_verts), 300)  # Fixed 300 for ankle/foot
    num_base = num_target - num_extra
    base_idx = np.linspace(0, n_total_verts - 1, num_base, dtype=int)
    np.random.seed(42)
    extra_idx = np.random.choice(ankle_foot_verts, num_extra, replace=False)

    # Merge and deduplicate, maintain total = num_target
    sample_idx = np.unique(np.concatenate([base_idx, extra_idx]))
    # After dedup may be less than num_target, fill from remaining vertices
    if len(sample_idx) < num_target:
        remaining = np.setdiff1d(np.arange(n_total_verts), sample_idx)
        fill = np.random.choice(remaining, num_target - len(sample_idx), replace=False)
        sample_idx = np.sort(np.concatenate([sample_idx, fill]))
    # After dedup may exceed num_target, remove excess non-ankle vertices
    elif len(sample_idx) > num_target:
        ankle_set = set(ankle_foot_verts)
        is_ankle = np.array([v in ankle_set for v in sample_idx])
        # Prioritize removing non-ankle duplicate points
        non_ankle_idx = np.where(~is_ankle)[0]
        n_remove = len(sample_idx) - num_target
        remove_pos = np.random.choice(non_ankle_idx, min(n_remove, len(non_ankle_idx)), replace=False)
        sample_idx = np.delete(sample_idx, remove_pos)[:num_target]

    print(f"  Final sample: {len(sample_idx)} verts "
          f"(ankle/foot region: {np.isin(sample_idx, ankle_foot_verts).sum()})")

    smpl_verts_sub = smpl_target_verts_np[:, sample_idx]  # [N, V_sub, 3] meters
    centroids = smpl_verts_sub.mean(axis=1, keepdims=True)  # [N, 1, 3]
    smpl_verts_centered = smpl_verts_sub - centroids  # [N, V_sub, 3] meters, centroid-centered
    print(f"  Subsampled: {smpl_verts_centered.shape}")
    print(f"  Centroid range: [{centroids.min():.4f}, {centroids.max():.4f}] m")

    # ===== Phase 5: Sanity check =====
    print(f"\n[Phase 5] Sanity check...")
    residual = smpl_verts_centered.mean(axis=1)  # should be ~0
    print(f"  Centroid residual (should be ~0): max={np.abs(residual).max():.2e}")
    joint_norms = np.sqrt((canon_joints ** 2).sum(-1))  # [N, 24]
    print(f"  Joint distance from pelvis: mean={joint_norms.mean()*1000:.1f}mm, "
          f"max={joint_norms.max()*1000:.1f}mm")

    # ===== Phase 5b: Visualize sampling distribution =====
    if args.no_viz:
        print(f"\n[Phase 5b] Skipped (--no_viz)")
    else:
        _viz_sampling(smpl_model, smpl_target_verts_np, sample_idx,
                      ankle_foot_verts, num_samples, args, device)

    # ===== Phase 6: Save training data (multi-view format) =====
    print(f"\n[Phase 6] Saving training data ({sup_label} supervision, multi-view format)...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap single-view data into multi-view format [N, V_max, V_sub, 3]
    V_max = args.max_views
    V_sub = len(sample_idx)
    N = num_samples
    mv_verts = np.zeros((N, V_max, V_sub, 3), dtype=np.float32)
    mv_verts[:, 0] = smpl_verts_centered  # single view → slot 0
    view_mask = np.zeros((N, V_max), dtype=bool)
    view_mask[:, 0] = True  # only view 0 is valid

    save_dict = {
        # Network input (multi-view format, single view in slot 0)
        'smpl_target_verts_sampled': mv_verts,                                 # [N, V_max, V_sub, 3] meters
        'view_mask': view_mask,                                                # [N, V_max] bool
        'num_views_total': np.array([V_max]),
        'smpl_vert_sample_indices': sample_idx,                                # [V_sub] into 6890

        # Canonical FK loss target (go=0, pelvis-centered)
        'smpl_joints_canonical': canon_joints.astype(np.float32),             # [N, 24, 3] meters, pelvis-centered

        # World-space FK loss target (go supervision via world-space joints)
        'smpl_joints_world': smpl_joints_world.astype(np.float32),            # [N, 24, 3] world space (J_reg @ bary_verts)

        # Param supervision targets
        'gt_body_pose': sup_body_pose.astype(np.float32),                     # [N, 69] axis-angle, radians
        'gt_betas': sup_betas.astype(np.float32),                             # [N, 10]
        'gt_global_orient': sup_go.astype(np.float32),                        # [N, 3] axis-angle, radians

        # Metadata
        'supervision_mode': np.array([1 if args.use_fitted else 0]),          # 1=fitted, 0=GT
    }

    np.savez(str(output_path), **save_dict)
    file_size_mb = os.path.getsize(str(output_path)) / 1024 / 1024
    print(f"\n  Saved: {output_path} ({file_size_mb:.1f} MB)")
    for k, v in save_dict.items():
        print(f"  {k:30s} {str(v.shape):20s} dtype={v.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
