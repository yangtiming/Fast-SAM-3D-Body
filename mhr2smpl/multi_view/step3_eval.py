#!/usr/bin/env python
"""
Step 3 (Multi-View): Evaluate multi-view MHR2SMPL on multi-camera data.

Pipeline per frame:
  1. Per view: MHR [18439, 3] → barycentric mapping → SMPL [6890, 3] → subsample → centroid-center
  2. Multi-view model: encode per view → confidence-weighted fusion → decode → SMPL params
  3. SMPL forward (go=0) → canonical joints [24, 3]
  4. Compare with GT (MPJPE, PA-MPJPE, PVE)

Reports:
  - Multi-view vs single-view comparison
  - Per-view confidence statistics
  - Degradation under view dropout

Usage:
  python step3_eval_multiview.py \
    --input_dirs /path/to/cam01/samples /path/to/cam02/samples \
    --model_dir ./experiments/mv_run_1 \
    --smpl_model ./data/SMPL_NEUTRAL.pkl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from tqdm import tqdm

from mhr.mhr import MHR
from conversion import Conversion
import smplx

from multiview_net import MinimalMultiViewNet, split_output


NUM_EVAL_JOINTS = 24


# ==================== Metrics (same as single-view step3) ====================

def procrustes_align(S1, S2):
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1**2).sum()
    U, s, Vt = np.linalg.svd(X1.T @ X2)
    Z = np.eye(3); Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = s.sum() / var1
    return scale * (S1 @ R.T) + (mu2 - scale * (R @ mu1))


def mpjpe(pred, gt):
    return np.sqrt(((pred - gt)**2).sum(1)).mean() * 1000


def pa_mpjpe(pred, gt):
    return mpjpe(procrustes_align(pred, gt), gt)


def pve(pred, gt):
    return np.sqrt(((pred - gt)**2).sum(1)).mean() * 1000


def compute_metrics(pred_joints, gt_joints, pred_verts, gt_verts, mask=None):
    num_samples = pred_joints.shape[0]
    if mask is None:
        mask = np.ones(num_samples, dtype=bool)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    all_mpjpe, all_pa, all_pve = [], [], []
    for i in indices:
        pred_j24 = pred_joints[i] - pred_joints[i, 0:1]
        gt_j24 = gt_joints[i] - gt_joints[i, 0:1]
        all_mpjpe.append(mpjpe(pred_j24, gt_j24))
        all_pa.append(pa_mpjpe(pred_j24, gt_j24))
        all_pve.append(pve(
            pred_verts[i] - pred_verts[i].mean(0),
            gt_verts[i] - gt_verts[i].mean(0)
        ))
    all_mpjpe, all_pa, all_pve = np.array(all_mpjpe), np.array(all_pa), np.array(all_pve)
    return {
        'MPJPE (mm)': float(np.mean(all_mpjpe)),
        'MPJPE_median': float(np.median(all_mpjpe)),
        'MPJPE_std': float(np.std(all_mpjpe)),
        'PA-MPJPE (mm)': float(np.mean(all_pa)),
        'PA-MPJPE_median': float(np.median(all_pa)),
        'PA-MPJPE_std': float(np.std(all_pa)),
        'PVE (mm)': float(np.mean(all_pve)),
        'PVE_std': float(np.std(all_pve)),
        'num_samples': int(len(indices)),
    }


# ==================== Multi-view sample matching ====================

def match_samples_across_views(input_dirs):
    """Match samples across views by filename. Returns frames with >= 2 views."""
    num_views = len(input_dirs)
    per_view_files = []
    for d in input_dirs:
        files = {f.stem: f for f in sorted(Path(d).glob("sample_*.npz"))}
        per_view_files.append(files)
        print(f"  View {len(per_view_files)-1}: {len(files)} samples")

    all_keys = set()
    for files in per_view_files:
        all_keys.update(files.keys())

    matched = []
    for key in sorted(all_keys):
        paths = [files.get(key, None) for files in per_view_files]
        num_valid = sum(p is not None for p in paths)
        if num_valid >= 1:  # eval: include single-view too for comparison
            matched.append((key, paths))

    multi_view_count = sum(1 for _, paths in matched if sum(p is not None for p in paths) >= 2)
    print(f"  Total frames: {len(matched)}, multi-view (>=2): {multi_view_count}")
    return matched, num_views


def pad_body_pose(bp_21, device):
    bs = bp_21.shape[0]
    bp_full = torch.zeros(bs, 69, device=device)
    bp_full[:, :63] = bp_21
    return bp_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="Per-camera Stage 1 sample dirs")
    parser.add_argument("--model_dir", required=True,
                        help="Trained multi-view model directory")
    parser.add_argument("--smpl_model", default="./data/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_views_eval", type=int, default=2,
                        help="Number of views to use at evaluation")
    parser.add_argument("--single_view_baseline", action="store_true",
                        help="Also run single-view for comparison")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    print(f"Model config: {json.dumps(config, indent=2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Match samples across views =====
    print(f"\n[Phase 0] Matching samples across {len(args.input_dirs)} views...")
    matched, num_views = match_samples_across_views(args.input_dirs)
    if args.max_samples > 0:
        matched = matched[:args.max_samples]
    num_frames = len(matched)

    # ===== Load models =====
    print("\n[Phase 1] Loading models...")
    mhr = MHR.from_files(lod=1, device=device)
    smpl_model = smplx.SMPL(model_path=args.smpl_model, gender='neutral').to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad_(False)
    converter = Conversion(mhr_model=mhr, smpl_model=smpl_model,
                           method='pytorch', batch_size=args.batch_size)

    # Load multi-view network
    print(f"Loading multi-view model from {model_dir / 'best_model.pth'}...")
    input_dim = config.get('V_sub', 1500) * 3
    if config.get('from_scratch', False) or config.get('pretrained_path') is None:
        net = MinimalMultiViewNet.from_scratch(device=device, input_dim=input_dim)
    else:
        net = MinimalMultiViewNet.from_pretrained(
            config['pretrained_path'], device=device, input_dim=input_dim,
        )
    net = net.to(device)
    net.load_state_dict(torch.load(model_dir / "best_model.pth",
                                    map_location=device, weights_only=True))
    net.eval()
    print(f"  Trainable params: {config.get('trainable_params', 'N/A'):,}")

    # Load sample indices
    sample_idx = np.load(model_dir / "sample_idx.npy")
    V_sub = len(sample_idx)
    sample_idx_t = torch.from_numpy(sample_idx.astype(np.int64)).to(device)

    V_sel = args.num_views_eval

    # ===== Phase 2: Inference =====
    print(f"\n[Phase 2] Multi-view inference on {num_frames} frames ({V_sel} views)...")

    pred_joints_mv = []       # multi-view predictions
    pred_verts_mv = []
    pred_joints_sv = []       # single-view predictions (view 0 only)
    pred_verts_sv = []
    gt_joints_all = []
    gt_verts_all = []
    all_confidences = []      # [N, V_sel] confidence weights
    valid_frames = []

    t_total_start = time.time()

    for fi, (frame_key, paths) in enumerate(tqdm(matched, desc="Inference")):
        # Load GT from first available view
        gt_params = None
        for path in paths:
            if path is not None:
                data = np.load(path, allow_pickle=True)
                # body_pose may be (1,63) or (63,); global_orient (1,3) or (3,)
                bp = data['gt_body_pose'].squeeze()    # (63,) or (69,)
                go = data['gt_global_orient'].squeeze() # (3,)
                betas = data['gt_betas'].squeeze()      # (10,)
                # pad body_pose to 69 if only 63 (21 joints → 23 joints)
                if bp.shape[0] < 69:
                    bp = np.concatenate([bp, np.zeros(69 - bp.shape[0])], axis=0)
                gt_params = {
                    'poses': np.concatenate([go, bp], axis=-1),  # (72,)
                    'betas': betas,
                }
                break
        if gt_params is None:
            continue

        # Process each view
        view_inputs = []  # list of [V_sub*3] tensors
        view_valid = []

        for vi, path in enumerate(paths):
            if path is None or vi >= V_sel:
                view_inputs.append(torch.zeros(V_sub * 3, device=device))
                view_valid.append(False)
                continue

            data = np.load(path, allow_pickle=True)
            pred_verts_raw = data['pred_vertices'].copy()
            pred_verts_raw[:, [1, 2]] *= -1
            pred_cam_t = data['pred_cam_t'].copy()
            pred_cam_t[[1, 2]] *= -1
            mhr_verts_cm = (pred_verts_raw + pred_cam_t[None, :]) * 100.0

            verts_tensor = torch.tensor(mhr_verts_cm).float().unsqueeze(0).to(device)
            with torch.no_grad():
                smpl_verts = converter._compute_target_vertices(
                    verts_tensor, direction="mhr2smpl"
                )[0]  # [6890, 3]

            verts_sub = smpl_verts[sample_idx_t]                    # [V_sub, 3]
            centroid = verts_sub.mean(0, keepdim=True)
            verts_centered = (verts_sub - centroid).reshape(-1)     # [V_sub*3]

            view_inputs.append(verts_centered)
            view_valid.append(True)

        # Pad to V_sel views
        while len(view_inputs) < V_sel:
            view_inputs.append(torch.zeros(V_sub * 3, device=device))
            view_valid.append(False)

        if sum(view_valid[:V_sel]) == 0:
            continue

        valid_frames.append(fi)

        # Multi-view forward
        views_t = torch.stack(view_inputs[:V_sel]).unsqueeze(0)     # [1, V_sel, D]
        mask_t = torch.tensor([view_valid[:V_sel]], dtype=torch.bool, device=device)

        with torch.no_grad():
            pred_mv, conf_w = net(views_t, mask_t)
        all_confidences.append(conf_w[0].cpu().numpy())

        pred_go, pred_bp, pred_betas = split_output(pred_mv)
        bp_full = pad_body_pose(pred_bp, device)
        with torch.no_grad():
            output = smpl_model(
                betas=pred_betas,
                body_pose=bp_full,
                global_orient=torch.zeros(1, 3, device=device),
            )
        pred_joints_mv.append(output.joints[0, :24].cpu().numpy())
        pred_verts_mv.append(output.vertices[0].cpu().numpy())

        # Single-view baseline (view 0 only)
        if args.single_view_baseline and view_valid[0]:
            sv_input = view_inputs[0].unsqueeze(0)  # [1, D]
            with torch.no_grad():
                pred_sv = net.forward_single_view(sv_input)
            sv_go, sv_bp, sv_betas = split_output(pred_sv)
            sv_bp_full = pad_body_pose(sv_bp, device)
            with torch.no_grad():
                sv_output = smpl_model(
                    betas=sv_betas,
                    body_pose=sv_bp_full,
                    global_orient=torch.zeros(1, 3, device=device),
                )
            pred_joints_sv.append(sv_output.joints[0, :24].cpu().numpy())
            pred_verts_sv.append(sv_output.vertices[0].cpu().numpy())
        elif args.single_view_baseline:
            pred_joints_sv.append(np.zeros((24, 3)))
            pred_verts_sv.append(np.zeros((6890, 3)))

        # GT joints
        poses_t = torch.tensor(gt_params['poses']).float().unsqueeze(0).to(device)
        betas_t = torch.tensor(gt_params['betas']).float().unsqueeze(0).to(device)
        with torch.no_grad():
            gt_output = smpl_model(
                betas=betas_t,
                body_pose=poses_t[:, 3:],
                global_orient=torch.zeros(1, 3, device=device),
            )
        gt_joints_all.append(gt_output.joints[0, :24].cpu().numpy())
        gt_verts_all.append(gt_output.vertices[0].cpu().numpy())

    t_total = time.time() - t_total_start

    # ===== Phase 3: Compute metrics =====
    pred_joints_mv = np.stack(pred_joints_mv)
    pred_verts_mv = np.stack(pred_verts_mv)
    gt_joints_all = np.stack(gt_joints_all)
    gt_verts_all = np.stack(gt_verts_all)
    all_confidences = np.stack(all_confidences)
    num_eval = len(pred_joints_mv)

    print(f"\n{'='*70}")
    print(f"Multi-View Results ({num_eval} frames, {V_sel} views)")
    print(f"{'='*70}")

    metrics_mv = compute_metrics(pred_joints_mv, gt_joints_all, pred_verts_mv, gt_verts_all)
    print(f"\nMulti-View ({V_sel} views):")
    print(f"  MPJPE:     {metrics_mv['MPJPE (mm)']:.1f} +/- {metrics_mv['MPJPE_std']:.1f} mm")
    print(f"  PA-MPJPE:  {metrics_mv['PA-MPJPE (mm)']:.1f} +/- {metrics_mv['PA-MPJPE_std']:.1f} mm")
    print(f"  PVE:       {metrics_mv['PVE (mm)']:.1f} +/- {metrics_mv['PVE_std']:.1f} mm")

    results = {'multi_view': metrics_mv}

    if args.single_view_baseline and len(pred_joints_sv) > 0:
        pred_joints_sv = np.stack(pred_joints_sv)
        pred_verts_sv = np.stack(pred_verts_sv)
        # Filter out zero entries (views where view 0 was invalid)
        sv_valid = pred_joints_sv.sum(axis=(1, 2)) != 0
        if sv_valid.sum() > 0:
            metrics_sv = compute_metrics(
                pred_joints_sv[sv_valid], gt_joints_all[sv_valid],
                pred_verts_sv[sv_valid], gt_verts_all[sv_valid]
            )
            print(f"\nSingle-View (view 0 only):")
            print(f"  MPJPE:     {metrics_sv['MPJPE (mm)']:.1f} +/- {metrics_sv['MPJPE_std']:.1f} mm")
            print(f"  PA-MPJPE:  {metrics_sv['PA-MPJPE (mm)']:.1f} +/- {metrics_sv['PA-MPJPE_std']:.1f} mm")
            print(f"  PVE:       {metrics_sv['PVE (mm)']:.1f} +/- {metrics_sv['PVE_std']:.1f} mm")

            improvement = metrics_sv['MPJPE (mm)'] - metrics_mv['MPJPE (mm)']
            print(f"\n  Multi-view improvement: {improvement:+.1f} mm MPJPE")
            results['single_view'] = metrics_sv
            results['improvement_mm'] = float(improvement)

    # Confidence statistics
    print(f"\nConfidence weights (mean per view):")
    for vi in range(V_sel):
        print(f"  View {vi}: {all_confidences[:, vi].mean():.3f} "
              f"+/- {all_confidences[:, vi].std():.3f}")
    results['confidence_mean'] = all_confidences.mean(axis=0).tolist()
    results['confidence_std'] = all_confidences.std(axis=0).tolist()

    # Timing
    ms_per_frame = t_total / num_eval * 1000
    print(f"\nTiming: {ms_per_frame:.2f} ms/frame (includes data loading)")
    results['timing_ms_per_frame'] = float(ms_per_frame)
    results['num_eval'] = num_eval
    results['model_config'] = config

    print(f"{'='*70}")

    out_file = output_dir / "results_multiview.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
