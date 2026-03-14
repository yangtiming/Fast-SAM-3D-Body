#!/usr/bin/env python
"""
Step 2: Multi-view MHR2SMPL training.

Supports:
  - from_scratch: random init, all params trainable
  - from pretrained: load encoder/decoder, optionally freeze encoder
  - SV+MV dual loss: each view gets single-view loss, fused output gets multi-view loss

Usage:
    python step2_train_multiview.py \
        --data_path ./data/mv_pairs_RICH.npz \
        --smpl_model ./data/SMPL_NEUTRAL.pkl \
        --save_dir ./experiments/multiview \
        --from_scratch \
        --sv_loss_weight 0.5 \
        --epochs 500 --lr 1e-3 --batch_size 64
"""

import argparse
import inspect
import json
from pathlib import Path

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
for _attr in ('bool', 'int', 'float', 'complex', 'object', 'str'):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))
if not hasattr(np, 'unicode'):
    np.unicode = str

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import smplx

from multiview_net import MinimalMultiViewNet, split_output, OUTPUT_DIM, rot6d_to_aa, aa_to_rot6d_np


# ==================== Helpers ====================

def pad_body_pose(bp_21, device):
    bs = bp_21.shape[0]
    bp_full = torch.zeros(bs, 69, device=device)
    bp_full[:, :63] = bp_21
    return bp_full


def smpl_verts_at(smpl_model, go6d, bp_21, betas, device, zero_go=False):
    bp_full = pad_body_pose(bp_21, device)
    if zero_go:
        go_aa = torch.zeros(bp_21.shape[0], 3, device=device)
    else:
        go_aa = rot6d_to_aa(go6d)
    output = smpl_model(global_orient=go_aa, body_pose=bp_full, betas=betas)
    return output.vertices


def smpl_fk_joints(smpl_model, bp_21, betas, device):
    bs = bp_21.shape[0]
    bp_full = pad_body_pose(bp_21, device)
    go_zero = torch.zeros(bs, 3, device=device)
    output = smpl_model(global_orient=go_zero, body_pose=bp_full, betas=betas)
    joints = output.joints[:, :24]
    return joints - joints[:, 0:1]


def mpjpe_mm(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).sum(-1)).mean() * 1000.0


def vert_error_mm(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).sum(-1)).mean() * 1000.0


def pose_error_deg(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).sum(-1)).mean() * (180.0 / np.pi)


# ==================== Dataset ====================

class MVDataset(Dataset):
    def __init__(self, verts, view_mask, canon_joints, gt_bp21, gt_betas, gt_go6d=None,
                 num_views_select=2, view_dropout_prob=0.15,
                 single_view_prob=0.15, noise_std=0.0, training=True):
        self.verts = verts
        self.view_mask = view_mask
        self.canon_joints = canon_joints
        self.gt_bp21 = gt_bp21
        self.gt_betas = gt_betas
        self.gt_go6d = gt_go6d  # [N, 6] or None
        self.num_views_select = num_views_select
        self.view_dropout_prob = view_dropout_prob
        self.single_view_prob = single_view_prob
        self.noise_std = noise_std
        self.training = training

    def __len__(self):
        return len(self.verts)

    def __getitem__(self, idx):
        all_views = self.verts[idx]          # [V_max, V_sub, 3]
        valid = self.view_mask[idx]          # [V_max]
        valid_indices = np.where(valid)[0]
        num_valid = len(valid_indices)

        V_sub = all_views.shape[1]
        V_sel = self.num_views_select

        if self.training:
            if np.random.random() < self.single_view_prob or num_valid == 1:
                pick = np.random.choice(valid_indices, 1)
                selected = np.zeros((V_sel, V_sub, 3), dtype=np.float32)
                sel_mask = np.zeros(V_sel, dtype=bool)
                selected[0] = all_views[pick[0]]
                sel_mask[0] = True
            else:
                n_pick = min(V_sel, num_valid)
                pick = np.random.choice(valid_indices, n_pick, replace=False)
                selected = np.zeros((V_sel, V_sub, 3), dtype=np.float32)
                sel_mask = np.zeros(V_sel, dtype=bool)
                for i, vi in enumerate(pick):
                    selected[i] = all_views[vi]
                    sel_mask[i] = True

                # View dropout
                if V_sel >= 2 and sel_mask.sum() >= 2:
                    for i in range(V_sel):
                        if sel_mask[i] and np.random.random() < self.view_dropout_prob:
                            if sel_mask.sum() > 1:
                                selected[i] = 0.0
                                sel_mask[i] = False

            # Asymmetric noise on one view
            if self.noise_std > 0 and sel_mask.sum() >= 2:
                noise_view = np.random.choice(np.where(sel_mask)[0])
                noise_scale = np.random.uniform(0.5, 2.0) * self.noise_std
                selected[noise_view] += np.random.randn(V_sub, 3).astype(np.float32) * noise_scale

            # Random permutation
            perm = np.random.permutation(V_sel)
            selected = selected[perm]
            sel_mask = sel_mask[perm]
        else:
            selected = np.zeros((V_sel, V_sub, 3), dtype=np.float32)
            sel_mask = np.zeros(V_sel, dtype=bool)
            n_pick = min(V_sel, num_valid)
            for i in range(n_pick):
                selected[i] = all_views[valid_indices[i]]
                sel_mask[i] = True

        views_flat = selected.reshape(V_sel, -1)
        go6d = self.gt_go6d[idx] if self.gt_go6d is not None else np.zeros(6, np.float32)
        return (
            views_flat.astype(np.float32),
            sel_mask,
            self.canon_joints[idx],
            self.gt_bp21[idx],
            self.gt_betas[idx],
            go6d,
        )


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", nargs="+", required=True,
                        help="One or more .npz data files (will be concatenated)")
    parser.add_argument("--smpl_model", required=True)
    parser.add_argument("--save_dir", default="./experiments/multiview")
    parser.add_argument("--pretrained_path", default=None,
                        help="Path to pretrained model (ignored if --from_scratch)")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train from random initialization (no pretrained)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder (only with pretrained, default: trainable)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_views_select", type=int, default=2)
    parser.add_argument("--view_dropout_prob", type=float, default=0.15)
    parser.add_argument("--single_view_prob", type=float, default=0.15)
    parser.add_argument("--noise_std", type=float, default=0.5)
    # Loss weights
    parser.add_argument("--vert_loss_weight", type=float, default=1.0)
    parser.add_argument("--fk_loss_weight", type=float, default=1.0)
    parser.add_argument("--param_loss_weight", type=float, default=0.1)
    parser.add_argument("--conf_entropy_weight", type=float, default=0.01)
    parser.add_argument("--go_loss_weight", type=float, default=1.0,
                        help="Weight for global_orient 6D MSE loss")
    parser.add_argument("--sv_loss_weight", type=float, default=0.5,
                        help="Weight for single-view loss (per-view independent prediction)")
    parser.add_argument("--mode", default='fuse_feat',
                        choices=['fuse_feat', 'avg_params', 'avg_pose'],
                        help="fuse_feat: pool feats then decode (default); "
                             "avg_params: decode per-view then weighted-avg params (freeze decoder)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SMPL
    smpl_model = smplx.SMPL(model_path=args.smpl_model, gender='neutral').to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad_(False)

    # Data — support multiple npz files (concatenated)
    print("Loading data...")
    all_verts, all_masks, all_cj, all_bp, all_be, all_go = [], [], [], [], [], []
    sample_indices = None
    for dp in args.data_path:
        data = np.load(dp)
        all_verts.append(data['smpl_target_verts_sampled'].astype(np.float32))
        all_masks.append(data['view_mask'].astype(bool))
        all_cj.append(data['smpl_joints_canonical'].astype(np.float32))
        all_bp.append(data['gt_body_pose'].astype(np.float32))
        all_be.append(data['gt_betas'].astype(np.float32))
        if 'gt_global_orient' in data:
            go_aa = data['gt_global_orient'].astype(np.float32)  # [N, 3] axis-angle, RICH world space
            all_go.append(aa_to_rot6d_np(go_aa))                 # [N, 6]
        else:
            n = len(data['smpl_target_verts_sampled'])
            all_go.append(np.zeros((n, 6), np.float32))
        if sample_indices is None:
            sample_indices = data['smpl_vert_sample_indices']
        n = len(data['smpl_target_verts_sampled'])
        print(f"  {dp}: {n} samples")

    verts = np.concatenate(all_verts, axis=0)
    view_mask = np.concatenate(all_masks, axis=0)
    canon_joints = np.concatenate(all_cj, axis=0)
    gt_body_pose = np.concatenate(all_bp, axis=0)
    gt_betas = np.concatenate(all_be, axis=0)
    gt_go6d = np.concatenate(all_go, axis=0)

    N, V_max, V_sub, _ = verts.shape
    gt_bp_21 = gt_body_pose[:, :63]

    print(f"  Total samples: {N}, Views: {V_max}, Verts: {V_sub}")
    print(f"  View availability: {view_mask.mean():.1%}")

    # Train/Val split
    indices = np.random.permutation(N)
    val_size = max(1, int(N * args.val_ratio))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_dataset = MVDataset(
        verts[train_idx], view_mask[train_idx],
        canon_joints[train_idx], gt_bp_21[train_idx], gt_betas[train_idx],
        gt_go6d=gt_go6d[train_idx],
        num_views_select=args.num_views_select,
        view_dropout_prob=args.view_dropout_prob,
        single_view_prob=args.single_view_prob,
        noise_std=args.noise_std,
        training=True,
    )
    val_dataset = MVDataset(
        verts[val_idx], view_mask[val_idx],
        canon_joints[val_idx], gt_bp_21[val_idx], gt_betas[val_idx],
        gt_go6d=gt_go6d[val_idx],
        num_views_select=args.num_views_select,
        training=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    sample_idx_tensor = torch.from_numpy(sample_indices.astype(np.int64)).to(device)

    # Model
    if args.from_scratch:
        print(f"\nInitializing from scratch  mode={args.mode}")
        model = MinimalMultiViewNet.from_scratch(
            device=device, input_dim=V_sub * 3,
        ).to(device)
    else:
        if args.pretrained_path is None:
            raise ValueError("--pretrained_path required when not using --from_scratch")
        print(f"\nLoading pretrained: {args.pretrained_path}  mode={args.mode}")
        model = MinimalMultiViewNet.from_pretrained(
            args.pretrained_path, device=device,
            input_dim=V_sub * 3,
            freeze_encoder=args.freeze_encoder,
            mode=args.mode,
            freeze_decoder=(args.mode in ('avg_params', 'avg_pose')),
        ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    # Weights
    w_vert = args.vert_loss_weight
    w_fk = args.fk_loss_weight
    w_param = args.param_loss_weight
    w_conf = args.conf_entropy_weight
    w_go = args.go_loss_weight
    w_sv = args.sv_loss_weight

    best_val_fk = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'val_vert_err': [], 'val_fk_mpjpe': [],
        'val_pose_err': [], 'val_betas_err': [],
    }

    print(f"\nTraining {args.epochs} epochs, lr={args.lr}")
    print(f"Loss: vert×{w_vert} + fk×{w_fk} + param×{w_param} + conf_ent×{w_conf} + go×{w_go} + sv×{w_sv}")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        train_loss_sum, n_train = 0.0, 0

        for batch in train_loader:
            views_b, mask_b, jb, bpb, bb, go6d_b = [x.to(device) for x in batch]

            pred, conf_weights = model(views_b, mask_b)
            pred_go, pred_bp, pred_betas = split_output(pred)

            # Loss 1: Canonical vertex L1 — compare zero-go pred vs zero-go GT (no camera-space mismatch)
            pred_verts = smpl_verts_at(smpl_model, pred_go, pred_bp, pred_betas, device, zero_go=True)
            gt_verts   = smpl_verts_at(smpl_model, None,    bpb,     bb,         device, zero_go=True)
            pred_verts_sub = pred_verts[:, sample_idx_tensor]
            gt_verts_sub   = gt_verts[:, sample_idx_tensor]
            pred_centered = pred_verts_sub - pred_verts_sub.mean(1, keepdim=True)
            gt_centered   = gt_verts_sub   - gt_verts_sub.mean(1, keepdim=True)
            loss_vert = F.l1_loss(pred_centered, gt_centered)

            # Loss 2: Canonical FK
            fk_joints = smpl_fk_joints(smpl_model, pred_bp, pred_betas, device)
            loss_fk = F.l1_loss(fk_joints, jb)

            # Loss 3: Param supervision
            loss_param = F.mse_loss(pred_bp, bpb) + F.mse_loss(pred_betas, bb)

            # Loss 4: Confidence entropy (maximize → don't collapse to single view)
            valid_weights = conf_weights[mask_b.sum(dim=1) > 1]
            if len(valid_weights) > 0:
                conf_entropy = -(valid_weights * torch.log(valid_weights + 1e-8)).sum(dim=1).mean()
                loss_conf = -w_conf * conf_entropy
            else:
                loss_conf = torch.tensor(0.0, device=device)

            # Loss 5: Global orient 6D MSE
            loss_go = F.mse_loss(pred_go, go6d_b)

            loss_mv = w_vert * loss_vert + w_fk * loss_fk + w_param * loss_param + loss_conf + w_go * loss_go

            # Loss 6: Single-view loss — each valid view independently predicts params
            loss_sv = torch.tensor(0.0, device=device)
            if w_sv > 0:
                B_cur, V_cur, D_cur = views_b.shape
                sv_count = 0
                for vi in range(V_cur):
                    vi_mask = mask_b[:, vi]  # [B] which samples have this view
                    if vi_mask.sum() == 0:
                        continue
                    sv_input = views_b[vi_mask, vi]  # [n, D]
                    sv_pred = model.forward_single_view(sv_input)
                    sv_go_i, sv_bp_i, sv_betas_i = split_output(sv_pred)

                    sv_fk = smpl_fk_joints(smpl_model, sv_bp_i, sv_betas_i, device)
                    sv_loss_fk = F.l1_loss(sv_fk, jb[vi_mask])
                    sv_loss_param = F.mse_loss(sv_bp_i, bpb[vi_mask]) + F.mse_loss(sv_betas_i, bb[vi_mask])
                    sv_loss_go = F.mse_loss(sv_go_i, go6d_b[vi_mask])
                    loss_sv = loss_sv + sv_loss_fk + 0.1 * sv_loss_param + sv_loss_go
                    sv_count += 1
                if sv_count > 0:
                    loss_sv = loss_sv / sv_count

            loss = loss_mv + w_sv * loss_sv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * views_b.size(0)
            n_train += views_b.size(0)

        train_loss = train_loss_sum / max(n_train, 1)

        # ---- Val ----
        model.eval()
        val_vert_errs, val_fk_errs, val_pose_errs = [], [], []
        val_loss_sum, n_val = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                views_b, mask_b, jb, bpb, bb, go6d_b = [x.to(device) for x in batch]
                pred, _ = model(views_b, mask_b)
                vp_go, vp_bp, vp_betas = split_output(pred)

                # Vertex error (canonical pred vs canonical GT)
                val_verts   = smpl_verts_at(smpl_model, vp_go, vp_bp, vp_betas, device, zero_go=True)
                val_gt_verts = smpl_verts_at(smpl_model, None,  bpb,   bb,       device, zero_go=True)
                val_verts_sub   = val_verts[:, sample_idx_tensor]
                val_gt_verts_sub = val_gt_verts[:, sample_idx_tensor]
                val_centered    = val_verts_sub   - val_verts_sub.mean(1, keepdim=True)
                val_gt_centered = val_gt_verts_sub - val_gt_verts_sub.mean(1, keepdim=True)
                vl_vert = F.l1_loss(val_centered, val_gt_centered)

                vl_fk = F.l1_loss(
                    smpl_fk_joints(smpl_model, vp_bp, vp_betas, device), jb)
                vl_param = F.mse_loss(vp_bp, bpb) + F.mse_loss(vp_betas, bb)

                vl = w_vert * vl_vert + w_fk * vl_fk + w_param * vl_param
                val_loss_sum += vl.item() * views_b.size(0)
                n_val += views_b.size(0)

                # Metrics (canonical vert error)
                val_vert_errs.append(vert_error_mm(val_centered, val_gt_centered).item())
                val_fk_errs.append(mpjpe_mm(
                    smpl_fk_joints(smpl_model, vp_bp, vp_betas, device), jb).item())
                val_pose_errs.append(pose_error_deg(
                    vp_bp.reshape(-1, 21, 3), bpb.reshape(-1, 21, 3)).item())

        val_loss = val_loss_sum / max(n_val, 1)
        val_vert_err = np.mean(val_vert_errs) if val_vert_errs else 0.0
        val_fk_mpjpe = np.mean(val_fk_errs) if val_fk_errs else 0.0
        val_pose_err = np.mean(val_pose_errs) if val_pose_errs else 0.0

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_vert_err'].append(val_vert_err)
        history['val_fk_mpjpe'].append(val_fk_mpjpe)
        history['val_pose_err'].append(val_pose_err)

        if val_fk_mpjpe < best_val_fk:
            best_val_fk = val_fk_mpjpe
            torch.save(model.state_dict(), save_dir / "best_model.pth")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"loss={train_loss:.6f}  "
                  f"vert={val_vert_err:.1f}mm  "
                  f"FK={val_fk_mpjpe:.1f}mm  "
                  f"pose={val_pose_err:.1f}°  "
                  f"best_FK={best_val_fk:.1f}mm  "
                  f"lr={lr_now:.1e}")

    # Save
    torch.save(model.state_dict(), save_dir / "last_model.pth")
    np.save(save_dir / "train_idx.npy", train_idx)
    np.save(save_dir / "val_idx.npy", val_idx)
    np.save(save_dir / "sample_idx.npy", sample_indices)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {
        'model_type': f'MinimalMultiViewNet_{args.mode}',
        'trainable_params': trainable,
        'total_params': sum(p.numel() for p in model.parameters()),
        'num_views_select': args.num_views_select,
        'view_dropout_prob': args.view_dropout_prob,
        'single_view_prob': args.single_view_prob,
        'noise_std': args.noise_std,
        'epochs': args.epochs,
        'lr': args.lr,
        'vert_loss_weight': w_vert,
        'fk_loss_weight': w_fk,
        'param_loss_weight': w_param,
        'conf_entropy_weight': w_conf,
        'from_scratch': args.from_scratch,
        'freeze_encoder': args.freeze_encoder,
        'sv_loss_weight': w_sv,
        'go_loss_weight': w_go,
        'pretrained_path': args.pretrained_path,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'V_sub': V_sub,
        'V_max': V_max,
        'best_val_fk_mpjpe_mm': best_val_fk,
    }
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    np.savez(save_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})

    print(f"\n{'='*60}")
    print(f"Done! Trainable: {trainable:,} params")
    print(f"  Best FK MPJPE: {best_val_fk:.2f} mm")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
