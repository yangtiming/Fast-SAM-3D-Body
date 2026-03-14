#!/usr/bin/env python
"""
Train temporal pose smoother on AMASS data.

Usage:
    python train_smoother.py --data data/amass_joints.npz
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SmoothDataset, load_amass_joints
from smoother_net import SmootherMLP

FOOT_JOINT_IDS = [7, 8, 10, 11]  # L_Ankle, R_Ankle, L_Foot, R_Foot


def foot_grounding_loss(pred_joints, margin=0.01):
    """Penalize foot joints that are above ground level.

    pred_joints: [B, 24, 3] — canonical joints (Y is vertical).
    Ground = min Y across all joints per sample.
    """
    ground_y = pred_joints[:, :, 1].min(dim=1, keepdim=True).values  # [B, 1]
    foot_y = pred_joints[:, FOOT_JOINT_IDS, 1]  # [B, 4]
    above_ground = F.relu(foot_y - ground_y - margin)  # [B, 4]
    return above_ground.mean()


def train_one_epoch(model, loader, optimizer, device, foot_weight=0.1):
    model.train()
    total_loss = 0
    total_l1 = 0
    total_foot = 0
    n = 0

    for noisy_window, clean_center in loader:
        noisy_window = noisy_window.to(device)
        clean_center = clean_center.to(device)
        B = noisy_window.shape[0]

        pred = model(noisy_window)  # [B, 72]

        # L1 reconstruction loss
        l1 = F.l1_loss(pred, clean_center)

        # Foot grounding loss
        pred_j = pred.reshape(B, 24, 3)
        foot_loss = foot_grounding_loss(pred_j)

        loss = l1 + foot_weight * foot_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_l1 += l1.item() * B
        total_foot += foot_loss.item() * B
        n += B

    return total_loss / n, total_l1 / n, total_foot / n


@torch.no_grad()
def validate(model, loader, device, foot_weight=0.1):
    model.eval()
    total_loss = 0
    total_l1 = 0
    total_foot = 0
    total_mpjpe = 0
    n = 0

    for noisy_window, clean_center in loader:
        noisy_window = noisy_window.to(device)
        clean_center = clean_center.to(device)
        B = noisy_window.shape[0]

        pred = model(noisy_window)

        l1 = F.l1_loss(pred, clean_center)
        pred_j = pred.reshape(B, 24, 3)
        clean_j = clean_center.reshape(B, 24, 3)
        foot_loss = foot_grounding_loss(pred_j)
        loss = l1 + foot_weight * foot_loss

        # MPJPE (mm)
        mpjpe = torch.norm(pred_j - clean_j, dim=2).mean() * 1000

        total_loss += loss.item() * B
        total_l1 += l1.item() * B
        total_foot += foot_loss.item() * B
        total_mpjpe += mpjpe.item() * B
        n += B

    return total_loss / n, total_l1 / n, total_foot / n, total_mpjpe / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Preprocessed AMASS joints npz")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--temporal_jitter_std", type=float, default=0.01)
    parser.add_argument("--outlier_prob", type=float, default=0.05)
    parser.add_argument("--outlier_std", type=float, default=0.05)
    parser.add_argument("--foot_weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(script_dir, data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {data_path}")
    joints, lengths = load_amass_joints(data_path)
    n_seq = len(lengths)
    print(f"  {n_seq} sequences, {joints.shape[0]} total frames")

    # Train/val split by sequence
    n_val = max(1, int(n_seq * args.val_ratio))
    perm = np.random.RandomState(42).permutation(n_seq)
    val_idx = set(perm[:n_val].tolist())

    # Split joints/lengths into train/val
    train_joints_list, val_joints_list = [], []
    train_lengths, val_lengths = [], []
    offset = 0
    for i, seq_len in enumerate(lengths):
        seq_joints = joints[offset:offset + seq_len]
        if i in val_idx:
            val_joints_list.append(seq_joints)
            val_lengths.append(seq_len)
        else:
            train_joints_list.append(seq_joints)
            train_lengths.append(seq_len)
        offset += seq_len

    train_joints = np.concatenate(train_joints_list, axis=0)
    val_joints = np.concatenate(val_joints_list, axis=0)
    train_lengths = np.array(train_lengths)
    val_lengths = np.array(val_lengths)
    print(f"  Train: {len(train_lengths)} seq, {train_joints.shape[0]} frames")
    print(f"  Val:   {len(val_lengths)} seq, {val_joints.shape[0]} frames")

    # Create datasets
    train_ds = SmoothDataset(
        train_joints, train_lengths, args.window_size,
        args.noise_std, args.temporal_jitter_std,
        args.outlier_prob, args.outlier_std,
    )
    val_ds = SmoothDataset(
        val_joints, val_lengths, args.window_size,
        args.noise_std, args.temporal_jitter_std,
        args.outlier_prob, args.outlier_std,
    )
    print(f"  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = SmootherMLP(
        window_size=args.window_size,
        joint_dim=72,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SmootherMLP, window={args.window_size}, hidden={args.hidden_dims}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Train
    best_val_mpjpe = float('inf')
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss, train_l1, train_foot = train_one_epoch(
            model, train_loader, optimizer, device, args.foot_weight
        )
        val_loss, val_l1, val_foot, val_mpjpe = validate(
            model, val_loader, device, args.foot_weight
        )
        scheduler.step(val_mpjpe)
        lr = optimizer.param_groups[0]['lr']

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "smoother_best.pth"))

        dt = time.time() - t_ep
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s)  "
                  f"train_l1={train_l1*1000:.2f}mm  "
                  f"val_l1={val_l1*1000:.2f}mm  val_mpjpe={val_mpjpe:.2f}mm  "
                  f"foot={val_foot*1000:.2f}mm  "
                  f"lr={lr:.1e}  best={best_val_mpjpe:.2f}mm@{best_epoch}")

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s. Best val MPJPE: {best_val_mpjpe:.2f}mm (epoch {best_epoch})")

    # Save config
    config = {
        'window_size': args.window_size,
        'hidden_dims': args.hidden_dims,
        'joint_dim': 72,
        'noise_std': args.noise_std,
        'temporal_jitter_std': args.temporal_jitter_std,
        'outlier_prob': args.outlier_prob,
        'outlier_std': args.outlier_std,
        'foot_weight': args.foot_weight,
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_val_mpjpe_mm': best_val_mpjpe,
        'n_params': n_params,
        'train_windows': len(train_ds),
        'val_windows': len(val_ds),
    }
    with open(os.path.join(output_dir, "smoother_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_dir}/smoother_config.json")


if __name__ == "__main__":
    main()
