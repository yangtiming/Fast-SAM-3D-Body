#!/usr/bin/env python
"""
Preprocess AMASS data → canonical SMPL joints sequences for smoother training.

Usage:
    python preprocess_amass.py \
        --amass_dir data/amass \
        --smpl_model_path ../../mhr2smpl/data/SMPL_NEUTRAL.pkl \
        --output data/amass_joints.npz \
        --target_fps 30
"""

import argparse
import inspect
import os
import sys
import time

import numpy as np
import torch

# Fix chumpy compatibility with Python 3.11+ / NumPy 1.24+
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
_np_compat = {'int': int, 'float': float, 'bool': bool, 'complex': complex,
              'object': object, 'str': str, 'unicode': str}
for attr, val in _np_compat.items():
    if not hasattr(np, attr):
        setattr(np, attr, val)

import smplx


def process_one_file(npz_path, smpl_model, device, target_fps=30):
    """Process one AMASS npz file → canonical joints [T', 24, 3]."""
    data = np.load(npz_path, allow_pickle=True)
    poses = data['poses']          # [T, 156] SMPL+H
    betas = data['betas'][:10]     # [10]
    fps = float(data['mocap_framerate'])
    T = poses.shape[0]

    if T < 10:
        return None

    # Downsample to target fps
    step = max(1, int(round(fps / target_fps)))
    poses = poses[::step]
    T = poses.shape[0]
    if T < 10:
        return None

    # Extract body-only params from SMPL+H
    global_orient = poses[:, :3]    # [T, 3]
    body_pose = poses[:, 3:66]      # [T, 63] (21 joints)

    # Pad to 69 dims (SMPL expects 23 joints x 3)
    body_pose_full = np.zeros((T, 69), dtype=np.float64)
    body_pose_full[:, :63] = body_pose

    # Run SMPL FK with go=0 for canonical joints
    with torch.no_grad():
        betas_t = torch.from_numpy(betas).float().unsqueeze(0).expand(T, -1).to(device)
        bp_t = torch.from_numpy(body_pose_full).float().to(device)
        go_zero = torch.zeros(T, 3, device=device)

        smpl_out = smpl_model(
            global_orient=go_zero,
            body_pose=bp_t,
            betas=betas_t,
        )
        joints = smpl_out.joints[:, :24].cpu().numpy()  # [T, 24, 3]

    # Pelvis-center each frame
    joints = joints - joints[:, 0:1, :]  # [T, 24, 3]

    return joints.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_dir", required=True, help="Root dir containing CMU/, KIT/, etc.")
    parser.add_argument("--smpl_model_path", required=True, help="Path to SMPL_NEUTRAL.pkl")
    parser.add_argument("--output", default="", help="Output path (default: data/amass_joints.npz)")
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--min_seq_len", type=int, default=64,
                        help="Minimum sequence length after downsampling (discard shorter)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.output:
        args.output = os.path.join(script_dir, "data", "amass_joints.npz")

    # Resolve paths
    amass_dir = args.amass_dir
    if not os.path.isabs(amass_dir):
        amass_dir = os.path.join(script_dir, amass_dir)
    smpl_path = args.smpl_model_path
    if not os.path.isabs(smpl_path):
        smpl_path = os.path.join(script_dir, smpl_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SMPL model
    print(f"Loading SMPL model from {smpl_path}")
    smpl_model = smplx.SMPL(model_path=smpl_path, gender='neutral').to(device)

    # Find all npz files
    npz_files = []
    for root, dirs, files in os.walk(amass_dir):
        for f in files:
            if f.endswith('.npz') and not f.startswith('.'):
                npz_files.append(os.path.join(root, f))
    npz_files.sort()
    print(f"Found {len(npz_files)} AMASS npz files in {amass_dir}")

    # Process all files
    all_sequences = []
    total_frames = 0
    skipped = 0
    t0 = time.time()

    for i, npz_path in enumerate(npz_files):
        try:
            joints = process_one_file(npz_path, smpl_model, device, args.target_fps)
        except Exception as e:
            skipped += 1
            continue

        if joints is None or joints.shape[0] < args.min_seq_len:
            skipped += 1
            continue

        all_sequences.append(joints)
        total_frames += joints.shape[0]

        if (i + 1) % 500 == 0 or i == len(npz_files) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(npz_files)}] sequences={len(all_sequences)}, "
                  f"frames={total_frames}, skipped={skipped}, {elapsed:.1f}s")

    print(f"\nTotal: {len(all_sequences)} sequences, {total_frames} frames")
    print(f"Skipped: {skipped} files (too short or errors)")

    # Compute stats
    all_lens = [s.shape[0] for s in all_sequences]
    print(f"Sequence lengths: min={min(all_lens)}, max={max(all_lens)}, "
          f"mean={np.mean(all_lens):.0f}, median={np.median(all_lens):.0f}")

    # Save as list of variable-length sequences using object array
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Store lengths and concatenated data for efficient loading
    lengths = np.array(all_lens, dtype=np.int32)
    concat = np.concatenate(all_sequences, axis=0)  # [total_frames, 24, 3]

    np.savez_compressed(
        args.output,
        joints=concat,             # [total_frames, 24, 3]
        lengths=lengths,           # [N_seq]
        target_fps=np.array(args.target_fps),
        n_sequences=np.array(len(all_sequences)),
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved to: {args.output} ({size_mb:.1f} MB)")
    print(f"  joints: [{total_frames}, 24, 3], lengths: [{len(all_sequences)}]")


if __name__ == "__main__":
    main()
