#!/usr/bin/env python
"""
Step 1c: Merge multiple training pair NPZ files into one.

All inputs must be in multi-view format:
  smpl_target_verts_sampled: [N, V_max, V_sub, 3]
  view_mask:                 [N, V_max] bool

Concatenates data from multiple step1 outputs (different datasets / supervision modes).
Validates that sample_indices and V_max are consistent across all files.

Usage:
  python step1c_merge.py \
    --inputs data/pairs_EMDB_fitted.npz data/pairs_3dpw_fitted.npz \
             data/mv_pairs_RICH.npz data/mv_pairs_AIST_stage1.npz \
    --output data/pairs_all_merged.npz
"""

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input NPZ files to merge (all must be multi-view format)")
    parser.add_argument("--output", required=True,
                        help="Output merged NPZ path")
    args = parser.parse_args()

    print(f"Merging {len(args.inputs)} files...")

    all_verts = []
    all_masks = []
    all_joints = []
    all_body_pose = []
    all_betas = []
    all_joints_world = []
    all_global_orient = []
    ref_sample_indices = None
    ref_v_max = None

    for path in args.inputs:
        print(f"\n  Loading: {path}")
        data = np.load(path)

        verts = data['smpl_target_verts_sampled']   # [N, V_max, V_sub, 3]
        joints = data['smpl_joints_canonical']      # [N, 24, 3]
        bp = data['gt_body_pose']                   # [N, 69]
        betas = data['gt_betas']                    # [N, 10]
        sample_idx = data['smpl_vert_sample_indices']  # [V_sub]

        if verts.ndim != 4:
            raise ValueError(f"{path}: expected 4D verts [N, V_max, V_sub, 3], got {verts.shape}. "
                             "Re-run step1 to output multi-view format.")

        view_mask = data['view_mask']               # [N, V_max] bool

        n = verts.shape[0]
        v_max = verts.shape[1]
        sup_mode = "fitted" if data['supervision_mode'][0] == 1 else "GT"
        views_per_sample = view_mask.sum(axis=1).mean()
        print(f"    samples: {n}, supervision: {sup_mode}, V_max: {v_max}")
        print(f"    verts: {verts.shape}, avg views/sample: {views_per_sample:.1f}")

        # Validate sample indices
        if ref_sample_indices is None:
            ref_sample_indices = sample_idx
        else:
            if not np.array_equal(ref_sample_indices, sample_idx):
                raise ValueError("All input files must use the same vertex sampling indices.")

        # Validate V_max
        if ref_v_max is None:
            ref_v_max = v_max
        else:
            if v_max != ref_v_max:
                raise ValueError(f"V_max mismatch: first file has {ref_v_max}, {path} has {v_max}. "
                                 "All files must use the same --max_views.")

        all_verts.append(verts)
        all_masks.append(view_mask)
        all_joints.append(joints)
        all_body_pose.append(bp)
        all_betas.append(betas)

        if 'smpl_joints_world' in data:
            all_joints_world.append(data['smpl_joints_world'])
        if 'gt_global_orient' in data:
            all_global_orient.append(data['gt_global_orient'])

    # Concatenate
    merged_verts = np.concatenate(all_verts, axis=0).astype(np.float32)
    merged_masks = np.concatenate(all_masks, axis=0)
    merged_joints = np.concatenate(all_joints, axis=0).astype(np.float32)
    merged_bp = np.concatenate(all_body_pose, axis=0).astype(np.float32)
    merged_betas = np.concatenate(all_betas, axis=0).astype(np.float32)

    has_world = len(all_joints_world) == len(args.inputs)
    has_go = len(all_global_orient) == len(args.inputs)

    total = merged_verts.shape[0]
    print(f"\n--- Merged ---")
    print(f"  Total samples: {total}")
    print(f"  verts:            {merged_verts.shape}")
    print(f"  view_mask:        {merged_masks.shape}")
    print(f"  joints:           {merged_joints.shape}")
    print(f"  body_pose:        {merged_bp.shape}")
    print(f"  betas:            {merged_betas.shape}")
    print(f"  joints_world:     {'YES' if has_world else 'NO (missing in some inputs)'}")
    print(f"  global_orient:    {'YES' if has_go else 'NO (missing in some inputs)'}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_sizes = np.array([v.shape[0] for v in all_verts], dtype=np.int64)
    dataset_names = np.array([Path(p).stem for p in args.inputs])
    print(f"  dataset_sizes: {list(zip(dataset_names, dataset_sizes))}")

    save_dict = {
        'smpl_target_verts_sampled': merged_verts,
        'view_mask': merged_masks,
        'num_views_total': np.array([ref_v_max]),
        'smpl_vert_sample_indices': ref_sample_indices,
        'smpl_joints_canonical': merged_joints,
        'gt_body_pose': merged_bp,
        'gt_betas': merged_betas,
        'supervision_mode': np.array([2]),  # 2 = mixed
        'dataset_sizes': dataset_sizes,
        'dataset_names': dataset_names,
    }

    if has_world:
        save_dict['smpl_joints_world'] = np.concatenate(all_joints_world, axis=0).astype(np.float32)
    if has_go:
        save_dict['gt_global_orient'] = np.concatenate(all_global_orient, axis=0).astype(np.float32)

    np.savez(str(output_path), **save_dict)
    file_size_mb = os.path.getsize(str(output_path)) / 1024 / 1024
    print(f"\n  Saved: {output_path} ({file_size_mb:.1f} MB)")
    for k, v in save_dict.items():
        print(f"  {k:30s} {str(v.shape):20s} dtype={v.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
