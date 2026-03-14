#!/usr/bin/env python
"""
Step 1 (Multi-View): Collect training pairs from AIST++ using real Stage1 predictions.

Unlike step1_collect_AIST.py (which used GT+noise), this script:
  1. Extracts frames from AIST++ videos
  2. Runs SAM-3D-Body Stage1 to get real MHR predictions in camera space
  3. Maps MHR → SMPL via barycentric mapping
  4. Groups by (motion_key, frame_idx) across cameras → multi-view pairs
  5. Saves mv_pairs_AIST_stage1.npz in same format as mv_pairs_RICH.npz

Usage:
  cd /home/jiawei/timingyang/CLEAN/Fast_sam-3d-body_mhr2smpl
  /home/jiawei/miniforge3/envs/sam_3d_body/bin/python3 \\
    mhr2smpl-multiview/step1_collect_AIST_stage1.py \\
    --video_dir /home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/AIST/videos \\
    --aist_dir   /home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/AIST \\
    --output_path mhr2smpl-multiview/data/mv_pairs_AIST_stage1.npz \\
    --frame_stride 18 \\
    --max_views 4
"""

import argparse
import inspect
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── compat patches ──────────────────────────────────────────────────────────
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import numpy as np
for _a in ('bool', 'int', 'float', 'complex', 'object', 'str'):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, object))
if not hasattr(np, 'unicode'):
    np.unicode = str

import cv2
import torch
import smplx
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # Fast_sam-3d-body/
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"  # mhr2smpl/data/
sys.path.insert(0, str(PROJECT_DIR))

from notebook.utils import setup_sam_3d_body
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_video_name(stem):
    """
    'gBR_sBM_c01_d04_mBR0_ch01' → (motion_key='gBR_sBM_cAll_d04_mBR0_ch01', cam_id='01')
    """
    m = re.match(r'(.+)_c(\d+)_(.+)', stem)
    if not m:
        return None, None
    prefix, cam, suffix = m.groups()
    motion_key = f"{prefix}_cAll_{suffix}"
    return motion_key, cam


def pick_largest_bbox(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.array(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = areas.argmax()
    return boxes[idx:idx+1]


@torch.no_grad()
def run_stage1(estimator, img_rgb, focal=1000.0):
    """Run Stage1 on one RGB frame. Returns (pred_vertices [18439,3], pred_cam_t [3]) or None."""
    H, W = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if estimator.detector is not None:
        det = estimator.detector.run_human_detection(
            img_bgr, det_cat_id=0, bbox_thr=0.4, nms_thr=0.3,
            default_to_full_image=False,
        )
        boxes = det['boxes'] if isinstance(det, dict) else det
    else:
        boxes = np.array([[0, 0, W, H]], dtype=np.float32)

    boxes = pick_largest_bbox(boxes)
    if boxes is None:
        return None

    cam_int = torch.tensor([[
        [focal,     0, W / 2.0],
        [0,     focal, H / 2.0],
        [0,         0,     1.0],
    ]]).float()

    batch = prepare_batch(img_rgb, estimator.transform, boxes)
    batch['cam_int'] = cam_int.to(batch['img'])
    batch = recursive_to(batch, 'cuda')
    estimator.model._initialize_batch(batch)
    pose_out = estimator.model.forward_step(batch, decoder_type='body')
    out = recursive_to(recursive_to(pose_out['mhr'], 'cpu'), 'numpy')
    return {
        'pred_vertices': out['pred_vertices'][0],  # [18439, 3]
        'pred_cam_t':    out['pred_cam_t'][0],     # [3]
    }


def preprocess_mhr(pred_vertices, pred_cam_t, mhr_vert_ids, baryc, sample_idx):
    """MHR camera-space → centroid-centered SMPL subsample [1500, 3]."""
    v = pred_vertices.copy().astype(np.float32)
    t = pred_cam_t.copy().astype(np.float32)
    # YZ-flip
    v[:, 1] *= -1; v[:, 2] *= -1
    t[1]    *= -1; t[2]    *= -1
    # → world space
    mhr_w = v + t[None, :]
    # barycentric MHR → SMPL
    face_v = mhr_w[mhr_vert_ids]                          # [6890, 3, 3]
    smpl_v = (face_v * baryc[:, :, None]).sum(axis=1)    # [6890, 3]
    # subsample + centroid-center
    smpl_v_sub = smpl_v[sample_idx]                       # [1500, 3]
    smpl_v_sub -= smpl_v_sub.mean(axis=0, keepdims=True)
    return smpl_v_sub                                     # [1500, 3] meters


def load_motion_gt(motion_pkl_path):
    """Load GT SMPL params from AIST++ motion pkl."""
    with open(motion_pkl_path, 'rb') as f:
        d = pickle.load(f)
    return {
        'smpl_poses':   d['smpl_poses'],    # [N, 72] axis-angle
        'smpl_scaling': float(d['smpl_scaling'][0]),  # ~93.78
        'smpl_trans':   d['smpl_trans'],    # [N, 3] cm
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',    required=True)
    parser.add_argument('--aist_dir',     required=True)
    parser.add_argument('--output_path',  required=True)
    parser.add_argument('--smpl_model',   default=str(DATA_DIR / 'SMPL_NEUTRAL.pkl'))
    parser.add_argument('--mapping_path', default=str(DATA_DIR / 'mhr2smpl_mapping.npz'))
    parser.add_argument('--sample_idx',   default=str(DATA_DIR / 'smpl_vert_sample_indices.npy'))
    parser.add_argument('--frame_stride', type=int,   default=18)
    parser.add_argument('--max_views',    type=int,   default=4)
    parser.add_argument('--focal',        type=float, default=None,
                        help='Override focal length (px). Default: read from AIST camera JSON per video.')
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--max_motions',  type=int,   default=0, help='0=all; limit number of unique motions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("[Phase 0] Loading models...")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=str(PROJECT_DIR / 'checkpoints/sam-3d-body-dinov3'),
        detector_name='yolo_pose',
        detector_model=str(PROJECT_DIR / 'checkpoints/yolo/yolo11m-pose.engine'),
        fov_name=None,
        device=str(device),
    )

    smpl_model = smplx.SMPL(model_path=args.smpl_model, gender='neutral').to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad_(False)
    J_reg = smpl_model.J_regressor.detach().cpu().numpy()  # [24, 6890]

    mapping    = np.load(args.mapping_path)
    mhr_vert_ids = mapping['mhr_vert_ids'].astype(np.int64)   # [6890, 3]
    baryc        = mapping['baryc_coords'].astype(np.float32)  # [6890, 3]
    sample_idx   = np.load(args.sample_idx).astype(np.int64)  # [1500]
    V_sub = len(sample_idx)
    print(f"  sample_idx: {V_sub} verts")

    # ── Load camera settings ──────────────────────────────────────────────────
    print("[Phase 1] Loading camera settings and mapping...")
    import json
    cam_dir = Path(args.aist_dir) / 'cameras'
    settings = {}
    for f in cam_dir.glob('setting*.json'):
        with open(f) as fp:
            cams_list = json.load(fp)
        settings[f.stem] = {c['name']: c for c in cams_list}

    seq_to_setting = {}
    with open(cam_dir / 'mapping.txt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                seq_to_setting[parts[0]] = parts[1]
    print(f"  Camera settings: {len(settings)}, Mapped sequences: {len(seq_to_setting)}")

    # ── Find videos + motion pkls ─────────────────────────────────────────────
    print("[Phase 1b] Scanning videos and motions...")
    video_dir  = Path(args.video_dir)
    motion_dir = Path(args.aist_dir) / 'motions'

    all_video_files = sorted(video_dir.glob('*.mp4'))

    # Limit by unique motions if requested
    if args.max_motions > 0:
        # First pass: collect all unique motion keys in order
        seen, ordered_motions = set(), []
        for vf in all_video_files:
            mk, _ = parse_video_name(vf.stem)
            if mk and mk not in seen:
                ordered_motions.append(mk)
                seen.add(mk)
        selected_motions = set(ordered_motions[:args.max_motions])
        # Second pass: include ALL videos for those motions (all cameras)
        video_files = [vf for vf in all_video_files
                       if parse_video_name(vf.stem)[0] in selected_motions]
    else:
        video_files = all_video_files
    print(f"  Videos to process: {len(video_files)}")

    # ── Process videos ────────────────────────────────────────────────────────
    # Buffer: {(motion_key, frame_idx): [(cam_id, verts_centered [1500,3])]}
    buf = defaultdict(list)
    skipped_no_motion = 0
    skipped_no_det    = 0
    total_frames      = 0

    for vf in tqdm(video_files, desc='Videos'):
        motion_key, cam_id = parse_video_name(vf.stem)
        if motion_key is None:
            continue

        pkl_path = motion_dir / f"{motion_key}.pkl"
        if not pkl_path.exists():
            skipped_no_motion += 1
            continue

        gt = load_motion_gt(pkl_path)
        N_gt = len(gt['smpl_poses'])

        # Get focal length from camera JSON
        setting_name = seq_to_setting.get(motion_key)
        cam_key = f"c{cam_id}"
        if args.focal is not None:
            focal = args.focal
        elif setting_name and setting_name in settings and cam_key in settings[setting_name]:
            K = settings[setting_name][cam_key]['matrix']
            focal = float(K[0][0])  # fx (== fy for AIST)
        else:
            focal = 1310.0  # AIST default fallback

        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % args.frame_stride == 0 and frame_idx < N_gt:
                img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pred = run_stage1(estimator, img_rgb, focal=focal)
                if pred is not None:
                    v_centered = preprocess_mhr(
                        pred['pred_vertices'], pred['pred_cam_t'],
                        mhr_vert_ids, baryc, sample_idx
                    )
                    buf[(motion_key, frame_idx)].append((cam_id, v_centered))
                    total_frames += 1
                else:
                    skipped_no_det += 1
            frame_idx += 1

        cap.release()

    print(f"\n  Total frame-views collected: {total_frames}")
    print(f"  Skipped (no motion pkl): {skipped_no_motion}")
    print(f"  Skipped (no detection):  {skipped_no_det}")

    # ── Build multi-view groups ───────────────────────────────────────────────
    print("[Phase 2] Building multi-view groups...")
    mv_groups = {k: v for k, v in buf.items() if len(v) >= 2}
    print(f"  Multi-view groups (>=2 views): {len(mv_groups)}")

    if len(mv_groups) == 0:
        print("No multi-view data collected!")
        return

    # ── Collect GT params ─────────────────────────────────────────────────────
    print("[Phase 3] Collecting GT params and building arrays...")
    V_max = args.max_views
    N = len(mv_groups)

    all_verts        = np.zeros((N, V_max, V_sub, 3), dtype=np.float32)
    all_mask         = np.zeros((N, V_max), dtype=bool)
    all_joints_world = np.zeros((N, 24, 3), dtype=np.float32)
    all_gt_bp        = []
    all_gt_betas     = []
    all_gt_go        = []

    motion_cache = {}

    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    cam_usage = defaultdict(int)      # track how many times each cam is selected

    for fi, ((motion_key, frame_idx), views) in enumerate(tqdm(sorted(mv_groups.items()), desc='Groups')):
        if motion_key not in motion_cache:
            pkl_path = motion_dir / f"{motion_key}.pkl"
            motion_cache[motion_key] = load_motion_gt(pkl_path)
        gt = motion_cache[motion_key]

        gt_pose = gt['smpl_poses'][frame_idx]   # [72]
        gt_go   = gt_pose[:3].astype(np.float32)
        gt_bp   = np.zeros(69, dtype=np.float32)
        gt_bp[:69] = gt_pose[3:72]
        gt_betas = np.zeros(10, dtype=np.float32)

        # Randomly shuffle views so all cameras get uniform coverage across frames
        views_shuffled = list(views)
        rng.shuffle(views_shuffled)

        all_joints_world[fi] = 0.0  # placeholder; not used in fuse_feat training

        for vi, (cam_id, v_centered) in enumerate(views_shuffled[:V_max]):
            all_verts[fi, vi] = v_centered
            all_mask[fi, vi]  = True
            cam_usage[cam_id] += 1

        all_gt_go.append(gt_go)
        all_gt_bp.append(gt_bp)
        all_gt_betas.append(gt_betas)

    all_gt_go    = np.stack(all_gt_go).astype(np.float32)
    all_gt_bp    = np.stack(all_gt_bp).astype(np.float32)
    all_gt_betas = np.stack(all_gt_betas).astype(np.float32)

    print(f"  View availability (slot usage): {all_mask.sum(axis=0)}")
    print(f"  Camera usage (uniform check): { {k: cam_usage[k] for k in sorted(cam_usage)} }")

    # ── Canonical joints ──────────────────────────────────────────────────────
    print("[Phase 4] Computing canonical joints...")
    canon_joints_list = []
    for i in tqdm(range(0, N, args.batch_size), desc='Canon joints'):
        end = min(i + args.batch_size, N)
        bs  = end - i
        bp_t = torch.tensor(all_gt_bp[i:end]).to(device)
        be_t = torch.tensor(all_gt_betas[i:end]).to(device)
        go_z = torch.zeros(bs, 3, device=device)
        with torch.no_grad():
            out = smpl_model(global_orient=go_z, body_pose=bp_t, betas=be_t)
        j = out.joints[:, :24].cpu().numpy()
        j -= j[:, 0:1]
        canon_joints_list.append(j)
    canon_joints = np.concatenate(canon_joints_list, axis=0).astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"[Phase 5] Saving to {args.output_path}...")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'smpl_target_verts_sampled': all_verts,         # [N, V, 1500, 3]
        'view_mask':                 all_mask,           # [N, V] bool
        'num_views_total':           np.array([V_max]),
        'smpl_vert_sample_indices':  sample_idx,         # [1500]
        'smpl_joints_canonical':     canon_joints,       # [N, 24, 3]
        'smpl_joints_world':         all_joints_world,   # [N, 24, 3]
        'gt_body_pose':              all_gt_bp,          # [N, 69]
        'gt_betas':                  all_gt_betas,       # [N, 10]
        'gt_global_orient':          all_gt_go,          # [N, 3]
        'supervision_mode':          np.array([0]),      # 0 = GT
    }

    np.savez(str(out_path), **save_dict)
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path} ({mb:.1f} MB)")
    for k, v in save_dict.items():
        print(f"    {k:30s} {str(v.shape):25s} dtype={v.dtype}")
    print("\nDone!")


if __name__ == '__main__':
    main()
