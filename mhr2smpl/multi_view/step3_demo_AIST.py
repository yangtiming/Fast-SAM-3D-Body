"""
Multi-Camera Full Pipeline Demo for AIST++ (1-3 cameras)
========================================================
N camera videos read frames synchronously, batch=N forward → multi-view fusion → (2N)-column visualization.

Col 1..N:   CamX image + MHR skeleton (green)
Col N+1:    1V SMPL (cyan) — single view
Col N+2:    2V SMPL (yellow) — 2-view fusion
Col N+3:    3V SMPL (orange) — 3-view fusion

Usage:
  cd /path/to/CLEAN/Fast_sam-3d-body_mhr2smpl
  python demo_multiview_aist.py --motion gBR_sBM_cAll_d04_mBR0_ch01 --cams 01 05 03
"""

import sys
import os
import argparse
import inspect
from pathlib import Path

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import numpy as np
for _attr in ('bool', 'int', 'float', 'complex', 'object', 'str'):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))
if not hasattr(np, 'unicode'):
    np.unicode = str

import cv2
import torch
from scipy.spatial.transform import Rotation as ScipyR

SHARE_DIR   = Path(__file__).parent.resolve()        # multi_view/
MHR2SMPL_DIR = SHARE_DIR.parent                      # mhr2smpl/
PROJECT_DIR = MHR2SMPL_DIR.parent                     # Fast_sam-3d-body/

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SHARE_DIR))

from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from notebook.utils import setup_sam_3d_body
from infer_multiview import MHR2SMPLMultiView

SKELETON = [
    (0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),
    (0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(20,22),
    (9,14),(14,17),(17,19),(19,21),(21,23),
]

AIST_VIDEO_DIR = '/home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/AIST/videos'


def pick_main_bbox(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.array(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = areas.argmax()
    return boxes[idx:idx+1]


def project_joints(j3d, focal, cx, cy):
    u = j3d[:, 0] / j3d[:, 2] * focal + cx
    v = j3d[:, 1] / j3d[:, 2] * focal + cy
    return u, v


def draw_skeleton(img, u, v, color, lw=4, r=8):
    for i, j in SKELETON:
        p1, p2 = (int(u[i]), int(v[i])), (int(u[j]), int(v[j]))
        if (0 < p1[0] < img.shape[1] and 0 < p2[0] < img.shape[1] and
                0 < p1[1] < img.shape[0] and 0 < p2[1] < img.shape[0]):
            cv2.line(img, p1, p2, color, lw, cv2.LINE_AA)
    for i in range(len(u)):
        pt = (int(u[i]), int(v[i]))
        if 0 < pt[0] < img.shape[1] and 0 < pt[1] < img.shape[0]:
            cv2.circle(img, pt, r, color, -1, cv2.LINE_AA)


def mhr_to_smpl_joints(pred_vertices, pred_cam_t, J_reg, mhr_faces, tri_ids, baryc):
    v_world = pred_vertices + pred_cam_t[None, :]
    face_v  = v_world[mhr_faces[tri_ids]]
    smpl_v  = (face_v * baryc[:, :, None]).sum(axis=1)
    return J_reg @ smpl_v


@torch.no_grad()
def process_images_batched(estimator, images_rgb, focal=1000.0):
    """Batch inference on N images, returns list of pred dicts (or None)."""
    N = len(images_rgb)
    results = []
    batches = []

    for img_rgb in images_rgb:
        H, W = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if estimator.detector is not None:
            det = estimator.detector.run_human_detection(
                img_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
                default_to_full_image=False,
            )
            boxes = det['boxes'] if isinstance(det, dict) else det
        else:
            boxes = np.array([[0, 0, W, H]], dtype=np.float32)

        boxes = pick_main_bbox(boxes)

        cam_int = torch.tensor([[
            [focal,     0, W / 2.0],
            [0,     focal, H / 2.0],
            [0,         0,     1.0],
        ]]).float()

        if boxes is None or len(boxes) == 0:
            results.append(None)
            batches.append(None)
        else:
            batch = prepare_batch(img_rgb, estimator.transform, boxes)
            batch['cam_int'] = cam_int.to(batch['img'])
            results.append('pending')
            batches.append(batch)

    valid_batches = [b for b in batches if b is not None]
    if not valid_batches:
        return [None] * N

    tensor_keys = ['img', 'img_size', 'ori_img_size', 'bbox_center', 'bbox_scale',
                   'bbox', 'affine_trans', 'mask', 'mask_score', 'cam_int', 'person_valid']
    if len(valid_batches) == 1:
        merged = valid_batches[0]
    else:
        merged = {}
        for k in tensor_keys:
            tensors = [b[k] for b in valid_batches if k in b]
            if tensors:
                merged[k] = torch.cat(tensors, dim=0)
        all_img_ori = []
        for b in valid_batches:
            all_img_ori.extend(b['img_ori'])
        merged['img_ori'] = all_img_ori

    merged = recursive_to(merged, 'cuda')
    estimator.model._initialize_batch(merged)
    pose_output = estimator.model.forward_step(merged, decoder_type='body')

    out = recursive_to(recursive_to(pose_output['mhr'], 'cpu'), 'numpy')
    num_valid = len(valid_batches)
    preds = [{'pred_vertices': out['pred_vertices'][i],
              'pred_cam_t':    out['pred_cam_t'][i]} for i in range(num_valid)]

    pi = 0
    final = []
    for r in results:
        if r is None:
            final.append(None)
        else:
            final.append(preds[pi])
            pi += 1
    return final


def motion_to_video_path(motion_key, cam_id, video_dir):
    stem = motion_key.replace('_cAll_', f'_c{cam_id}_')
    return Path(video_dir) / f'{stem}.mp4'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion',     required=True,
                        help='AIST++ motion key, e.g. gBR_sBM_cAll_d04_mBR0_ch01')
    parser.add_argument('--cams',       nargs='+', default=['01', '05'],
                        help='Camera IDs (1-3 cameras)')
    # Backward compat
    parser.add_argument('--cam0',       default=None)
    parser.add_argument('--cam1',       default=None)
    parser.add_argument('--video_dir',  default=AIST_VIDEO_DIR)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--out',        default=None)
    parser.add_argument('--focal',      type=float, default=1000.0)
    parser.add_argument('--model_path', default=None)
    args = parser.parse_args()

    # Handle backward compat
    if args.cam0 is not None and args.cam1 is not None:
        args.cams = [args.cam0, args.cam1]
    elif args.cam0 is not None:
        args.cams = [args.cam0]

    cam_ids = args.cams
    N_cams = len(cam_ids)
    assert 1 <= N_cams <= 3, f"Support 1-3 cameras, got {N_cams}"

    if args.out is None:
        cam_str = 'c' + 'c'.join(cam_ids)
        out_dir = MHR2SMPL_DIR / 'output_visualization'
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / f'output_aist_{args.motion}_{cam_str}.mp4')

    # Resolve video paths
    vid_paths = []
    for c in cam_ids:
        vp = motion_to_video_path(args.motion, c, args.video_dir)
        print(f"Video cam{c}: {vp}")
        if not vp.exists():
            print(f"ERROR: {vp} not found!"); return
        vid_paths.append(vp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading SAM3DBody estimator...")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=str(PROJECT_DIR / 'checkpoints/sam-3d-body-dinov3'),
        detector_name='yolo_pose',
        detector_model=str(PROJECT_DIR / 'checkpoints/yolo/yolo11m-pose.engine'),
        fov_name=None,
        device=str(device),
    )

    print("Loading MHR2SMPL multi-view model...")
    mv_model = MHR2SMPLMultiView(
        model_path=args.model_path if args.model_path else str(MHR2SMPL_DIR / 'weights/best_model.pth'),
        mapping_path=str(MHR2SMPL_DIR / 'data/mhr2smpl_mapping.npz'),
        sample_idx_path=str(MHR2SMPL_DIR / 'data/smpl_vert_sample_indices.npy'),
        device=str(device),
        smoother_dir=str(MHR2SMPL_DIR / 'experiments/smoother_w5'),
    )

    import pickle
    from scipy import sparse
    with open(str(MHR2SMPL_DIR / 'data/SMPL_NEUTRAL.pkl'), 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')
    J_reg = smpl_data['J_regressor']
    if sparse.issparse(J_reg):
        J_reg = J_reg.toarray()
    mapping   = np.load(str(MHR2SMPL_DIR / 'data/mhr2smpl_mapping.npz'))
    tri_ids   = mapping['triangle_ids']
    baryc     = mapping['baryc_coords']
    mhr_faces = np.load('/home/jiawei/timingyang/sam-3d-body/eval_ECCV/mhr_faces.npy')

    import smplx
    smpl_model = smplx.SMPL(str(MHR2SMPL_DIR / 'data/SMPL_NEUTRAL.pkl'),
                             gender='neutral').to(device)
    smpl_model.eval()
    for p in smpl_model.parameters():
        p.requires_grad_(False)

    # Open videos
    caps = []
    for vp in vid_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"ERROR: Cannot open {vp}!"); return
        caps.append(cap)

    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    totals = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    total = min(*totals, args.max_frames)
    print(f"FPS={fps:.0f}  Total frames: {totals}, using {total}")

    ret, first_frame = caps[0].read()
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    H, W = first_frame.shape[:2] if ret else (1080, 1920)
    scale = min(1.0, 640.0 / W)
    OW, OH = int(W * scale), int(H * scale)

    N_cols = 2 * N_cams  # N cam panels + N result panels (1V, 2V, ..., NV)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                             min(fps, 30), (OW * N_cols, OH))
    print(f"Writing → {args.out}  ({OW * N_cols}x{OH})  [{N_cols} columns]")

    mv_model.reset()
    prev_q_go = None
    focal_v, cx, cy = args.focal, W / 2, H / 2

    for fi in range(total):
        frames_bgr = []
        ok = True
        for cap in caps:
            ret, frame_bgr = cap.read()
            if not ret:
                ok = False; break
            frames_bgr.append(frame_bgr)
        if not ok:
            break

        imgs_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

        preds = process_images_batched(estimator, imgs_rgb, focal=args.focal)

        views = []
        for pred in preds:
            if pred is not None:
                views.append((pred['pred_vertices'].astype(np.float32),
                              pred['pred_cam_t'].astype(np.float32)))

        if not views:
            blank = np.zeros((H, W, 3), np.uint8)
            frame = np.concatenate([blank] * N_cols, axis=1)
            if scale < 1.0:
                frame = cv2.resize(frame, (OW * N_cols, OH))
            writer.write(frame)
            continue

        # Infer SMPL for each view count: 1V, 2V, ..., NV
        infer_results = []
        for nv in range(1, len(views) + 1):
            go_nv, bp_nv, betas_nv, w_nv = mv_model.infer(views[:nv])
            bp69 = np.zeros(69, np.float32); bp69[:63] = bp_nv
            with torch.no_grad():
                out_nv = smpl_model(
                    global_orient=torch.zeros(1, 3, device=device),
                    body_pose=torch.from_numpy(bp69).unsqueeze(0).to(device),
                    betas=torch.from_numpy(betas_nv).unsqueeze(0).to(device),
                )
            j_nv = out_nv.joints[0, :24].cpu().numpy()
            j_nv -= j_nv[0:1]
            infer_results.append((j_nv, w_nv))

        # Col 1..N: MHR skeleton (green) per camera
        panels = []
        j_cam0 = None
        view_idx = 0
        for ci, (c, pred, bgr) in enumerate(zip(cam_ids, preds, frames_bgr)):
            panel = bgr.copy()
            if pred is not None:
                j_mhr = mhr_to_smpl_joints(pred['pred_vertices'], pred['pred_cam_t'],
                                            J_reg, mhr_faces, tri_ids, baryc)
                u, v = project_joints(j_mhr, focal_v, cx, cy)
                draw_skeleton(panel, u, v, (0, 255, 0))
                cv2.putText(panel, f'c{c} MHR',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 7)
                if ci == 0:
                    j_cam0 = j_mhr
                view_idx += 1
            if ci == 0:
                cv2.putText(panel, f'frame {fi}',
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (200, 200, 200), 6)
            panels.append(panel)

        # Compute R_cam from full-view result for consistent projection
        pelvis_cam = j_cam0[0] if j_cam0 is not None else np.array([0.0, 0.0, 3.0])
        j_full = infer_results[-1][0]

        if j_cam0 is not None:
            j_mhr_rel = j_cam0 - j_cam0[0]
            A = j_full[1:].T
            B = j_mhr_rel[1:].T
            Cov = A @ B.T
            U, _, Vt = np.linalg.svd(Cov)
            R_cam = Vt.T @ U.T
            q_go = ScipyR.from_matrix(R_cam).as_quat()
            if prev_q_go is not None:
                if np.dot(q_go, prev_q_go) < 0:
                    q_go = -q_go
                q_go = 0.7 * q_go + 0.3 * prev_q_go
                q_go /= np.linalg.norm(q_go)
            prev_q_go = q_go
            R_cam = ScipyR.from_quat(q_go).as_matrix()
        else:
            R_cam = np.eye(3)

        # Result panels: 1V, 2V, ..., NV
        NV_COLORS = [(255, 255, 0), (0, 255, 255), (0, 100, 255)]
        NV_LABELS = ['1V SMPL', '2V SMPL', '3V SMPL']

        for nv_idx, (j_nv, w_nv) in enumerate(infer_results):
            j_rot = (R_cam @ j_nv.T).T
            j_cam = j_rot + pelvis_cam
            us_nv, vs_nv = project_joints(j_cam, focal_v, cx, cy)

            panel_nv = frames_bgr[0].copy()
            draw_skeleton(panel_nv, us_nv, vs_nv, NV_COLORS[nv_idx])
            w_str = ', '.join(f'{w:.2f}' for w in w_nv)
            cv2.putText(panel_nv, f'{NV_LABELS[nv_idx]}  w=[{w_str}]',
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, NV_COLORS[nv_idx], 6)
            panels.append(panel_nv)

        # Pad if some views failed detection
        while len(panels) < N_cols:
            panels.append(np.zeros((H, W, 3), np.uint8))

        frame = np.concatenate(panels, axis=1)
        if scale < 1.0:
            frame = cv2.resize(frame, (OW * N_cols, OH))
        writer.write(frame)

        if fi % 10 == 0:
            print(f"  [{fi+1}/{total}] views={len(views)}")

    for cap in caps:
        cap.release()
    writer.release()
    print(f"\nDone → {args.out}")


if __name__ == '__main__':
    main()
