import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _alias, _target in (
    ("bool", np.bool_), ("int", np.int_), ("float", np.float64),
    ("complex", np.complex128), ("object", np.object_), ("str", np.str_),
    ("unicode", np.str_),
):
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)

import smplx  # noqa: E402  (must follow compat patch)

from mocap.utils.renderer import Renderer


# SMPL kinematic tree: parent index for each of the 24 joints
SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)



def load_smpl_model(smpl_model_path):
    model_path = Path(smpl_model_path)
    if not model_path.is_file():
        raise RuntimeError(f"SMPL model file not found: {smpl_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = smplx.SMPL(model_path=str(model_path), gender="neutral", batch_size=1)
    smpl_model = smpl_model.to(device)
    smpl_model.eval()
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    num_betas = int(getattr(smpl_model, "num_betas", 10))
    return smpl_model, faces, device, num_betas


def smpl_vertices_joints_from_pose(smpl_pose, smpl_model, device, num_betas, body_quat=None):
    pose = np.asarray(smpl_pose, dtype=np.float32)
    if pose.shape != (21, 3):
        raise RuntimeError(f"Expected smpl_pose shape (21,3), got {pose.shape}")

    if body_quat is None:
        global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
    else:
        quat_xyzw = np.asarray(body_quat, dtype=np.float32).reshape(4)
        global_rotvec = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
        global_orient = torch.from_numpy(global_rotvec.reshape(1, 3)).to(device=device, dtype=torch.float32)

    required_body_joints = int(getattr(smpl_model, "NUM_BODY_JOINTS", 23))
    if required_body_joints == 21:
        body_pose_np = pose
    elif required_body_joints == 23:
        body_pose_np = np.zeros((23, 3), dtype=np.float32)
        body_pose_np[:21] = pose
    else:
        raise RuntimeError(f"Unsupported SMPL body joint count: {required_body_joints}")

    body_pose = torch.from_numpy(body_pose_np.reshape(1, required_body_joints * 3)).to(device=device, dtype=torch.float32)
    betas = torch.zeros((1, num_betas), dtype=torch.float32, device=device)
    transl = torch.zeros((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
    root = joints[0:1].copy()
    verts -= root
    joints -= root
    return verts, joints


def _project_joints(joints, cam_t, focal_length, width, height, side_view=False):
    """Project joints to 2D matching Renderer.__call__'s camera model.

    Renderer places camera at [-cam_t[0], cam_t[1], cam_t[2]] with identity
    rotation (looking down -Z), and applies 180° X then optional 90° Y (side_view)
    to the mesh. We apply the same transforms to joints before projecting.
    """
    j = np.asarray(joints, dtype=np.float64)
    if side_view:
        # 90° Y rotation first: (x, y, z) -> (z, y, -x)
        j = np.stack([j[:, 2], j[:, 1], -j[:, 0]], axis=1)
    # 180° X second: negate Y and Z
    j = j * np.array([1.0, -1.0, -1.0])
    # Camera position (Renderer negates cam_t X)
    cam_pos = np.array([-float(cam_t[0]), float(cam_t[1]), float(cam_t[2])])
    # To camera space (identity rotation)
    p = j - cam_pos
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    mask = z < -0.01
    safe_z = np.where(mask, z, -1.0)
    cx, cy = width / 2.0, height / 2.0
    u = focal_length * (x / -safe_z) + cx
    v = focal_length * (y / safe_z) + cy
    return np.stack([u, v], axis=1), mask


def draw_skeleton(image, joints_2d, mask, color=(0, 0, 255), thickness=2):
    h, w = image.shape[:2]
    for i, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx == -1:
            continue
        if not mask[i] or not mask[parent_idx]:
            continue
        pt1 = tuple(joints_2d[i].astype(int))
        pt2 = tuple(joints_2d[parent_idx].astype(int))
        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
            cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)
    for i in range(len(joints_2d)):
        if mask[i]:
            pt = tuple(joints_2d[i].astype(int))
            if 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]:
                cv2.circle(image, pt, thickness + 1, (255, 0, 0), -1, cv2.LINE_AA)


def render_smpl_records_video(records, output_path, smpl_model_path, fps, width, height, show_joints=True):
    if not records:
        raise RuntimeError("No records to render")

    smpl_model, faces, device, num_betas = load_smpl_model(smpl_model_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        float(fps),
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    focal_length = float(max(width, height))
    renderer = Renderer(focal_length=focal_length, faces=faces)
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)

    for index, record in enumerate(tqdm(records, total=len(records), desc="Render", unit="frame")):
        verts, mesh_joints = smpl_vertices_joints_from_pose(
            record["smpl_pose"],
            smpl_model=smpl_model,
            device=device,
            num_betas=num_betas,
            body_quat=record.get("body_quat"),
        )

        rendered_front = renderer(verts, cam_t, white_bg.copy(),
                                  mesh_base_color=(0.65, 0.74, 0.86), scene_bg_color=(1, 1, 1))
        frame_front = (rendered_front * 255).astype(np.uint8)

        rendered_side = renderer(verts, cam_t, white_bg.copy(), side_view=True,
                                 mesh_base_color=(0.65, 0.74, 0.86), scene_bg_color=(1, 1, 1))
        frame_side = (rendered_side * 255).astype(np.uint8)

        if show_joints:
            j2d_front, mask_front = _project_joints(mesh_joints, cam_t, focal_length, width, height)
            draw_skeleton(frame_front, j2d_front, mask_front)
            j2d_side, mask_side = _project_joints(mesh_joints, cam_t, focal_length, width, height, side_view=True)
            draw_skeleton(frame_side, j2d_side, mask_side)

        cv2.putText(frame_front, "Front", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(frame_side, "Side", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)

        combined = np.concatenate([frame_front, frame_side], axis=1)
        cv2.putText(
            combined,
            f"msg={index + 1}/{len(records)} frame_idx={record['frame_index']}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA,
        )
        writer.write(combined)

    writer.release()
    renderer.delete()
