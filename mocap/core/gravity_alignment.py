import numpy as np
from scipy.spatial.transform import Rotation


def build_camera_to_world_rotation(gravity_cam):
    """
    Build rotation matrix from camera frame to gravity-aligned world frame (Z-up, right-hand).
    Returns R_world_cam where v_cam = R_world_cam @ v_world.
    """
    col0 = gravity_cam / np.linalg.norm(gravity_cam)
    cam_z = np.array([0.0, 0.0, 1.0])
    col1 = np.cross(col0, cam_z)
    col1_norm = np.linalg.norm(col1)
    if col1_norm < 1e-6:
        col1 = np.cross(col0, np.array([1.0, 0.0, 0.0]))
    col1 = col1 / np.linalg.norm(col1)
    col2 = np.cross(col0, col1)
    col2 = col2 / np.linalg.norm(col2)
    R_final = np.column_stack([col0, col1, col2])
    R_zup_inv = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=np.float64)
    return R_zup_inv @ R_final


def transform_pose_to_world(body_quat_cam, joints_cam, R_world_cam):
    """
    Transform body quaternion (wxyz) and joints from camera frame to world frame.
    Returns (body_quat_world wxyz, joints_world).
    """
    R_cam_world = R_world_cam.T
    R_quat = Rotation.from_matrix(R_cam_world)
    quat_cam_xyzw = np.array([body_quat_cam[1], body_quat_cam[2], body_quat_cam[3], body_quat_cam[0]])
    R_world_body = R_quat * Rotation.from_quat(quat_cam_xyzw)
    quat_world_xyzw = R_world_body.as_quat()
    body_quat_world = np.array([quat_world_xyzw[3], quat_world_xyzw[0], quat_world_xyzw[1], quat_world_xyzw[2]])
    joints_world = (R_cam_world @ joints_cam.T).T
    return body_quat_world, joints_world
