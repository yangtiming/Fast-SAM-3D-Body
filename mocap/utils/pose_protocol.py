import numpy as np
from scipy.spatial.transform import Rotation

from mocap.core.gravity_alignment import transform_pose_to_world

# Coordinate transform from SMPL body-local frame to protocol frame.
JOINTS_COORD_TRANSFORM = np.eye(3, dtype=np.float64)

# Match PICO protocol: remove SMPL fixed base rotation from published body quaternion.
SMPL_BASE_REMOVE_QUAT_XYZW = np.array([-0.5, -0.5, -0.5, 0.5], dtype=np.float64)

# Global orientation adjustment (reversed order): apply X-90 first, then Y+90.
GLOBAL_ORIENT_EXTRA_ROT = Rotation.from_euler("y", 90.0, degrees=True) * Rotation.from_euler(
    "x", -90.0, degrees=True
)


def quat_apply(quat, vec):
    qw, qx, qy, qz = quat
    qvec = np.array([qx, qy, qz])
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (uv * qw + uuv)


def quat_inverse(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=quat.dtype)


def quat_wxyz_to_xyzw(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def prepare_publish_pose(body_quat_xyzw, joints_world_or_canonical, smpl_pose, R_world_cam,
                         *, joints_are_world=False):
    """Convert converter output to the final published representation.

    Args:
        body_quat_xyzw: body orientation in camera frame, xyzw convention.
        joints_world_or_canonical: (24, 3) joints either already in world space
            (joints_are_world=True) or in canonical (zero-pose) space that need
            rotating into camera frame first.
        smpl_pose: (21, 3) body pose array.
        R_world_cam: (3, 3) rotation matrix from camera to world frame.
        joints_are_world: if False (default), joints are canonical and will be
            rotated by body_quat before world-transform.

    Returns:
        q_world_wxyz: body quaternion in world frame, wxyz convention.
        joints_local: (24, 3) joints in root-relative local frame.
        pose: smpl_pose cast to float64.
    """
    body_quat_xyzw = np.asarray(body_quat_xyzw, dtype=np.float64)
    pose = np.asarray(smpl_pose, dtype=np.float64)

    if joints_are_world:
        joints_cam = np.asarray(joints_world_or_canonical, dtype=np.float64)
    else:
        joints_cam = Rotation.from_quat(body_quat_xyzw).apply(
            np.asarray(joints_world_or_canonical, dtype=np.float64)
        )

    q_cam_wxyz = quat_xyzw_to_wxyz(body_quat_xyzw)
    q_world, joints_world = transform_pose_to_world(q_cam_wxyz, joints_cam, R_world_cam)

    q_world_xyzw = quat_wxyz_to_xyzw(q_world)
    q_world_xyzw = (
        Rotation.from_quat(q_world_xyzw) * Rotation.from_quat(SMPL_BASE_REMOVE_QUAT_XYZW)
    ).as_quat()
    q_world = quat_xyzw_to_wxyz(q_world_xyzw)

    root_pos = joints_world[0]
    joints_local = quat_apply(quat_inverse(q_world), joints_world - root_pos)
    joints_local = joints_local @ JOINTS_COORD_TRANSFORM.T

    q_world_xyzw = quat_wxyz_to_xyzw(q_world)
    q_world_xyzw = (GLOBAL_ORIENT_EXTRA_ROT * Rotation.from_quat(q_world_xyzw)).as_quat()
    q_world = quat_xyzw_to_wxyz(q_world_xyzw)

    return q_world.astype(np.float64), joints_local.astype(np.float64), pose
