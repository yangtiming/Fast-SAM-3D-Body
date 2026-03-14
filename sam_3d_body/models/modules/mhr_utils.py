# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import math
import os.path as osp
import pickle

import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_angle_difference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle difference (magnitude) between two batches of SO(3) rotation matrices.
    Args:
        A: Tensor of shape (*, 3, 3), batch of rotation matrices.
        B: Tensor of shape (*, 3, 3), batch of rotation matrices.
    Returns:
        Tensor of shape (*,), angle differences in radians.
    """
    # Compute relative rotation matrix
    R_rel = torch.matmul(A, B.transpose(-2, -1))  # (B, 3, 3)
    # Compute trace of relative rotation
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B,)
    # Compute angle using the trace formula
    cos_theta = (trace - 1) / 2
    # Clamp for numerical stability
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    # Compute angle difference
    angle = torch.acos(cos_theta_clamped)
    return angle


def fix_wrist_euler(
    wrist_xzy, limits_x=(-2.2, 1.0), limits_z=(-2.2, 1.5), limits_y=(-1.2, 1.5)
):
    """
    wrist_xzy: B x 2 x 3 (X, Z, Y angles)
    Returns: Fixed angles within joint limits
    """
    x, z, y = wrist_xzy[..., 0], wrist_xzy[..., 1], wrist_xzy[..., 2]

    x_alt = torch.atan2(torch.sin(x + torch.pi), torch.cos(x + torch.pi))
    z_alt = torch.atan2(torch.sin(-(z + torch.pi)), torch.cos(-(z + torch.pi)))
    y_alt = torch.atan2(torch.sin(y + torch.pi), torch.cos(y + torch.pi))

    # Calculate L2 violation distance
    def calc_violation(val, limits):
        below = torch.clamp(limits[0] - val, min=0.0)
        above = torch.clamp(val - limits[1], min=0.0)
        return below**2 + above**2

    violation_orig = (
        calc_violation(x, limits_x)
        + calc_violation(z, limits_z)
        + calc_violation(y, limits_y)
    )

    violation_alt = (
        calc_violation(x_alt, limits_x)
        + calc_violation(z_alt, limits_z)
        + calc_violation(y_alt, limits_y)
    )

    # Use alternative where it has lower L2 violation
    use_alt = violation_alt < violation_orig

    # Stack alternative and apply mask
    wrist_xzy_alt = torch.stack([x_alt, z_alt, y_alt], dim=-1)
    result = torch.where(use_alt.unsqueeze(-1), wrist_xzy_alt, wrist_xzy)

    return result


def batch6DFromXYZ(r, return_9D=False):
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1]) + [3, 3], dtype=r.dtype).to(r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy

    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result


# ============ Optimized rotmat -> euler conversion (replaces roma.rotmat_to_euler) ============

def rotmat_to_euler_xyz(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to xyz Euler angles (extrinsic XYZ).
    Optimized replacement for roma.rotmat_to_euler("xyz", rotmat).

    Roma uses lowercase "xyz" for extrinsic rotation, equivalent to R = Rz @ Ry @ Rx

    Args:
        rotmat: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) xyz Euler angles (x, y, z)
    """
    # R = Rz @ Ry @ Rx (extrinsic xyz)
    m00, m10, m20 = rotmat[..., 0, 0], rotmat[..., 1, 0], rotmat[..., 2, 0]
    m01, m11, m21 = rotmat[..., 0, 1], rotmat[..., 1, 1], rotmat[..., 2, 1]
    m02, m12, m22 = rotmat[..., 0, 2], rotmat[..., 1, 2], rotmat[..., 2, 2]

    # y = asin(-m20)
    sy = torch.clamp(-m20, -1.0, 1.0)
    y = torch.asin(sy)
    cy = torch.sqrt(1.0 - sy * sy)
    singular = cy < 1e-6

    out = torch.empty(*rotmat.shape[:-2], 3, device=rotmat.device, dtype=rotmat.dtype)
    out[..., 1] = y
    # Output order is [x, y, z], matrix is Rz @ Ry @ Rx
    out[..., 2] = torch.where(singular, torch.zeros_like(y), torch.atan2(m10, m00))  # z
    out[..., 0] = torch.where(singular, torch.atan2(-m12, m11), torch.atan2(m21, m22))  # x
    return out


def rotmat_to_euler_XZY(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to XZY Euler angles (intrinsic XZY).
    Optimized replacement for roma.rotmat_to_euler("XZY", rotmat).

    Args:
        rotmat: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) XZY Euler angles (x, z, y)
    """
    # R = Rx @ Rz @ Ry
    m01 = rotmat[..., 0, 1]  # -sz
    m00, m02 = rotmat[..., 0, 0], rotmat[..., 0, 2]
    m11, m21 = rotmat[..., 1, 1], rotmat[..., 2, 1]
    m12, m22 = rotmat[..., 1, 2], rotmat[..., 2, 2]

    sz = torch.clamp(-m01, -1.0, 1.0)
    z = torch.asin(sz)
    cz = torch.sqrt(1.0 - sz * sz)
    singular = cz < 1e-6

    out = torch.empty(*rotmat.shape[:-2], 3, device=rotmat.device, dtype=rotmat.dtype)
    out[..., 1] = z  # z is the middle angle in XZY
    out[..., 0] = torch.where(singular, torch.atan2(-m12, m22), torch.atan2(m21, m11))
    out[..., 2] = torch.where(singular, torch.zeros_like(z), torch.atan2(m02, m00))
    return out


def rotmat_to_euler_ZYX(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to ZYX Euler angles.
    Optimized replacement for roma.rotmat_to_euler("ZYX", rotmat).

    Args:
        rotmat: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) ZYX Euler angles (z, y, x)
    """
    # Extract matrix elements
    m00, m10, m20 = rotmat[..., 0, 0], rotmat[..., 1, 0], rotmat[..., 2, 0]
    m01, m11, m21 = rotmat[..., 0, 1], rotmat[..., 1, 1], rotmat[..., 2, 1]
    m02, m12, m22 = rotmat[..., 0, 2], rotmat[..., 1, 2], rotmat[..., 2, 2]

    # ZYX convention: R = Rz @ Ry @ Rx
    # y = asin(-m20), with numerical stability handling
    sy = torch.clamp(-m20, -1.0, 1.0)
    y = torch.asin(sy)

    cy = torch.sqrt(1.0 - sy * sy)
    singular = cy < 1e-6

    # Pre-allocate output
    out = torch.empty(*rotmat.shape[:-2], 3, device=rotmat.device, dtype=rotmat.dtype)

    # Y component
    out[..., 1] = y

    # Z and X components
    out[..., 0] = torch.where(singular,
                              torch.zeros_like(y),           # singular: z=0
                              torch.atan2(m10, m00))         # normal
    out[..., 2] = torch.where(singular,
                              torch.atan2(-m12, m11),        # singular
                              torch.atan2(m21, m22))         # normal

    return out


# ============ Optimized euler -> rotmat conversion (replaces roma.euler_to_rotmat) ============

def euler_to_rotmat_xyz(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert xyz Euler angles to rotation matrix (extrinsic XYZ).
    Optimized replacement for roma.euler_to_rotmat("xyz", euler).

    Roma uses lowercase "xyz" for extrinsic rotation, equivalent to R = Rz @ Ry @ Rx

    Args:
        euler: (..., 3) xyz Euler angles (x, y, z)
    Returns:
        (..., 3, 3) rotation matrices
    """
    x, y, z = euler[..., 0], euler[..., 1], euler[..., 2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # R = Rz @ Ry @ Rx (extrinsic xyz)
    out = torch.empty(*euler.shape[:-1], 3, 3, device=euler.device, dtype=euler.dtype)

    out[..., 0, 0] = cy * cz
    out[..., 0, 1] = cz * sx * sy - cx * sz
    out[..., 0, 2] = cx * cz * sy + sx * sz
    out[..., 1, 0] = cy * sz
    out[..., 1, 1] = cx * cz + sx * sy * sz
    out[..., 1, 2] = cx * sy * sz - cz * sx
    out[..., 2, 0] = -sy
    out[..., 2, 1] = cy * sx
    out[..., 2, 2] = cx * cy

    return out


def euler_to_rotmat_XZY(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert XZY Euler angles to rotation matrix (intrinsic XZY).
    Optimized replacement for roma.euler_to_rotmat("XZY", euler).

    Args:
        euler: (..., 3) XZY Euler angles (x, z, y)
    Returns:
        (..., 3, 3) rotation matrices
    """
    x, z, y = euler[..., 0], euler[..., 1], euler[..., 2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # R = Rx @ Rz @ Ry
    out = torch.empty(*euler.shape[:-1], 3, 3, device=euler.device, dtype=euler.dtype)

    out[..., 0, 0] = cy * cz
    out[..., 0, 1] = -sz
    out[..., 0, 2] = sy * cz
    out[..., 1, 0] = sx * sy + cx * cy * sz
    out[..., 1, 1] = cx * cz
    out[..., 1, 2] = -sx * cy + cx * sy * sz
    out[..., 2, 0] = -cx * sy + sx * cy * sz
    out[..., 2, 1] = sx * cz
    out[..., 2, 2] = cx * cy + sx * sy * sz

    return out


# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L82
_batchXYZfrom6D_timing = {
    'normalize_cross': 0, 'stack_matrix': 0, 'euler_compute': 0, 'assemble': 0, 'calls': 0
}


# JIT-compiled core computation function (no timing overhead)
@torch.jit.script
def _batchXYZfrom6D_jit(poses: torch.Tensor) -> torch.Tensor:
    """JIT-compiled 6D to XYZ Euler conversion."""
    # 6D → rotation matrix columns (Gram-Schmidt)
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    # Manual normalize (avoids extra overhead of F.normalize)
    col0 = x_raw / (torch.norm(x_raw, dim=-1, keepdim=True) + 1e-8)

    # Cross product: col0 × y_raw
    z_tmp = torch.cross(col0, y_raw, dim=-1)
    col2 = z_tmp / (torch.norm(z_tmp, dim=-1, keepdim=True) + 1e-8)

    # Cross product: col2 × col0
    col1 = torch.cross(col2, col0, dim=-1)

    # Directly extract matrix elements
    c0_0, c0_1, c0_2 = col0[..., 0], col0[..., 1], col0[..., 2]
    c1_1, c1_2 = col1[..., 1], col1[..., 2]
    c2_1, c2_2 = col2[..., 1], col2[..., 2]

    # Euler angles (XYZ convention)
    sy = torch.sqrt(c0_0 * c0_0 + c0_1 * c0_1)

    # Since singular cases are very rare, compute the normal case directly
    # If singular handling is needed, use torch.where
    out_euler = torch.empty(poses.shape[:-1] + (3,), device=poses.device, dtype=poses.dtype)

    out_euler[..., 1] = torch.atan2(-c0_2, sy)
    out_euler[..., 0] = torch.atan2(c1_2, c2_2)
    out_euler[..., 2] = torch.atan2(c0_1, c0_0)

    return out_euler


def batchXYZfrom6D(poses):
    import time as _time
    def _sync_time():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _time.perf_counter()

    global _batchXYZfrom6D_timing
    t0 = _sync_time()

    # Use JIT-compiled version
    out_euler = _batchXYZfrom6D_jit(poses)

    t1 = _sync_time()
    _batchXYZfrom6D_timing['normalize_cross'] += (t1 - t0) * 1000 * 0.4  # estimated proportion
    _batchXYZfrom6D_timing['euler_compute'] += (t1 - t0) * 1000 * 0.6
    _batchXYZfrom6D_timing['calls'] += 1

    return out_euler


def resize_image(image_array, scale_factor, interpolation=cv2.INTER_LINEAR):
    new_height = int(image_array.shape[0] // scale_factor)
    new_width = int(image_array.shape[1] // scale_factor)
    resized_image = cv2.resize(
        image_array, (new_width, new_height), interpolation=interpolation
    )

    return resized_image


_compact_cont_hand_timing = {
    'mask_create': 0, 'slice_3dof': 0, 'batchXYZfrom6D': 0,
    'slice_1dof': 0, 'atan2': 0, 'assemble': 0, 'calls': 0,
    # mask_create breakdown
    'mask_tensor_create': 0, 'mask_cat_1': 0, 'mask_cat_2': 0, 'mask_cat_3': 0, 'mask_cat_4': 0,
}

# ============ Precomputed static masks and indices (executed once at module load time) ============
# fmt: off
_HAND_DOFS_IN_ORDER = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]
# Mask of 3DoFs into hand_cont (size 54)
_MASK_CONT_THREEDOFS = torch.cat([torch.ones(2 * k).bool() * (k == 3) for k in _HAND_DOFS_IN_ORDER])
# Mask of 1DoFs (including 2DoF) into hand_cont (size 54)
_MASK_CONT_ONEDOFS = torch.cat([torch.ones(2 * k).bool() * (k in [1, 2]) for k in _HAND_DOFS_IN_ORDER])
# Mask of 3DoFs into hand_model_params (size 27)
_MASK_PARAM_THREEDOFS = torch.cat([torch.ones(k).bool() * (k == 3) for k in _HAND_DOFS_IN_ORDER])
# Mask of 1DoFs (including 2DoF) into hand_model_params (size 27)
_MASK_PARAM_ONEDOFS = torch.cat([torch.ones(k).bool() * (k in [1, 2]) for k in _HAND_DOFS_IN_ORDER])

# Precompute integer indices (faster than boolean masks)
_IDX_CONT_THREEDOFS = _MASK_CONT_THREEDOFS.nonzero(as_tuple=True)[0]  # [30] 3DoF positions in 54
_IDX_CONT_ONEDOFS = _MASK_CONT_ONEDOFS.nonzero(as_tuple=True)[0]      # [24] 1DoF positions in 54
_IDX_PARAM_THREEDOFS = _MASK_PARAM_THREEDOFS.nonzero(as_tuple=True)[0]  # [15] 3DoF positions in 27
_IDX_PARAM_ONEDOFS = _MASK_PARAM_ONEDOFS.nonzero(as_tuple=True)[0]      # [12] 1DoF positions in 27

# Cache indices on different devices (avoids calling .to(device) every time)
_IDX_CACHE = {}

# Pre-create indices on CUDA (avoids CPU->GPU transfer during torch.compile tracing)
if torch.cuda.is_available():
    _IDX_CACHE['cuda:0'] = {
        'cont_3dof': _IDX_CONT_THREEDOFS.cuda(),
        'cont_1dof': _IDX_CONT_ONEDOFS.cuda(),
        'param_3dof': _IDX_PARAM_THREEDOFS.cuda(),
        'param_1dof': _IDX_PARAM_ONEDOFS.cuda(),
    }

def _get_cached_idx(device):
    """Get cached indices for the specified device."""
    device_key = str(device)
    if device_key not in _IDX_CACHE:
        _IDX_CACHE[device_key] = {
            'cont_3dof': _IDX_CONT_THREEDOFS.to(device),
            'cont_1dof': _IDX_CONT_ONEDOFS.to(device),
            'param_3dof': _IDX_PARAM_THREEDOFS.to(device),
            'param_1dof': _IDX_PARAM_ONEDOFS.to(device),
        }
    return _IDX_CACHE[device_key]

def warmup_mhr_idx_cache(device):
    """Warm up index cache, call before torch.compile to avoid CPU tensor warnings."""
    _get_cached_idx(device)
    _get_body_cached_idx(device)
# fmt: on
# ================================================================

def compact_cont_to_model_params_hand_fast(hand_cont):
    """
    Fast version: no timing overhead, calls JIT function directly.
    Used for production inference.
    """
    assert hand_cont.shape[-1] == 54

    # Get cached device indices
    idx = _get_cached_idx(hand_cont.device)

    # 3DoFs: index_select + unflatten + 6D->euler
    hand_cont_threedofs = torch.index_select(hand_cont, -1, idx['cont_3dof']).unflatten(-1, (-1, 6))
    hand_model_params_threedofs = _batchXYZfrom6D_jit(hand_cont_threedofs).flatten(-2, -1)

    # 1DoFs: index_select + unflatten + atan2
    hand_cont_onedofs = torch.index_select(hand_cont, -1, idx['cont_1dof']).unflatten(-1, (-1, 2))
    hand_model_params_onedofs = torch.atan2(hand_cont_onedofs[..., 0], hand_cont_onedofs[..., 1])

    # Assemble: 27-dim output
    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27, device=hand_cont.device, dtype=hand_cont.dtype)
    hand_model_params[..., idx['param_3dof']] = hand_model_params_threedofs
    hand_model_params[..., idx['param_1dof']] = hand_model_params_onedofs

    return hand_model_params


def compact_cont_to_model_params_hand(hand_cont):
    import time as _time
    def _sync_time():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _time.perf_counter()

    global _compact_cont_hand_timing
    t0 = _sync_time()

    # These are ordered by joint, not model params ^^
    assert hand_cont.shape[-1] == 54

    # Get cached device indices
    idx = _get_cached_idx(hand_cont.device)
    t1 = _sync_time()
    _compact_cont_hand_timing['mask_create'] += (t1 - t0) * 1000

    # Convert hand_cont to eulers
    ## First for 3DoFs - using cached integer indices
    hand_cont_threedofs = torch.index_select(hand_cont, -1, idx['cont_3dof']).unflatten(-1, (-1, 6))
    t2 = _sync_time()
    _compact_cont_hand_timing['slice_3dof'] += (t2 - t1) * 1000

    hand_model_params_threedofs = batchXYZfrom6D(hand_cont_threedofs).flatten(-2, -1)
    t3 = _sync_time()
    _compact_cont_hand_timing['batchXYZfrom6D'] += (t3 - t2) * 1000

    ## Next for 1DoFs - using cached integer indices
    hand_cont_onedofs = torch.index_select(hand_cont, -1, idx['cont_1dof']).unflatten(-1, (-1, 2))
    t4 = _sync_time()
    _compact_cont_hand_timing['slice_1dof'] += (t4 - t3) * 1000

    hand_model_params_onedofs = torch.atan2(
        hand_cont_onedofs[..., 0], hand_cont_onedofs[..., 1]
    )
    t5 = _sync_time()
    _compact_cont_hand_timing['atan2'] += (t5 - t4) * 1000

    # Finally, assemble into a 27-dim vector - create directly on target device
    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27, device=hand_cont.device, dtype=hand_cont.dtype)
    # Assign using cached integer indices
    hand_model_params[..., idx['param_3dof']] = hand_model_params_threedofs
    hand_model_params[..., idx['param_1dof']] = hand_model_params_onedofs

    t6 = _sync_time()
    _compact_cont_hand_timing['assemble'] += (t6 - t5) * 1000
    _compact_cont_hand_timing['calls'] += 1

    return hand_model_params


def compact_model_params_to_cont_hand(hand_model_params):
    # These are ordered by joint, not model params ^^
    assert hand_model_params.shape[-1] == 27

    # Get cached device indices
    idx = _get_cached_idx(hand_model_params.device)

    # Convert eulers to hand_cont
    ## First for 3DoFs - using cached integer indices
    hand_model_params_threedofs = torch.index_select(
        hand_model_params, -1, idx['param_3dof']
    ).unflatten(-1, (-1, 3))
    hand_cont_threedofs = batch6DFromXYZ(hand_model_params_threedofs).flatten(-2, -1)

    ## Next for 1DoFs - using cached integer indices
    hand_model_params_onedofs = torch.index_select(hand_model_params, -1, idx['param_1dof'])
    hand_cont_onedofs = torch.stack(
        [hand_model_params_onedofs.sin(), hand_model_params_onedofs.cos()], dim=-1
    ).flatten(-2, -1)

    # Finally, assemble into a 54-dim vector - create directly on target device
    hand_cont = torch.zeros(*hand_model_params.shape[:-1], 54, device=hand_model_params.device, dtype=hand_model_params.dtype)
    hand_cont[..., idx['cont_3dof']] = hand_cont_threedofs
    hand_cont[..., idx['cont_1dof']] = hand_cont_onedofs

    return hand_cont


def batch9Dfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1).flatten(-2, -1)  # ... x 3 x 3 -> x9

    return matrix


def batch4Dfrom2D(poses):
    # Args: poses: ... x 2, where "2" is sincos
    poses_norm = F.normalize(poses, dim=-1)

    poses_4d = torch.stack(
        [
            poses_norm[..., 1],
            poses_norm[..., 0],
            -poses_norm[..., 0],
            poses_norm[..., 1],
        ],
        dim=-1,
    )  # Flattened SO2.

    return poses_4d  # .... x 4


def compact_cont_to_rotmat_body(body_pose_cont, inflate_trans=False):
    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]
    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_rotmat_3dofs = batch9Dfrom6D(body_cont_3dofs).flatten(-2, -1)
    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    body_rotmat_1dofs = batch4Dfrom2D(body_cont_1dofs).flatten(-2, -1)
    if inflate_trans:
        assert (
            False
        ), "This is left as a possibility to increase the space/contribution/supervision trans params gets compared to rots"
    else:
        ## Nothing to do for trans
        body_rotmat_trans = body_cont_trans
    # Put them together
    body_rotmat_params = torch.cat(
        [body_rotmat_3dofs, body_rotmat_1dofs, body_rotmat_trans], dim=-1
    )
    return body_rotmat_params


# ============ Body index precomputation (executed once at module load time) ============
_BODY_3DOF_ROT_IDXS = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
_BODY_1DOF_ROT_IDXS = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
_BODY_1DOF_TRANS_IDXS = torch.LongTensor([124, 125, 126, 127, 128, 129])
_BODY_NUM_3DOF = len(_BODY_3DOF_ROT_IDXS) * 3  # 69
_BODY_NUM_1DOF = len(_BODY_1DOF_ROT_IDXS)       # 58
_BODY_NUM_TRANS = len(_BODY_1DOF_TRANS_IDXS)    # 6
_BODY_3DOF_FLAT_IDXS = _BODY_3DOF_ROT_IDXS.flatten()

# Body index device cache
_BODY_IDX_CACHE = {}

# Pre-create indices on CUDA (avoids CPU->GPU transfer during torch.compile tracing)
if torch.cuda.is_available():
    _BODY_IDX_CACHE['cuda:0'] = {
        '3dof_flat': _BODY_3DOF_FLAT_IDXS.cuda(),
        '1dof_rot': _BODY_1DOF_ROT_IDXS.cuda(),
        '1dof_trans': _BODY_1DOF_TRANS_IDXS.cuda(),
    }

def _get_body_cached_idx(device):
    """Get device cache for body indices."""
    device_key = str(device)
    if device_key not in _BODY_IDX_CACHE:
        _BODY_IDX_CACHE[device_key] = {
            '3dof_flat': _BODY_3DOF_FLAT_IDXS.to(device),
            '1dof_rot': _BODY_1DOF_ROT_IDXS.to(device),
            '1dof_trans': _BODY_1DOF_TRANS_IDXS.to(device),
        }
    return _BODY_IDX_CACHE[device_key]


def compact_cont_to_model_params_body_fast(body_pose_cont):
    """
    Fast version: no timing overhead, uses pre-cached indices.
    Used for production inference.
    """
    num_3dof_angles = _BODY_NUM_3DOF
    num_1dof_angles = _BODY_NUM_1DOF

    # Get subsets (slicing does not need cached indices)
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]

    # 3DoFs: unflatten + 6D→euler
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_params_3dofs = _batchXYZfrom6D_jit(body_cont_3dofs).flatten(-2, -1)

    # 1DoFs: unflatten + atan2
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))
    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])

    # Assemble: 133-dim output
    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133, device=body_pose_cont.device, dtype=body_pose_cont.dtype)
    idx = _get_body_cached_idx(body_pose_cont.device)
    body_pose_params[..., idx['3dof_flat']] = body_params_3dofs
    body_pose_params[..., idx['1dof_rot']] = body_params_1dofs
    body_pose_params[..., idx['1dof_trans']] = body_cont_trans

    return body_pose_params


_compact_cont_body_timing = {
    'slice': 0, 'unflatten_3dof': 0, 'batchXYZfrom6D': 0,
    'unflatten_1dof': 0, 'atan2': 0, 'assemble': 0, 'calls': 0
}

def compact_cont_to_model_params_body(body_pose_cont):
    import time as _time
    def _sync_time():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _time.perf_counter()

    global _compact_cont_body_timing
    t0 = _sync_time()

    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]

    t1 = _sync_time()
    _compact_cont_body_timing['slice'] += (t1 - t0) * 1000

    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    t2 = _sync_time()
    _compact_cont_body_timing['unflatten_3dof'] += (t2 - t1) * 1000

    body_params_3dofs = batchXYZfrom6D(body_cont_3dofs).flatten(-2, -1)
    t3 = _sync_time()
    _compact_cont_body_timing['batchXYZfrom6D'] += (t3 - t2) * 1000

    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    t4 = _sync_time()
    _compact_cont_body_timing['unflatten_1dof'] += (t4 - t3) * 1000

    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])
    t5 = _sync_time()
    _compact_cont_body_timing['atan2'] += (t5 - t4) * 1000

    ## Nothing to do for trans
    body_params_trans = body_cont_trans
    # Put them together
    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133).to(body_pose_cont)
    body_pose_params[..., all_param_3dof_rot_idxs.flatten()] = body_params_3dofs
    body_pose_params[..., all_param_1dof_rot_idxs] = body_params_1dofs
    body_pose_params[..., all_param_1dof_trans_idxs] = body_params_trans

    t6 = _sync_time()
    _compact_cont_body_timing['assemble'] += (t6 - t5) * 1000
    _compact_cont_body_timing['calls'] += 1

    return body_pose_params


def compact_model_params_to_cont_body(body_pose_params):
    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_params.shape[-1] == (
        num_3dof_angles + num_1dof_angles + num_1dof_trans
    )
    # Take out params
    body_params_3dofs = body_pose_params[..., all_param_3dof_rot_idxs.flatten()]
    body_params_1dofs = body_pose_params[..., all_param_1dof_rot_idxs]
    body_params_trans = body_pose_params[..., all_param_1dof_trans_idxs]
    # params to cont
    body_cont_3dofs = batch6DFromXYZ(body_params_3dofs.unflatten(-1, (-1, 3))).flatten(
        -2, -1
    )
    body_cont_1dofs = torch.stack(
        [body_params_1dofs.sin(), body_params_1dofs.cos()], dim=-1
    ).flatten(-2, -1)
    body_cont_trans = body_params_trans
    # Put them together
    body_pose_cont = torch.cat(
        [body_cont_3dofs, body_cont_1dofs, body_cont_trans], dim=-1
    )
    return body_pose_cont


# fmt: off
mhr_param_hand_idxs = [62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
mhr_cont_hand_idxs = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237]
mhr_param_hand_mask = torch.zeros(133).bool(); mhr_param_hand_mask[mhr_param_hand_idxs] = True
mhr_cont_hand_mask = torch.zeros(260).bool(); mhr_cont_hand_mask[mhr_cont_hand_idxs] = True
# fmt: on
