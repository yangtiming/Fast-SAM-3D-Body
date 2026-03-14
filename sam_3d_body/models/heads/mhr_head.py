# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time
import warnings
from typing import Optional

import roma
import torch
import torch.nn as nn

# MHRHead detailed timing control
_MHR_DETAIL_TIMING = os.environ.get("INTERM_TIMING", "0") == "1"
_MHR_TIMING_WARMUP = 3
_MHR_TIMING_COUNT = [0]  # Use list to simulate mutable reference


def _sync_time():
    """Synchronize CUDA and return the current time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

from ..modules import rot6d_to_rotmat
from ..modules.mhr_utils import (
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_body_fast,
    compact_cont_to_model_params_hand,
    compact_cont_to_model_params_hand_fast,
    compact_model_params_to_cont_body,
    euler_to_rotmat_xyz,
    mhr_param_hand_idxs,
    rotmat_to_euler_xyz,
    rotmat_to_euler_ZYX,
)

from ..modules.transformer import FFN

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR

        MOMENTUM_ENABLED = True
        warnings.warn("Momentum is enabled")
    else:
        warnings.warn("Momentum is not enabled")
        raise ImportError
except:
    MOMENTUM_ENABLED = False
    warnings.warn("Momentum is not enabled")


@torch.jit.script
def _fast_quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Fast unit quaternion to rotation matrix conversion.
    q: (..., 4) in (x, y, z, w) format (roma convention).
    Returns: (..., 3, 3)
    """
    # roma uses (x, y, z, w) format
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Precompute squares and products
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Build rotation matrix
    r00 = 1.0 - 2.0 * (y2 + z2)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (x2 + z2)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (x2 + y2)

    # Stack to (..., 3, 3)
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class MHRHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        mhr_model_path: str = "",
        extra_joint_regressor: str = "",
        ffn_zero_bias: bool = True,
        mlp_channel_div_factor: int = 8,
        enable_hand_model=False,
    ):
        super().__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.enable_hand_model = enable_hand_model
        # apply_correctives=False gives ~10x speedup (18ms -> 1.9ms)
        # Can be disabled via MHR_NO_CORRECTIVES=1 environment variable
        import os
        self.apply_correctives = os.environ.get('MHR_NO_CORRECTIVES', '0') != '1'

        # CUDA Graph toggle (off by default - copy overhead offsets the speedup)
        # Enable via MHR_USE_CUDA_GRAPH=1 environment variable
        self.use_cuda_graph = os.environ.get('MHR_USE_CUDA_GRAPH', '0') == '1'
        # Support multiple batch_size CUDA Graphs (body=1, hand=2)
        self._cuda_graphs = {}  # {batch_size: graph}
        self._cuda_graph_captured = {}  # {batch_size: bool}
        self._cuda_graph_warmup_count = 0
        self._cuda_graph_warmup_needed = 3  # number of warmup iterations
        self._supported_batch_sizes = [1, 2]  # supported batch sizes

        self.body_cont_dim = 260
        self.npose = (
            6  # Global Rotation
            + self.body_cont_dim  # then body
            + self.num_shape_comps
            + self.num_scale_comps
            + self.num_hand_comps * 2
            + self.num_face_comps
        )

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)

        # MHR Parameters
        self.model_data_dir = mhr_model_path
        self.num_hand_scale_comps = self.num_scale_comps - 18
        self.num_hand_pose_comps = self.num_hand_comps

        # Buffers to be filled in by model state dict
        self.joint_rotation = nn.Parameter(torch.zeros(127, 3, 3), requires_grad=False)
        self.scale_mean = nn.Parameter(torch.zeros(68), requires_grad=False)
        self.scale_comps = nn.Parameter(torch.zeros(28, 68), requires_grad=False)
        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
        self._faces_numpy_cache = None  # Cache numpy version to avoid repeated conversion
        self.hand_pose_mean = nn.Parameter(torch.zeros(54), requires_grad=False)
        self.hand_pose_comps = nn.Parameter(torch.eye(54), requires_grad=False)
        self.hand_joint_idxs_left = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.hand_joint_idxs_right = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.keypoint_mapping = nn.Parameter(
            torch.zeros(308, 18439 + 127), requires_grad=False
        )
        # Some special buffers for the hand-version
        self.right_wrist_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.root_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.local_to_world_wrist = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
        self.nonhand_param_idxs = nn.Parameter(
            torch.zeros(145).long(), requires_grad=False
        )

        # Load MHR itself
        if MOMENTUM_ENABLED:
            self.mhr = MHR.from_files(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                lod=1,
            )
        else:
            self.mhr = torch.jit.load(
                mhr_model_path,
                map_location=("cuda" if torch.cuda.is_available() else "cpu"),
            )

        for param in self.mhr.parameters():
            param.requires_grad = False

        # CUDA Graph static buffers (lazy init) - support multiple batch sizes
        self._static_buffers_initialized = {}  # {batch_size: bool}
        self._static_buffers = {}  # {batch_size: {'shape': ..., 'model': ..., ...}}

        # torch.compile support
        self._compiled = False
        self._compiled_mhr_forward = None
        self._compiled_head_forward = None

    def _init_cuda_graph_buffers(self, device, batch_size=1):
        """Initialize static buffers required for CUDA Graph (per batch_size)."""
        if self._static_buffers_initialized.get(batch_size, False):
            return

        # Fixed-size input buffers
        self._static_buffers[batch_size] = {
            'shape_params': torch.zeros(batch_size, 45, device=device, dtype=torch.float32),
            'model_params': torch.zeros(batch_size, 204, device=device, dtype=torch.float32),
            'expr_params': torch.zeros(batch_size, 72, device=device, dtype=torch.float32),
            # Output buffers (sized according to MHR output)
            # skinned_verts: (B, 18439, 3), skel_state: (B, 127, 8)
            'output_verts': torch.zeros(batch_size, 18439, 3, device=device, dtype=torch.float32),
            'output_skel': torch.zeros(batch_size, 127, 8, device=device, dtype=torch.float32),
        }

        self._static_buffers_initialized[batch_size] = True

    def _get_faces_numpy(self):
        """Get numpy version of faces, cached to avoid repeated GPU->CPU transfer."""
        if self._faces_numpy_cache is None:
            self._faces_numpy_cache = self.faces.cpu().numpy()
        return self._faces_numpy_cache

    def _capture_cuda_graph(self, batch_size=1):
        """Capture MHR forward pass as CUDA Graph (per batch_size)."""
        if self._cuda_graph_captured.get(batch_size, False):
            return

        try:
            buffers = self._static_buffers[batch_size]
            device = buffers['shape_params'].device

            # Create CUDA Graph
            graph = torch.cuda.CUDAGraph()

            # Warmup stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.mhr(
                            buffers['shape_params'],
                            buffers['model_params'],
                            buffers['expr_params'],
                            self.apply_correctives
                        )
            torch.cuda.current_stream().wait_stream(s)

            # Capture graph
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    buffers['output_verts'], buffers['output_skel'] = self.mhr(
                        buffers['shape_params'],
                        buffers['model_params'],
                        buffers['expr_params'],
                        self.apply_correctives
                    )

            self._cuda_graphs[batch_size] = graph
            self._cuda_graph_captured[batch_size] = True
            print(f"[MHRHead] CUDA Graph captured for batch_size={batch_size} (apply_correctives={self.apply_correctives})")
        except Exception as e:
            print(f"[MHRHead] CUDA Graph capture failed for batch_size={batch_size}: {e}")
            print(f"[MHRHead] Falling back to regular execution for this batch_size")
            self._cuda_graph_captured[batch_size] = False
            # Clean up failed graph object and CUDA state
            self._cuda_graphs.pop(batch_size, None)
            try:
                torch.cuda.synchronize()
            except:
                pass

    def warmup_cuda_graph(self, batch_size=1):
        """Call before inference to warm up and capture CUDA Graph (per batch_size)."""
        if not self.use_cuda_graph:
            return

        device = next(self.parameters()).device
        self._init_cuda_graph_buffers(device, batch_size)

        buffers = self._static_buffers[batch_size]

        # Warmup
        for _ in range(self._cuda_graph_warmup_needed):
            with torch.no_grad():
                _ = self.mhr(
                    buffers['shape_params'],
                    buffers['model_params'],
                    buffers['expr_params'],
                    self.apply_correctives
                )

        # Capture
        self._capture_cuda_graph(batch_size)

    def _mhr_with_cuda_graph(self, shape_params, model_params, expr_params):
        """Call MHR using CUDA Graph (auto-selects graph for the given batch_size)."""
        batch_size = shape_params.shape[0]

        # Check if this batch_size is supported
        if batch_size not in self._supported_batch_sizes:
            return self.mhr(shape_params, model_params, expr_params, self.apply_correctives)

        # Initialize static buffers (if needed)
        if not self._static_buffers_initialized.get(batch_size, False):
            self._init_cuda_graph_buffers(shape_params.device, batch_size)

        # If graph not yet captured, capture it (includes warmup)
        if not self._cuda_graph_captured.get(batch_size, False):
            self.warmup_cuda_graph(batch_size)

        # If capture failed, fall back to regular execution
        if not self._cuda_graph_captured.get(batch_size, False):
            return self.mhr(shape_params, model_params, expr_params, self.apply_correctives)

        buffers = self._static_buffers[batch_size]

        # Copy inputs to static buffers
        buffers['shape_params'].copy_(shape_params)
        buffers['model_params'].copy_(model_params)
        buffers['expr_params'].copy_(expr_params)

        # Replay graph
        self._cuda_graphs[batch_size].replay()

        # Return copies of outputs (to avoid buffer overwrite)
        return buffers['output_verts'].clone(), buffers['output_skel'].clone()

    def get_zero_pose_init(self, factor=1.0):
        # Initialize pose token with zero-initialized learnable params
        # Note: bias/initial value should be zero-pose in cont, not all-zeros
        weights = torch.zeros(1, self.npose)
        weights[:, : 6 + self.body_cont_dim] = torch.cat(
            [
                torch.FloatTensor([1, 0, 0, 0, 1, 0]),
                compact_model_params_to_cont_body(torch.zeros(1, 133)).squeeze()
                * factor,
            ],
            dim=0,
        )
        return weights

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136
        batch_size = hand_pose_params.shape[0]

        # Optimization: batch-process left and right hands together
        left_hand, right_hand = torch.split(
            hand_pose_params, [self.num_hand_pose_comps, self.num_hand_pose_comps], dim=1
        )
        # After cat: rows 0~B-1 are left hand, rows B~2B-1 are right hand
        both_hands = torch.cat([left_hand, right_hand], dim=0)  # (B*2, 54)

        # Batch PCA decode: (B*2, 54) @ (54, 54) -> (B*2, 54)
        # Optimization: use mm instead of einsum (einsum "da,ab->db" = matmul)
        both_pca_decoded = self.hand_pose_mean + both_hands.mm(self.hand_pose_comps)

        # Batch 6D->euler: (B*2, 54) -> (B*2, 27)
        both_model_params = compact_cont_to_model_params_hand_fast(both_pca_decoded)

        # Split back to left/right hands: (B*2, 27) -> (B, 27), (B, 27)
        left_model_params = both_model_params[:batch_size]
        right_model_params = both_model_params[batch_size:]

        # Drop it in
        full_pose_params[:, self.hand_joint_idxs_left] = left_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_model_params

        return full_pose_params  # B x 207

    def apply_compile(self, mode: str = "reduce-overhead"):
        """
        Apply torch.compile to MHRHead core computation for faster inference.

        Args:
            mode: torch.compile mode, one of "default", "reduce-overhead", "max-autotune"
        """
        if self._compiled:
            print(f"[MHRHead] Already compiled, skipping")
            return

        print(f"[MHRHead] Applying torch.compile with mode='{mode}'")

        # Compile core computation (excludes timing and CUDA Graph logic)
        # Use dynamic=True to support different batch sizes (single/multi-person)
        self._compiled_mhr_forward = torch.compile(
            self._mhr_forward_core,
            mode=mode,
            fullgraph=False,  # Allow graph breaks for flexibility
            dynamic=True,
        )

        # Compile full head forward core (proj + rot_convert + pose_convert + mhr_forward + postprocess)
        self._compiled_head_forward = torch.compile(
            self._head_forward_core,
            mode=mode,
            fullgraph=False,
            dynamic=True,
        )

        self._compiled = True
        print(f"[MHRHead] torch.compile applied successfully (mhr_forward + head_forward)")

    def _head_forward_core(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Core computation of MHRHead.forward (no timing, used for torch.compile).
        Includes: proj + rot_convert + pose_convert + mhr_forward_core + postprocess

        Returns: (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
                  pred_shape, pred_scale, pred_hand, pred_face,
                  verts, j3d, jcoords, mhr_model_params, joint_global_rots)
        """
        batch_size = x.shape[0]

        # === proj stage ===
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        # === rot_convert stage ===
        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = rotmat_to_euler_ZYX(global_rot_rotmat)
        global_trans = torch.zeros_like(global_rot_euler)

        # === pose_convert stage ===
        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim
        pred_pose_euler = compact_cont_to_model_params_body_fast(pred_pose_cont)
        pred_pose_euler[:, mhr_param_hand_idxs] = 0
        pred_pose_euler[:, -3:] = 0

        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0

        # === mhr_forward_core stage ===
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = self._mhr_forward_core(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            return_keypoints=True,
        )

        # === postprocess stage ===
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        if verts is not None:
            verts = verts.clone()
            verts[..., [1, 2]] *= -1  # Camera system difference
        j3d = j3d.clone()
        j3d[..., [1, 2]] *= -1
        if jcoords is not None:
            jcoords = jcoords.clone()
            jcoords[..., [1, 2]] *= -1

        return (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
                pred_shape, pred_scale, pred_hand, pred_face,
                verts, j3d, jcoords, mhr_model_params, joint_global_rots)

    def _head_forward_core_slim(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        """
        Slim core computation of MHRHead.forward (for intermediate layers).
        Skips computations not needed by intermediate layers: joint_global_rots, verts/jcoords postprocess.

        Returns: (global_rot_6d, pred_pose_cont, j3d)
        - Only returns outputs needed by intermediate layers, saving memory bandwidth.
        """
        # === proj stage ===
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        # === rot_convert stage ===
        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = rotmat_to_euler_ZYX(global_rot_rotmat)
        global_trans = torch.zeros_like(global_rot_euler)

        # === pose_convert stage ===
        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim
        pred_pose_euler = compact_cont_to_model_params_body_fast(pred_pose_cont)
        pred_pose_euler[:, mhr_param_hand_idxs] = 0
        pred_pose_euler[:, -3:] = 0

        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0

        # === mhr_forward_core stage (slim_mode=True) ===
        verts, j3d, jcoords, mhr_model_params, _ = self._mhr_forward_core(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            return_keypoints=True,
            slim_mode=True,  # Skip joint_rots computation
        )

        # === Slim postprocess ===
        # Only process j3d (for projection and keypoint update)
        # Skip verts/jcoords clone and coordinate flipping
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        j3d = j3d.clone()
        j3d[..., [1, 2]] *= -1

        return (global_rot_6d, pred_pose_cont, j3d)

    def _mhr_forward_core(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params,
        return_keypoints=True,
        slim_mode=False,
    ):
        """
        Core computation of MHR forward pass (no timing, used for torch.compile).
        Returns: (skinned_verts, keypoints_pred, joint_coords, model_params, joint_rots)

        Args:
            slim_mode: If True, skip computations not needed by intermediate layers (joint_rots).
        """
        # === prep_params stage ===
        if self.enable_hand_model:
            # Transfer wrist-centric predictions to the body.
            global_rot_ori = global_rot.clone()
            global_trans_ori = global_trans.clone()
            global_rot = rotmat_to_euler_xyz(
                euler_to_rotmat_xyz(global_rot_ori) @ self.local_to_world_wrist
            )
            global_trans = (
                -(
                    euler_to_rotmat_xyz(global_rot)
                    @ (self.right_wrist_coords - self.root_coords)
                    + self.root_coords
                )
                + global_trans_ori
            )

        body_pose_params = body_pose_params[..., :130]

        # Convert scale
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps

        # Assemble pose params
        full_pose_params = torch.cat(
            [global_trans * 10, global_rot, body_pose_params], dim=1
        )

        # Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(
                full_pose_params, hand_pose_params
            )

        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.enable_hand_model:
            model_params[:, self.nonhand_param_idxs] = 0

        # === mhr_call stage ===
        apply_correctives = getattr(self, 'apply_correctives', True)
        curr_skinned_verts, curr_skel_state = self.mhr(
            shape_params, model_params, expr_params, apply_correctives
        )

        # === post_skinning stage ===
        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        curr_skinned_verts = curr_skinned_verts * 0.01
        curr_joint_coords = curr_joint_coords * 0.01

        # slim_mode: skip joint_rots computation (not needed by intermediate layers)
        if slim_mode:
            curr_joint_rots = None
        else:
            curr_joint_rots = _fast_quat_to_rotmat(curr_joint_quats)

        # === keypoint_extraction stage ===
        model_keypoints_pred = None
        if return_keypoints:
            model_vert_joints = torch.cat(
                [curr_skinned_verts, curr_joint_coords], dim=1
            )
            model_keypoints_pred = torch.einsum(
                'kv,bvc->bkc', self.keypoint_mapping, model_vert_joints
            )
            if self.enable_hand_model:
                model_keypoints_pred[:, :21] = 0
                model_keypoints_pred[:, 42:] = 0

        return curr_skinned_verts, model_keypoints_pred, curr_joint_coords, model_params, curr_joint_rots

    def mhr_forward(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params=None,
        return_keypoints=False,
        do_pcblend=True,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
        scale_offsets=None,
        vertex_offsets=None,
        _do_timing=False,
    ):
        # ========== Use compiled version (if available) ==========
        # Conditions: compiled, no timing, no scale_offsets/vertex_offsets, no CUDA Graph
        use_compiled = (
            self._compiled
            and self._compiled_mhr_forward is not None
            and not _do_timing
            and scale_offsets is None
            and vertex_offsets is None
            and not getattr(self, 'use_cuda_graph', False)
        )

        if use_compiled:
            # Call compiled core function
            curr_skinned_verts, model_keypoints_pred, curr_joint_coords, model_params, curr_joint_rots = \
                self._compiled_mhr_forward(
                    global_trans, global_rot, body_pose_params, hand_pose_params,
                    scale_params, shape_params, expr_params, return_keypoints
                )

            # Build return values
            to_return = [curr_skinned_verts]
            if return_keypoints and model_keypoints_pred is not None:
                to_return.append(model_keypoints_pred)
            if return_joint_coords:
                to_return.append(curr_joint_coords)
            if return_model_params:
                to_return.append(model_params)
            if return_joint_rotations:
                to_return.append(curr_joint_rots)

            if len(to_return) == 1:
                return to_return[0]
            else:
                return tuple(to_return)
        # ========== End compiled version ==========

        if _do_timing:
            t0 = _sync_time()

        if self.enable_hand_model:
            # Transfer wrist-centric predictions to the body.
            global_rot_ori = global_rot.clone()
            global_trans_ori = global_trans.clone()
            global_rot = rotmat_to_euler_xyz(
                euler_to_rotmat_xyz(global_rot_ori) @ self.local_to_world_wrist
            )
            global_trans = (
                -(
                    euler_to_rotmat_xyz(global_rot)
                    @ (self.right_wrist_coords - self.root_coords)
                    + self.root_coords
                )
                + global_trans_ori
            )

        body_pose_params = body_pose_params[..., :130]

        # Convert from scale and shape params to actual scales and vertices
        ## Add singleton batches in case...
        if len(scale_params.shape) == 1:
            scale_params = scale_params[None]
        if len(shape_params.shape) == 1:
            shape_params = shape_params[None]
        ## Convert scale...
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
        if scale_offsets is not None:
            scales = scales + scale_offsets

        # Now, figure out the pose.
        ## 10 here is because it's more stable to optimize global translation in meters.
        full_pose_params = torch.cat(
            [global_trans * 10, global_rot, body_pose_params], dim=1
        )  # B x 127

        ## Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(
                full_pose_params, hand_pose_params
            )

        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.enable_hand_model:
            # Zero out non-hand parameters
            model_params[:, self.nonhand_param_idxs] = 0

        if _do_timing:
            t_prep_params = _sync_time()

        # apply_correctives=False gives ~10x speedup (18ms -> 1.9ms)
        # but sacrifices some pose correction accuracy.
        # CUDA Graph provides additional 4-6x speedup (by reducing kernel launch overhead)
        batch_size = shape_params.shape[0]
        supported_sizes = getattr(self, '_supported_batch_sizes', [1])
        if getattr(self, 'use_cuda_graph', False) and batch_size in supported_sizes:
            # CUDA Graph supports batch_size in [1, 2] (body=1, hand=2)
            curr_skinned_verts, curr_skel_state = self._mhr_with_cuda_graph(
                shape_params, model_params, expr_params
            )
        else:
            apply_correctives = getattr(self, 'apply_correctives', True)
            curr_skinned_verts, curr_skel_state = self.mhr(
                shape_params, model_params, expr_params, apply_correctives
            )

        if _do_timing:
            t_mhr_call = _sync_time()

        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        # Use multiplication instead of division (faster)
        curr_skinned_verts = curr_skinned_verts * 0.01
        curr_joint_coords = curr_joint_coords * 0.01
        # Use fast quaternion -> rotmat (faster than roma)
        curr_joint_rots = _fast_quat_to_rotmat(curr_joint_quats)

        if _do_timing:
            t_post_skinning = _sync_time()

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            # Use einsum instead of permute/flatten/reshape (faster)
            model_vert_joints = torch.cat(
                [curr_skinned_verts, curr_joint_coords], dim=1
            )  # B x (num_verts + 127) x 3
            # keypoint_mapping: (308, 18566), model_vert_joints: (B, 18566, 3)
            # Output: (B, 308, 3)
            model_keypoints_pred = torch.einsum(
                'kv,bvc->bkc', self.keypoint_mapping, model_vert_joints
            )

            if self.enable_hand_model:
                # Zero out everything except for the right hand
                model_keypoints_pred[:, :21] = 0
                model_keypoints_pred[:, 42:] = 0

            to_return = to_return + [model_keypoints_pred]

        if _do_timing:
            t_keypoint_ext = _sync_time()
            print(f"        [mhr_forward] "
                  f"prep_params: {(t_prep_params-t0)*1000:.2f}ms | "
                  f"mhr_call: {(t_mhr_call-t_prep_params)*1000:.2f}ms | "
                  f"post_skinning: {(t_post_skinning-t_mhr_call)*1000:.2f}ms | "
                  f"keypoint_ext: {(t_keypoint_ext-t_post_skinning)*1000:.2f}ms")

        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rots]

        if isinstance(to_return, list) and len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        do_pcblend=True,
        slim_keypoints=False,
    ):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.npose]
        """
        # Timing control
        _MHR_TIMING_COUNT[0] += 1
        do_timing = _MHR_DETAIL_TIMING and _MHR_TIMING_COUNT[0] > _MHR_TIMING_WARMUP

        batch_size = x.shape[0]

        # ========== Use compiled version (if available) ==========
        use_compiled_head = (
            self._compiled
            and self._compiled_head_forward is not None
            and not do_timing
            and do_pcblend  # default parameter
            and not slim_keypoints  # default parameter
            and not getattr(self, 'use_cuda_graph', False)
        )

        if use_compiled_head:
            # Call compiled full head forward core function
            (global_rot_6d, global_rot_euler, pred_pose_cont, pred_pose_euler,
             pred_shape, pred_scale, pred_hand, pred_face,
             verts, j3d, jcoords, mhr_model_params, joint_global_rots) = \
                self._compiled_head_forward(x, init_estimate)

            # Assemble output dict (clone tensors to avoid cudagraph buffer reuse issues)
            output = {
                "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1).clone(),
                "pred_pose_rotmat": None,
                "global_rot": global_rot_euler.clone(),
                "body_pose": pred_pose_euler.clone(),
                "shape": pred_shape.clone(),
                "scale": pred_scale.clone(),
                "hand": pred_hand.clone(),
                "face": pred_face.clone(),
                "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3).clone(),
                "pred_vertices": verts.reshape(batch_size, -1, 3).clone() if verts is not None else None,
                "pred_joint_coords": jcoords.reshape(batch_size, -1, 3).clone() if jcoords is not None else None,
                "faces": self._get_faces_numpy(),
                "joint_global_rots": joint_global_rots.clone(),
                "mhr_model_params": mhr_model_params.clone(),
            }
            return output

        # ========== Original path (with timing or special parameters) ==========
        if do_timing:
            t0 = _sync_time()

        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        if do_timing:
            t_proj = _sync_time()

        # From pred, we want to pull out individual predictions.

        ## First, get globals
        ### Global rotation is first 6.
        count = 6
        global_rot_6d = pred[:, :count]

        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)  # B x 3 x 3
        global_rot_euler = rotmat_to_euler_ZYX(global_rot_rotmat)  # B x 3

        if do_timing:
            t_rot_convert = _sync_time()
        global_trans = torch.zeros_like(global_rot_euler)

        ## Next, get body pose.
        ### Hold onto raw, continuous version for iterative correction.
        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim

        ### Convert to eulers (and trans)
        pred_pose_euler = compact_cont_to_model_params_body_fast(pred_pose_cont)

        ### Zero-out hands (use index list instead of boolean mask for torch.compile compatibility)
        pred_pose_euler[:, mhr_param_hand_idxs] = 0
        ### Zero-out jaw
        pred_pose_euler[:, -3:] = 0

        ## Get remaining parameters
        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0
        count += self.num_face_comps

        if do_timing:
            t_pose_convert = _sync_time()

        # Run everything through mhr
        output = self.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            do_pcblend=do_pcblend,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
            _do_timing=do_timing,
        )

        if do_timing:
            t_mhr_forward = _sync_time()

        # Some existing code to get joints and fix camera system
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = output
        j3d = j3d[:, :70]  # 308 --> 70 keypoints

        if verts is not None:
            verts[..., [1, 2]] *= -1  # Camera system difference
        j3d[..., [1, 2]] *= -1  # Camera system difference
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1

        # Prep outputs
        output = {
            "pred_pose_raw": torch.cat(
                [global_rot_6d, pred_pose_cont], dim=1
            ),  # Both global rot and continuous pose
            "pred_pose_rotmat": None,  # This normally used for mhr pose param rotmat supervision.
            "global_rot": global_rot_euler,
            "body_pose": pred_pose_euler,  # Unused during training
            "shape": pred_shape,
            "scale": pred_scale,
            "hand": pred_hand,
            "face": pred_face,
            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
            "pred_vertices": (
                verts.reshape(batch_size, -1, 3) if verts is not None else None
            ),
            "pred_joint_coords": (
                jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None
            ),
            "faces": self._get_faces_numpy(),
            "joint_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
        }

        if do_timing:
            t_postprocess = _sync_time()
            total = (t_postprocess - t0) * 1000
            print(f"    [MHRHead.forward] "
                  f"proj: {(t_proj-t0)*1000:.2f}ms | "
                  f"rot_convert: {(t_rot_convert-t_proj)*1000:.2f}ms | "
                  f"pose_convert: {(t_pose_convert-t_rot_convert)*1000:.2f}ms | "
                  f"mhr_forward: {(t_mhr_forward-t_pose_convert)*1000:.2f}ms | "
                  f"postprocess: {(t_postprocess-t_mhr_forward)*1000:.2f}ms | "
                  f"TOTAL: {total:.2f}ms")

        return output
