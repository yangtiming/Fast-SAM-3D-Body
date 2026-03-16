import inspect
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# chumpy (used by smplx to load .pkl models) has two compatibility issues with
# modern Python/NumPy that must be patched before any smplx import:
#   1. inspect.getargspec removed in Python 3.11+
#   2. np.bool/int/float/etc aliases removed in NumPy 1.24+
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _alias, _target in (
    ("bool", np.bool_), ("int", np.int_), ("float", np.float64),
    ("complex", np.complex128), ("object", np.object_), ("str", np.str_),
    ("unicode", np.str_),
):
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)


def _load_ply_faces(ply_path):
    """Read triangle face indices from a binary little-endian PLY file."""
    with open(ply_path, "rb") as f:
        num_vertices = num_faces = 0
        vertex_stride = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                num_faces = int(line.split()[-1])
            elif line.startswith("property") and num_faces == 0:
                # accumulate per-vertex property sizes
                parts = line.split()
                type_sizes = {"float": 4, "double": 8, "int": 4, "uint": 4,
                              "short": 2, "ushort": 2, "uchar": 1, "char": 1}
                vertex_stride += type_sizes.get(parts[1], 4)
            elif line == "end_header":
                break
        f.read(num_vertices * vertex_stride)
        faces = []
        for _ in range(num_faces):
            n = struct.unpack("B", f.read(1))[0]
            faces.append(struct.unpack(f"{n}i", f.read(n * 4)))
    return np.array(faces, dtype=np.int64)


GO_DIM = 6
POSE_DIM = 63
BETAS_DIM = 10
OUTPUT_DIM = GO_DIM + POSE_DIM + BETAS_DIM


def rot6d_to_rotmat(r6d):
    v1 = r6d[..., :3]
    v2 = r6d[..., 3:6]
    a1 = F.normalize(v1, dim=-1)
    a2 = F.normalize(v2 - (a1 * v2).sum(-1, keepdim=True) * a1, dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)


def rot6d_to_aa(r6d):
    mat = rot6d_to_rotmat(r6d)
    trace = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    rx = mat[:, 2, 1] - mat[:, 1, 2]
    ry = mat[:, 0, 2] - mat[:, 2, 0]
    rz = mat[:, 1, 0] - mat[:, 0, 1]
    skew = torch.stack([rx, ry, rz], dim=-1)
    safe_sin = torch.sin(angle).clamp(min=1e-7)
    aa = skew / (2.0 * safe_sin.unsqueeze(-1)) * angle.unsqueeze(-1)
    small = (angle < 1e-4).unsqueeze(-1)
    return torch.where(small, skew / 2.0, aa)


def split_output(y):
    return y[:, :GO_DIM], y[:, GO_DIM:GO_DIM + POSE_DIM], y[:, GO_DIM + POSE_DIM:]


class MinimalMultiViewNet(nn.Module):
    def __init__(self, input_dim=4500, hidden_dim=512, feat_dim=256, output_dim=OUTPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(),
        )
        self.confidence = nn.Linear(feat_dim, 1)
        self.decoder = nn.Linear(feat_dim, output_dim)
        self.feat_dim = feat_dim

    def forward(self, views, view_mask=None):
        batch_size, num_views, dim = views.shape
        feats_flat = self.encoder(views.reshape(batch_size * num_views, dim))
        feats = feats_flat.reshape(batch_size, num_views, self.feat_dim)
        conf = self.confidence(feats_flat).reshape(batch_size, num_views)
        if view_mask is not None:
            conf = conf.masked_fill(~view_mask, float("-inf"))
        weights = torch.softmax(conf, dim=1)
        fused = (feats * weights.unsqueeze(-1)).sum(dim=1)
        params = self.decoder(fused)
        return params, weights

    def forward_single_view(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)


class SmootherMLP(nn.Module):
    def __init__(self, window_size=1, joint_dim=72, hidden_dims=(512, 256)):
        super().__init__()
        layers = []
        in_dim = window_size * joint_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, joint_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:
            x = x.reshape(x.shape[0], -1)
        elif x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


class MultiViewFusionRunner:
    def __init__(
        self,
        smpl_model_path,
        model_dir,
        mapping_path,
        mhr_mesh_path=None,
        device=None,
        smoother_dir=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MinimalMultiViewNet(input_dim=1500 * 3).to(self.device)

        model_dir = Path(model_dir)
        mapping_path = Path(mapping_path)
        model_path = model_dir / "best_model.pth"
        sample_idx_path = model_dir / "sample_idx.npy"

        if not model_dir.is_dir():
            raise RuntimeError(f"Multi-view model directory not found: {model_dir}")
        if not model_path.is_file():
            raise RuntimeError(f"Multi-view model not found: {model_path}")
        if not sample_idx_path.is_file():
            raise RuntimeError(f"Sample index file not found: {sample_idx_path}")
        if not mapping_path.is_file():
            raise RuntimeError(f"Mapping file not found: {mapping_path}")

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        mapping = np.load(mapping_path)
        self._baryc = np.asarray(mapping["baryc_coords"], dtype=np.float32)  # (6890, 3)
        if "mhr_vert_ids" in mapping:
            self._mhr_vert_ids = np.asarray(mapping["mhr_vert_ids"], dtype=np.int64)  # (6890, 3)
        elif "triangle_ids" in mapping:
            if mhr_mesh_path is None:
                raise RuntimeError(
                    "Mapping file uses triangle_ids format: mhr_mesh_path (PLY) is required to expand it"
                )
            faces = _load_ply_faces(mhr_mesh_path)
            triangle_ids = np.asarray(mapping["triangle_ids"], dtype=np.int64)  # (6890,)
            self._mhr_vert_ids = faces[triangle_ids].astype(np.int64)           # (6890, 3)
        else:
            raise RuntimeError(
                f"Mapping file must contain mhr_vert_ids or triangle_ids: {mapping_path}"
            )
        self._sample_idx = np.load(sample_idx_path).astype(np.int64)

        import smplx

        self._smpl = smplx.SMPL(model_path=str(smpl_model_path), gender="neutral").to(self.device)
        self._smpl.eval()
        for p in self._smpl.parameters():
            p.requires_grad_(False)

        self._smoother = None
        self._smoother_buffer = None
        self._smoother_window_size = None
        if smoother_dir is not None:
            self._load_smoother(smoother_dir)

    def _load_smoother(self, smoother_dir):
        import json

        smoother_dir = Path(smoother_dir)
        ckpt_path = smoother_dir / "smoother_best.pth"
        config_path = smoother_dir / "smoother_config.json"
        if not smoother_dir.is_dir():
            raise RuntimeError(f"Smoother directory not found: {smoother_dir}")
        if not ckpt_path.is_file():
            raise RuntimeError(f"Smoother checkpoint not found: {ckpt_path}")
        if not config_path.is_file():
            raise RuntimeError(f"Smoother config not found: {config_path}")

        cfg = json.loads(config_path.read_text())
        window_size = int(cfg["window_size"])
        smoother = SmootherMLP(
            window_size=window_size,
            joint_dim=int(cfg["joint_dim"]),
            hidden_dims=tuple(int(v) for v in cfg["hidden_dims"]),
        ).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        smoother.load_state_dict(state_dict)
        smoother.eval()
        self._smoother = smoother
        self._smoother_window_size = window_size
        self._smoother_buffer = []

    def _preprocess_view(self, pred_vertices, pred_cam_t):
        verts = np.asarray(pred_vertices, dtype=np.float32).copy()
        cam_t = np.asarray(pred_cam_t, dtype=np.float32).copy()
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0
        cam_t[1] *= -1.0
        cam_t[2] *= -1.0
        mhr_world = verts + cam_t[None, :]
        face_verts = mhr_world[self._mhr_vert_ids]
        smpl_verts = (face_verts * self._baryc[:, :, None]).sum(axis=1)
        smpl_sub = smpl_verts[self._sample_idx]
        smpl_sub -= smpl_sub.mean(axis=0, keepdims=True)
        return smpl_sub.reshape(-1).astype(np.float32)

    def infer(self, views):
        if len(views) < 1:
            raise RuntimeError(f"infer() requires at least 1 view, got {len(views)}")

        feats_list = []
        mask_list = []
        for v_t in views:
            if v_t is None:
                feats_list.append(np.zeros(1500 * 3, dtype=np.float32))
                mask_list.append(False)
            else:
                v, t = v_t
                feats_list.append(self._preprocess_view(v, t))
                mask_list.append(True)

        feats = np.stack(feats_list, axis=0)
        inputs = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        mask = torch.tensor([mask_list], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            params, weights = self.model(inputs, mask)
            go6d, body_pose, betas = split_output(params)
            go_aa = rot6d_to_aa(go6d)

            body_pose_np = body_pose[0].cpu().numpy().astype(np.float32)
            betas_np = betas[0].cpu().numpy().astype(np.float32)
            weights_np = weights[0].cpu().numpy().astype(np.float32)

            bp_full = np.zeros(69, dtype=np.float32)
            bp_full[:63] = body_pose_np
            smpl_out = self._smpl(
                global_orient=torch.zeros(1, 3, device=self.device),
                body_pose=torch.from_numpy(bp_full).unsqueeze(0).to(self.device),
                betas=torch.from_numpy(betas_np).unsqueeze(0).to(self.device),
            )

        joints = smpl_out.joints[0, :24].cpu().numpy().astype(np.float32)
        joints -= joints[0:1]
        if self._smoother is not None:
            self._smoother_buffer.append(joints.flatten())
            if len(self._smoother_buffer) > self._smoother_window_size:
                self._smoother_buffer.pop(0)
            pad = [self._smoother_buffer[0]] * (self._smoother_window_size - len(self._smoother_buffer))
            window = np.stack(pad + self._smoother_buffer, axis=0)
            x = torch.from_numpy(window.reshape(1, -1).astype(np.float32)).to(self.device)
            with torch.no_grad():
                joints = self._smoother(x)[0].cpu().numpy().reshape(24, 3).astype(np.float32)
        return body_pose_np.reshape(21, 3), joints, betas_np, weights_np
