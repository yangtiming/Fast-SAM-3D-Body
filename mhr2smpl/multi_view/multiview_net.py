"""
Minimal Multi-View MHR2SMPL Network.

Encoder frozen from 46K pretrained. Only confidence + decoder trainable.
Total trainable params: ~20K (vs ~2.5M full model).

    View 1: [4500] → FrozenEncoder(4500→512→256) → feat_1 [256] ─┐
                                                                    ├→ Softmax → weighted sum → fused [256]
    View 2: [4500] → FrozenEncoder(4500→512→256) → feat_2 [256] ─┘
                                                                          ↓
                                                              Decoder(256→76) → SMPL params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

GO_DIM = 6   # 6D continuous rotation (Zhou et al.)
POSE_DIM = 63
BETAS_DIM = 10
OUTPUT_DIM = GO_DIM + POSE_DIM + BETAS_DIM  # 79


def rot6d_to_rotmat(r6d):
    """6D → 3×3 rotation matrix via Gram-Schmidt (differentiable). r6d: [..., 6]"""
    v1 = r6d[..., :3]
    v2 = r6d[..., 3:6]
    a1 = F.normalize(v1, dim=-1)
    a2 = F.normalize(v2 - (a1 * v2).sum(-1, keepdim=True) * a1, dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)  # [..., 3, 3]  columns


def rot6d_to_aa(r6d):
    """6D continuous rotation → axis-angle (differentiable). r6d: [B, 6]"""
    mat = rot6d_to_rotmat(r6d)  # [B, 3, 3]
    trace = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # [B]
    rx = mat[:, 2, 1] - mat[:, 1, 2]
    ry = mat[:, 0, 2] - mat[:, 2, 0]
    rz = mat[:, 1, 0] - mat[:, 0, 1]
    skew = torch.stack([rx, ry, rz], dim=-1)  # [B, 3]
    safe_sin = torch.sin(angle).clamp(min=1e-7)
    aa = skew / (2.0 * safe_sin.unsqueeze(-1)) * angle.unsqueeze(-1)
    small = (angle < 1e-4).unsqueeze(-1)
    return torch.where(small, skew / 2.0, aa)


class MinimalMultiViewNet(nn.Module):
    """Minimal multi-view MHR2SMPL.

    - Encoder: pretrained, frozen (4500→512→256)
    - Confidence: single linear (256→1), ~257 params
    - Decoder: single linear (256→76), ~19.5K params
    - Total trainable: ~20K
    """
    def __init__(self, input_dim=4500, hidden_dim=512, feat_dim=256,
                 output_dim=OUTPUT_DIM):
        super().__init__()
        # Encoder (will be frozen after loading pretrained weights)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(),
        )
        # Confidence: single linear → scalar per view
        self.confidence = nn.Linear(feat_dim, 1)
        # Decoder: single linear → SMPL params
        self.decoder = nn.Linear(feat_dim, output_dim)
        self.feat_dim = feat_dim

    def forward(self, views, view_mask=None):
        """
        Args:
            views: [B, V, D]
            view_mask: [B, V] bool
        Returns:
            params: [B, 76], weights: [B, V]
        """
        B, V, D = views.shape

        # Encode (frozen)
        feats_flat = self.encoder(views.reshape(B * V, D))  # [B*V, feat_dim]
        feats = feats_flat.reshape(B, V, self.feat_dim)      # [B, V, feat_dim]

        # Confidence scores
        conf = self.confidence(feats_flat).reshape(B, V)      # [B, V]
        if view_mask is not None:
            conf = conf.masked_fill(~view_mask, float('-inf'))
        weights = torch.softmax(conf, dim=1)                   # [B, V]

        # Weighted pooling
        fused = (feats * weights.unsqueeze(-1)).sum(dim=1)     # [B, feat_dim]

        # Decode
        params = self.decoder(fused)                            # [B, output_dim]
        return params, weights

    def forward_single_view(self, x):
        """Single-view fallback. x: [B, D]"""
        feat = self.encoder(x)
        return self.decoder(feat)

    @staticmethod
    def _filter_init_kwargs(kwargs):
        valid = {'input_dim', 'hidden_dim', 'feat_dim', 'output_dim'}
        return {k: v for k, v in kwargs.items() if k in valid}

    @staticmethod
    def from_scratch(device='cpu', **kwargs):
        """Random initialization — all parameters trainable."""
        model = MinimalMultiViewNet(**MinimalMultiViewNet._filter_init_kwargs(kwargs))
        # Initialize go6d decoder bias to identity rotation
        with torch.no_grad():
            model.decoder.bias.data[:6] = torch.tensor([1., 0., 0., 0., 1., 0.])
        trainable = sum(p.numel() for p in model.parameters())
        print(f"  From scratch: {trainable:,} params (all trainable)")
        return model

    @staticmethod
    def from_pretrained(checkpoint_path, device='cpu', freeze_encoder=True, **kwargs):
        """Load encoder + decoder from pretrained single-view model."""
        sv_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = MinimalMultiViewNet(**MinimalMultiViewNet._filter_init_kwargs(kwargs))

        # Load encoder: sv net.0 → encoder.0, sv net.2 → encoder.2
        encoder_map = {
            'net.0.weight': '0.weight', 'net.0.bias': '0.bias',
            'net.2.weight': '2.weight', 'net.2.bias': '2.bias',
        }
        enc_state = {v: sv_state[k] for k, v in encoder_map.items() if k in sv_state}
        model.encoder.load_state_dict(enc_state, strict=False)
        print(f"  Loaded encoder ({len(enc_state)} params) from pretrained")

        # Load decoder output layer: sv net.4 (76-dim) → decoder (79-dim, 6D go)
        # Old layout: [go3 | pose63 | betas10]  (positions 0:3, 3:66, 66:76)
        # New layout: [go6 | pose63 | betas10]  (positions 0:6, 6:69, 69:79)
        if 'net.4.weight' in sv_state:
            old_w = sv_state['net.4.weight']  # [76, feat_dim]
            old_b = sv_state['net.4.bias']    # [76]
            with torch.no_grad():
                # Copy pose + betas from old [3:76] → new [6:79]
                model.decoder.weight.data[6:] = old_w[3:]
                model.decoder.bias.data[6:]   = old_b[3:]
                # go6 head: small random init, bias = identity 6D (1,0,0,0,1,0)
                nn.init.normal_(model.decoder.weight.data[:6], std=0.01)
                model.decoder.bias.data[:6] = torch.tensor(
                    [1., 0., 0., 0., 1., 0.])
            print("  Loaded decoder (pose+betas) from pretrained; go6d head re-initialized")

        if freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            print("  Encoder frozen")
        else:
            print("  Encoder trainable")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total:,}, Trainable: {trainable:,}")

        return model


def split_output(y):
    """Split [N, 79] → go6d [N,6], pose [N,63], betas [N,10]"""
    return y[:, :GO_DIM], y[:, GO_DIM:GO_DIM + POSE_DIM], y[:, GO_DIM + POSE_DIM:]


def aa_to_rot6d_np(aa_np):
    """axis-angle [N, 3] numpy → 6D rotation [N, 6] numpy (for GT conversion, no grad)."""
    import numpy as _np
    from scipy.spatial.transform import Rotation as _R
    mat = _R.from_rotvec(aa_np).as_matrix()        # [N, 3, 3]
    return _np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=1).astype(_np.float32)  # [N, 6]
