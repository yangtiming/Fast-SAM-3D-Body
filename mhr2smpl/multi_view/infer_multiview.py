"""
Multi-View MHR2SMPL Inference Interface
=======================================

Input: MHR mesh from multiple cameras per frame → Output: SMPL parameters.
SmootherMLP is optional (applied after multi-view fusion to denoise joint positions).

Required files (under share/ directory):
  data/mhr2smpl_mapping.npz          -- MHR→SMPL barycentric mapping
  data/smpl_vert_sample_indices.npy  -- 1500 vertex subsampling indices
  data/SMPL_NEUTRAL.pkl              -- SMPL model (needed when calling infer_smpl_joints)
  multiview_net.py                   -- Network definition
  smoother_net.py                    -- SmootherMLP definition
  weights/best_model.pth             -- Multi-view inference weights

Usage:
  from infer_multiview import MHR2SMPLMultiView

  model = MHR2SMPLMultiView()                    # No post-processing
  model = MHR2SMPLMultiView(smoother_dir='...')  # Enable SmootherMLP

  go, body_pose, betas, weights = model.infer(views)
  # views: [(pred_vertices [18439,3], pred_cam_t [3]), ...]
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()        # multi_view/
BASE_DIR   = SCRIPT_DIR.parent                      # mhr2smpl/
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BASE_DIR / "smooth"))

from multiview_net import MinimalMultiViewNet, split_output, rot6d_to_aa


class MHR2SMPLMultiView:
    """Multi-view MHR mesh → SMPL parameter inference engine.

    If smoother_dir is provided, SmootherMLP is enabled to denoise fused joint positions.
    """

    def __init__(
        self,
        model_path: str | None = None,
        mapping_path: str | None = None,
        sample_idx_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        smoother_dir: str | None = None,
    ):
        self.device = torch.device(device)

        model_path      = model_path      or str(BASE_DIR / "weights/best_model.pth")
        mapping_path    = mapping_path    or str(BASE_DIR / "data/mhr2smpl_mapping.npz")
        sample_idx_path = sample_idx_path or str(BASE_DIR / "data/smpl_vert_sample_indices.npy")

        # Barycentric mapping: MHR [18439] → SMPL [6890]
        mapping = np.load(mapping_path)
        self._mhr_vert_ids = mapping["mhr_vert_ids"].astype(np.int64)   # [6890, 3]
        self._baryc        = mapping["baryc_coords"].astype(np.float32) # [6890, 3]

        # Subsampling indices: SMPL [6890] → [1500]
        self._sample_idx = np.load(sample_idx_path).astype(np.int64)   # [1500]

        # Multi-view inference network
        self.model = MinimalMultiViewNet(input_dim=1500 * 3).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[MHR2SMPLMultiView] model={model_path}  device={device}")

        # SmootherMLP (optional, denoises joints after fusion)
        self._smoother     = None
        self._sm_window    = 1
        self._sm_joint_dim = 72
        self._joint_buf    = None
        if smoother_dir is not None:
            from smoother_net import SmootherMLP
            sm_dir = Path(smoother_dir)
            with open(sm_dir / "smoother_config.json") as f:
                cfg = json.load(f)
            self._smoother = SmootherMLP(
                window_size=cfg["window_size"],
                joint_dim=cfg["joint_dim"],
                hidden_dims=cfg["hidden_dims"],
            ).to(self.device)
            self._smoother.load_state_dict(
                torch.load(sm_dir / "smoother_best.pth",
                           map_location=self.device, weights_only=True)
            )
            self._smoother.eval()
            self._sm_window    = cfg["window_size"]
            self._sm_joint_dim = cfg["joint_dim"]
            self._joint_buf    = []
            print(f"  SmootherMLP: window={self._sm_window}, "
                  f"val_mpjpe={cfg.get('best_val_mpjpe_mm', '?'):.1f}mm")

    def reset(self):
        """Reset SmootherMLP frame buffer (call when switching scene/person)."""
        if self._joint_buf is not None:
            self._joint_buf.clear()

    # ── Preprocessing: single-view MHR → centroid-centered vertex vector ──
    def _preprocess_view(
        self,
        pred_vertices: np.ndarray,  # [18439, 3]  camera coordinate system, m
        pred_cam_t: np.ndarray,     # [3]          camera translation, m
    ) -> np.ndarray:                # [4500]        centroid-centered flat
        v = pred_vertices.copy()
        t = pred_cam_t.copy()

        # YZ-flip (Stage1 coordinate system convention)
        v[:, 1] *= -1; v[:, 2] *= -1
        t[1]    *= -1; t[2]    *= -1

        # → World coordinate system
        mhr_w = v + t[None, :]

        # Barycentric → SMPL [6890, 3]
        face_v = mhr_w[self._mhr_vert_ids]
        smpl_v = (face_v * self._baryc[:, :, None]).sum(axis=1)

        # Subsample [1500, 3] → centroid-center → flatten [4500]
        smpl_v_sub = smpl_v[self._sample_idx]
        smpl_v_sub -= smpl_v_sub.mean(axis=0, keepdims=True)
        return smpl_v_sub.flatten().astype(np.float32)

    # ── Main inference interface ────────────────────────────────────────────
    def infer(
        self,
        views: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            views: [(pred_vertices [18439,3], pred_cam_t [3]), ...]
                   One or more views are supported, 2 views recommended

        Returns:
            go        [3]   global orient, axis-angle
            body_pose [63]  body pose, 21 joints x 3, axis-angle
            betas     [10]  shape coefficients
            weights   [V]   per-view confidence (softmax, sum to 1)
        """
        assert len(views) >= 1

        feats  = np.stack([self._preprocess_view(v, t) for v, t in views], axis=0)
        V      = len(views)
        views_t = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        mask_t  = torch.ones(1, V, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            if V == 1:
                params  = self.model.forward_single_view(views_t[:, 0])
                weights = torch.ones(1, 1, device=self.device)
            else:
                params, weights = self.model(views_t, mask_t)

        go6d, body_pose, betas = split_output(params)
        go_aa = rot6d_to_aa(go6d)  # 6D → axis-angle for downstream use
        return (
            go_aa[0].cpu().numpy(),
            body_pose[0].cpu().numpy(),
            betas[0].cpu().numpy(),
            weights[0].cpu().numpy(),
        )

    def infer_smpl_joints(
        self,
        views: list[tuple[np.ndarray, np.ndarray]],
        smpl_model_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Infer and return canonical 3D joints (optional SmootherMLP denoising).

        Returns:
            go, body_pose, betas, weights — same as infer()
            joints [24, 3] — canonical space, root-relative
        """
        import smplx

        smpl_path = smpl_model_path or str(BASE_DIR / "data/SMPL_NEUTRAL.pkl")
        if not hasattr(self, "_smpl"):
            self._smpl = smplx.SMPL(model_path=smpl_path, gender="neutral").to(self.device)
            self._smpl.eval()
            for p in self._smpl.parameters():
                p.requires_grad_(False)

        go, body_pose, betas, weights = self.infer(views)

        bp_full = np.zeros(69, dtype=np.float32)
        bp_full[:63] = body_pose

        with torch.no_grad():
            out = self._smpl(
                global_orient=torch.zeros(1, 3, device=self.device),
                body_pose=torch.from_numpy(bp_full).unsqueeze(0).to(self.device),
                betas=torch.from_numpy(betas).unsqueeze(0).to(self.device),
            )
        j = out.joints[0, :24].cpu().numpy()
        j -= j[0:1]

        # SmootherMLP denoising (post-fusion correction, optional)
        if self._smoother is not None:
            j_flat = j.flatten().astype(np.float32)
            self._joint_buf.append(j_flat)
            if len(self._joint_buf) > self._sm_window:
                self._joint_buf.pop(0)
            pad    = [self._joint_buf[0]] * (self._sm_window - len(self._joint_buf))
            window = np.stack(pad + self._joint_buf, axis=0)
            x = torch.from_numpy(window.flatten()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                j = self._smoother(x)[0].cpu().numpy().reshape(24, 3)

        return go, body_pose, betas, weights, j


# ── Command-line quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_npz", nargs="+", required=True,
                        help="Stage1 sample_*.npz, each corresponds to one camera view")
    parser.add_argument("--smoother_dir", default=None,
                        help="SmootherMLP checkpoint directory (optional)")
    parser.add_argument("--smpl_joints", action="store_true")
    args = parser.parse_args()

    model = MHR2SMPLMultiView(smoother_dir=args.smoother_dir)

    views = []
    for npz_path in args.stage1_npz:
        d = np.load(npz_path, allow_pickle=True)
        views.append((d["pred_vertices"].astype(np.float32),
                      d["pred_cam_t"].astype(np.float32)))
        print(f"  Loaded: {npz_path}  cam_id={d.get('cam_id', '?')}")

    if args.smpl_joints:
        go, bp, betas, weights, joints = model.infer_smpl_joints(views)
        print(f"\ngo={go}\nbp[:6]={bp[:6]}\nbetas={betas}\nweights={weights}")
        print(f"joints[0]={joints[0]}  joints[15]={joints[15]}")
    else:
        go, bp, betas, weights = model.infer(views)
        print(f"\ngo={go}\nbp[:6]={bp[:6]}\nbetas={betas}\nweights={weights}")
