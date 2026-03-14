import numpy as np
import torch
from torch.utils.data import Dataset


class SmoothDataset(Dataset):
    """Dataset for training temporal pose smoother.

    Takes clean AMASS joint sequences, adds synthetic jitter,
    and creates sliding window training pairs.

    Input:  noisy window [W, 72]
    Target: clean center frame [72]
    """

    def __init__(self, joints, lengths, window_size=32,
                 noise_std=0.02, temporal_jitter_std=0.01,
                 outlier_prob=0.05, outlier_std=0.05):
        """
        Args:
            joints: [total_frames, 24, 3] concatenated sequences
            lengths: [N_seq] length of each sequence
            window_size: sliding window size W
            noise_std: per-joint Gaussian noise std (meters)
            temporal_jitter_std: per-frame global offset std
            outlier_prob: probability of large-noise outlier frames
            outlier_std: outlier noise std
        """
        self.window_size = window_size
        self.noise_std = noise_std
        self.temporal_jitter_std = temporal_jitter_std
        self.outlier_prob = outlier_prob
        self.outlier_std = outlier_std

        # Build index: (seq_start, frame_offset) for each valid window
        self.windows = []
        offset = 0
        for seq_len in lengths:
            n_windows = seq_len - window_size + 1
            if n_windows > 0:
                for j in range(n_windows):
                    self.windows.append((offset + j, offset + j + window_size))
            offset += seq_len

        self.joints = joints.reshape(-1, 72).astype(np.float32)  # [total, 72]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        clean_window = self.joints[start:end].copy()  # [W, 72]

        # Add noise to simulate prediction errors
        noisy_window = clean_window.copy()

        # 1. Per-joint Gaussian noise
        noisy_window += np.random.randn(*noisy_window.shape).astype(np.float32) * self.noise_std

        # 2. Per-frame global offset (simulates camera/detection jitter)
        frame_offsets = np.random.randn(self.window_size, 1).astype(np.float32) * self.temporal_jitter_std
        # Broadcast to all 24 joints (repeat for x,y,z of each joint)
        frame_offsets_3d = np.repeat(frame_offsets, 72, axis=1)  # [W, 72]
        noisy_window += frame_offsets_3d

        # 3. Outlier frames (occasional large noise)
        outlier_mask = np.random.rand(self.window_size) < self.outlier_prob
        if outlier_mask.any():
            n_outliers = outlier_mask.sum()
            noisy_window[outlier_mask] += (
                np.random.randn(n_outliers, 72).astype(np.float32) * self.outlier_std
            )

        # Target: clean center frame
        center = self.window_size // 2
        clean_center = clean_window[center]  # [72]

        return (
            torch.from_numpy(noisy_window.reshape(-1)),  # [W*72]
            torch.from_numpy(clean_center),               # [72]
        )


def load_amass_joints(npz_path):
    """Load preprocessed AMASS joints data."""
    data = np.load(npz_path)
    joints = data['joints']    # [total_frames, 24, 3]
    lengths = data['lengths']  # [N_seq]
    return joints, lengths
