import torch
import torch.nn as nn


class SmootherMLP(nn.Module):
    """Sliding-window MLP for temporal pose smoothing.

    Input:  [B, W*72]  — window of W frames, each 24 joints x 3
    Output: [B, 72]    — smoothed center frame (24 joints x 3)
    """

    def __init__(self, window_size=32, joint_dim=72, hidden_dims=(512, 256)):
        super().__init__()
        self.window_size = window_size
        self.joint_dim = joint_dim

        layers = []
        in_dim = window_size * joint_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, joint_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: [B, W*72] or [B, W, 24, 3] → [B, 72]"""
        if x.dim() == 4:
            B = x.shape[0]
            x = x.reshape(B, -1)
        elif x.dim() == 3:
            B = x.shape[0]
            x = x.reshape(B, -1)
        return self.net(x)
