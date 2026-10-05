import torch
import torch.nn as nn
from .snn_branch import SNNTemporalBranch


class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        groups = 4 if out_ch % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.net(x)


class TriBranchTESSNet(nn.Module):
    """Morphology-only three-branch TESS classifier used by v9.

    The model deliberately does not consume the rule-based physical-vetting
    measurements. Real-TESS fine-tuning therefore teaches the network the
    morphology distribution directly, while BLS/odd-even/secondary vetting
    remains an independent evidence stream.
    """
    def __init__(self, tabular_dim=None):
        super().__init__()
        self.global_conv = nn.Sequential(
            Conv1DBlock(1, 16, 5),
            Conv1DBlock(16, 32, 5),
            nn.AdaptiveAvgPool1d(4)
        )
        self.local_conv = nn.Sequential(
            Conv1DBlock(1, 16, 3),
            Conv1DBlock(16, 32, 3),
            nn.AdaptiveAvgPool1d(4)
        )
        self.snn_branch = SNNTemporalBranch(in_features=61, hidden_dim=32)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 16, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 2)
        )

    def forward(self, global_x, local_x, tab_x=None):
        g = self.global_conv(global_x).flatten(1)
        l = self.local_conv(local_x).flatten(1)
        s = self.snn_branch(local_x)
        fused = torch.cat([g, l, s], dim=1)
        return self.classifier(fused)
