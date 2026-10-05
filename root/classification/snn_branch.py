import torch
import torch.nn as nn
from .snn_core import LIFNeuron


class SNNTemporalBranch(nn.Module):
    """
    Spiking temporal branch over the 61-point local transit view.

    The previous implementation squeezed the 61 samples into one timestep
    before the LIF neuron, so the "temporal" branch actually saw a sequence
    of length one.  We now apply the same small feature projection to each
    local-view sample and let the LIF state evolve across all 61 positions.
    """
    def __init__(self, in_features=61, hidden_dim=32):
        super().__init__()
        self.in_features = in_features
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )
        self.lif = LIFNeuron()
        self.out_dense = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, 1, 61] -> [B, 61, 1]
        seq = x.squeeze(1).unsqueeze(-1)
        h = self.input_proj(seq)          # [B, 61, hidden]
        spikes = self.lif(h)              # temporal membrane dynamics
        rate = spikes.mean(dim=1)         # [B, hidden]
        return self.out_dense(rate)
