import torch
import torch.nn as nn
import numpy as np

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, threshold=1.0):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        grad_v = grad_output * (1.0 / (1.0 + (np.pi * (v - ctx.threshold)) ** 2))
        return grad_v, None

class LIFNeuron(nn.Module):
    def __init__(self, decay=0.85, threshold=1.0):
        super().__init__()
        self.decay = decay
        self.threshold = threshold
        self.spike_fn = SurrogateSpike.apply

    def forward(self, x):
        mem = 0.0
        spikes = []
        for t in range(x.shape[1]):
            mem = mem * self.decay + x[:, t]
            spike = self.spike_fn(mem, self.threshold)
            mem = mem - spike * self.threshold
            spikes.append(spike)
        return torch.stack(spikes, dim=1)