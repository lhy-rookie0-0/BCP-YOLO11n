import torch
from torch import nn


class Bi_FPN(nn.Module):
    """Paper's nonnegative, normalized weighted feature sum."""

    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32))
        self.epsilon = 1e-4

    def forward(self, x):
        if len(x) != self.weight.numel():
            raise ValueError("BiFPN input count does not match fusion weights")
        weights = torch.relu(self.weight)
        weights = weights / (weights.sum() + self.epsilon)
        return sum(w * feature for w, feature in zip(weights, x))
