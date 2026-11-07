import torch
from torch import nn


class CBAM(nn.Module):
    """Channel attention (average/max, shared MLP) followed by spatial attention."""

    def __init__(self, c1, kernel_size=7, reduction=16):
        super().__init__()
        hidden = max(1, c1 // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
        )
        self.channel_gate = nn.Sigmoid()
        if kernel_size != 7:
            raise ValueError("The manuscript specifies a 7x7 spatial convolution")
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_gate = nn.Sigmoid()

    def forward(self, x):
        channel = self.channel_gate(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * channel
        spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * self.spatial_gate(self.spatial_conv(spatial))
