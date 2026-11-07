from torch import nn


class PPLCStem(nn.Module):
    """PP-LCNet-inspired 3x3 convolution, batch norm and h-swish stem."""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        if k != 3:
            raise ValueError("The manuscript specifies a 3x3 stem")
        self.conv = nn.Conv2d(c1, c2, k, s, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
