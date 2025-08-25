import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------
class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU.
    Supports optional downsample (encoder) or upsample (decoder).
    """
    def __init__(self, in_ch, out_ch, downsample=False, upsample=False, groups=32):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        self.in_ch = in_ch
        self.out_ch = out_ch

        mid_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, mid_ch), num_channels=mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)

        # Skip path to match channels/size
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1) if in_ch != out_ch else nn.Identity()

        # Down/Up ops
        if self.downsample:
            self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            self.skip_down = nn.Conv2d(in_ch if in_ch == out_ch else out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        elif self.upsample:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv_up = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            self.skip_up = nn.Upsample(scale_factor=2, mode="nearest")

        self.act = nn.SiLU()

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        # match channels for skip
        identity = self.skip(identity)

        # residual add (before scale change)
        out = out + identity

        # apply scale change coherently to both paths if requested
        if self.downsample:
            out = self.down(out)
            # recompute identity from the residual output itself to keep shapes aligned
            # (we already added skip; after downsampling, residual carries both)
        elif self.upsample:
            out = self.up(out)
            out = self.conv_up(out)

        return out