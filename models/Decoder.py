import torch
from torch import nn
import torch.nn.functional as F

from models.ResBlocks import ResidualBlock


class Decoder(nn.Module):
    """
    4x4 latent feature map -> 64x64 RGB reconstruction.
    """
    def __init__(self, out_ch=3, base_ch=64, latent_ch=128):
        super().__init__()
        C = base_ch

        self.proj = nn.Conv2d(latent_ch, 8*C, kernel_size=1)

        # Sequence of residual ups: 4->8->16->32->64
        self.dec = nn.Sequential(
            ResidualBlock(8*C,   8*C,   upsample=False),
            ResidualBlock(8*C,   8*C,   upsample=True),    # 4  -> 8
            ResidualBlock(8*C,   4*C,   upsample=False),
            ResidualBlock(4*C,   4*C,   upsample=True),    # 8  -> 16
            ResidualBlock(4*C,   2*C,   upsample=False),
            ResidualBlock(2*C,   2*C,   upsample=True),    # 16 -> 32
            ResidualBlock(2*C,   C,     upsample=False),
            ResidualBlock(C,     C,     upsample=True),    # 32 -> 64
            ResidualBlock(C,     C,     upsample=False),
        )

        self.out_norm = nn.GroupNorm(num_groups=min(32, C), num_channels=C)
        self.out_conv = nn.Conv2d(C, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.proj(z)
        h = self.dec(h)
        h = self.out_norm(h)
        h = F.silu(h)
        logits = self.out_conv(h)
        # If your images are [0,1], keep Sigmoid. If they're scaled to [-1,1], switch to Tanh.
        x_recon = torch.sigmoid(logits)
        return x_recon
