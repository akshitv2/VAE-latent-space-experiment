import torch
from torch import nn

from models.ResBlocks import ResidualBlock


class Encoder(nn.Module):
    """
    64x64 -> ... -> 4x4 spatial feature map.
    Produces mu and logvar tensors of shape (B, latent_ch, 4, 4).
    """
    def __init__(self, in_ch=3, base_ch=64, latent_ch=128):
        super().__init__()
        C = base_ch

        self.stem = nn.Conv2d(in_ch, C, kernel_size=3, stride=1, padding=1)

        # Sequence of residual downs: 64->32->16->8->4
        self.enc = nn.Sequential(
            ResidualBlock(C,     C,     downsample=False),
            ResidualBlock(C,     C,     downsample=True),   # 64 -> 32
            ResidualBlock(C,     2*C,   downsample=False),
            ResidualBlock(2*C,   2*C,   downsample=True),   # 32 -> 16
            ResidualBlock(2*C,   4*C,   downsample=False),
            ResidualBlock(4*C,   4*C,   downsample=True),   # 16 -> 8
            ResidualBlock(4*C,   8*C,   downsample=False),
            ResidualBlock(8*C,   8*C,   downsample=True),   # 8  -> 4
            ResidualBlock(8*C,   8*C,   downsample=False),
        )

        # Heads for mean and log-variance; both output (B, latent_ch, 4, 4)
        self.mu_head     = nn.Conv2d(8*C, latent_ch, kernel_size=1)
        self.logvar_head = nn.Conv2d(8*C, latent_ch, kernel_size=1)

    def forward(self, x):
        h = self.stem(x)
        h = self.enc(h)              # (B, 8*C, 4, 4)
        mu = self.mu_head(h)         # (B, latent_ch, 4, 4)
        logvar = self.logvar_head(h) # (B, latent_ch, 4, 4)
        return mu, logvar
