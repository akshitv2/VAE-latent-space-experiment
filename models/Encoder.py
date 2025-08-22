import torch
from torch import nn

from models.ResModels import ResDown


class Encoder(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()
        # Input: 3x64x64
        self.res1 = ResDown(3, 64)     # 64x64 -> 32x32
        self.res2 = ResDown(64, 128)   # 32x32 -> 16x16
        self.res3 = ResDown(128, 256)  # 16x16 -> 8x8
        self.res4 = ResDown(256, 512)  # 8x8 -> 4x4

        # Use 1x1 conv to produce mu and logvar
        self.conv_mu = nn.Conv2d(512, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(512, latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        mu = self.conv_mu(x)        # shape: [B, latent_channels, 4, 4]
        logvar = self.conv_logvar(x) # shape: [B, latent_channels, 4, 4]
        return mu, logvar
