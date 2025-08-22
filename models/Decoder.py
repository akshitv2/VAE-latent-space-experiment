import torch
from torch import nn

from models.ResModels import ResUp


class Decoder(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()
        # Input: latent feature map [B, latent_channels, 4, 4]
        self.resup1 = ResUp(latent_channels, 512)             # 4x4 -> 8x8
        self.resup2 = ResUp(512, 256)               # 8x8 -> 16x16
        self.resup3 = ResUp(256, 128)               # 16x16 -> 32x32
        self.resup4 = ResUp(128, 64)                # 32x32 -> 64x64

        # Final conv to RGB
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()  # scale output to [0,1]

    def forward(self, z):
        x = self.resup1(z)   # 4 -> 8
        x = self.resup2(x)   # 8 -> 16
        x = self.resup3(x)   # 16 -> 32
        x = self.resup4(x)   # 32 -> 64
        x = self.final_conv(x)
        x = self.output_act(x)
        return x

