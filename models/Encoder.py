import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 4, 4]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# [B, 512, 2, 2]
            nn.ReLU()
        )

        # Use 1x1 convolutions to map into latent channels
        self.conv_mean   = nn.Conv2d(512, latent_dim, kernel_size=1)   # [B, latent_dim, 2, 2]
        self.conv_logvar = nn.Conv2d(512, latent_dim, kernel_size=1)   # [B, latent_dim, 2, 2]

    def forward(self, x):
        h = self.conv_layers(x)       # [B, 512, 2, 2]
        mu = self.conv_mean(h)        # [B, latent_dim, 2, 2]
        logvar = self.conv_logvar(h)  # [B, latent_dim, 2, 2]
        return mu, logvar
