import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.deconv_layers = nn.Sequential(
            # input: [B, latent_dim, 2, 2]
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),         # [B, 256, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),         # [B, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # [B, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),           # [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),                     # [B, 3, 64, 64]
            nn.Sigmoid()  # Output in [0,1] for images
        )

    def forward(self, z):
        return self.deconv_layers(z)   # [B, 3, 64, 64]
