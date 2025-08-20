import torch
from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Project latent vector back to feature map [512, 7, 7]
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)

        # Deconvolutional layers (reverse of encoder)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [B, 64, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # [B, 32, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # [B, 3, 224, 224]
            nn.Sigmoid()  # constrain output to [0, 1] pixel range
        )

    def forward(self, z):
        h = self.fc(z)                          # [B, 512*7*7]
        h = h.view(z.size(0), 512, 7, 7)        # reshape -> [B, 512, 7, 7]
        x_recon = self.deconv_layers(h)         # [B, 3, 224, 224]
        return x_recon
