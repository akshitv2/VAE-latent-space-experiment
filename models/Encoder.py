import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 112, 112]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 56, 56]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 28, 28]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 14, 14]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # [B, 512, 7, 7]
            nn.ReLU()
        )

        # Flattened feature size: 512 * 7 * 7 = 25088
        self.fc_mean = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)          # [B, 512, 7, 7]
        h = h.view(h.size(0), -1)        # Flatten -> [B, 25088]
        mu = self.fc_mean(h)             # [B, latent_dim]
        logvar = self.fc_logvar(h)       # [B, latent_dim]
        return mu, logvar
