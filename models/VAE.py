import torch
from torch import nn

from models.Decoder import Decoder
from models.Encoder import Encoder


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick applied element-wise on feature maps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)  # [B, C, 4, 4]
        z = self.reparameterize(mu, logvar)  # same shape
        recon = self.decoder(z)  # [B, 3, 64, 64]
        return recon, mu, logvar