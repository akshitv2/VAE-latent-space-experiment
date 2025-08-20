import torch
from torch import nn

from models.Decoder import Decoder
from models.Encoder import Encoder


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)  # conv encoder
        self.decoder = Decoder(latent_dim=latent_dim)  # conv decoder
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)          # Encode image
        z = self.reparameterize(mu, logvar)   # Sample latent
        x_recon = self.decoder(z)             # Decode back
        return x_recon, mu, logvar
