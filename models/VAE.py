import torch
from torch import nn

from models.Decoder import Decoder
from models.Encoder import Encoder


class VAE(nn.Module):
    """
    Deep residual CNN-VAE for 64x64 RGB.
    Latent distribution is spatial: (latent_ch, 4, 4).
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, latent_ch=128):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.decoder = Decoder(out_ch=out_ch, base_ch=base_ch, latent_ch=latent_ch)

    @staticmethod
    def reparameterize(mu, logvar):
        # Clamp logvar to avoid numerical issues (helps prevent exploding KL)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)         # (B, latent_ch, 4, 4)
        z = self.reparameterize(mu, logvar)  # (B, latent_ch, 4, 4)
        x_recon = self.decoder(z)            # (B, 3, 64, 64)
        return x_recon, mu, logvar