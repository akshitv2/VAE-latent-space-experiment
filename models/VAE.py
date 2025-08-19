from torch import nn
import torch
from models.Encoder import Encoder
from models.Decoder import Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.latent_dim = latent_dim


    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar
