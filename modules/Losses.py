import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LossOut:
    total: torch.Tensor
    recon: torch.Tensor
    kld: torch.Tensor


def vae_loss(x_recon, x, mu, logvar, beta: float = 1.0) -> LossOut:
    # Reconstruction loss
    # BCE can sometimes lead to vanishing gradients for pixel values near 0 or 1.
    # MSE (L2) is often smoother for VAEs on images.
    recon = F.mse_loss(x_recon, x, reduction="sum")

    # KL Divergence term
    # Summed over all pixels in latent map
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Beta-VAE objective
    total = recon + beta * kld

    return LossOut(total, recon, kld)

