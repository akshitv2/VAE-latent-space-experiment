from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossOut:
    total: torch.Tensor
    recon: torch.Tensor
    kld: torch.Tensor


def vae_loss(x_recon, x, mu, logvar, beta: float = 1.0) -> LossOut:
    # Reconstruction loss (BCE over all pixels, summed across batch)
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss (beta-VAE)
    total = recon + beta * kld

    return LossOut(total, recon, kld)
