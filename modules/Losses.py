import torch
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class LossOut:
    total: torch.Tensor
    recon: torch.Tensor
    kld: torch.Tensor


def vae_loss(logits, x, mu, logvar, beta: float = 1.0) -> LossOut:
    # Computing VAE Loss, Using Binary Cross Entropy for 0/1
    B = x.size(0)
    x_flat = x.view(B, -1)
    recon = F.binary_cross_entropy_with_logits(logits, x_flat, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kld
    return LossOut(total, recon, kld)