import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as vutils
import os

from models.VAE import VAE


@torch.no_grad()
def save_reconstructions(model: VAE, batch, out_dir: str, step: int, device: torch.device):
    model.eval()
    x, _ = batch
    x = x.to(device)
    logits, _, _ = model(x)
    x_recon = torch.sigmoid(logits).view(-1, 1, 28, 28)
    grid = vutils.make_grid(torch.cat([x[:8].cpu(), x_recon[:8].cpu()], dim=0), nrow=8)
    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(grid, os.path.join(out_dir, f"recon_step_{step:06d}.png"))


@torch.no_grad()
def save_samples(model: VAE, out_dir: str, device: torch.device, n: int = 64):
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    logits = model.decoder(z)
    x = torch.sigmoid(logits).view(n, 1, 28, 28)
    grid = vutils.make_grid(x, nrow=int(math.sqrt(n)))
    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(grid, os.path.join(out_dir, "samples.png"))