import math
import torch
import os
from torchvision import utils as vutils
from models.VAE import VAE

@torch.no_grad()
def save_reconstructions(model: VAE, batch, out_dir: str, step: int, device: torch.device):
    model.eval()
    x, _ = batch
    x = x.to(device)

    # Forward pass (x_recon already in [0,1], shape [B, 3, 224, 224])
    x_recon, _, _ = model(x)

    # Show originals (top row) + reconstructions (bottom row)
    grid = vutils.make_grid(
        torch.cat([x[:8].cpu(), x_recon[:8].cpu()], dim=0),
        nrow=8,
        normalize=True,  # scales to [0,1] for saving
        value_range=(0, 1)
    )

    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(grid, os.path.join(out_dir, f"recon_step_{step:06d}.png"))


@torch.no_grad()
def save_samples(model: VAE, out_dir: str, device: torch.device, n: int = 64):
    model.eval()

    # Sample latent vectors
    z = torch.randn(n, model.latent_dim, device=device)

    # Decode to images (already [0,1], shape [n, 3, 224, 224])
    x = model.decoder(z)

    # Arrange into a grid
    grid = vutils.make_grid(
        x.cpu(),
        nrow=int(math.sqrt(n)),
        normalize=True,
        value_range=(0, 1)
    )

    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(grid, os.path.join(out_dir, "samples.png"))
