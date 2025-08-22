import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LossOut:
    total: torch.Tensor
    recon: torch.Tensor
    perceptual: torch.Tensor
    kld: torch.Tensor

import torch
import torch.nn as nn
import torchvision.models as models

# Perceptual VGG loss
class VGGLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2'], device='cuda'):
        super().__init__()
        self.device = device
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.layer_names = layers
        self.layer_map = {'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15}
        self.slices = nn.ModuleList()
        prev_idx = 0
        for name in layers:
            idx = self.layer_map[name]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:idx + 1]))
            prev_idx = idx + 1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Upsample to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        loss = 0
        x_feat, y_feat = x, y
        for slice in self.slices:
            x_feat = slice(x_feat)
            y_feat = slice(y_feat)
            loss += F.mse_loss(x_feat, y_feat)
        return loss


# Full VAE loss
def vae_loss(recon, x, mu, logvar, alpha = 10.0, beta=1.0, gamma=0.001, vgg_loss_fn=None, pixel_loss='mse'):
    """
    recon: reconstructed images [B,3,H,W]
    x: original images [B,3,H,W]
    mu, logvar: latent maps [B,C,H',W']
    beta: KL weight
    gamma: VGG weight
    vgg_loss_fn: instance of VGGLoss
    pixel_loss: 'mse' or 'bce'
    """
    # Reconstruction loss
    if pixel_loss == 'mse':
        recon_loss = nn.functional.mse_loss(recon, x)
    elif pixel_loss == 'bce':
        recon_loss = nn.functional.binary_cross_entropy(recon, x)
    else:
        raise ValueError("pixel_loss must be 'mse' or 'bce'")

    # VGG perceptual loss
    if vgg_loss_fn is not None:
        perceptual_loss = vgg_loss_fn(recon, x)
    else:
        perceptual_loss = 0.0

    # KL divergence (sum over latent channels & spatial dims, mean over batch)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]))

    # Total loss
    loss = recon_loss * alpha + gamma * perceptual_loss + beta * kl
    return LossOut(loss, recon_loss, perceptual_loss, kl)

