import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], device='cuda'):
        super().__init__()
        self.device = device
        vgg = models.vgg16(pretrained=True).features.to(device)  # <-- move to device
        self.layers = nn.ModuleList([vgg[:i+1] for i in layer_ids])
        for param in vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        x_norm = (x - mean) / std
        y_norm = (y - mean) / std

        loss = 0
        for layer in self.layers:
            loss += F.mse_loss(layer(x_norm), layer(y_norm))
        return loss

# -------------------------------
# VAE Loss combining L1 + Perceptual + KL
# -------------------------------
class VAEVggLoss(nn.Module):
    def __init__(self, recon_weight=10.0, perc_weight=0.001, kl_weight=1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.perc_weight = perc_weight
        self.kl_weight = kl_weight
        self.perc_loss = VGGPerceptualLoss()

    def forward(self, x_recon, x, mu, logvar, func = "l1"):
        # L1 reconstruction loss
        if func == "l1":
            recon_loss = F.l1_loss(x_recon, x)
        elif func == "mse":
            recon_loss = F.mse_loss(x_recon, x)

        # Perceptual loss
        perc_loss = self.perc_loss(x_recon, x)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # normalize by batch size

        total_loss = self.recon_weight * recon_loss + \
                     self.perc_weight * perc_loss + \
                     self.kl_weight * kl_loss
        return total_loss, self.recon_weight *recon_loss, self.perc_weight *perc_loss, self.kl_weight *kl_loss