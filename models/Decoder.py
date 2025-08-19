from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):

    def __init__(self, output_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        logits = self.fc_out(h)  # logits, not passed through sigmoid
        return logits
