import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from experiments.Checkpointing import save_checkpoint
from models.VAE import VAE
from modules.Losses import vae_loss
from modules.SaveOutputs import save_reconstructions, save_samples


def train(epochs: int = 10, dataset_dir: str = "./data/raw", out_dir: str = "./outputs/",
          checkpoint_dir = "./experiments/checkpoints", batch_size: int = 128,
          latent_dim: int = 20, hidden_dim: int = 400, lr: float = 1e-3,
          beta: float = 1.0) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()
                                    # , transforms.Normalize((0.5,), (0.5,))
                                    ])

    train_ds = datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model & Optimizer
    model = VAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kld = 0.0

        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mean, logvar = model(x)
            loss = vae_loss(logits, x, mean, logvar, beta=beta)
            loss.total.backward()
            optimizer.step()

            running_total += loss.total.item()
            running_recon += loss.recon.item()
            running_kld += loss.kld.item()

            if global_step % 500 == 0:
                save_reconstructions(model, (x.cpu(), None), out_dir, global_step, device)
            global_step += 1

        n_train = len(train_loader.dataset)
        print(
            f"Epoch {epoch:02d} | total: {running_total / n_train:.4f} | "
            f"recon: {running_recon / n_train:.4f} | kld: {running_kld / n_train:.4f}"
        )

        model.eval()
        test_total = test_recon = test_kld = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                logits, mean, logvar = model(x)
                loss = vae_loss(logits, x, mean, logvar, beta=beta)
                test_total += loss.total.item()
                test_recon += loss.recon.item()
                test_kld += loss.kld.item()
        n_test = len(test_loader.dataset)
        print(
            f"  [val] total: {test_total / n_test:.4f} | recon: {test_recon / n_test:.4f} | kld: {test_kld / n_test:.4f}"
        )
    save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    # Save samples from prior
    save_samples(model, out_dir, device, n=64)
    torch.save({
        "model_state_dict": model.state_dict(),
    }, os.path.join(out_dir, "vae_mnist.pt"))
    print(f"Saved samples to {os.path.join(out_dir, 'samples.png')} and model to vae_mnist.pt")
