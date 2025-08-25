import torch
from torch import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

from experiments.Checkpointing import save_checkpoint
from models.VAE import VAE
from modules.Losses import VAEVggLoss
from modules.SaveOutputs import save_reconstructions, save_samples


def load_data(dataset_dir:str = "./data/raw", batch_size:int=64, num_workers:int=4):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # resize to 224x224
        transforms.ToTensor()  # convert to tensor & scale to [0,1]
    ])
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    train_test_split_var = 0.99
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(train_test_split_var * len(dataset)),
                                                                         len(dataset) - int(
                                                                             train_test_split_var * len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    n_train = len(train_loader.dataset)
    print("Loaded datasets, number of samples: ", n_train)
    return train_loader, test_loader, n_train

def train(epochs: int = 10, dataset_dir: str = "./data/raw", out_dir: str = "./outputs/",
          checkpoint_dir="./experiments/checkpoints", batch_size: int = 128,
          latent_dim: int = 128, lr: float = 1e-3,
          beta: float = 1.0, variant: str = "") -> None:

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to 224x224
        transforms.ToTensor()  # convert to tensor & scale to [0,1]
    ])

    train_loader, test_loader, n_train = load_data(dataset_dir=dataset_dir, batch_size=batch_size)

    # Model & Optimizer
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vgg_loss = VAEVggLoss(recon_weight=0.1, perc_weight=1.0, kl_weight=0.01, recon_loss_function="mse")

    global_step = 0
    os.makedirs(out_dir, exist_ok=True)

    print("Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_total = running_recon = running_kld = running_perceptual =  0.0
        progress_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc="Training")
        x = None
        for batch_idx, (x, _) in progress_bar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mean, logvar = model(x)
            loss, l1_loss, perc_loss, kl_loss = vgg_loss(logits, x, mean, logvar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(
                loss=f"{loss.item():.3f}",
                l1=f"{l1_loss.item():.3f}",
                kld=f"{kl_loss.item():.3f}",
                percep=f"{perc_loss.item():.3f}",

            )
            running_total += loss.total.item()
            running_recon += loss.recon.item()
            running_kld += loss.kld.item()

        save_reconstructions(model=model, x=x, out_dir=out_dir, step = epoch, device=device, variant=variant)
        n_train = len(train_loader.dataset)
        print(
            f"Epoch {epoch:02d} | total: {running_total / n_train:.4f} | "
            f"recon: {running_recon / n_train:.4f} | kld: {running_kld / n_train:.4f} | "
            f"perceptual: {running_perceptual / n_train:.4f}"
        )

        model.eval()
        test_total = test_recon = test_kld = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                logits, mean, logvar = model(x)
                loss = vgg_loss(logits, x, mean, logvar, beta=beta)
                test_total += loss.total.item()
                test_recon += loss.recon.item()
                test_kld += loss.kld.item()
        n_test = len(test_loader.dataset)
        print(
            f"  [val] total: {test_total / n_test:.4f} | recon: {test_recon / n_test:.4f} | kld: {test_kld / n_test:.4f}"
        )
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)