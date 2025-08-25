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


def load_data(dataset_dir: str = "./data/raw", batch_size: int = 64, num_workers: int = 4):
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


def vae_train(model: torch.nn.Module, train_loader, test_loader, loss_function, epochs: int = 10,
              out_dir: str = "./outputs/",
              checkpoint_dir="./experiments/checkpoints",
              lr: float = 1e-3,
              beta: float = 1.0, variant: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_total = running_recon = running_kld = running_perceptual = 0.0
        progress_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc="Epoch" + str(epoch))

        x = None
        for batch_idx, (x, _) in progress_bar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mean, logvar = model(x)
            loss, recon_loss, kl_loss = loss_function(logits, x, mean, logvar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(
                loss=f"{loss.item():.3f}",
                l1=f"{recon_loss.item():.3f}",
                kld=f"{kl_loss.item():.3f}"
            )
            running_total += loss.item()
            running_recon += loss.item()
            running_kld += loss.item()
        save_reconstructions(model=model, x=x, out_dir=out_dir, step=epoch, device=device, variant=variant)
        n_train = len(train_loader.dataset)
        print(
            f"Epoch {epoch:02d} | total: {running_total / n_train:.4f} | "
            f"recon: {running_recon / n_train:.4f} | kld: {running_kld / n_train:.4f} | "
            f"perceptual: {running_perceptual / n_train:.4f}"
        )
        if test_loader is not None:
            model.eval()
            test_total = test_recon = test_kld = 0.0
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    logits, mean, logvar = model(x)
                    loss, recon_loss, kl_loss = loss_function(logits, x, mean, logvar, beta=beta)
                    test_total += loss.item()
                    test_recon += recon_loss.item()
                    test_kld += kl_loss.item()
            n_test = len(test_loader.dataset)
            print(
                f"  [val] total: {test_total / n_test:.4f} | recon: {test_recon / n_test:.4f} | kld: {test_kld / n_test:.4f}"
            )
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
