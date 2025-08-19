import os
import torch

def save_checkpoint(model, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"vae_checkpoint_epoch_{epoch:02d}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")