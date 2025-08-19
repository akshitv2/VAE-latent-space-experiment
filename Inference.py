import torch
from experiments.ModelLoader import load_encoder_decoder

def inference(checkpoint_path: str = "./experiments/checkpoints/vae_checkpoint_epoch_10.pt", device: torch.device = None):
    encoder, decoder = load_encoder_decoder(checkpoint_path)

if __name__ == "__main__":
    inference()