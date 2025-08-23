import torch

from models.Decoder import Decoder
from models.Encoder import Encoder
from models.VAE import VAE


def load_vae_model(checkpoint_path: str, device: torch.device = None) -> VAE:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    latent_dim = args.get('latent_dim', 20)
    hidden_dim = args.get('hidden_dim', 400)
    model = VAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_encoder_decoder(checkpoint_path: str, device: torch.device = None, latent_dim: int = 128) :
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})

    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    model_state = checkpoint['model_state_dict']

    # Extract encoder and decoder states
    encoder_state = {k.replace('encoder.', ''): v for k, v in model_state.items() if k.startswith('encoder.')}
    decoder_state = {k.replace('decoder.', ''): v for k, v in model_state.items() if k.startswith('decoder.')}

    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    encoder.eval()
    decoder.eval()
    return encoder, decoder