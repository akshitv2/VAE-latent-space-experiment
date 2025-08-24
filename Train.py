import torch

from trainers.Training import train

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(epochs=30)