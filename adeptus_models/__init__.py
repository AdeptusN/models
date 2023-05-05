import torch

from .autoencoder import Autoencoder
from .segmentation import UNet
from .unet_autoencoder import UNetAutoencoder


def save_model(model, path: str):
    """
    Saves model weights
    Args:
        model: model to save
        path: path to save
    """
    torch.save(model.state_dict(), path)


def load_model(model, path: str):
    """
    Loads model weights
    Args:
        model: model to load
        path: weights path
    """
    model.load_state_dict(torch.load(path))
    return model