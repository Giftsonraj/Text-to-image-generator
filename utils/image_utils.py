"""
Image utilities: grids, denormalization, and plotting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] (Tanh output) to [0, 1]."""
    return (tensor.clamp(-1, 1) + 1) / 2


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single CHW tensor in [-1, 1] to a PIL image."""
    img = denormalize_image(tensor).cpu().detach()
    if img.dim() == 4:
        img = img[0]
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """Save a batch of images (B, C, H, W) in [-1, 1] as a grid PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(
        denormalize_image(images) if normalize else images,
        nrow=nrow,
        padding=2,
    )
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def plot_losses(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    """Plot generator and discriminator losses from train_loss.csv."""
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    epochs, d_losses, g_losses = [], [], []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            d_losses.append(float(row["d_loss"]))
            g_losses.append(float(row["g_loss"]))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, d_losses, label="Discriminator loss")
    plt.plot(epochs, g_losses, label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Loss")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=120)
        plt.close()
    else:
        plt.show()
