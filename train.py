"""
Train Conditional GAN for text-to-image generation on CUB Birds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.discriminator import ConditionalDiscriminator
from models.generator import ConditionalGenerator
from utils.checkpoint import save_checkpoint
from utils.dataset import CUBTextImageDataset
from utils.image_utils import save_image_grid
from utils.text_encoder import TextEncoder

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "demo_cgan_64.pt"
LOG_CSV = PROJECT_ROOT / "logs" / "train_loss.csv"
SAMPLES_DIR = PROJECT_ROOT / "outputs" / "samples"
PROJECTION_CACHE = PROJECT_ROOT / "data_cache" / "text_projection.pt"


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_cond(cond: torch.Tensor) -> torch.Tensor:
    """L2-normalize text conditioning (stabilizes cGAN training)."""
    return F.normalize(cond, p=2, dim=1)


def instance_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return images
    return images + torch.randn_like(images) * std


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Using device: {device}")

    dataset = CUBTextImageDataset(
        image_size=args.image_size,
        subset=args.subset,
        require_cache=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    config = {
        "z_dim": args.z_dim,
        "cond_dim": args.cond_dim,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "image_size": args.image_size,
        "model_name": "distilbert-base-uncased",
    }

    generator = ConditionalGenerator(
        z_dim=args.z_dim, cond_dim=args.cond_dim, ngf=args.ngf
    ).to(device)
    discriminator = ConditionalDiscriminator(
        cond_dim=args.cond_dim, ndf=args.ndf
    ).to(device)
    text_encoder = TextEncoder(cond_dim=args.cond_dim).to(device)

    if PROJECTION_CACHE.exists():
        text_encoder.projection.load_state_dict(
            torch.load(PROJECTION_CACHE, map_location=device, weights_only=True)
        )
        print(f"Loaded text projection from {PROJECTION_CACHE}")

    use_lsgan = args.loss == "lsgan"
    criterion = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()
    # Cached embeddings are fixed; only generator/discriminator are trained.
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999)
    )

    real_label = args.label_smoothing
    fake_label = 0.0
    noise_std = args.instance_noise

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists() or args.overwrite_logs
    if args.overwrite_logs and LOG_CSV.exists():
        LOG_CSV.unlink()

    fixed_batch = next(iter(dataloader))
    fixed_cond = normalize_cond(fixed_batch["embedding"][:8].to(device))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        n_batches = 0

        # Decay instance noise so D does not overpower G early on.
        epoch_noise = noise_std * max(0.2, 1.0 - epoch / max(args.epochs, 1))

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            real_images = batch["image"].to(device)
            cond = normalize_cond(batch["embedding"].to(device))
            batch_size = real_images.size(0)

            loss_d_val = 0.0
            loss_g_val = 0.0

            # --- Train Discriminator (once) ---
            discriminator.zero_grad()
            z = torch.randn(batch_size, args.z_dim, device=device)
            with torch.no_grad():
                fake_images = generator(z, cond)

            noisy_real = instance_noise(real_images, epoch_noise)
            noisy_fake = instance_noise(fake_images.detach(), epoch_noise)

            if use_lsgan:
                output_real, _ = discriminator(noisy_real, cond, return_features=True)
                loss_d_real = 0.5 * criterion(output_real, torch.ones_like(output_real))
                output_fake, _ = discriminator(noisy_fake, cond, return_features=True)
                loss_d_fake = 0.5 * criterion(output_fake, torch.zeros_like(output_fake))
            else:
                output_real = discriminator(noisy_real, cond)
                loss_d_real = criterion(
                    output_real, torch.full_like(output_real, real_label)
                )
                output_fake = discriminator(noisy_fake, cond)
                loss_d_fake = criterion(output_fake, torch.zeros_like(output_fake))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
            loss_d_val = loss_d.item()

            # --- Train Generator (multiple steps + feature matching) ---
            for _ in range(args.g_steps):
                generator.zero_grad()
                z = torch.randn(batch_size, args.z_dim, device=device)
                fake_images = generator(z, cond)

                output_fake, feat_fake = discriminator(
                    fake_images, cond, return_features=True
                )
                _, feat_real = discriminator(real_images, cond, return_features=True)

                if use_lsgan:
                    loss_g_adv = 0.5 * criterion(output_fake, torch.ones_like(output_fake))
                else:
                    loss_g_adv = criterion(output_fake, torch.ones_like(output_fake))

                loss_g_fm = F.mse_loss(feat_fake, feat_real.detach())
                loss_g = loss_g_adv + args.fm_weight * loss_g_fm
                loss_g.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                optimizer_g.step()
                loss_g_val = loss_g.item()

            d_loss_epoch += loss_d_val
            g_loss_epoch += loss_g_val
            n_batches += 1
            global_step += 1
            pbar.set_postfix(d_loss=f"{loss_d_val:.4f}", g_loss=f"{loss_g_val:.4f}")

        avg_d = d_loss_epoch / max(n_batches, 1)
        avg_g = g_loss_epoch / max(n_batches, 1)

        with LOG_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "d_loss", "g_loss", "global_step"])
                write_header = False
            writer.writerow([epoch, f"{avg_d:.6f}", f"{avg_g:.6f}", global_step])

        print(f"Epoch {epoch}: D_loss={avg_d:.4f}, G_loss={avg_g:.4f}")

        # Sample grid
        with torch.no_grad():
            z = torch.randn(8, args.z_dim, device=device)
            samples = generator(z, fixed_cond)
        save_image_grid(
            samples,
            SAMPLES_DIR / f"epoch_{epoch:03d}.png",
            nrow=4,
        )

        save_checkpoint(
            args.save,
            generator,
            discriminator,
            text_encoder,
            epoch,
            config,
            optimizer_g,
            optimizer_d,
        )

    print(f"Training complete. Checkpoint: {args.save}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-to-image cGAN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4, help="Deprecated: use --lr-g")
    parser.add_argument("--lr-g", type=float, default=None, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=None, help="Discriminator learning rate")
    parser.add_argument("--g-steps", type=int, default=2, help="Generator steps per discriminator step")
    parser.add_argument("--instance-noise", type=float, default=0.08, help="Noise added to D inputs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm for G (0=off)")
    parser.add_argument("--loss", type=str, default="lsgan", choices=["lsgan", "bce"])
    parser.add_argument("--fm-weight", type=float, default=10.0, help="Feature matching weight")
    parser.add_argument("--label-smoothing", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--overwrite-logs", action="store_true")
    return parser.parse_args()


def _apply_lr_defaults(args: argparse.Namespace) -> None:
    if args.lr_g is None:
        args.lr_g = args.lr
    if args.lr_d is None:
        args.lr_d = args.lr * 0.25  # weaker discriminator by default


if __name__ == "__main__":
    parsed = parse_args()
    _apply_lr_defaults(parsed)
    train(parsed)
