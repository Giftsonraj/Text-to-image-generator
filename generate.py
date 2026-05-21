"""
Generate an image from a text description using a trained cGAN checkpoint.

Modes:
  gan       - pure GAN synthesis (64x64, may look noisy)
  retrieve  - best real CUB photo matching the text (clear demo output)
  best      - retrieval upscaled (default for presentation quality)
  both      - save GAN + retrieval side by side paths
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint import load_checkpoint
from utils.image_utils import tensor_to_pil
from utils.retrieval import load_retrieved_image, retrieve_best_match
from utils.text_encoder import TextEncoder


def normalize_cond(cond: torch.Tensor) -> torch.Tensor:
    return F.normalize(cond, p=2, dim=1)


def generate_gan(
    text: str,
    generator,
    discriminator,
    text_encoder: TextEncoder,
    config: dict,
    device: str,
    seed: int | None = None,
    num_samples: int = 16,
) -> Image.Image:
    if seed is not None:
        torch.manual_seed(seed)

    z_dim = config.get("z_dim", 100)

    with torch.no_grad():
        cond = normalize_cond(text_encoder.encode([text], device=device))
        best = None
        best_score = float("-inf")
        for _ in range(max(1, num_samples)):
            z = torch.randn(1, z_dim, device=device)
            fake = generator(z, cond)
            score = discriminator(fake, cond).mean().item()
            if score > best_score:
                best_score = score
                best = fake
        return tensor_to_pil(best[0])


def generate(
    text: str,
    checkpoint: str | Path,
    output: str | Path,
    seed: int | None = None,
    device: str = "cpu",
    num_samples: int = 16,
    mode: str = "best",
    display_size: int = 256,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if mode in ("retrieve", "best"):
        matches = retrieve_best_match(text, device=device, top_k=1)
        if not matches:
            raise FileNotFoundError("No matching CUB images found. Check dataset/cub/images/")
        img = load_retrieved_image(matches[0], size=display_size if mode == "best" else 0)
        img.save(output)
        print(f"Retrieved: {matches[0]['id']} (similarity={matches[0]['similarity']:.3f})")
        print(f"Dataset caption: {matches[0]['caption']}")
        return output

    ckpt = load_checkpoint(checkpoint, device=device)
    gan_img = generate_gan(
        text,
        ckpt["generator"],
        ckpt["discriminator"],
        ckpt["text_encoder"],
        ckpt["config"],
        device,
        seed,
        num_samples,
    )

    if mode == "gan":
        if display_size > 64:
            gan_img = gan_img.resize((display_size, display_size), Image.Resampling.NEAREST)
        gan_img.save(output)
        return output

    # mode == "both"
    matches = retrieve_best_match(text, device=device, top_k=1)
    gan_path = output.with_name(output.stem + "_gan.png")
    gan_img.save(gan_path)
    if matches:
        ref_img = load_retrieved_image(matches[0], size=display_size)
        ref_path = output.with_name(output.stem + "_reference.png")
        ref_img.save(ref_path)
        ref_img.save(output)
        print(f"GAN saved: {gan_path}")
        print(f"Reference saved: {ref_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image from text")
    parser.add_argument("--text", type=str, required=True, help="Bird description")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "demo_cgan_64.pt"),
    )
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "outputs" / "generated.png"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--mode",
        type=str,
        default="best",
        choices=["gan", "retrieve", "best", "both"],
        help="best=clear dataset match (recommended for demo)",
    )
    parser.add_argument("--display-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    path = generate(
        text=args.text,
        checkpoint=args.checkpoint,
        output=args.out,
        seed=args.seed,
        device=args.device,
        num_samples=args.samples,
        mode=args.mode,
        display_size=args.display_size,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
