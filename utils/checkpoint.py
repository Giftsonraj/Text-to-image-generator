"""
Save and load GAN checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.discriminator import ConditionalDiscriminator
from models.generator import ConditionalGenerator
from utils.text_encoder import TextEncoder


def save_checkpoint(
    path: str | Path,
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    text_encoder: TextEncoder,
    epoch: int,
    config: dict,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "config": config,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "text_projection": text_encoder.projection.state_dict(),
    }
    if optimizer_g is not None:
        payload["optimizer_g"] = optimizer_g.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    torch.save(payload, path)

    config_path = path.with_suffix(".json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    load_optimizers: bool = False,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("config", {})

    z_dim = config.get("z_dim", 100)
    cond_dim = config.get("cond_dim", 256)
    ngf = config.get("ngf", 64)
    ndf = config.get("ndf", 64)

    generator = ConditionalGenerator(z_dim=z_dim, cond_dim=cond_dim, ngf=ngf).to(device)
    discriminator = ConditionalDiscriminator(cond_dim=cond_dim, ndf=ndf).to(device)
    text_encoder = TextEncoder(cond_dim=cond_dim, model_name=config.get("model_name", "distilbert-base-uncased")).to(device)

    generator.load_state_dict(payload["generator"])
    discriminator.load_state_dict(payload["discriminator"])
    text_encoder.projection.load_state_dict(payload["text_projection"])

    generator.eval()
    discriminator.eval()
    text_encoder.eval()

    result = {
        "generator": generator,
        "discriminator": discriminator,
        "text_encoder": text_encoder,
        "epoch": payload.get("epoch", 0),
        "config": config,
    }
    if load_optimizers and "optimizer_g" in payload:
        result["optimizer_g_state"] = payload["optimizer_g"]
        result["optimizer_d_state"] = payload["optimizer_d"]
    return result
