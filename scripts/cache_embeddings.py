"""
Precompute DistilBERT conditioning vectors for all captions (CPU-friendly training).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import DEFAULT_CACHE_DIR, DEFAULT_INDEX_PATH, load_cub_index
from utils.text_encoder import TextEncoder


def cache_all(
    index_path: Path,
    cache_dir: Path,
    batch_size: int = 16,
    device: str = "cpu",
    subset: int = 0,
) -> None:
    entries = load_cub_index(index_path)
    if subset > 0:
        entries = entries[:subset]

    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = TextEncoder().to(device)
    encoder.eval()

    # Flatten (entry, caption_idx) pairs
    pairs: list[tuple[dict, int, str]] = []
    for entry in entries:
        for i, caption in enumerate(entry["captions"]):
            pairs.append((entry, i, caption))

    print(f"Caching {len(pairs)} caption embeddings to {cache_dir}")

    for start in tqdm(range(0, len(pairs), batch_size)):
        batch = pairs[start : start + batch_size]
        texts = [p[2] for p in batch]
        with torch.no_grad():
            embeddings = encoder.encode(texts, device=device)

        for (entry, cap_idx, _), emb in zip(batch, embeddings):
            out_path = cache_dir / f"{entry['id']}_{cap_idx}.pt"
            torch.save(emb.cpu(), out_path)

    projection_path = cache_dir.parent / "text_projection.pt"
    torch.save(encoder.projection.state_dict(), projection_path)
    print(f"Saved text projection weights to {projection_path}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache text embeddings for CUB captions")
    parser.add_argument("--index", type=str, default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--subset", type=int, default=0, help="0 = all entries")
    args = parser.parse_args()

    cache_all(
        index_path=Path(args.index),
        cache_dir=Path(args.cache_dir),
        batch_size=args.batch_size,
        device=args.device,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
