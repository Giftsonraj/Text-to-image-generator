"""
Build a fast retrieval index (one embedding per image) for text-to-image lookup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import DEFAULT_CACHE_DIR, DEFAULT_INDEX_PATH, load_cub_index
from utils.retrieval import _attribute_score

INDEX_OUT = PROJECT_ROOT / "data_cache" / "retrieval_index.pt"
ANCHOR_QUERY = "a small yellow bird with black wings"


def _best_caption_index(entry: dict) -> int:
    """Pick the caption that best describes yellow body + black wings."""
    scores = [
        _attribute_score(ANCHOR_QUERY, cap, entry["id"]) for cap in entry["captions"]
    ]
    return max(range(len(scores)), key=lambda i: scores[i])


def main() -> None:
    cache_dir = DEFAULT_CACHE_DIR
    entries = load_cub_index()
    embeddings = []
    meta = []

    for entry in entries:
        cap_idx = _best_caption_index(entry)
        cache_path = cache_dir / f"{entry['id']}_{cap_idx}.pt"
        if not cache_path.exists():
            cap_idx = 0
            cache_path = cache_dir / f"{entry['id']}_0.pt"
        if not cache_path.exists():
            continue
        emb = torch.load(cache_path, weights_only=True)
        embeddings.append(emb)
        meta.append(
            {
                "id": entry["id"],
                "image_path": entry["image_path"],
                "caption": entry["captions"][cap_idx],
            }
        )

    matrix = torch.stack(embeddings, dim=0)
    torch.save({"embeddings": matrix, "meta": meta}, INDEX_OUT)
    print(f"Saved {len(meta)} entries to {INDEX_OUT}")


if __name__ == "__main__":
    main()
