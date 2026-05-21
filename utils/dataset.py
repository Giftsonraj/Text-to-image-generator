"""
CUB-200 Birds dataset with text captions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "cub"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "dataset" / "cub_index.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data_cache" / "embeddings"


def _find_images_dir(cub_root: Path) -> Path | None:
    """Locate bird images under common CUB folder layouts."""
    candidates = [
        cub_root / "images",
        cub_root / "CUB_200_2011" / "images",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _find_text_dir(cub_root: Path) -> Path | None:
    """Locate caption text files."""
    candidates = [
        cub_root / "text",
        cub_root / "text_c10",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*.txt")):
            return path
    return None


def _load_captions(text_file: Path) -> list[str]:
    lines = text_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_cub_index(
    cub_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[dict]:
    """
    Scan CUB images and caption files, write JSON index.

    Expected layout after download:
      dataset/cub/images/<class>/<image>.jpg
      dataset/cub/text/<image_id>.txt   (10 captions per file)
    """
    cub_root = Path(cub_root or DEFAULT_DATASET_DIR)
    output_path = Path(output_path or DEFAULT_INDEX_PATH)
    images_dir = _find_images_dir(cub_root)
    text_dir = _find_text_dir(cub_root)

    if images_dir is None:
        raise FileNotFoundError(
            f"Images not found under {cub_root}. Run scripts/download_cub.py for setup steps."
        )
    if text_dir is None:
        raise FileNotFoundError(
            f"Caption text not found under {cub_root}. "
            "Download captions from https://github.com/reedscot/cvpr2016"
        )

    # Map stem -> relative image path
    image_map: dict[str, str] = {}
    for img_path in images_dir.rglob("*.jpg"):
        stem = img_path.stem
        rel = str(img_path.relative_to(cub_root)).replace("\\", "/")
        image_map[stem] = rel

    entries: list[dict] = []
    for text_file in sorted(text_dir.glob("*.txt")):
        stem = text_file.stem
        if stem not in image_map:
            continue
        captions = _load_captions(text_file)
        if not captions:
            continue
        entries.append(
            {
                "id": stem,
                "image_path": image_map[stem],
                "captions": captions,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    return entries


def load_cub_index(index_path: str | Path | None = None) -> list[dict]:
    index_path = Path(index_path or DEFAULT_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run: python scripts/download_cub.py"
        )
    with index_path.open(encoding="utf-8") as f:
        return json.load(f)


class CUBTextImageDataset(Dataset):
    """Image-caption pairs with optional precomputed text embeddings."""

    def __init__(
        self,
        cub_root: str | Path | None = None,
        index_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        image_size: int = 64,
        subset: int = 0,
        require_cache: bool = False,
    ):
        self.cub_root = Path(cub_root or DEFAULT_DATASET_DIR)
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.image_size = image_size
        self.require_cache = require_cache

        entries = load_cub_index(index_path)
        if subset > 0:
            entries = entries[:subset]
        self.entries = entries

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _cache_path(self, entry_id: str, caption_idx: int) -> Path:
        return self.cache_dir / f"{entry_id}_{caption_idx}.pt"

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        img_path = self.cub_root / entry["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_idx = random.randint(0, len(entry["captions"]) - 1)
        caption = entry["captions"][caption_idx]

        cache_path = self._cache_path(entry["id"], caption_idx)
        if cache_path.exists():
            embedding = torch.load(cache_path, map_location="cpu", weights_only=True)
        elif self.require_cache:
            raise FileNotFoundError(
                f"Missing cached embedding: {cache_path}. "
                "Run: python scripts/cache_embeddings.py"
            )
        else:
            embedding = torch.zeros(256)  # placeholder if cache not built yet

        return {
            "image": image,
            "caption": caption,
            "embedding": embedding.float(),
            "id": entry["id"],
            "caption_idx": caption_idx,
        }
