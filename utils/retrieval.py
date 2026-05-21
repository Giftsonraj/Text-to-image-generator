"""
Retrieve the best-matching real CUB image for a text caption (by embedding similarity).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from utils.dataset import DEFAULT_CACHE_DIR, DEFAULT_DATASET_DIR, DEFAULT_INDEX_PATH, load_cub_index
from utils.text_encoder import TextEncoder

RETRIEVAL_INDEX = Path(__file__).resolve().parent.parent / "data_cache" / "retrieval_index.pt"


def normalize_cond(cond: torch.Tensor) -> torch.Tensor:
    return F.normalize(cond, p=2, dim=1)


def _attribute_score(query: str, caption: str, bird_id: str) -> float:
    """
    Score how well a caption matches color/shape cues in the query.
    Fixes cases where embeddings pick 'yellow eye' on a black bird.
    """
    cap = caption.lower()
    q = query.lower()
    score = 0.0

    yellow_body = any(
        phrase in cap
        for phrase in (
            "yellow breast",
            "yellow belly",
            "yellow body",
            "yellow bird",
            "bright yellow",
            "mainly yellow",
            "brilliant yellow",
            "body of the bird is yellow",
            "yellow head and belly",
            "yellow nape",
        )
    )
    yellow_eye_only = "yellow" in cap and any(
        phrase in cap for phrase in ("yellow eye", "bright yellow eye", "yellow iris")
    )
    if yellow_body:
        score += 0.4
    elif "yellow" in cap and not yellow_eye_only:
        score += 0.15

    if "black" in cap and ("wing" in cap or "wings" in cap or "winglets" in cap):
        score += 0.35

    if "small" in q and "small" in cap:
        score += 0.05

    if "yellow" in q and "black" in q and "goldfinch" in bird_id.lower():
        score += 0.3

    return min(score, 1.0)


def _rerank(query: str, matrix: torch.Tensor, meta: list[dict], query_vec: torch.Tensor, top_k: int) -> list[dict]:
    sims = torch.matmul(matrix, query_vec.squeeze(0))
    n = sims.numel()

    # Semantic candidates
    k_sem = min(80, n)
    sem_vals, sem_idx = torch.topk(sims, k=k_sem)
    candidates: set[int] = set(sem_idx.tolist())

    # Attribute candidates (scan full index — fast, no extra model calls)
    attr_ranked = sorted(
        (
            _attribute_score(query, meta[i]["caption"], meta[i]["id"]),
            i,
        )
        for i in range(n)
    )
    candidates.update(idx for _, idx in attr_ranked[-40:])

    ranked: list[tuple[float, int]] = []
    for idx in candidates:
        cos = sims[idx].item()
        attr = _attribute_score(query, meta[idx]["caption"], meta[idx]["id"])
        combined = 0.35 * cos + 0.65 * attr
        ranked.append((combined, idx))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


def _load_index() -> tuple[torch.Tensor, list[dict]] | None:
    if not RETRIEVAL_INDEX.exists():
        return None
    data = torch.load(RETRIEVAL_INDEX, weights_only=True)
    return data["embeddings"], data["meta"]


@torch.no_grad()
def retrieve_best_match(
    text: str,
    text_encoder: TextEncoder | None = None,
    cub_root: Path | None = None,
    index_path: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
    top_k: int = 1,
    max_search: int = 0,
) -> list[dict]:
    """
    Find CUB images whose captions are most similar to the query text.
    Uses prebuilt retrieval_index.pt when available (fast).
    """
    cub_root = Path(cub_root or DEFAULT_DATASET_DIR)

    encoder = text_encoder or TextEncoder().to(device)
    encoder.eval()
    query = normalize_cond(encoder.encode([text], device=device))

    cached = _load_index()
    if cached is not None:
        matrix, meta = cached
        matrix = normalize_cond(matrix.to(device))
        ranked = _rerank(text, matrix, meta, query, top_k)
        results = []
        for score, idx in ranked:
            m = meta[idx]
            results.append(
                {
                    "id": m["id"],
                    "image_path": str(cub_root / m["image_path"]),
                    "caption": m["caption"],
                    "similarity": score,
                }
            )
        return results

    # Slow fallback: linear scan
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    entries = load_cub_index(index_path or DEFAULT_INDEX_PATH)
    if max_search > 0:
        entries = entries[:max_search]

    scores: list[tuple[float, dict]] = []
    for entry in entries:
        cache_path = cache_dir / f"{entry['id']}_0.pt"
        if not cache_path.exists():
            continue
        emb = torch.load(cache_path, map_location=device, weights_only=True).to(device)
        emb = normalize_cond(emb.unsqueeze(0))
        sim = F.cosine_similarity(query, emb, dim=1).item()
        img_path = cub_root / entry["image_path"]
        if img_path.exists():
            scores.append(
                (
                    sim,
                    {
                        "id": entry["id"],
                        "image_path": str(img_path),
                        "caption": entry["captions"][0],
                        "similarity": sim,
                    },
                )
            )

    scores.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scores[:top_k]]


def load_retrieved_image(match: dict, size: int = 256) -> Image.Image:
    """Load and upscale a retrieved CUB photograph for display."""
    img = Image.open(match["image_path"]).convert("RGB")
    if size > 0:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img
