"""
Extract plain-text captions from the ICML2016 birds archive (cub_captions.t7).

The file is a gzip-compressed tar of per-image .t7 caption files.
Run setup_cub_step1.py first to download it.
"""

from __future__ import annotations

import gzip
import io
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE = PROJECT_ROOT / "dataset" / "cub_captions.t7"
TEXT_OUT = PROJECT_ROOT / "dataset" / "cub" / "text"

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "


def _read_q(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack("q", data[offset : offset + 8])[0], offset + 8


def _read_i(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack("i", data[offset : offset + 4])[0], offset + 4


def parse_char_tensor(raw: bytes) -> np.ndarray | None:
    """Parse torch.ByteTensor with shape (seq_len, num_captions) from a .t7 file."""
    marker = b"torch.ByteTensor"
    idx = raw.find(marker)
    if idx < 0:
        return None
    pos = idx + len(marker)

    ndim, pos = _read_i(raw, pos)
    dims = []
    for _ in range(ndim):
        d, pos = _read_q(raw, pos)
        dims.append(d)
    for _ in range(ndim):
        _, pos = _read_q(raw, pos)
    storage_offset, pos = _read_q(raw, pos)
    storage_offset -= 1

    storage_marker = b"torch.ByteStorage"
    sidx = raw.find(storage_marker)
    if sidx < 0:
        return None
    pos = sidx + len(storage_marker)

    # Skip optional version / class name string objects (type 2)
    while pos + 8 < len(raw):
        t, pos2 = _read_i(raw, pos)
        if t == 2:
            sz, pos2 = _read_i(raw, pos2)
            pos = pos2 + sz
            continue
        if t == 4:
            pos = pos2 + 4  # skip TORCH index
            continue
        break

    size, pos = _read_q(raw, pos)
    storage = np.frombuffer(raw[pos : pos + size], dtype=np.uint8)
    if storage_offset + int(np.prod(dims)) > len(storage):
        return None
    return storage[storage_offset : storage_offset + int(np.prod(dims))].reshape(dims)


def decode_captions(arr: np.ndarray) -> list[str]:
    """Decode (seq_len, 10) index tensor to caption strings."""
    captions: list[str] = []
    ncol = arr.shape[1] if arr.ndim == 2 else 1
    for c in range(min(10, ncol)):
        chars = []
        for j in range(arr.shape[0]):
            idx = int(arr[j, c]) if arr.ndim == 2 else int(arr[j])
            if idx == 0:
                break
            if 1 <= idx <= len(ALPHABET):
                chars.append(ALPHABET[idx - 1])
        text = "".join(chars).strip()
        if text:
            captions.append(text)
    return captions


def extract_all(archive: Path = ARCHIVE, out_dir: Path = TEXT_OUT, limit: int = 0) -> int:
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing {archive}. Run: python scripts/setup_cub_step1.py"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(archive, "rb") as gz:
        tar_bytes = gz.read()

    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes))
    members = [m for m in tar.getmembers() if m.name.endswith(".t7") and m.isfile()]
    if limit > 0:
        members = members[:limit]

    count = 0
    for member in tqdm(members, desc="Extracting captions"):
        stem = Path(member.name).stem
        raw = tar.extractfile(member).read()
        tensor = parse_char_tensor(raw)
        if tensor is None:
            continue
        captions = decode_captions(tensor)
        if not captions:
            continue
        (out_dir / f"{stem}.txt").write_text("\n".join(captions), encoding="utf-8")
        count += 1

    return count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract CUB captions to .txt files")
    parser.add_argument("--archive", type=str, default=str(ARCHIVE))
    parser.add_argument("--out", type=str, default=str(TEXT_OUT))
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    n = extract_all(Path(args.archive), Path(args.out), args.limit)
    print(f"Wrote {n} caption files to {args.out}")


if __name__ == "__main__":
    main()
