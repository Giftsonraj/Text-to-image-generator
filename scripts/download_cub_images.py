"""
Download and extract CUB-200-2011 images (~1.1 GB).
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "dataset" / "CUB_200_2011.tgz"
CUB_ROOT = PROJECT_ROOT / "dataset" / "cub"

URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url}")
    print(f"Saving to {dest} (this may take several minutes)...")

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 / total_size)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\nDownload complete.")


def extract(archive: Path, dest: Path) -> None:
    print(f"Extracting {archive} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest.parent)
    print(f"Extracted under {dest.parent}")


def images_ready(cub_root: Path) -> bool:
    for candidate in [
        cub_root / "images",
        cub_root / "CUB_200_2011" / "images",
        cub_root.parent / "CUB_200_2011" / "images",
    ]:
        if candidate.is_dir() and any(candidate.rglob("*.jpg")):
            return True
    return False


def main() -> None:
    if images_ready(CUB_ROOT):
        print("CUB images already present. Skipping download.")
        return

    if not ARCHIVE.exists():
        last_err = None
        for url in URLS:
            try:
                download(url, ARCHIVE)
                break
            except Exception as exc:
                last_err = exc
                print(f"Failed: {exc}")
        else:
            raise RuntimeError(f"All download URLs failed. Last error: {last_err}")

    extract(ARCHIVE, CUB_ROOT)

    if images_ready(CUB_ROOT):
        print("CUB images ready.")
    else:
        print("Extract finished but images/ not found. Check dataset/ folder layout.")
        sys.exit(1)


if __name__ == "__main__":
    main()
