"""
Step 1: Set up CUB text captions (and create folder layout).

What this does:
  1. Downloads ICML2016 bird captions (if missing) -> dataset/cub_captions.t7
  2. Extracts .txt files -> dataset/cub/text/
  3. Creates dataset/cub/images/ (you add CUB photos here in step 2)

Note: The GitHub folder cvpr2016-master is TRAINING CODE, not caption files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "dataset" / "cub_captions.t7"
TEXT_DIR = PROJECT_ROOT / "dataset" / "cub" / "text"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "cub" / "images"


def download_captions_archive() -> None:
    if ARCHIVE.exists():
        print(f"Captions archive already exists: {ARCHIVE}")
        return

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading bird captions from Google Drive (ICML2016, ~459 MB)...")
    subprocess.run(
        [sys.executable, "-m", "gdown", "0B0ywwgffWnLLLUc2WHYzM0Q2eWc", "-O", str(ARCHIVE)],
        check=True,
    )


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)

    download_captions_archive()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    print("\nExtracting captions to dataset/cub/text/ ...")
    from extract_icml_captions import extract_all

    n = extract_all(ARCHIVE, TEXT_DIR)
    print(f"Done: {n} caption .txt files in {TEXT_DIR}")

    print(
        f"""
Step 1 complete.

Next (Step 2 — you do this manually):
  1. Download CUB-200-2011 images from:
     https://www.vision.caltech.edu/datasets/cub_200_2011/
  2. Extract so photos are under:
     {IMAGES_DIR}\\<bird_class>\\<photo>.jpg
  3. Run:
     python scripts/setup_cub_step2.py

You can ignore the cvpr2016-master folder in the project root (it is not the caption data).
"""
    )


if __name__ == "__main__":
    main()
