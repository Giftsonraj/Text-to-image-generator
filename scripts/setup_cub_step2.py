"""
Step 2: Verify CUB images + captions and build dataset/cub_index.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import DEFAULT_DATASET_DIR, DEFAULT_INDEX_PATH, build_cub_index


def main() -> None:
    cub_root = DEFAULT_DATASET_DIR
    images_dir = cub_root / "images"
    text_dir = cub_root / "text"

    print(f"Dataset root: {cub_root}")
    print(f"  images: {images_dir} ({'found' if images_dir.is_dir() else 'MISSING'})")
    print(f"  text:   {text_dir} ({'found' if text_dir.is_dir() else 'MISSING'})")

    if not images_dir.is_dir():
        print(
            """
ERROR: CUB images not found.

Download CUB-200-2011 from:
  https://www.vision.caltech.edu/datasets/cub_200_2011/

Extract the zip so you have paths like:
  dataset/cub/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg

Or extract the full CUB_200_2011 folder under dataset/cub/ (the script accepts that layout too).
"""
        )
        sys.exit(1)

    n_txt = len(list(text_dir.glob("*.txt"))) if text_dir.is_dir() else 0
    if n_txt == 0:
        print("ERROR: No caption .txt files. Run first: python scripts/setup_cub_step1.py")
        sys.exit(1)

    entries = build_cub_index(cub_root=cub_root, output_path=DEFAULT_INDEX_PATH)
    print(f"\nSuccess: indexed {len(entries)} image-caption pairs.")
    print(f"Index saved to: {DEFAULT_INDEX_PATH}")
    print("\nStep 2 complete. You can now run step 3:")
    print("  python scripts/cache_embeddings.py")


if __name__ == "__main__":
    main()
