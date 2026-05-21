"""
Verify CUB dataset layout and build the JSON index.

CUB-200-2011 images: https://www.vision.caltech.edu/datasets/cub_200_2011/
Captions (cvpr2016): https://github.com/reedscot/cvpr2016
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import DEFAULT_DATASET_DIR, DEFAULT_INDEX_PATH, build_cub_index


def print_instructions() -> None:
    print(
        """
=== CUB Birds Dataset Setup ===

1. Download CUB-200-2011 images from:
   https://www.vision.caltech.edu/datasets/cub_200_2011/

2. Extract so images are at one of:
   dataset/cub/images/<class_name>/<image>.jpg
   dataset/cub/CUB_200_2011/images/...

3. Download text captions from:
   https://github.com/reedscot/cvpr2016
   Place .txt files (10 captions each) in:
   dataset/cub/text/

4. Re-run this script to build dataset/cub_index.json
"""
    )


def main() -> None:
    cub_root = DEFAULT_DATASET_DIR
    cub_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {cub_root}")
    print_instructions()

    try:
        entries = build_cub_index(cub_root=cub_root, output_path=DEFAULT_INDEX_PATH)
        print(f"\nSuccess: indexed {len(entries)} image-caption pairs.")
        print(f"Index saved to: {DEFAULT_INDEX_PATH}")
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
