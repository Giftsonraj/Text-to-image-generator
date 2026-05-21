"""
Download CUB-200-2011 images from Hugging Face into dataset/cub/images/.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_OUT = PROJECT_ROOT / "dataset" / "cub" / "images"


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading CUB-200-2011 from Hugging Face...")
    ds = load_dataset("birder-project/CUB_200_2011", split="train")

    n = len(ds)
    print(f"Exporting {n} images to {IMAGES_OUT}")

    for i, row in enumerate(ds):
        key = row["__key__"]  # e.g. training/064.Ring_billed_Gull/Ring_Billed_Gull_0028_51454
        parts = Path(str(key).replace("\\", "/")).parts
        class_name = parts[-2] if len(parts) >= 2 else "unknown"
        stem = parts[-1]
        out_path = IMAGES_OUT / class_name / f"{stem}.jpg"
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            row["jpg"].convert("RGB").save(out_path, quality=95)

        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}")

    print("Done.")


if __name__ == "__main__":
    main()
