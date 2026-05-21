"""
Quick demo training on a CPU-friendly subset (5 epochs, 500 images).
Produces checkpoints/demo_cgan_64.pt for the Streamlit app.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    train_script = PROJECT_ROOT / "train.py"
    checkpoint = PROJECT_ROOT / "checkpoints" / "demo_cgan_64.pt"

    cmd = [
        sys.executable,
        str(train_script),
        "--epochs",
        "50",
        "--loss",
        "bce",
        "--fm-weight",
        "1.0",
        "--subset",
        "2000",
        "--g-steps",
        "2",
        "--lr-d",
        "5e-5",
        "--lr-g",
        "2e-4",
        "--instance-noise",
        "0.05",
        "--batch-size",
        "8",
        "--device",
        "cpu",
        "--save",
        str(checkpoint),
        "--overwrite-logs",
    ]

    print("Quick demo training (CPU, 50 epochs, 2000 images, fixed generator)")
    print("Expected runtime: ~45-90 minutes depending on your CPU.")
    print("Command:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
