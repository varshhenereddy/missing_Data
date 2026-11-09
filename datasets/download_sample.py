"""Download CIFAR-10 and save one RGB sample as a PNG image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from torchvision.datasets import CIFAR10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/data"))
    parser.add_argument("--output", type=Path, default=Path("datasets/sample.png"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = CIFAR10(root=str(args.root), train=False, download=True)
    if not 0 <= args.index < len(dataset):
        parser.error(f"--index must be between 0 and {len(dataset) - 1}")
    image, label = dataset[args.index]
    rgb = np.asarray(image.convert("RGB"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"Could not write sample image to {args.output}")
    print(f"Saved CIFAR-10 sample (label {label}) to {args.output}")


if __name__ == "__main__":
    main()
