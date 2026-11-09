"""Compare classical and probabilistic image inpainting methods."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.classical_baseline import inpaint_method
from src.metrics import evaluate
from src.probabilistic_inpainter import ProbabilisticInpainter
from src.synthetic_corruptor import apply_mask, generate_mask


def _read_rgb(path: Path) -> NDArray[np.float32]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def _print_report(
    results: dict[str, NDArray[np.float32]],
    original: NDArray[np.float32],
    elapsed: dict[str, float],
) -> None:
    headers = ("Method", "PSNR (dB)", "SSIM", "MAE", "Time (s)")
    rows = []
    for name, image in results.items():
        scores = evaluate(original, image)
        rows.append(
            (
                name,
                f"{scores['PSNR']:.3f}",
                f"{scores['SSIM']:.4f}",
                f"{scores['MAE']:.5f}",
                f"{elapsed[name]:.3f}",
            )
        )
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print("Metrics are computed over the complete image.")


def _show_comparison(
    images: list[tuple[str, NDArray[np.float32]]],
    mask: NDArray[np.bool_],
    save_path: Path | None,
    display: bool,
) -> None:
    figure, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for axis, (title, image) in zip(axes, images):
        shown = image.copy()
        if title == "Corrupted":
            shown[mask] = 0.0
        axis.imshow(shown)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison figure to {save_path}")
    if display:
        plt.show()
    else:
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to an RGB image")
    parser.add_argument(
        "--mask", choices=("noise", "block", "irregular"), default="block"
    )
    parser.add_argument("--amount", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--save-figure", type=Path)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    try:
        original = _read_rgb(args.image)
        mask = generate_mask(original.shape, args.mask, args.amount, args.seed)
        corrupted = apply_mask(original, mask).astype(np.float32)

        results: dict[str, NDArray[np.float32]] = {}
        elapsed: dict[str, float] = {}
        for name in ("Telea", "Navier-Stokes"):
            start = time.perf_counter()
            results[name] = inpaint_method(
                corrupted, mask, name, radius=args.radius
            )
            elapsed[name] = time.perf_counter() - start

        probabilistic = ProbabilisticInpainter(
            patch_size=args.patch_size,
            components=args.components,
            inference_iterations=args.iterations,
            seed=args.seed,
        )
        start = time.perf_counter()
        results["Probabilistic GMM"] = probabilistic.inpaint(corrupted, mask)
        elapsed["Probabilistic GMM"] = time.perf_counter() - start

        _print_report(results, original, elapsed)
        _show_comparison(
            [
                ("Original", original),
                ("Corrupted", corrupted),
                ("Telea", results["Telea"]),
                ("Navier-Stokes", results["Navier-Stokes"]),
                ("Probabilistic GMM", results["Probabilistic GMM"]),
            ],
            mask,
            args.save_figure,
            display=not args.no_display,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
