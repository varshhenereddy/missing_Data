"""Synthetic spatial masks for evaluating image inpainting methods."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def generate_mask(
    image_shape: tuple[int, ...],
    kind: str = "block",
    amount: float = 0.25,
    seed: int | None = None,
) -> NDArray[np.bool_]:
    """Return a boolean HxW mask in which True denotes a missing pixel.

    ``amount`` is the dropout fraction for noise, approximate area fraction
    for a block, and a scale controlling stroke count for irregular masks.
    """
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = image_shape[:2]
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < amount <= 1.0:
        raise ValueError("amount must be in the interval (0, 1]")
    if kind not in {"noise", "block", "irregular"}:
        raise ValueError("kind must be 'noise', 'block', or 'irregular'")

    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.uint8)

    if kind == "noise":
        mask = (rng.random((height, width)) < amount).astype(np.uint8)
    elif kind == "block":
        side_scale = np.sqrt(amount)
        block_h = min(height, max(1, int(round(height * side_scale))))
        block_w = min(width, max(1, int(round(width * side_scale))))
        top = int(rng.integers(0, height - block_h + 1))
        left = int(rng.integers(0, width - block_w + 1))
        mask[top : top + block_h, left : left + block_w] = 1
    else:
        stroke_count = max(1, int(round(amount * 20)))
        thickness = max(1, int(round(min(height, width) * 0.02)))
        for _ in range(stroke_count):
            point = (
                int(rng.integers(0, width)),
                int(rng.integers(0, height)),
            )
            segment_count = int(rng.integers(2, 6))
            for _ in range(segment_count):
                angle = float(rng.uniform(0, 2 * np.pi))
                length = float(rng.uniform(0.04, 0.2) * min(height, width))
                next_point = (
                    int(np.clip(point[0] + np.cos(angle) * length, 0, width - 1)),
                    int(np.clip(point[1] + np.sin(angle) * length, 0, height - 1)),
                )
                cv2.line(mask, point, next_point, color=1, thickness=thickness)
                point = next_point

    if not np.any(mask):
        mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = 1
    return mask.astype(bool)


def apply_mask(
    image: NDArray[np.generic], mask: NDArray[np.bool_]
) -> NDArray[np.generic]:
    """Replace masked pixels with zero while preserving the image dtype."""
    if image.ndim != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Expected an HxWxC image and a matching HxW mask")
    corrupted = image.copy()
    corrupted[mask] = 0
    return corrupted
