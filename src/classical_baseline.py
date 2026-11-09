"""OpenCV Telea and Navier-Stokes inpainting benchmark baselines."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def inpaint_method(
    image_rgb: NDArray[np.generic],
    missing_mask: NDArray[np.bool_],
    method_name: str,
    radius: float = 3.0,
) -> NDArray[np.float32]:
    """Run one named OpenCV method, returning an RGB image in [0, 1]."""
    methods = {
        "Telea": cv2.INPAINT_TELEA,
        "Navier-Stokes": cv2.INPAINT_NS,
    }
    if method_name not in methods:
        raise ValueError(f"Unknown inpainting method: {method_name}")
    image = np.asarray(image_rgb, dtype=np.float32)
    mask = np.asarray(missing_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Expected an HxWx3 image and a matching HxW mask")
    if not 0.0 < radius:
        raise ValueError("radius must be positive")

    image_bgr = cv2.cvtColor(
        np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    restored_bgr = cv2.inpaint(
        image_bgr, mask.astype(np.uint8) * 255, radius, methods[method_name]
    )
    return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def inpaint_classical(
    image_rgb: NDArray[np.generic],
    missing_mask: NDArray[np.bool_],
    radius: float = 3.0,
) -> dict[str, NDArray[np.float32]]:
    """Run both OpenCV classical algorithms on an RGB [0, 1] image."""
    return {
        name: inpaint_method(image_rgb, missing_mask, name, radius)
        for name in ("Telea", "Navier-Stokes")
    }
