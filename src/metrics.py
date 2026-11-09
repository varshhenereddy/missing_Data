"""Image reconstruction metrics for RGB images normalized to [0, 1]."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity


def _validate_pair(
    reference: NDArray[np.generic], estimate: NDArray[np.generic]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference_array = np.asarray(reference, dtype=np.float64)
    estimate_array = np.asarray(estimate, dtype=np.float64)
    if reference_array.shape != estimate_array.shape or reference_array.ndim != 3:
        raise ValueError("Images must have identical HxWxC shapes")
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(estimate_array)):
        raise ValueError("Images must contain only finite values")
    if reference_array.min() < 0.0 or reference_array.max() > 1.0:
        raise ValueError("reference values must be normalized to [0, 1]")
    if estimate_array.min() < 0.0 or estimate_array.max() > 1.0:
        raise ValueError("estimate values must be normalized to [0, 1]")
    return reference_array, estimate_array


def mean_absolute_error(
    reference: NDArray[np.generic], estimate: NDArray[np.generic]
) -> float:
    """Mean absolute error over all channels and pixels."""
    reference_array, estimate_array = _validate_pair(reference, estimate)
    return float(np.mean(np.abs(reference_array - estimate_array)))


def peak_signal_to_noise_ratio(
    reference: NDArray[np.generic], estimate: NDArray[np.generic]
) -> float:
    """PSNR in dB for images with a unit dynamic range."""
    reference_array, estimate_array = _validate_pair(reference, estimate)
    mse = float(np.mean((reference_array - estimate_array) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def structural_similarity_index(
    reference: NDArray[np.generic], estimate: NDArray[np.generic]
) -> float:
    """Compute RGB SSIM with the [0, 1] dynamic range."""
    reference_array, estimate_array = _validate_pair(reference, estimate)
    shortest_side = min(reference_array.shape[:2])
    if shortest_side < 3:
        raise ValueError("SSIM requires images at least 3 pixels tall and wide")
    window_size = min(7, shortest_side if shortest_side % 2 else shortest_side - 1)
    return float(
        structural_similarity(
            reference_array,
            estimate_array,
            data_range=1.0,
            channel_axis=-1,
            win_size=window_size,
        )
    )


def evaluate(
    reference: NDArray[np.generic], estimate: NDArray[np.generic]
) -> dict[str, float]:
    """Return the requested metrics in a stable report order."""
    return {
        "PSNR": peak_signal_to_noise_ratio(reference, estimate),
        "SSIM": structural_similarity_index(reference, estimate),
        "MAE": mean_absolute_error(reference, estimate),
    }
