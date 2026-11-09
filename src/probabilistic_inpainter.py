"""Patch-GMM probabilistic inpainting using damped mean-field inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


@dataclass
class _GaussianMixture:
    """A small full-covariance Gaussian mixture fitted by EM."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    covariances: NDArray[np.float64]

    @classmethod
    def fit(
        cls,
        samples: NDArray[np.float64],
        component_count: int,
        regularization: float,
        max_iterations: int,
        seed: int,
    ) -> _GaussianMixture:
        sample_count, dimension = samples.shape
        rng = np.random.default_rng(seed)
        if sample_count > 10_000:
            samples = samples[rng.choice(sample_count, size=10_000, replace=False)]
            sample_count = samples.shape[0]
        count = min(component_count, sample_count)
        means = samples[rng.choice(sample_count, size=count, replace=False)].copy()

        # A few Lloyd steps give EM better-separated starting component means.
        for _ in range(5):
            distances = np.sum((samples[:, None, :] - means[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            for component in range(count):
                members = samples[labels == component]
                if members.size:
                    means[component] = members.mean(axis=0)

        centered = samples - samples.mean(axis=0)
        global_covariance = centered.T @ centered / max(sample_count, 1)
        global_covariance += regularization * np.eye(dimension)
        covariances = np.repeat(global_covariance[None, :, :], count, axis=0)
        weights = np.full(count, 1.0 / count, dtype=np.float64)

        for _ in range(max_iterations):
            log_joint = np.column_stack(
                [
                    np.log(max(weights[k], 1e-12))
                    + _log_gaussian_density(samples, means[k], covariances[k])
                    for k in range(count)
                ]
            )
            log_evidence = _logsumexp(log_joint, axis=1)
            responsibilities = np.exp(log_joint - log_evidence[:, None])
            effective_counts = responsibilities.sum(axis=0)

            for component in range(count):
                effective_count = effective_counts[component]
                if effective_count < 1e-8:
                    means[component] = samples[int(rng.integers(sample_count))]
                    covariances[component] = global_covariance
                    weights[component] = 1.0 / sample_count
                    continue
                weights[component] = effective_count / sample_count
                means[component] = (
                    responsibilities[:, component] @ samples / effective_count
                )
                residuals = samples - means[component]
                weighted_residuals = residuals * responsibilities[:, component, None]
                covariance = weighted_residuals.T @ residuals / effective_count
                covariances[component] = (
                    covariance + regularization * np.eye(dimension)
                )
            weights /= weights.sum()

        return cls(weights=weights, means=means, covariances=covariances)


def _log_gaussian_density(
    samples: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate log N(x | mean, covariance) using a stable Cholesky solve."""
    factor = np.linalg.cholesky(covariance)
    residuals = (samples - mean).T
    whitened = np.linalg.solve(factor, residuals)
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    return -0.5 * (
        samples.shape[1] * np.log(2.0 * np.pi)
        + log_determinant
        + np.sum(whitened**2, axis=0)
    )


def _logsumexp(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


class ProbabilisticInpainter:
    """Infer missing color values using a GMM prior over local image patches.

    The mixture is learned with EM from fully observed patches in the input
    image. At inference time, each missing pixel is updated from the
    conditional Gaussian-mixture posterior of its neighborhood. Damped,
    simultaneous updates form a stable mean-field approximation to the
    posterior over the entire image.
    """

    def __init__(
        self,
        patch_size: int = 3,
        components: int = 4,
        em_iterations: int = 20,
        inference_iterations: int = 8,
        regularization: float = 1e-4,
        damping: float = 0.7,
        seed: int = 0,
    ) -> None:
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 3")
        if components < 1 or em_iterations < 1 or inference_iterations < 1:
            raise ValueError("component and iteration counts must be positive")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in the interval (0, 1]")
        self.patch_size = patch_size
        self.components = components
        self.em_iterations = em_iterations
        self.inference_iterations = inference_iterations
        self.regularization = regularization
        self.damping = damping
        self.seed = seed

    def inpaint(
        self, image: NDArray[np.generic], missing_mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """Return a float32 RGB image in [0, 1] with observed pixels preserved."""
        normalized = np.asarray(image, dtype=np.float32)
        mask = np.asarray(missing_mask, dtype=bool)
        if normalized.ndim != 3 or normalized.shape[2] != 3:
            raise ValueError("image must have shape HxWx3 (RGB)")
        if mask.shape != normalized.shape[:2]:
            raise ValueError("missing_mask must have the image's HxW shape")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("image values must be finite")
        if normalized.min() < 0.0 or normalized.max() > 1.0:
            raise ValueError("image values must be normalized to [0, 1]")
        if not np.any(mask):
            return normalized.copy()
        if np.all(mask):
            raise ValueError("at least one known pixel is required")

        height, width, channels = normalized.shape
        radius = self.patch_size // 2
        padded_image = np.pad(
            normalized, ((radius, radius), (radius, radius), (0, 0)), mode="reflect"
        )
        padded_mask = np.pad(
            mask,
            ((radius, radius), (radius, radius)),
            mode="reflect",
        )
        patch_values = _extract_patches(padded_image, height, width, self.patch_size)
        patch_missing = _extract_mask_patches(
            padded_mask, height, width, self.patch_size
        )
        complete_patches = patch_values.reshape(-1, patch_values.shape[-1])[
            ~np.any(patch_missing, axis=-1).reshape(-1)
        ]
        if complete_patches.shape[0] == 0:
            raise ValueError(
                "No fully observed patches are available to learn the image prior; "
                "use a smaller mask or a larger source image."
            )

        mixture = _GaussianMixture.fit(
            complete_patches.astype(np.float64),
            component_count=self.components,
            regularization=self.regularization,
            max_iterations=self.em_iterations,
            seed=self.seed,
        )
        estimate = normalized.copy()
        nearest_known = distance_transform_edt(mask, return_distances=False, return_indices=True)
        estimate[mask] = normalized[nearest_known[0][mask], nearest_known[1][mask]]

        center = (self.patch_size * self.patch_size // 2) * channels
        target_indices = np.arange(center, center + channels)
        observed_indices = np.concatenate(
            (np.arange(center), np.arange(center + channels, complete_patches.shape[1]))
        )
        conditional_parameters = _prepare_conditionals(
            mixture,
            target_indices,
            observed_indices,
            self.regularization,
        )

        for _ in range(self.inference_iterations):
            padded_estimate = np.pad(
                estimate,
                ((radius, radius), (radius, radius), (0, 0)),
                mode="reflect",
            )
            current_patches = _extract_patches(
                padded_estimate, height, width, self.patch_size
            )
            predictions = _conditional_mixture_means(
                current_patches[mask].astype(np.float64),
                conditional_parameters,
                observed_indices,
            )
            previous = estimate[mask].copy()
            estimate[mask] = (
                (1.0 - self.damping) * previous + self.damping * predictions
            ).astype(np.float32)
            if np.max(np.abs(estimate[mask] - previous)) < 1e-4:
                break

        estimate[~mask] = normalized[~mask]
        return np.clip(estimate, 0.0, 1.0).astype(np.float32)


def _extract_patches(
    padded_image: NDArray[np.generic], height: int, width: int, patch_size: int
) -> NDArray[np.generic]:
    """Flatten each HxWxC neighborhood in row/column/channel order."""
    radius = patch_size // 2
    patches = np.stack(
        [
            padded_image[row : row + height, column : column + width, :]
            for row in range(patch_size)
            for column in range(patch_size)
        ],
        axis=2,
    )
    return patches.reshape(height, width, -1)


def _extract_mask_patches(
    padded_mask: NDArray[np.bool_], height: int, width: int, patch_size: int
) -> NDArray[np.bool_]:
    return np.stack(
        [
            padded_mask[row : row + height, column : column + width]
            for row in range(patch_size)
            for column in range(patch_size)
        ],
        axis=2,
    ).reshape(height, width, -1)


def _conditional_mixture_means(
    patches: NDArray[np.float64],
    parameters: list[
        tuple[
            float,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ],
    observed_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute E[target | neighborhood] under the conditional GMM posterior."""
    if patches.shape[0] == 0:
        return np.empty((0, parameters[0][1].size), dtype=np.float64)

    log_probabilities = []
    conditional_means = []
    observed_values = patches[:, observed_indices]
    for log_normalizer, target_mean, observed_mean, coefficient, precision in parameters:
        residuals = observed_values - observed_mean
        conditional_mean = target_mean + residuals @ coefficient.T
        mahalanobis = np.sum((residuals @ precision) * residuals, axis=1)
        log_probabilities.append(log_normalizer - 0.5 * mahalanobis)
        conditional_means.append(conditional_mean)

    component_log_probabilities = np.column_stack(log_probabilities)
    posterior = np.exp(
        component_log_probabilities
        - _logsumexp(component_log_probabilities, axis=1)[:, None]
    )
    return np.einsum("nk,nkc->nc", posterior, np.stack(conditional_means, axis=1))


def _prepare_conditionals(
    mixture: _GaussianMixture,
    target_indices: NDArray[np.int64],
    observed_indices: NDArray[np.int64],
    regularization: float,
) -> list[
    tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
]:
    """Precompute each component's Gaussian conditional and evidence precision."""
    parameters = []
    for weight, mean, covariance in zip(
        mixture.weights, mixture.means, mixture.covariances
    ):
        covariance_oo = covariance[np.ix_(observed_indices, observed_indices)]
        covariance_oo = covariance_oo + regularization * np.eye(observed_indices.size)
        covariance_to = covariance[np.ix_(target_indices, observed_indices)]
        factor = np.linalg.cholesky(covariance_oo)
        solved_cross = np.linalg.solve(
            factor.T, np.linalg.solve(factor, covariance_to.T)
        )
        precision = np.linalg.solve(
            factor.T,
            np.linalg.solve(factor, np.eye(observed_indices.size)),
        )
        log_determinant = 2.0 * np.log(np.diag(factor)).sum()
        log_normalizer = np.log(max(weight, 1e-12)) - 0.5 * (
            observed_indices.size * np.log(2.0 * np.pi) + log_determinant
        )
        parameters.append(
            (
                float(log_normalizer),
                mean[target_indices],
                mean[observed_indices],
                solved_cross.T,
                precision,
            )
        )
    return parameters
