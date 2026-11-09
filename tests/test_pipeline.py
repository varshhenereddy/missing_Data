"""Focused tests for synthetic masks, probabilistic restoration, and metrics."""

import unittest

import numpy as np

from src.metrics import evaluate
from src.probabilistic_inpainter import ProbabilisticInpainter
from src.synthetic_corruptor import apply_mask, generate_mask


class InpaintingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        height, width = 20, 20
        y, x = np.mgrid[:height, :width]
        self.image = np.stack(
            (x / (width - 1), y / (height - 1), (x + y) / (height + width - 2)),
            axis=-1,
        ).astype(np.float32)

    def test_mask_types_are_reproducible_and_nonempty(self) -> None:
        for kind in ("noise", "block", "irregular"):
            first = generate_mask(self.image.shape, kind, amount=0.15, seed=4)
            second = generate_mask(self.image.shape, kind, amount=0.15, seed=4)
            self.assertEqual(first.shape, self.image.shape[:2])
            self.assertTrue(np.array_equal(first, second))
            self.assertTrue(first.any())

    def test_probabilistic_inpainting_preserves_observations(self) -> None:
        mask = generate_mask(self.image.shape, "block", amount=0.1, seed=3)
        corrupted = apply_mask(self.image, mask)
        restored = ProbabilisticInpainter(
            components=2,
            em_iterations=4,
            inference_iterations=2,
            seed=3,
        ).inpaint(corrupted, mask)

        self.assertEqual(restored.shape, self.image.shape)
        self.assertTrue(np.isfinite(restored).all())
        self.assertGreaterEqual(float(restored.min()), 0.0)
        self.assertLessEqual(float(restored.max()), 1.0)
        self.assertTrue(np.array_equal(restored[~mask], self.image[~mask]))

    def test_metrics_return_expected_fields(self) -> None:
        scores = evaluate(self.image, self.image.copy())
        self.assertEqual(set(scores), {"PSNR", "SSIM", "MAE"})
        self.assertEqual(scores["PSNR"], float("inf"))
        self.assertEqual(scores["SSIM"], 1.0)
        self.assertEqual(scores["MAE"], 0.0)


if __name__ == "__main__":
    unittest.main()
