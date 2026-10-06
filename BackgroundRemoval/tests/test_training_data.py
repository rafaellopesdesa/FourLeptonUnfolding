from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import uproot

from BackgroundRemoval.common import FEATURES
from BackgroundRemoval.Training.train_background_ratio import (
    MemberResult,
    _bad_members,
    _normalization_bias,
    _selected_chunks,
    fit_logit_calibrator,
)


def _write_sample(
    path: Path,
    *,
    weights: np.ndarray,
    reconstructed: np.ndarray,
    fiducial: np.ndarray,
) -> None:
    size = weights.size
    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight_nominal_pb": weights.astype(np.float64),
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "reconstructed": reconstructed.astype(np.bool_),
        "fiducial": fiducial.astype(np.bool_),
    }
    for index, name in enumerate(FEATURES):
        arrays[name] = np.linspace(index, index + 0.5, size, dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


class TrainingDataTest(unittest.TestCase):
    def test_common_score_saturation_is_not_treated_as_a_failed_member(self):
        def result(slot: int, loss: float) -> MemberResult:
            return MemberResult(
                slot=slot,
                attempt=0,
                seed=slot,
                state_dict={},
                history={},
                validation_loss=loss,
                validation_score_mean=0.5,
                validation_score_std=0.5,
                validation_saturated_fraction=1.0,
            )

        saturated_members = [result(slot, 0.2) for slot in range(4)]
        self.assertEqual(_bad_members(saturated_members), [])

        with_loss_outlier = [
            result(0, 0.2),
            result(1, 0.2),
            result(2, 0.2),
            result(3, 0.4),
        ]
        self.assertEqual(_bad_members(with_loss_outlier), [3])

    def test_exact_signal_and_combined_background_masks(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                gg_h,
                weights=np.array([2.0, 3.0, 5.0, 0.0]),
                reconstructed=np.array([True, True, False, True]),
                fiducial=np.array([True, False, True, True]),
            )
            _write_sample(
                zz,
                weights=np.array([7.0, 11.0]),
                reconstructed=np.array([True, False]),
                fiducial=np.array([True, True]),
            )
            chunks = list(
                _selected_chunks(
                    gg_h,
                    zz,
                    weight_branch="weight_nominal_pb",
                    split_seed=4,
                    expected_luminosity_fb=312.0,
                    step_size="1 MB",
                )
            )
            by_component = {chunk.component_name: chunk for chunk in chunks}
            self.assertEqual(
                by_component["gg_H_reconstructed_and_fiducial"].weights.tolist(),
                [2.0],
            )
            self.assertEqual(
                by_component["gg_H_reconstructed_and_fiducial"].zero_count, 1
            )
            self.assertEqual(
                by_component["gg_H_reconstructed_not_fiducial"].weights.tolist(),
                [3.0],
            )
            self.assertEqual(by_component["ZZ_reconstructed"].weights.tolist(), [7.0])
            background_yield = sum(
                np.sum(chunk.weights)
                for chunk in chunks
                if chunk.class_name == "background"
            )
            self.assertEqual(background_yield, 10.0)

    def test_negative_training_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                gg_h,
                weights=np.array([-1.0, 2.0]),
                reconstructed=np.array([True, True]),
                fiducial=np.array([True, False]),
            )
            _write_sample(
                zz,
                weights=np.array([1.0]),
                reconstructed=np.array([True]),
                fiducial=np.array([True]),
            )
            with self.assertRaisesRegex(ValueError, "signed weighted BCE"):
                list(
                    _selected_chunks(
                        gg_h,
                        zz,
                        weight_branch="weight_nominal_pb",
                        split_seed=4,
                        expected_luminosity_fb=312.0,
                        step_size="1 MB",
                    )
                )

    def test_affine_calibration_and_ratio_normalization(self):
        signal_scores = np.array([0.55, 0.65, 0.75, 0.85])
        background_scores = np.array([0.15, 0.25, 0.35, 0.45])
        scores = np.concatenate([signal_scores, background_scores])
        labels = np.concatenate([np.ones(4), np.zeros(4)])
        weights = np.full(8, 1.0 / 8.0)
        scale, bias = fit_logit_calibrator(
            scores, labels, weights, logit_clip=20.0
        )
        self.assertGreater(scale, 0.0)
        self.assertLessEqual(scale, 5.0)
        raw_nll = -float(
            np.sum(
                weights
                * (labels * np.log(scores) + (1.0 - labels) * np.log1p(-scores))
            )
        )
        calibrated_scores = 1.0 / (
            1.0 + np.exp(-(scale * np.log(scores / (1.0 - scores)) + bias))
        )
        calibrated_nll = -float(
            np.sum(
                weights
                * (
                    labels * np.log(calibrated_scores)
                    + (1.0 - labels) * np.log1p(-calibrated_scores)
                )
            )
        )
        self.assertLessEqual(calibrated_nll, raw_nll + 1.0e-12)
        normalized_bias, before = _normalization_bias(
            background_scores,
            np.ones(4),
            scale=scale,
            initial_bias=bias,
            logit_clip=20.0,
        )
        self.assertGreater(before, 0.0)
        logits = np.log(background_scores / (1.0 - background_scores))
        ratios = np.exp(np.clip(scale * logits + normalized_bias, -20.0, 20.0))
        self.assertAlmostEqual(float(np.mean(ratios)), 1.0, places=8)

    def test_normalization_brackets_extreme_calibration_slopes(self):
        scores = np.array([0.9, 0.8])
        bias, _ = _normalization_bias(
            scores,
            np.ones(2),
            scale=100.0,
            initial_bias=0.0,
            logit_clip=20.0,
        )
        logits = np.log(scores / (1.0 - scores))
        ratios = np.exp(np.clip(100.0 * logits + bias, -20.0, 20.0))
        self.assertAlmostEqual(float(np.mean(ratios)), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
