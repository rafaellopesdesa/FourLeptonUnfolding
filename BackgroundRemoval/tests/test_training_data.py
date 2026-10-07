from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from BackgroundRemoval.common import FEATURES, weight_measure_contract
from BackgroundRemoval.Training.train_background_ratio import (
    MemberResult,
    _parser,
    _bad_members,
    _normalization_bias,
    _selected_chunks,
    fit_logit_calibrator,
    train,
)


def _write_sample(
    path: Path,
    *,
    weights: np.ndarray,
    reconstructed: np.ndarray,
    fiducial: np.ndarray,
    masses: np.ndarray | None = None,
) -> None:
    size = weights.size
    if masses is None:
        masses = np.full(size, 125.0, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight_nominal_pb": weights.astype(np.float64),
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "reconstructed": reconstructed.astype(np.bool_),
        "fiducial": fiducial.astype(np.bool_),
        "reco_m_ZZ": np.asarray(masses, dtype=np.float32),
    }
    for index, name in enumerate(FEATURES):
        arrays[name] = np.linspace(index, index + 0.5, size, dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


class _FakeCorrectionBundle:
    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def predict(self, raw_features: np.ndarray, *, batch_size: int):
        return {
            "physical_ratio": np.full(raw_features.shape[0], self.factor),
        }


class TrainingDataTest(unittest.TestCase):
    def test_custom_weight_measure_pair_is_rejected_without_override(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            gg_h.write_bytes(b"ggH")
            zz.write_bytes(b"ZZ")
            correction_dir = directory / "correction"
            correction_dir.mkdir()
            (correction_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            correction_branch = "correction_custom_weight"
            correction_bundle = types.SimpleNamespace(
                root=correction_dir,
                manifest={
                    "luminosity_fb": 312.0,
                    "weight_branch": correction_branch,
                    "weight_measure": weight_measure_contract(
                        correction_branch, luminosity_fb=312.0
                    ),
                    "selections": {
                        "mass_window_gev": {
                            "branch": "reco_m_ZZ",
                            "low_exclusive": 130.0,
                            "high_exclusive": 160.0,
                        }
                    },
                },
            )
            args = _parser().parse_args(
                [
                    "--gg-h-root",
                    str(gg_h),
                    "--zz-root",
                    str(zz),
                    "--correction-model-dir",
                    str(correction_dir),
                    "--output-dir",
                    str(directory / "background"),
                    "--weight-branch",
                    "training_custom_weight",
                ]
            )
            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            fake_torch.device = lambda name: name
            fake_diagnostics = types.ModuleType(
                "BackgroundRemoval.Training.diagnostics"
            )
            fake_diagnostics.make_diagnostics_pdf = lambda *args, **kwargs: None

            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "BackgroundRemoval.Training.diagnostics": fake_diagnostics,
                    },
                ),
                patch(
                    "BackgroundRemoval.Training.train_background_ratio."
                    "toolkit_runtime_provenance",
                    return_value={"runtime_commit_verified": True},
                ),
                patch(
                    "BackgroundRemoval.Training.train_background_ratio.ModelBundle.load",
                    return_value=correction_bundle,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "allow-weight-measure-mismatch"
                ):
                    train(args)

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
                    correction_bundle=_FakeCorrectionBundle(),
                    inference_batch_size=16,
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
            self.assertEqual(
                by_component[
                    "ZZ_reconstructed_correction_weighted"
                ].weights.tolist(),
                [14.0],
            )
            background_yield = sum(
                np.sum(chunk.weights)
                for chunk in chunks
                if chunk.class_name == "background"
            )
            self.assertEqual(background_yield, 17.0)

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
                        correction_bundle=_FakeCorrectionBundle(),
                        inference_batch_size=16,
                    )
                )

    def test_training_uses_strict_115_to_130_mass_window(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                gg_h,
                weights=np.ones(5),
                reconstructed=np.ones(5, dtype=np.bool_),
                fiducial=np.ones(5, dtype=np.bool_),
                masses=np.array([114.9, 115.0, 120.0, 130.0, 130.1]),
            )
            _write_sample(
                zz,
                weights=np.ones(1),
                reconstructed=np.ones(1, dtype=np.bool_),
                fiducial=np.ones(1, dtype=np.bool_),
                masses=np.array([125.0]),
            )
            chunks = list(
                _selected_chunks(
                    gg_h,
                    zz,
                    weight_branch="weight_nominal_pb",
                    split_seed=4,
                    expected_luminosity_fb=312.0,
                    step_size="1 MB",
                    correction_bundle=_FakeCorrectionBundle(),
                    inference_batch_size=16,
                )
            )
            signal = next(chunk for chunk in chunks if chunk.class_name == "signal")
            self.assertEqual(signal.weights.tolist(), [1.0])

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
