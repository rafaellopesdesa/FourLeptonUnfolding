from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from BackgroundRemoval.common import (
    FEATURES,
    transformed_features,
    weight_measure_contract,
)
from BackgroundRemoval.Training.train_background_ratio import (
    MemberResult,
    SelectedChunk,
    WeightedStatistics,
    _absolute_weight_sampling_plan,
    _parser,
    _bad_members,
    _draw_absolute_weighted,
    _scan_inputs,
    _selected_chunks,
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
    def test_signed_statistics_keep_net_sum_sumw2_and_absolute_sum_distinct(self):
        statistics = WeightedStatistics()
        weights = np.array([2.0, -1.0, 0.5])
        splits = np.array([0, 0, 1], dtype=np.uint8)
        statistics.add(weights, splits, zero_count=2)

        self.assertEqual(statistics.entries, 3)
        self.assertEqual(statistics.zero_weight_entries, 2)
        self.assertEqual(statistics.negative_weight_entries, 1)
        self.assertAlmostEqual(statistics.sum_weights, 1.5)
        self.assertAlmostEqual(statistics.sum_absolute_weights, 3.5)
        self.assertAlmostEqual(statistics.sum_squared_weights, 5.25)
        self.assertAlmostEqual(statistics.split_sum_weights["fit"], 1.0)
        self.assertAlmostEqual(statistics.split_sum_absolute_weights["fit"], 3.0)
        self.assertAlmostEqual(statistics.split_sum_squared_weights["fit"], 5.0)
        self.assertEqual(statistics.split_negative_weight_entries["fit"], 1)
        self.assertAlmostEqual(statistics.effective_entries(), 1.5**2 / 5.25)

    def test_absolute_weight_sampling_is_unbiased_for_signed_bce_risk(self):
        weights = np.array([2.0, -1.0, 3.0])
        indices = np.arange(weights.size, dtype=np.int64)
        cumulative, coefficient_scale = _absolute_weight_sampling_plan(
            weights,
            indices,
            class_name="test",
        )
        self.assertAlmostEqual(cumulative[-1], 6.0)
        self.assertAlmostEqual(coefficient_scale, 1.5)

        # Treat these as fixed per-event BCE values. The importance-sampled
        # expectation must reproduce sum(w_i * loss_i) / sum(w_i).
        losses = np.array([0.2, 2.0, 1.0])
        expected = float(np.sum(weights * losses) / np.sum(weights))
        draws = _draw_absolute_weighted(
            np.random.default_rng(19110907),
            indices,
            cumulative,
            size=200_000,
        )
        estimate = float(
            np.mean(np.sign(weights[draws]) * coefficient_scale * losses[draws])
        )
        self.assertAlmostEqual(estimate, expected, delta=0.01)

    def test_scan_uses_absolute_weights_only_for_scaler_moments(self):
        raw_signal = np.arange(6 * len(FEATURES), dtype=np.float32).reshape(
            6, len(FEATURES)
        ) / 20.0
        raw_background = raw_signal + np.float32(0.3)
        weights = np.array([2.0, -1.0, 2.0, -1.0, 2.0, -1.0])
        splits = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8)
        chunks = [
            SelectedChunk(
                class_name="signal",
                component_name="gg_H_reconstructed_and_fiducial",
                raw_features=raw_signal,
                weights=weights,
                splits=splits,
                zero_count=0,
            ),
            SelectedChunk(
                class_name="background",
                component_name="gg_H_reconstructed_not_fiducial",
                raw_features=raw_background,
                weights=weights,
                splits=splits,
                zero_count=0,
            ),
        ]
        with patch(
            "BackgroundRemoval.Training.train_background_ratio._selected_chunks",
            return_value=iter(chunks),
        ):
            _, classes, mean, scale = _scan_inputs(
                Path("gg.root"),
                Path("zz.root"),
                weight_branch="weight_nominal_pb",
                split_seed=1,
                expected_luminosity_fb=312.0,
                step_size="1 MB",
                correction_bundle=_FakeCorrectionBundle(),
                inference_batch_size=16,
            )

        transformed_signal = transformed_features(raw_signal[:2]).astype(np.float64)
        transformed_background = transformed_features(raw_background[:2]).astype(
            np.float64
        )
        absolute_fit_weights = np.array([2.0, 1.0])
        signal_mean = np.sum(
            transformed_signal * absolute_fit_weights[:, np.newaxis], axis=0
        ) / 3.0
        background_mean = np.sum(
            transformed_background * absolute_fit_weights[:, np.newaxis], axis=0
        ) / 3.0
        expected_mean = 0.5 * (signal_mean + background_mean)
        np.testing.assert_allclose(mean, expected_mean)
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale > 0.0))
        self.assertAlmostEqual(classes["signal"].sum_weights, 3.0)
        self.assertAlmostEqual(classes["signal"].sum_absolute_weights, 9.0)
        self.assertEqual(classes["signal"].negative_weight_entries, 3)

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

        negative_signed_losses = [
            result(0, -1.0),
            result(1, -1.0),
            result(2, -1.0),
            result(3, -0.5),
        ]
        self.assertEqual(_bad_members(negative_signed_losses), [3])

        runaway_negative_signed_loss = [
            result(0, -100.0),
            result(1, -1.0),
            result(2, -1.0),
            result(3, -1.0),
        ]
        self.assertEqual(_bad_members(runaway_negative_signed_loss), [0])

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

    def test_negative_training_weight_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                gg_h,
                weights=np.array([-1.0, 2.0, 3.0]),
                reconstructed=np.array([True, True, True]),
                fiducial=np.array([True, True, False]),
            )
            _write_sample(
                zz,
                weights=np.array([1.0]),
                reconstructed=np.array([True]),
                fiducial=np.array([True]),
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
                [-1.0, 2.0],
            )
            self.assertEqual(
                by_component["gg_H_reconstructed_not_fiducial"].weights.tolist(),
                [3.0],
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



if __name__ == "__main__":
    unittest.main()
