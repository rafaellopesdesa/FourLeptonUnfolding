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
    _normalization_bias,
    _scan_inputs,
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

    def test_affine_calibration_and_normalization_accept_signed_weights(self):
        signal_scores = np.array([0.55, 0.65, 0.75, 0.85])
        background_scores = np.array([0.15, 0.25, 0.35, 0.45])
        scores = np.concatenate([signal_scores, background_scores])
        labels = np.concatenate([np.ones(4), np.zeros(4)])
        signal_weights = np.array([1.2, -0.2, 0.6, 0.4])
        background_weights = np.array([0.8, -0.1, 0.7, 0.6])
        weights = np.concatenate(
            [
                0.5 * signal_weights / np.sum(signal_weights),
                0.5 * background_weights / np.sum(background_weights),
            ]
        )

        scale, bias = fit_logit_calibrator(
            scores, labels, weights, logit_clip=20.0
        )
        self.assertTrue(np.isfinite(scale))
        self.assertTrue(np.isfinite(bias))
        self.assertGreaterEqual(scale, 0.1)
        self.assertLessEqual(scale, 5.0)

        normalized_bias, before = _normalization_bias(
            background_scores,
            background_weights,
            scale=scale,
            initial_bias=bias,
            logit_clip=20.0,
        )
        self.assertGreater(before, 0.0)
        logits = np.log(background_scores / (1.0 - background_scores))
        ratios = np.exp(np.clip(scale * logits + normalized_bias, -20.0, 20.0))
        signed_mean = np.sum(background_weights * ratios) / np.sum(
            background_weights
        )
        self.assertAlmostEqual(float(signed_mean), 1.0, places=8)

    def test_signed_calibrator_does_not_accept_stationary_saddle(self):
        # The identity transform has a vanishing gradient for this signed
        # sample but a negative Hessian eigenvalue.  A single-start first-order
        # optimizer therefore used to report the saddle as the calibration.
        scores = np.array(
            [
                0.918267037869059,
                0.06953007083461765,
                0.7381518650657852,
                0.45890993719647033,
                0.7435622726360329,
                0.05940961477265993,
                0.063403439307473,
                0.12555789029861167,
                0.9464457224930242,
                0.23378705706335223,
                0.6322003817947188,
                0.9479936956983981,
            ]
        )
        labels = np.r_[np.ones(6), np.zeros(6)]
        weights = np.array(
            [
                -0.2010311413190771,
                0.1271358732893945,
                0.34754364935437865,
                0.6062672674870114,
                -0.18777723868263732,
                -0.19213841012907037,
                -0.28382206445867825,
                0.3229752543622726,
                -0.1873604927262832,
                0.07303493712079821,
                0.3658319448764966,
                0.2093404208253944,
            ]
        )

        scale, bias = fit_logit_calibrator(
            scores, labels, weights, logit_clip=8.0
        )

        logits = np.log(scores) - np.log1p(-scores)

        def signed_bce(candidate_scale: float, candidate_bias: float) -> float:
            prediction = candidate_scale * logits + candidate_bias
            losses = (
                np.maximum(prediction, 0.0)
                - labels * prediction
                + np.log1p(np.exp(-np.abs(prediction)))
            )
            return float(np.sum(weights * losses, dtype=np.float64))

        self.assertLess(signed_bce(scale, bias), signed_bce(1.0, 0.0) - 0.05)
        self.assertAlmostEqual(scale, 0.1, places=7)
        self.assertAlmostEqual(bias, -0.009065, places=5)

    def test_signed_calibrator_explores_scale_edge_midpoints(self):
        # Identity plus only the four parameter-box corners converges to a
        # higher local minimum for this signed sample.  In particular, the
        # lower-scale, zero-bias start is needed to reach the better basin.
        scores = np.array(
            [
                0.8778938505769383,
                0.14294507628952396,
                0.6473892156318168,
                0.4983937881143941,
                0.6258941418372559,
                0.6290994887888002,
                0.9670859205530947,
                0.4318020032011376,
                0.9005976971418931,
                0.9864467256266499,
                0.7515352588124318,
                0.7363341035260051,
                0.488742603209187,
                0.4618496861824923,
                0.17300099762703328,
                0.20485979696708823,
                0.7139783193300612,
                0.06907435963959835,
                0.9898696634906776,
                0.7704305184657461,
                0.6308761761615372,
                0.40173416835586184,
                0.24584325755991449,
                0.5290039975143489,
                0.43635868399244954,
                0.15803061049234407,
                0.4754958718649156,
                0.6334375360021309,
                0.8777650018468728,
                0.02109577703864221,
            ]
        )
        labels = np.r_[np.ones(15), np.zeros(15)]
        weights = np.array(
            [
                -0.06955701864058762,
                -0.0695254981511428,
                0.1714793419029644,
                0.00169070481402511,
                0.26880292354675805,
                0.22388784068831866,
                -0.08494329692108131,
                -0.14556622360867488,
                0.1340675916276195,
                0.02607033180969284,
                -0.06086385774318614,
                0.07079254660111414,
                0.09492565401535233,
                -0.0894671163599366,
                0.02820607641876469,
                0.3615721076966149,
                0.07774187122166216,
                -0.10043969651567981,
                0.06130199558623531,
                -0.07484318524436996,
                -0.37370283725801107,
                -0.05960719076449892,
                0.22020155786846704,
                0.1404682551822135,
                -0.04788854893055955,
                -0.20826114808601592,
                0.23745646645905938,
                0.5719377429244279,
                0.02593401208793181,
                -0.33187140222747674,
            ]
        )

        scale, bias = fit_logit_calibrator(
            scores, labels, weights, logit_clip=8.0
        )
        logits = np.log(scores) - np.log1p(-scores)
        prediction = scale * logits + bias
        fitted_loss = float(
            np.sum(
                weights
                * (
                    np.maximum(prediction, 0.0)
                    - labels * prediction
                    + np.log1p(np.exp(-np.abs(prediction)))
                ),
                dtype=np.float64,
            )
        )

        self.assertLess(fitted_loss, 0.7393)
        self.assertAlmostEqual(scale, 0.1, places=7)
        self.assertAlmostEqual(bias, -0.202475, places=5)

    def test_normalization_rejects_shape_changing_output_clipping(self):
        scores = np.array([0.9, 0.8])
        with self.assertRaisesRegex(RuntimeError, "would require clipping"):
            _normalization_bias(
                scores,
                np.ones(2),
                scale=100.0,
                initial_bias=0.0,
                logit_clip=20.0,
            )

    def test_signed_normalization_brackets_root_beyond_score_extrema(self):
        logits = np.array([-1.06996896, -0.33517923, -5.44417737])
        scores = 1.0 / (1.0 + np.exp(-logits))
        weights = np.array([5.0, 2.0, -3.0])
        bias, _ = _normalization_bias(
            scores,
            weights,
            scale=1.0,
            initial_bias=0.0,
            logit_clip=3.0,
        )
        ratios = np.exp(np.clip(np.clip(logits, -3.0, 3.0) + bias, -3.0, 3.0))
        signed_mean = float(np.sum(weights * ratios) / np.sum(weights))
        self.assertAlmostEqual(signed_mean, 1.0, places=8)

    def test_signed_normalization_rejects_nonpositive_ratio_moment(self):
        logits = np.array([-1.0, 1.0])
        scores = 1.0 / (1.0 + np.exp(-logits))
        with self.assertRaisesRegex(RuntimeError, "no stable positive"):
            _normalization_bias(
                scores,
                np.array([1.0, -0.9]),
                scale=1.0,
                initial_bias=0.0,
                logit_clip=30.0,
            )

    def test_signed_normalization_is_one_shape_preserving_multiplier(self):
        logits = np.array([-0.61475135, -2.57105235, 0.36060179])
        scores = 1.0 / (1.0 + np.exp(-logits))
        weights = np.array([-3.90384409, 4.79554235, 1.41211051])
        bias, _ = _normalization_bias(
            scores,
            weights,
            scale=1.0,
            initial_bias=-0.4,
            logit_clip=3.0,
        )
        raw = np.exp(logits - 0.4)
        normalized = np.exp(logits + bias)
        factors = normalized / raw
        self.assertTrue(np.allclose(factors, factors[0], rtol=1.0e-12))
        self.assertAlmostEqual(
            float(np.sum(weights * normalized) / np.sum(weights)),
            1.0,
            places=10,
        )

    def test_signed_normalization_rejects_catastrophic_cancellation(self):
        logits = np.array([0.0, 1.0])
        scores = 1.0 / (1.0 + np.exp(-logits))
        weights = np.array([1.0, -(1.0 - 1.0e-14) / np.e])
        with self.assertRaisesRegex(RuntimeError, "numerically cancelled"):
            _normalization_bias(
                scores,
                weights,
                scale=1.0,
                initial_bias=0.0,
                logit_clip=20.0,
            )

    def test_calibrator_recovers_broad_logit_affine_map(self):
        logits = np.linspace(-30.0, 30.0, 601)
        probabilities = 1.0 / (1.0 + np.exp(-(0.7 * logits - 0.3)))
        raw_scores = 1.0 / (1.0 + np.exp(-logits))
        scores = np.repeat(raw_scores, 2)
        labels = np.tile(np.array([1.0, 0.0]), logits.size)
        weights = np.column_stack(
            [probabilities, 1.0 - probabilities]
        ).reshape(-1)
        weights /= np.sum(weights)
        scale, bias = fit_logit_calibrator(
            scores,
            labels,
            weights,
            logit_clip=40.0,
        )
        self.assertAlmostEqual(scale, 0.7, places=6)
        self.assertAlmostEqual(bias, -0.3, places=6)

    def test_calibrator_uses_a_locally_descending_active_set_direction(self):
        # At the optimum the scale is at its upper bound.  A coupled Fisher
        # step from that face can be locally ascending even if its fully
        # clipped unit displacement looks descending; the optimizer must
        # switch to a feasible descent direction before backtracking.
        scores = np.array(
            [
                0.37377823576471214, 0.5895666076554016,
                0.5002905883535771, 0.8996083518166426,
                0.9159325326667178, 0.8359656655593696,
                0.4946534479931835, 0.20915537172191176,
                0.6229013843007501, 0.7220679486332228,
                0.5331375389361955, 0.18379297354938892,
                0.7264994588703463, 0.8596126610286606,
                0.7769394716587615, 0.20020495365844312,
                0.22913300554087423, 0.8891329199754977,
                0.9094779129269105, 0.4356586355777468,
                0.8753713570385461,
            ]
        )
        labels = np.array(
            [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            dtype=np.float64,
        )
        weights = np.array(
            [
                -1.8650415753457352, 0.1301756888842444,
                2.276369507747734, 0.515925255814574,
                0.585498485446444, 0.21057120104975274,
                -1.6425139940155864, 1.9707596370948721,
                0.4221771777416838, 2.29353421708021,
                1.7389670021218193, 0.3930791802455779,
                0.22984583876431025, 1.5945201049385624,
                -0.6039170625414393, 0.3580726098608785,
                0.786516592242419, -1.277686990410254,
                2.510300267383626, 1.5761445122233435,
                0.799796458874286,
            ]
        )
        scale, bias = fit_logit_calibrator(
            scores, labels, weights, logit_clip=20.0
        )
        self.assertAlmostEqual(scale, 5.0, places=7)
        self.assertAlmostEqual(bias, -9.75684008, places=6)

    def test_calibrator_converges_while_approaching_lower_scale_bound(self):
        scores = np.array(
            [
                0.92710741, 0.36908691, 0.73244695, 0.18873275, 0.29681648,
                0.29003363, 0.42538642, 0.17333295, 0.27599952, 0.77622814,
                0.66233643, 0.36985046, 0.50296398, 0.16154975, 0.09942076,
                0.71342641, 0.44543308, 0.06745538, 0.22111999, 0.79476442,
            ]
        )
        labels = np.r_[np.ones(10), np.zeros(10)]
        weights = np.array(
            [
                0.02899405, 0.05323777, 0.01855952, 0.03126482, 0.05925754,
                0.08204296, 0.05668093, 0.02189712, 0.10100465, 0.04706063,
                0.01717648, 0.00684964, 0.1221241, 0.0742262, -0.02531187,
                0.07977622, 0.11739438, -0.03208125, 0.05327715, 0.08656894,
            ]
        )
        scale, bias = fit_logit_calibrator(
            scores, labels, weights, logit_clip=10.0
        )
        self.assertAlmostEqual(scale, 0.1, places=7)
        self.assertAlmostEqual(bias, 0.00729438, places=6)


if __name__ == "__main__":
    unittest.main()
