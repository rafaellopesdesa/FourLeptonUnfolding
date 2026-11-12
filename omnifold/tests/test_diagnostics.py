"""Numerical and notebook-contract checks for signed-weight diagnostics."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omnifold.diagnostics import (
    FEATURES,
    _histogram,
    _learning_figure,
    _normalize,
    _ratio,
    _reliability,
    histogram_discrepancy,
    plot_distributions,
    plot_ratio_diagnostics,
    save_figures,
)


class _ConstantRatio:
    def __init__(self):
        self.predicted_features = []
        self.history = [{"slot": 0, "history": {
            "training_loss": [0.7, -0.1],
            "validation_loss": [0.71, 0.69],
            "learning_rate": [0.01, 0.001],
        }}]

    def predict(self, features):
        self.predicted_features.append(features.copy())
        size = len(features)
        return {
            "shape_ratio": np.ones(size), "full_ratio": np.full(size, 2.0),
            "score_raw": np.full(size, 0.5),
            "ensemble_std": np.zeros(size),
        }


class DiagnosticsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_learning_plot_uses_irregular_updates_and_preserves_initial_nn(self):
        history = {"optimizer_step": [0, 3, 7, 8], "epoch": [0, 0.375, 0.875, 1],
                   "training_loss": [None, 0.69, 0.64, 0.61],
                   "validation_loss": [0.6932, 0.694, 0.697, 0.699],
                   "learning_rate": [0.001, 0.0009, 0.0008, 0.0007]}
        summary = {"selected_optimizer_step": 0, "selected_epoch": 0,
                   "constant_baseline_loss": np.log(2.0), "initial_validation_loss": 0.6932,
                   "within_baseline_tolerance": True, "improved_over_constant": False,
                   "initial_checkpoint_selected": True, "stop_reason": "validation_patience",
                   "trainable_parameters": 173, "weight_decay": 0.001,
                   "target_fit_entries": 20, "reference_fit_entries": 30,
                   "target_draws_per_fit_event": 1.5, "reference_draws_per_fit_event": 1.0,
                   "optimizer_steps_completed": 8, "planned_optimizer_steps": 24}
        record = {"slot": 0, "history": history, "training_summary": summary}
        manifest = {"classes": {"target": {"split_sum_weights": {"fit": 6.0},
                                            "split_sum_squared_weights": {"fit": 12.0}}}}
        figure = _learning_figure([record], title="Synthetic", manifest=manifest)
        for axis in figure.axes:
            self.assertEqual(axis.get_xlabel(), "Optimizer updates")
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [0, 3, 7, 8])
            self.assertLessEqual(axis.get_xlim()[0], 0.0)
        self.assertTrue(np.isnan(figure.axes[0].lines[0].get_ydata()[0]))
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), history["validation_loss"])
        np.testing.assert_allclose(figure.axes[1].lines[1].get_ydata(), np.log(2.0))
        np.testing.assert_allclose(figure.axes[1].collections[0].get_offsets(), [[0, 0.6932]])
        text = "\n".join(item.get_text() for item in figure.texts)
        self.assertIn("accepted initial NN; no improvement beyond min delta", text)
        self.assertIn("selected update 0; stop: validation patience", text)
        self.assertIn("Parameters/member: 173", text)
        self.assertIn("Fit entries target/reference: 20/30", text)
        self.assertIn("Fit effective entries (sumw)^2/sumw2 — target: 3", text)
        self.assertIn("Mean draws per fit event target/reference: 1.5/1", text)

    def test_learning_plot_labels_failed_safeguard_without_claiming_acceptance(self):
        record = {"slot": 0, "history": {"optimizer_step": [0, 1],
                  "training_loss": [None, 0.6], "validation_loss": [0.8, 0.9],
                  "learning_rate": [0.001, 0.001]},
                  "training_summary": {"within_baseline_tolerance": False,
                                       "initial_checkpoint_selected": True}}
        figure = _learning_figure([record], title="Rejected fixture", manifest={})
        text = "\n".join(item.get_text() for item in figure.texts)
        self.assertIn("REJECTED: failed constant-baseline safeguard", text)
        self.assertNotIn("accepted initial NN", text)

    def test_legacy_learning_plot_uses_epochs_without_requiring_training_summary(self):
        figure = _learning_figure(_ConstantRatio().history, title="Legacy raw", manifest={})
        for axis in figure.axes:
            self.assertEqual(axis.get_xlabel(), "Epoch")
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 2])
        np.testing.assert_allclose(figure.axes[1].lines[1].get_ydata(), np.log(2.0))

    def test_histogram_preserves_signed_bins_sumw2_and_flow(self):
        values, variance = _histogram(
            np.array([-2.0, 0.25, 0.75, 7.0, np.nan]),
            np.array([2.0, -3.0, 4.0, -1.0, 99.0]),
            np.array([0.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(values, [-1.0, 3.0])
        np.testing.assert_allclose(variance, [13.0, 17.0])

    def test_shape_covariance_accounts_for_random_denominator(self):
        values, covariance, jacobian = _normalize(
            np.array([30.0, 70.0]), np.diag([30.0, 70.0])
        )
        np.testing.assert_allclose(values, [0.3, 0.7])
        np.testing.assert_allclose(covariance, [[0.0021, -0.0021], [-0.0021, 0.0021]])
        np.testing.assert_allclose(np.sum(covariance, axis=0), 0.0, atol=1e-16)
        np.testing.assert_allclose(np.sum(jacobian, axis=0), 0.0, atol=1e-16)

    def test_shared_weight_ratio_has_zero_error_for_constant_multiplier(self):
        nominal = np.array([-1.0, 3.0])
        nominal_covariance = np.diag([13.0, 17.0])
        ratio, error = _ratio(2 * nominal, nominal, 4 * nominal_covariance,
                              nominal_covariance, 2 * nominal_covariance)
        np.testing.assert_allclose(ratio, [2.0, 2.0])
        np.testing.assert_allclose(error, 0.0)

    def test_signed_reliability_keeps_negative_and_above_one_fractions(self):
        result = _reliability(
            np.array([0.25, 0.75]), np.array([0.25, 0.75]),
            np.array([-1.0, 3.0]), np.array([2.0, 1.0]), bins=2,
        )
        self.assertAlmostEqual(result["fractions"][0], -3.0)
        self.assertTrue(np.all(np.isfinite(result["errors"])))
        negative_mass = _reliability(
            np.array([0.25, 0.75]), np.array([0.25, 0.75]),
            np.array([-3.0, 4.0]), np.array([1.0, 1.0]), bins=2,
        )
        self.assertLess(negative_mass["net_bin_mass"][0], 0.0)
        self.assertAlmostEqual(negative_mass["fractions"][0], 1.2)
        self.assertTrue(negative_mass["valid"][0])

    def test_reliability_identifies_zero_signed_denominator(self):
        result = _reliability(
            np.array([0.25, 0.75]), np.array([0.25, 0.75]),
            np.array([-1.0, 3.0]), np.array([1.0, 1.0]), bins=2,
        )
        self.assertFalse(result["valid"][0])
        self.assertTrue(np.isnan(result["fractions"][0]))

    def test_shared_event_reliability_and_closure_include_cross_covariance(self):
        scores = np.array([0.25, 0.75, 0.75])
        weights = np.array([-1.0, 3.0, 2.0])
        reliability = _reliability(scores, scores, 2 * weights, weights,
                                   bins=2, shared_events=True)
        np.testing.assert_allclose(reliability["fractions"], 0.5)
        np.testing.assert_allclose(reliability["errors"], 0.0, atol=1e-15)
        features = scores[:, None]
        figures = plot_ratio_diagnostics(
            _ConstantRatio(), features, 2 * weights, features, weights,
            target_splits=np.full(3, 2), reference_splits=np.full(3, 2),
            shared_events=True,
        )
        reweighted = figures["closure"].diagnostic_metrics["variable_0"]["Reweighted reference"]
        self.assertEqual(reweighted["chi2"], 0.0)
        self.assertEqual(reweighted["covariance_rank"], 0)

    def test_shape_discrepancy_uses_full_covariance_and_null_mode(self):
        target, target_covariance, _ = _normalize(np.array([50.0, 50.0]), np.diag([50.0, 50.0]))
        estimate, estimate_covariance, _ = _normalize(np.array([60.0, 40.0]), np.diag([60.0, 40.0]))
        result = histogram_discrepancy(target, estimate, target_covariance, estimate_covariance)
        self.assertEqual(result["covariance_rank"], 1)
        self.assertAlmostEqual(result["chi2"], 0.02 / 0.0098)

    def test_ratio_diagnostics_predict_only_heldout_and_preserve_yields(self):
        model = _ConstantRatio()
        features = np.column_stack([np.linspace(-0.5, 0.5, 6)] * 8)
        features[:, 6:] += [90.0, 25.0]
        splits = np.array([0, 1, 0, 2, 2, 2])
        target_weights = np.array([1000.0, 2000.0, 3000.0, -1.0, 3.0, 4.0])
        reference_weights = np.array([9999.0, 9999.0, 9999.0, -1.0, 1.0, 2.0])
        figures = plot_ratio_diagnostics(
            model, features, target_weights, features, reference_weights,
            target_splits=splits, reference_splits=splits, title="Synthetic signed fixture",
        )
        self.assertEqual(set(figures), {"closure", "reliability", "scores", "ensemble", "learning"})
        for predicted in model.predicted_features:
            np.testing.assert_array_equal(predicted, features[splits == 2])
        normalization = figures["closure"].ratio_normalization
        self.assertEqual(normalization["target_yield"], 6.0)
        self.assertEqual(normalization["reference_yield"], 2.0)
        # Deliberately mismatched: diagnostics must not fix a failing closure.
        self.assertEqual(normalization["reweighted_reference_yield"], 4.0)
        self.assertEqual(normalization["reference_shape_mean"], 1.0)
        self.assertEqual(len(figures["closure"].axes), 16)
        self.assertTrue(all(axis.get_yscale() == "linear" for axis in figures["closure"].axes))
        self.assertEqual(figures["learning"].axes[0].get_yscale(), "linear")
        self.assertEqual(figures["learning"].axes[0].lines[0].get_ydata()[1], -0.1)

    def test_reliability_and_scores_show_only_unchanged_raw_predictions(self):
        class RawRatio:
            def predict(self, features):
                scores = features[:, 0]
                odds = scores / (1.0 - scores)
                return {"score_raw": scores, "shape_ratio": odds,
                        "full_ratio": 2.0 * odds, "ensemble_std": np.zeros(len(scores))}

        features = np.array([[0.15], [0.35], [0.65], [0.85]])
        target_weights = np.array([1.0, 3.0, 2.0, 1.0])
        reference_weights = np.array([2.0, 2.0, 1.0, 3.0])
        figures = plot_ratio_diagnostics(
            RawRatio(), features, target_weights, features, reference_weights,
            target_splits=np.full(4, 2), reference_splits=np.full(4, 2),
        )
        expected = _reliability(features[:, 0], features[:, 0], target_weights, reference_weights)
        self.assertEqual(len(figures["reliability"].axes), 1)
        axis = figures["reliability"].axes[0]
        self.assertEqual(len(axis.containers), 1)
        point_line = axis.containers[0].lines[0]
        np.testing.assert_allclose(point_line.get_xdata(), expected["centers"], equal_nan=True)
        np.testing.assert_allclose(point_line.get_ydata(), expected["fractions"], equal_nan=True)
        self.assertEqual([axis.get_xlabel() for axis in figures["scores"].axes],
                         ["Raw score", "log(shape ratio)"])
        expected_mean = np.average(features[:, 0] / (1.0 - features[:, 0]),
                                   weights=reference_weights)
        self.assertNotAlmostEqual(expected_mean, 1.0)
        self.assertAlmostEqual(figures["closure"].ratio_normalization["reference_shape_mean"],
                               expected_mean)

    def test_distribution_default_is_absolute_and_pdf_is_saved(self):
        features = np.array([[0.25], [0.75]])
        figure = plot_distributions(features, np.array([-1.0, 3.0]), np.array([-2.0, 6.0]),
                                    feature_names=["test"], title="Signed weights")
        metrics = figure.diagnostic_metrics["test"]["Updated"]
        self.assertEqual(metrics["target_yield"], 2.0)
        self.assertEqual(metrics["estimate_yield"], 4.0)
        self.assertEqual(figure.axes[0].get_ylabel(), "Expected events / bin")
        with tempfile.TemporaryDirectory() as directory:
            output = save_figures({"signed": figure}, Path(directory) / "plots.pdf")
            self.assertEqual(output.read_bytes()[:4], b"%PDF")

    def test_explicit_shape_only_and_extra_observables(self):
        names = (*FEATURES, "m_ZZ", "y_ZZ", "pT_ZZ")
        features = np.zeros((3, len(names)))
        figure = plot_distributions(features, np.array([2.0, -1.0, 2.0]),
                                    np.array([4.0, -1.0, 2.0]),
                                    feature_names=names, shape_only=True)
        self.assertEqual(len(figure.axes), 22)
        for metric in figure.diagnostic_metrics.values():
            self.assertAlmostEqual(metric["Updated"]["target_yield"], 1.0)
            self.assertAlmostEqual(metric["Updated"]["estimate_yield"], 1.0)

    def test_validation_rejects_missing_target_and_heldout_events(self):
        features = np.zeros((3, 1))
        weights = np.ones(3)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            plot_distributions(features, weights, weights, truth_weights=weights,
                               feature_names=["x"])
        with self.assertRaisesRegex(ValueError, "no held-out"):
            plot_ratio_diagnostics(_ConstantRatio(), features, weights, features, weights,
                                   target_splits=np.zeros(3), reference_splits=np.zeros(3))


if __name__ == "__main__":
    unittest.main()
