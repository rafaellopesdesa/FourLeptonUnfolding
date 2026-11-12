from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from BackgroundRemoval.common import FEATURES, MODEL_FEATURES, MODEL_KIND_BACKGROUND_REMOVAL
from BackgroundRemoval.Training.diagnostics import (
    _draw_comparison,
    _histogram,
    _learning_curve_figure,
    _normalize_histogram_statistics,
    _reliability,
    _training_audit_lines,
    make_diagnostics_pdf,
)


class SignedDiagnosticsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_learning_curves_include_initial_checkpoint_and_irregular_updates(self):
        history = {"optimizer_step": [0, 3, 7, 8], "epoch": [0, 0.375, 0.875, 1],
                   "training_loss": [None, 0.69, 0.64, 0.61],
                   "validation_loss": [0.6932, 0.694, 0.697, 0.699]}
        summary = {"selected_optimizer_step": 0, "selected_epoch": 0,
                   "constant_baseline_loss": np.log(2.0),
                   "within_baseline_tolerance": True, "improved_over_constant": False,
                   "initial_checkpoint_selected": True, "stop_reason": "validation_patience"}
        member = {"slot": 0, "attempt": 0, "seed": 1,
                  "history": history, "training_summary": summary}
        figure = _learning_curve_figure([member])
        axis = figure.axes[0]
        lines = {line.get_label(): line for line in axis.lines}
        self.assertEqual(axis.get_xlabel(), "Optimizer updates")
        np.testing.assert_array_equal(lines["validation"].get_xdata(), [0, 3, 7, 8])
        np.testing.assert_allclose(lines["validation"].get_ydata(), history["validation_loss"])
        self.assertTrue(np.isnan(lines["fit batches"].get_ydata()[0]))
        np.testing.assert_allclose(lines["Constant score 0.5 (ln 2)"].get_ydata(), np.log(2.0))
        np.testing.assert_array_equal(lines["Selected: 0"].get_xdata(), [0, 0])
        np.testing.assert_allclose(axis.collections[0].get_offsets(), [[0, 0.6932]])
        self.assertIn("Accepted initial NN", axis.get_title())
        self.assertIn("validation patience", axis.get_title())

    def test_legacy_learning_curves_keep_epoch_coordinates(self):
        member = {"slot": 0, "attempt": 0, "seed": 1,
                  "history": {"training_loss": [0.7, 0.6], "validation_loss": [0.71, 0.68]}}
        axis = _learning_curve_figure([member]).axes[0]
        self.assertEqual(axis.get_xlabel(), "Epoch")
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 2])
        np.testing.assert_array_equal(axis.lines[1].get_xdata(), [1, 2])
        self.assertEqual(_training_audit_lines({"members": [member]}), [])

    def test_training_audit_distinguishes_failed_safeguard_and_reports_fit_statistics(self):
        summary = {"within_baseline_tolerance": False, "initial_checkpoint_selected": True,
                   "trainable_parameters": 173, "weight_decay": 0.001,
                   "target_fit_entries": 20, "reference_fit_entries": 30,
                   "target_draws_per_fit_event": 1.5, "reference_draws_per_fit_event": 1.0,
                   "selected_optimizer_step": 0, "selected_epoch": 0,
                   "stop_reason": "validation_patience"}
        manifest = {"members": [{"slot": 0, "training_summary": summary}],
                    "classes": {"signal": {"split_sum_weights": {"fit": 6.0},
                                           "split_sum_squared_weights": {"fit": 12.0}}}}
        text = "\n".join(_training_audit_lines(manifest))
        self.assertIn("Rejected: failed constant-baseline safeguard", text)
        self.assertNotIn("Accepted initial NN", text)
        self.assertIn("Trainable parameters / weight decay: 173 / 0.001", text)
        self.assertIn("Fit entries target / reference: 20 / 30", text)
        self.assertIn("Mean draws per fit event target / reference: 1.5 / 1", text)
        self.assertIn("signal fit effective entries (sumw)^2/sumw2: 3", text)

    def test_report_has_one_raw_reliability_curve_without_calibration_metadata(self):
        scores = {"signal": np.array([0.15, 0.35, 0.85]),
                  "background": np.array([0.25, 0.55, 0.75])}
        weights = {"signal": np.array([1.0, 2.0, 3.0]),
                   "background": np.array([2.0, 3.0, 2.0])}
        predictions = {name: SimpleNamespace(
            member_scores=np.repeat(score[:, None], 4, axis=1),
            indices=np.arange(3),
        ) for name, score in scores.items()}
        cache = {name: SimpleNamespace(weights=weight,
                                      features=np.zeros((3, len(FEATURES))))
                 for name, weight in weights.items()}
        manifest = {
            "model_kind": MODEL_KIND_BACKGROUND_REMOVAL,
            "yields": {"signal": 6.0, "background": 7.0,
                       "signal_to_background": 6.0 / 7.0, "units": "events"},
            "luminosity_fb": 312.0,
            "toolkit": {"pinned_commit": "fixture"},
            "ensemble": {"size": 4},
            "architecture": {"hidden_layers": 1, "neurons": 8, "loss_logit_clip": 30.0},
            "members": [{"slot": 0, "attempt": 0, "seed": 1,
                         "history": {"training_loss": [0.7], "validation_loss": [0.71]}}],
        }
        figures = []
        with patch("BackgroundRemoval.Training.diagnostics.PdfPages") as pdf, patch(
            "BackgroundRemoval.Training.diagnostics._draw_comparison",
            return_value={"chi2": 0.0, "covariance_rank": 1},
        ) as comparison:
            pdf.return_value.__enter__.return_value.savefig.side_effect = figures.append
            metrics = make_diagnostics_pdf(Path("unused.pdf"), manifest=manifest,
                                           cache=cache, validation=predictions, closure=predictions)
        reliability_axis = next(axis for figure in figures for axis in figure.axes
                                if axis.get_title() == "Raw balanced-prior reliability (closure)")
        self.assertEqual(len(reliability_axis.containers), 1)
        line = reliability_axis.containers[0].lines[0]
        expected = _reliability(scores["signal"], scores["background"],
                                weights["signal"], weights["background"])
        valid = np.isfinite(expected[0]) & np.isfinite(expected[1])
        np.testing.assert_allclose(line.get_xdata(), expected[0][valid])
        np.testing.assert_allclose(line.get_ydata(), expected[1][valid])
        self.assertEqual(set(metrics["reliability"]), {"closure_raw"})
        self.assertNotIn("calibration", metrics)
        self.assertEqual(set(metrics["shape_closure"]), set(MODEL_FEATURES))
        # Raw azimuths are zero in this fixture, so cosine feature plots must
        # receive ones and use [-1, 1] bins, not draw the excluded signed angles.
        for feature_index in (0, 1):
            plotted = comparison.call_args_list[2 * feature_index].kwargs
            np.testing.assert_allclose(plotted["values_target"], 1.0)
            np.testing.assert_allclose(plotted["values_estimate"], 1.0)
            self.assertEqual(plotted["edges"][0], -1.0)
            self.assertEqual(plotted["edges"][-1], 1.0)
            self.assertIn(r"\cos", plotted["xlabel"])
        expected_mean = np.average(scores["background"] / (1.0 - scores["background"]),
                                   weights=weights["background"])
        self.assertNotAlmostEqual(expected_mean, 1.0)
        self.assertAlmostEqual(metrics["ratio_normalization"]["closure"]["forward"], expected_mean)

    def test_normalized_histogram_propagates_denominator_covariance(self):
        normalized, covariance, jacobian = _normalize_histogram_statistics(
            np.array([30.0, 70.0]),
            np.diag([30.0, 70.0]),
        )

        np.testing.assert_allclose(normalized, [0.3, 0.7])
        np.testing.assert_allclose(
            covariance,
            [[0.0021, -0.0021], [-0.0021, 0.0021]],
        )
        np.testing.assert_allclose(np.sum(covariance, axis=0), 0.0, atol=1.0e-15)
        np.testing.assert_allclose(np.ones(2) @ jacobian, 0.0, atol=1.0e-15)

    def test_normalized_comparison_chi2_uses_singular_full_covariance(self):
        with tempfile.TemporaryDirectory() as directory_name:
            output = Path(directory_name) / "normalized_comparison.pdf"
            target_values = np.concatenate(
                [np.full(50, 0.25), np.full(50, 0.75)]
            )
            estimate_values = np.concatenate(
                [np.full(60, 0.25), np.full(40, 0.75)]
            )
            with PdfPages(output) as pdf:
                metrics = _draw_comparison(
                    pdf,
                    values_target=target_values,
                    weights_target=np.ones(target_values.size),
                    values_estimate=estimate_values,
                    weights_estimate=np.ones(estimate_values.size),
                    edges=np.array([0.0, 0.5, 1.0]),
                    xlabel="observable",
                    title="normalized comparison",
                    target_label="target",
                    estimate_label="estimate",
                    normalize=True,
                )

        # The difference [0.1, -0.1] lies in the sole non-null covariance
        # mode.  A diagonal-only treatment would incorrectly give twice this.
        self.assertAlmostEqual(metrics["chi2"], 0.02 / 0.0098, places=12)
        self.assertEqual(metrics["bins"], 2)
        self.assertEqual(metrics["covariance_rank"], 1)

    def test_histogram_uses_signed_sum_and_unsigned_sumw2(self):
        contents, variances = _histogram(
            np.array([0.25, 0.25, 0.75]),
            np.array([2.0, -3.0, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(contents, [-1.0, 1.0])
        np.testing.assert_allclose(variances, [13.0, 1.0])

    def test_reliability_skips_nonpositive_net_bin_but_keeps_signed_metrics(self):
        centers, fractions, errors, nll, brier, ece = _reliability(
            signal_scores=np.array([0.1, 0.9]),
            background_scores=np.array([0.1, 0.9]),
            signal_weights=np.array([-1.0, 2.0]),
            background_weights=np.array([1.0, 1.0]),
            bins=2,
        )

        self.assertTrue(np.isnan(centers[0]))
        self.assertTrue(np.isnan(fractions[0]))
        self.assertTrue(np.isnan(errors[0]))
        self.assertTrue(np.isfinite(centers[1]))
        self.assertTrue(np.isfinite(fractions[1]))
        self.assertTrue(np.isfinite(errors[1]))
        self.assertTrue(np.isfinite(nll))
        self.assertTrue(np.isfinite(brier))
        self.assertTrue(np.isfinite(ece))

    def test_comparison_pdf_accepts_negative_histogram_bins(self):
        with tempfile.TemporaryDirectory() as directory_name:
            output = Path(directory_name) / "signed_comparison.pdf"
            with PdfPages(output) as pdf:
                metrics = _draw_comparison(
                    pdf,
                    values_target=np.array([0.25, 0.75]),
                    weights_target=np.array([-1.0, 3.0]),
                    values_estimate=np.array([0.25, 0.75]),
                    weights_estimate=np.array([-0.5, 2.5]),
                    edges=np.array([0.0, 0.5, 1.0]),
                    xlabel="signed observable",
                    title="signed comparison",
                    target_label="target",
                    estimate_label="estimate",
                    normalize=True,
                )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertTrue(np.isfinite(metrics["chi2"]))
            self.assertAlmostEqual(metrics["target_integral"], 1.0)
            self.assertAlmostEqual(metrics["estimate_integral"], 1.0)


if __name__ == "__main__":
    unittest.main()
