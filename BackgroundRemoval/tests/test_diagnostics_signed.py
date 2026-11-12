from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from BackgroundRemoval.common import FEATURES, MODEL_KIND_BACKGROUND_REMOVAL
from BackgroundRemoval.Training.diagnostics import (
    _draw_comparison,
    _histogram,
    _normalize_histogram_statistics,
    _reliability,
    make_diagnostics_pdf,
)


class SignedDiagnosticsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

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
        ):
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
