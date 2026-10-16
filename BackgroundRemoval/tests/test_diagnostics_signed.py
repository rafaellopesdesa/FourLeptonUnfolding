from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from BackgroundRemoval.Training.diagnostics import (
    _draw_comparison,
    _histogram,
    _normalize_histogram_statistics,
    _reliability,
)


class SignedDiagnosticsTest(unittest.TestCase):
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
