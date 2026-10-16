from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import uproot

from Plotting.plot_data_mc import (
    HistogramValues,
    KINEMATIC_FIELDS,
    OBSERVABLES,
    _draw_stack,
    _read_histogram,
    _ratio_quantities,
    _weighted_histogram,
    create_comparison_pdf,
)


def write_plot_sample(
    path: Path,
    *,
    weights: np.ndarray,
    lumi: float = 312_000.0,
    data: bool = False,
) -> None:
    weights = np.asarray(weights, dtype=np.float64)
    size = len(weights)
    index = np.arange(size)
    reconstructed = np.ones(size, dtype=np.bool_)
    fiducial = np.asarray(index % 3 != 2, dtype=np.bool_)
    arrays: dict[str, np.ndarray] = {
        "event_id": index.astype(np.uint64),
        "event_number": (100 + index).astype(np.int64),
        "weight": weights,
        "weight_shape": np.sign(weights),
        "weight_nominal_pb": weights / lumi,
        "lumi": np.full(size, lumi, dtype=np.float64),
        "luminosity_fb": np.full(size, lumi / 1000.0, dtype=np.float64),
        "cross_section_pb": np.full(
            size, np.nan if data else np.sum(weights) / lumi, dtype=np.float64
        ),
        "reconstructed": reconstructed,
        "fiducial": fiducial,
        "reco_type": (index % 4).astype(np.int8),
        "truth_type": (index % 4).astype(np.int8),
        "type": (index % 4).astype(np.int8),
    }
    for field_index, field in enumerate(KINEMATIC_FIELDS):
        if field.startswith("cos_theta"):
            values = np.linspace(-0.8, 0.8, size)
        elif field in {"Phi", "Phi1", "Psi"}:
            values = np.linspace(-2.5, 2.5, size)
        elif field == "m_Z1":
            values = 75.0 + index
        elif field == "m_Z2":
            values = 25.0 + index
        elif field == "m_ZZ":
            values = 120.0 + index
        elif field == "y_ZZ":
            values = np.linspace(-1.5, 1.5, size)
        elif field == "pT_ZZ":
            values = 10.0 + 12.0 * index
        else:  # pragma: no cover - guarded by KINEMATIC_FIELDS
            raise AssertionError(field_index)
        arrays[f"reco_{field}"] = np.asarray(values, dtype=np.float32)
        arrays[f"truth_{field}"] = np.asarray(values + 0.1, dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


class PlotDataMCTest(unittest.TestCase):
    def test_signed_histogram_uses_sumw2_uncertainty_and_folds_flow(self):
        result = _weighted_histogram(
            np.array([-2.0, 0.2, 0.2, 0.2, 5.0]),
            np.array([2.0, 1.0, 1.0, -1.0, -2.0]),
            np.array([0.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(result.values, [3.0, -2.0])
        np.testing.assert_allclose(result.variances, [7.0, 4.0])
        np.testing.assert_allclose(result.errors, [np.sqrt(7.0), 2.0])

    def test_ratio_keeps_negative_denominators_and_uses_positive_errors(self):
        reference = HistogramValues(
            values=np.array([3.0, -2.0, 1.0]),
            variances=np.array([9.0, 4.0, 1.0]),
        )
        denominator = HistogramValues(
            values=np.array([-2.0, 4.0, 0.0]),
            variances=np.array([1.0, 16.0, 25.0]),
        )

        ratio, ratio_error, relative_error = _ratio_quantities(
            reference, denominator
        )

        np.testing.assert_allclose(ratio[:2], [-1.5, -0.5])
        np.testing.assert_allclose(ratio_error[:2], [1.5, 0.5])
        np.testing.assert_allclose(relative_error[:2], [0.5, 1.0])
        self.assertTrue(np.isnan(ratio[2]))
        self.assertTrue(np.isnan(ratio_error[2]))
        self.assertTrue(np.isnan(relative_error[2]))

    def test_stack_adds_signed_contents_and_sumw2_variances(self):
        total = _draw_stack(
            Mock(),
            np.array([0.0, 1.0, 2.0]),
            [
                (
                    "first",
                    "blue",
                    HistogramValues(
                        values=np.array([3.0, -1.0]),
                        variances=np.array([5.0, 1.0]),
                    ),
                ),
                (
                    "second",
                    "orange",
                    HistogramValues(
                        values=np.array([-2.0, 4.0]),
                        variances=np.array([4.0, 10.0]),
                    ),
                ),
            ],
        )

        np.testing.assert_allclose(total.values, [1.0, 3.0])
        np.testing.assert_allclose(total.variances, [9.0, 11.0])

    def test_rejects_a_luminosity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.root"
            write_plot_sample(path, weights=np.array([1.0, 2.0]), lumi=300_000.0)
            with self.assertRaisesRegex(ValueError, "not normalized"):
                _read_histogram(
                    path,
                    value_branch="reco_m_ZZ",
                    selection_branch="reconstructed",
                    edges=np.array([105.0, 130.0, 160.0]),
                    expected_lumi_pb_inverse=312_000.0,
                    step_size="1 MB",
                )

    def test_handles_undefined_values_without_accepting_invalid_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            edges = np.array([105.0, 130.0, 160.0])

            bad_weight = directory / "bad_weight.root"
            with uproot.recreate(bad_weight) as root_file:
                root_file["Analysis"] = {
                    "reco_m_ZZ": np.array([120.0, 121.0, np.nan]),
                    "reconstructed": np.array([True, True, False]),
                    "weight": np.array([1.0, np.nan, np.nan]),
                    "lumi": np.full(3, 312_000.0),
                }
            with self.assertRaisesRegex(ValueError, "selected event weights"):
                _read_histogram(
                    bad_weight,
                    value_branch="reco_m_ZZ",
                    selection_branch="reconstructed",
                    edges=edges,
                    expected_lumi_pb_inverse=312_000.0,
                    step_size="1 MB",
                )

            bad_value = directory / "bad_value.root"
            with uproot.recreate(bad_value) as root_file:
                root_file["Analysis"] = {
                    "reco_m_ZZ": np.array([120.0, np.nan, np.nan]),
                    "reconstructed": np.array([True, True, False]),
                    "weight": np.array([1.0, 1.0, np.nan]),
                    "lumi": np.full(3, 312_000.0),
                }
            histogram = _read_histogram(
                bad_value,
                value_branch="reco_m_ZZ",
                selection_branch="reconstructed",
                edges=edges,
                expected_lumi_pb_inverse=312_000.0,
                step_size="1 MB",
            )
            np.testing.assert_allclose(histogram.values, [1.0, 0.0])
            np.testing.assert_allclose(histogram.variances, [1.0, 0.0])

            infinite_value = directory / "infinite_value.root"
            with uproot.recreate(infinite_value) as root_file:
                root_file["Analysis"] = {
                    "reco_m_ZZ": np.array([120.0, np.inf]),
                    "reconstructed": np.array([True, True]),
                    "weight": np.array([1.0, 1.0]),
                    "lumi": np.full(2, 312_000.0),
                }
            with self.assertRaisesRegex(ValueError, "infinite selected values"):
                _read_histogram(
                    infinite_value,
                    value_branch="reco_m_ZZ",
                    selection_branch="reconstructed",
                    edges=edges,
                    expected_lumi_pb_inverse=312_000.0,
                    step_size="1 MB",
                )

            sentinel = directory / "sentinel.root"
            with uproot.recreate(sentinel) as root_file:
                root_file["Analysis"] = {
                    "reco_m_ZZ": np.array([120.0, np.nan]),
                    "reconstructed": np.array([True, False]),
                    "weight": np.array([-1.0, np.nan]),
                    "lumi": np.full(2, 312_000.0),
                }
            histogram = _read_histogram(
                sentinel,
                value_branch="reco_m_ZZ",
                selection_branch="reconstructed",
                edges=edges,
                expected_lumi_pb_inverse=312_000.0,
                step_size="1 MB",
            )
            np.testing.assert_allclose(histogram.values, [-1.0, 0.0])
            np.testing.assert_allclose(histogram.variances, [1.0, 0.0])

    def test_creates_all_reco_and_fiducial_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_plot_sample(
                directory / "data.root",
                weights=np.array([1.0, 1.0, -1.0, 1.0]),
                data=True,
            )
            write_plot_sample(
                directory / "ZZ_pythia.root",
                weights=np.array([1.2, 0.8, 1.1, 0.9]),
            )
            write_plot_sample(
                directory / "gg_H_pythia.root",
                weights=np.array([0.3, 0.4, 0.2, 0.1]),
            )
            # Make truth-only Herwig content deliberately different from data;
            # the report must read these files for its fiducial column.
            write_plot_sample(
                directory / "ZZ_herwig.root",
                weights=np.array([1.0, 1.0, 1.0, 1.0]),
            )
            write_plot_sample(
                directory / "gg_H_herwig.root",
                weights=np.array([0.2, 0.2, 0.2, 0.2]),
            )
            output = directory / "comparison.pdf"
            pages = create_comparison_pdf(
                directory,
                output,
                luminosity_fb=312.0,
                step_size="1 MB",
            )
            self.assertEqual(pages, len(OBSERVABLES))
            payload = output.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF"))
            page_objects = re.findall(rb"/Type\s*/Page\b", payload)
            self.assertEqual(len(page_objects), len(OBSERVABLES))

    def test_refuses_to_overwrite_an_input_or_write_a_non_pdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in (
                "data.root",
                "ZZ_pythia.root",
                "gg_H_pythia.root",
                "ZZ_herwig.root",
                "gg_H_herwig.root",
            ):
                write_plot_sample(directory / name, weights=np.array([1.0, 1.0]))
            with self.assertRaisesRegex(ValueError, r"\.pdf extension"):
                create_comparison_pdf(
                    directory,
                    directory / "report.root",
                    overwrite=True,
                )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                create_comparison_pdf(
                    directory,
                    directory / "data.root",
                    overwrite=True,
                )


if __name__ == "__main__":
    unittest.main()
