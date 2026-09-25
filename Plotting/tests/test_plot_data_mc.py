from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import uproot

from Plotting.plot_data_mc import (
    KINEMATIC_FIELDS,
    OBSERVABLES,
    _read_histogram,
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
