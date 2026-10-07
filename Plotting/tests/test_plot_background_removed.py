from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import uproot

from Plotting.plot_background_removed import (
    APPLICATION_MANIFEST,
    _assert_campaign_unchanged,
    _read_signal_region_histogram,
    _sha256_file,
    _validate_campaign,
    create_background_removed_comparison_pdf,
)
from Plotting.plot_data_mc import KINEMATIC_FIELDS, OBSERVABLES


BACKGROUND_MODEL_SHA256 = "a" * 64
CORRECTION_MODEL_SHA256 = "b" * 64


def write_decorated_sample(
    path: Path,
    *,
    sample_kind: str,
    weights: np.ndarray,
    removal_weights: np.ndarray,
    masses: np.ndarray,
    reconstructed: np.ndarray,
    fiducial: np.ndarray,
    analysis_region: np.ndarray | None = None,
    phi_values: np.ndarray | None = None,
    lumi: float = 312_000.0,
) -> None:
    weights = np.asarray(weights, dtype=np.float64)
    removal_weights = np.asarray(removal_weights, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.bool_)
    fiducial = np.asarray(fiducial, dtype=np.bool_)
    size = len(weights)
    if analysis_region is None:
        analysis_region = (
            reconstructed & (masses > 115.0) & (masses < 130.0)
        ).astype(np.uint8)
    else:
        analysis_region = np.asarray(analysis_region, dtype=np.uint8)

    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight": weights,
        "lumi": np.full(size, lumi, dtype=np.float64),
        "luminosity_fb": np.full(size, lumi / 1000.0, dtype=np.float64),
        "reconstructed": reconstructed,
        "fiducial": fiducial,
        "analysis_region": analysis_region,
        "background_removal_weight": removal_weights,
        "reco_type": (np.arange(size) % 4).astype(np.int8),
    }
    for field in KINEMATIC_FIELDS:
        if field.startswith("cos_theta"):
            values = np.linspace(-0.8, 0.8, size)
        elif field == "Phi" and phi_values is not None:
            values = np.asarray(phi_values, dtype=np.float32)
        elif field in {"Phi", "Phi1", "Psi"}:
            values = np.linspace(-2.5, 2.5, size)
        elif field == "m_Z1":
            values = 70.0 + np.arange(size)
        elif field == "m_Z2":
            values = 20.0 + np.arange(size)
        elif field == "m_ZZ":
            values = masses
        elif field == "y_ZZ":
            values = np.linspace(-1.5, 1.5, size)
        elif field == "pT_ZZ":
            values = 10.0 + 10.0 * np.arange(size)
        else:  # pragma: no cover - guarded by the shared observable list
            raise AssertionError(field)
        arrays[f"reco_{field}"] = np.asarray(values, dtype=np.float32)

    metadata = {
        "format_version": 2,
        "sample_kind": sample_kind,
        "analysis_mass_window_GeV": [115.0, 130.0],
        "analysis_mass_window_strict": True,
        "background_model_manifest_sha256": BACKGROUND_MODEL_SHA256,
        "correction_model_manifest_sha256": CORRECTION_MODEL_SHA256,
    }
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays
        root_file["background_removal_metadata"] = json.dumps(metadata)


def write_application_manifest(
    directory: Path, *, data_file: str = "data.root"
) -> None:
    files = []
    for name, sample_kind in (
        (data_file, "data"),
        ("gg_H_pythia.root", "gg-h"),
    ):
        path = directory / name
        files.append(
            {
                "output": str(path),
                "output_sha256": _sha256_file(path),
                "sample_kind": sample_kind,
            }
        )
    manifest = {
        "format_version": 1,
        "background_model_manifest_sha256": BACKGROUND_MODEL_SHA256,
        "correction_model_manifest_sha256": CORRECTION_MODEL_SHA256,
        "files": files,
    }
    (directory / APPLICATION_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class PlotBackgroundRemovedTest(unittest.TestCase):
    def test_uses_data_purity_and_fiducial_ggh_target_with_sumw2(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            common = {
                "masses": np.array([120.0, 121.0, 122.0, 115.0, 130.0, 140.0]),
                "reconstructed": np.ones(6, dtype=np.bool_),
            }
            data_path = directory / "data.root"
            write_decorated_sample(
                data_path,
                sample_kind="data",
                weights=np.array([2.0, -1.0, 3.0, 100.0, 100.0, 100.0]),
                removal_weights=np.array([0.5, 0.25, 0.5, 1.0, 1.0, 1.0]),
                fiducial=np.ones(6, dtype=np.bool_),
                phi_values=np.array([0.2, 0.2, 1.2, 0.2, 0.2, 0.2]),
                **common,
            )

            data_histogram = _read_signal_region_histogram(
                data_path,
                value_branch="reco_Phi",
                edges=np.array([0.0, 1.0, 2.0]),
                expected_lumi_pb_inverse=312_000.0,
                step_size="1 MB",
                sample_kind="data",
            )
            np.testing.assert_allclose(data_histogram.values, [0.75, 1.5])
            np.testing.assert_allclose(data_histogram.variances, [1.0625, 2.25])

            gg_h_path = directory / "gg_H_pythia.root"
            write_decorated_sample(
                gg_h_path,
                sample_kind="gg-h",
                weights=np.array([2.0, 100.0, -1.0, 100.0, 100.0, 100.0]),
                removal_weights=np.ones(6),
                fiducial=np.array([True, False, True, True, True, True]),
                phi_values=np.array([0.2, 0.2, 1.2, 0.2, 0.2, 0.2]),
                **common,
            )

            gg_h_histogram = _read_signal_region_histogram(
                gg_h_path,
                value_branch="reco_Phi",
                edges=np.array([0.0, 1.0, 2.0]),
                expected_lumi_pb_inverse=312_000.0,
                step_size="1 MB",
                sample_kind="gg-h",
            )
            np.testing.assert_allclose(gg_h_histogram.values, [2.0, -1.0])
            np.testing.assert_allclose(gg_h_histogram.variances, [4.0, 1.0])

    def test_rejects_inconsistent_region_and_nonunity_ggh_factor(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inconsistent = directory / "data.root"
            write_decorated_sample(
                inconsistent,
                sample_kind="data",
                weights=np.array([1.0]),
                removal_weights=np.array([0.5]),
                masses=np.array([120.0]),
                reconstructed=np.array([True]),
                fiducial=np.array([True]),
                analysis_region=np.array([0], dtype=np.uint8),
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                _read_signal_region_histogram(
                    inconsistent,
                    value_branch="reco_Phi",
                    edges=np.array([-3.2, 3.2]),
                    expected_lumi_pb_inverse=312_000.0,
                    step_size="1 MB",
                    sample_kind="data",
                )

            gg_h = directory / "gg_H_pythia.root"
            write_decorated_sample(
                gg_h,
                sample_kind="gg-h",
                weights=np.array([1.0]),
                removal_weights=np.array([0.9]),
                masses=np.array([120.0]),
                reconstructed=np.array([True]),
                fiducial=np.array([True]),
            )
            with self.assertRaisesRegex(ValueError, "non-unity ggH"):
                _read_signal_region_histogram(
                    gg_h,
                    value_branch="reco_Phi",
                    edges=np.array([-3.2, 3.2]),
                    expected_lumi_pb_inverse=312_000.0,
                    step_size="1 MB",
                    sample_kind="gg-h",
                )

    def test_creates_one_reco_page_per_shared_observable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            common = {
                "weights": np.array([1.0, 1.0, 1.0, 1.0]),
                "masses": np.array([116.0, 120.0, 125.0, 129.0]),
                "reconstructed": np.ones(4, dtype=np.bool_),
                "fiducial": np.array([True, True, False, True]),
            }
            write_decorated_sample(
                directory / "data_0001.root",
                sample_kind="data",
                removal_weights=np.array([0.7, 0.8, 0.6, 0.9]),
                **common,
            )
            write_decorated_sample(
                directory / "gg_H_pythia.root",
                sample_kind="gg-h",
                removal_weights=np.ones(4),
                **common,
            )
            write_application_manifest(directory, data_file="data_0001.root")
            output = directory / "background_removed.pdf"
            pages = create_background_removed_comparison_pdf(
                directory,
                output,
                data_file="data_0001.root",
                step_size="1 MB",
            )
            self.assertEqual(pages, len(OBSERVABLES))
            payload = output.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF"))
            self.assertEqual(
                len(re.findall(rb"/Type\s*/Page\b", payload)), len(OBSERVABLES)
            )
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                create_background_removed_comparison_pdf(
                    directory, output, data_file="data_0001.root"
                )

    def test_rejects_stale_file_and_detects_change_during_plotting(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            common = {
                "masses": np.array([120.0]),
                "reconstructed": np.array([True]),
                "fiducial": np.array([True]),
            }
            write_decorated_sample(
                directory / "data.root",
                sample_kind="data",
                weights=np.array([1.0]),
                removal_weights=np.array([0.8]),
                **common,
            )
            write_decorated_sample(
                directory / "gg_H_pythia.root",
                sample_kind="gg-h",
                weights=np.array([1.0]),
                removal_weights=np.array([1.0]),
                **common,
            )
            write_application_manifest(directory)
            paths = {
                "data": directory / "data.root",
                "gg_H_pythia": directory / "gg_H_pythia.root",
            }
            snapshot = _validate_campaign(directory, paths)

            write_decorated_sample(
                directory / "gg_H_pythia.root",
                sample_kind="gg-h",
                weights=np.array([2.0]),
                removal_weights=np.array([1.0]),
                **common,
            )
            with self.assertRaisesRegex(RuntimeError, "changed while plotting"):
                _assert_campaign_unchanged(snapshot)
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                create_background_removed_comparison_pdf(
                    directory, directory / "comparison.pdf"
                )

    def test_rejects_undecorated_input_and_non_pdf_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with uproot.recreate(directory / "data.root") as root_file:
                root_file["Analysis"] = {"reco_Phi": np.array([0.0])}
            write_decorated_sample(
                directory / "gg_H_pythia.root",
                sample_kind="gg-h",
                weights=np.array([1.0]),
                removal_weights=np.array([1.0]),
                masses=np.array([120.0]),
                reconstructed=np.array([True]),
                fiducial=np.array([True]),
            )
            with self.assertRaisesRegex(ValueError, r"\.pdf extension"):
                create_background_removed_comparison_pdf(
                    directory, directory / "comparison.root"
                )
            with self.assertRaisesRegex(KeyError, "Application"):
                _read_signal_region_histogram(
                    directory / "data.root",
                    value_branch="reco_Phi",
                    edges=np.array([-3.2, 3.2]),
                    expected_lumi_pb_inverse=312_000.0,
                    step_size="1 MB",
                    sample_kind="data",
                )


if __name__ == "__main__":
    unittest.main()
