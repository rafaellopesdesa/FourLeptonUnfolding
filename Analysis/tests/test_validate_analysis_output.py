from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import uproot

from Analysis.validate_analysis_output import KINEMATIC_FIELDS, validate_analysis_output


def write_output(
    path: Path,
    *,
    entries: int = 3,
    cross_section: float = 0.2,
    weights: np.ndarray | None = None,
    missing_branch: str | None = None,
) -> None:
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        entries = len(weights)
    else:
        weights = np.ones(entries, dtype=np.float64)
    index = np.arange(entries)
    arrays: dict[str, np.ndarray] = {
            "event_id": index.astype(np.uint64),
            "event_number": index.astype(np.int64),
            "weight": weights,
            "cross_section_pb": np.full(entries, cross_section, dtype=np.float64),
            "fiducial": np.ones(entries, dtype=np.bool_),
            "reconstructed": np.ones(entries, dtype=np.bool_),
            "truth_type": np.zeros(entries, dtype=np.int8),
            "reco_type": np.zeros(entries, dtype=np.int8),
            "type": np.zeros(entries, dtype=np.int8),
    }
    for level in ("truth", "reco"):
        for field in KINEMATIC_FIELDS:
            arrays[f"{level}_{field}"] = np.ones(entries, dtype=np.float32)
    if missing_branch is not None:
        del arrays[missing_branch]
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


class ValidateAnalysisOutputTest(unittest.TestCase):
    def test_accepts_a_complete_compact_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.root"
            write_output(path)
            summary = validate_analysis_output(path, expected_entries=3)
            self.assertEqual(summary.entries, 3)
            self.assertEqual(summary.sum_weights, 3.0)
            self.assertEqual(summary.sum_squared_weights, 3.0)
            self.assertEqual(summary.cross_section_pb, 0.2)

    def test_accepts_signed_weights_with_negative_net_sum_and_reports_sumw2(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.root"
            write_output(path, weights=np.array([1.0, -2.0, 0.0]))

            summary = validate_analysis_output(path, expected_entries=3)

            self.assertEqual(summary.entries, 3)
            self.assertAlmostEqual(summary.sum_weights, -1.0)
            self.assertAlmostEqual(summary.sum_squared_weights, 5.0)
            self.assertEqual(summary.cross_section_pb, 0.2)

    def test_accepts_signed_weights_with_exactly_zero_net_sum(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.root"
            write_output(path, weights=np.array([1.0, -1.0]))

            summary = validate_analysis_output(path, expected_entries=2)

            self.assertEqual(summary.sum_weights, 0.0)
            self.assertEqual(summary.sum_squared_weights, 2.0)

    def test_rejects_wrong_entry_count_and_missing_cross_section(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.root"
            write_output(path, cross_section=float("nan"))
            with self.assertRaisesRegex(ValueError, "contains 3 entries"):
                validate_analysis_output(path, expected_entries=4)
            with self.assertRaisesRegex(ValueError, "cross section"):
                validate_analysis_output(path, expected_entries=3)

    def test_rejects_an_incomplete_physics_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.root"
            write_output(path, missing_branch="truth_m_ZZ")
            with self.assertRaisesRegex(KeyError, "truth_m_ZZ"):
                validate_analysis_output(path, expected_entries=3)


if __name__ == "__main__":
    unittest.main()
