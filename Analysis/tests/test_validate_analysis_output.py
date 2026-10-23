from __future__ import annotations

import json
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
    lhe_columns: dict[str, np.ndarray] | None = None,
    lhe_metadata: dict | None = None,
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
    if lhe_columns is not None:
        arrays.update(lhe_columns)
    if missing_branch is not None:
        del arrays[missing_branch]
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays
        if lhe_metadata is not None:
            root_file["lhe_metadata"] = json.dumps(lhe_metadata)


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
            self.assertIsNone(summary.lhe_matched_entries)

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


def matched_lhe_columns(entries: int = 3) -> dict[str, np.ndarray]:
    """An undecayed gg_H hard event has no direct four-lepton candidate."""
    from Analysis.lhe import empty_columns

    columns = empty_columns(entries)
    columns["lhe_available"][:] = True
    columns["lhe_event_index"][:] = np.arange(1, entries + 1)
    columns["lhe_source_id"][:] = np.uint64(0xFEDCBA9876543210)
    columns["lhe_process_id"][:] = 42
    columns["lhe_weight"][:] = np.resize(np.array([-2.5, 0.0, 3.25]), entries)
    columns["lhe_scale"][:] = 125.0
    columns["lhe_has_higgs"][:] = True
    columns["lhe_higgs_kinematics_valid"][:] = True
    columns["lhe_higgs_m"][:] = 125.0
    columns["lhe_higgs_y"][:] = 0.5
    columns["lhe_higgs_pT"][:] = 20.0
    return columns


class ValidateLHEOutputTest(unittest.TestCase):
    def test_optional_full_digest_metadata_must_match_row_source_ids_and_indices(self):
        identifier = "fedcba9876543210"
        source = {"lhe_source_id": identifier, "lhe_sha256": identifier + "0" * 48,
                  "lhe_event_count": 3}
        metadata = {"schema_version": 1, "event_number": "lhe_event_index_1based", "sources": [source]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.root"
            columns = matched_lhe_columns()
            write_output(path, lhe_columns=columns, lhe_metadata=metadata)
            self.assertEqual(validate_analysis_output(path, require_lhe=True).lhe_matched_entries, 3)
            for bad_metadata, error in (
                ({**metadata, "sources": [{**source, "lhe_sha256": "0" * 64}]}, "source ID, SHA-256"),
                ({**metadata, "sources": []}, "absent from lhe_metadata"),
                ({**metadata, "sources": [{**source, "lhe_event_count": 2}]}, "exceeds its metadata"),
            ):
                with self.subTest(error=error):
                    write_output(path, lhe_columns=columns, lhe_metadata=bad_metadata)
                    with self.assertRaisesRegex(ValueError, error):
                        validate_analysis_output(path)

    def test_legacy_is_accepted_by_default_but_not_require_lhe(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.root"
            write_output(path)
            summary = validate_analysis_output(path)
            self.assertIsNone(summary.lhe_matched_entries)
            self.assertIsNone(summary.lhe_four_lepton_entries)
            self.assertIsNone(summary.lhe_higgs_entries)
            with self.assertRaisesRegex(KeyError, "missing required LHE branches"):
                validate_analysis_output(path, require_lhe=True)

    def test_any_partial_lhe_schema_is_rejected_even_without_require_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.root"
            write_output(path, lhe_columns={"lhe_weight": np.ones(3, dtype=np.float64)})
            with self.assertRaisesRegex(KeyError, "lhe_available"):
                validate_analysis_output(path)

    def test_matched_undecayed_higgs_and_signed_raw_weights_are_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "undecayed.root"
            write_output(path, lhe_columns=matched_lhe_columns())
            summary = validate_analysis_output(path, require_lhe=True, step_size=1)
            self.assertEqual(summary.lhe_matched_entries, 3)
            self.assertEqual(summary.lhe_four_lepton_entries, 0)
            self.assertEqual(summary.lhe_higgs_entries, 3)
            self.assertEqual(summary.sum_weights, 3.0)
            with uproot.open(path) as root_file:
                np.testing.assert_array_equal(root_file["Analysis"]["lhe_weight"].array(library="np"),
                                              [-2.5, 0.0, 3.25])

    def test_unavailable_lhe_schema_is_explicit_and_requirement_is_strict(self):
        from Analysis.lhe import empty_columns

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unavailable.root"
            write_output(path, lhe_columns=empty_columns(3))
            summary = validate_analysis_output(path)
            self.assertEqual(summary.lhe_matched_entries, 0)
            self.assertEqual(summary.lhe_four_lepton_entries, 0)
            with self.assertRaisesRegex(ValueError, "matched LHE event for every row"):
                validate_analysis_output(path, require_lhe=True)
            bad = empty_columns(3)
            bad["lhe_weight"][0] = 1.0
            write_output(path, lhe_columns=bad)
            with self.assertRaisesRegex(ValueError, "unavailable LHE rows require NaN"):
                validate_analysis_output(path)

    def test_valid_and_degenerate_four_lepton_records_have_distinct_flags(self):
        columns = matched_lhe_columns()
        columns["lhe_has_four_leptons"][:2] = True
        columns["lhe_n_final_leptons"][:2] = 4
        columns["lhe_type"][:2] = 2
        columns["lhe_kinematics_valid"][0] = True
        for field in KINEMATIC_FIELDS:
            columns[f"lhe_{field}"][0] = 1.0
        for field in ("m_Z1", "m_Z2", "m_ZZ", "y_ZZ", "pT_ZZ"):
            columns[f"lhe_{field}"][1] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "four_leptons.root"
            write_output(path, lhe_columns=columns)
            summary = validate_analysis_output(path, require_lhe=True)
            self.assertEqual(summary.lhe_four_lepton_entries, 2)
            columns["lhe_kinematics_valid"][1] = True
            write_output(path, lhe_columns=columns)
            with self.assertRaisesRegex(ValueError, "finite four-lepton kinematics"):
                validate_analysis_output(path)

    def test_higgs_presence_without_identifiable_kinematics_is_allowed(self):
        columns = matched_lhe_columns()
        columns["lhe_higgs_kinematics_valid"][0] = False
        for field in ("m", "y", "pT"):
            columns[f"lhe_higgs_{field}"][0] = np.nan
        columns["lhe_has_higgs"][1] = False
        columns["lhe_higgs_kinematics_valid"][1] = False
        for field in ("m", "y", "pT"):
            columns[f"lhe_higgs_{field}"][1] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ambiguous_higgs.root"
            write_output(path, lhe_columns=columns)
            summary = validate_analysis_output(path, require_lhe=True)
            self.assertEqual(summary.lhe_higgs_entries, 2)
            columns["lhe_higgs_kinematics_valid"][1] = True
            write_output(path, lhe_columns=columns)
            with self.assertRaisesRegex(ValueError, "requires lhe_has_higgs"):
                validate_analysis_output(path)

    def test_identity_dtype_and_one_based_source_index_are_checked(self):
        modifications = (
            ("lhe_event_index", np.arange(3, dtype=np.int64), "one-based source event index"),
            ("lhe_source_id", np.ones(3, dtype=np.uint32), "scalar uint64 branch"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity.root"
            for name, values, error in modifications:
                with self.subTest(name=name, values=values):
                    columns = matched_lhe_columns()
                    columns[name] = values
                    write_output(path, lhe_columns=columns)
                    with self.assertRaisesRegex(ValueError, error):
                        validate_analysis_output(path)
            columns = matched_lhe_columns()
            columns["lhe_source_id"][:] = 0  # A truncated SHA-256 may be zero.
            write_output(path, lhe_columns=columns)
            self.assertEqual(validate_analysis_output(path, require_lhe=True).lhe_matched_entries, 3)

    def test_flag_inconsistencies_and_nonfinite_lhe_metadata_are_rejected(self):
        modifications = (
            ("lhe_available", False, "flags require lhe_available"),
            ("lhe_weight", np.nan, "finite lhe_weight"),
            ("lhe_scale", np.inf, "infinite LHE values"),
            ("lhe_type", 0, "channel type"),
            ("lhe_m_ZZ", 125.0, "without an LHE four-lepton candidate"),
            ("lhe_higgs_pT", np.nan, "finite Higgs kinematics"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inconsistent.root"
            for name, value, error in modifications:
                with self.subTest(name=name):
                    columns = matched_lhe_columns()
                    columns[name][0] = value
                    write_output(path, lhe_columns=columns)
                    with self.assertRaisesRegex(ValueError, error):
                        validate_analysis_output(path)


if __name__ == "__main__":
    unittest.main()
