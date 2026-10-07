from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from BackgroundRemoval.common import (
    FEATURES,
    CORRECTION_MASS_WINDOW,
    assert_file_unchanged,
    capture_file_provenance,
    deterministic_split,
    known_weight_measures_compatible,
    ratio_outputs,
    row_fingerprint_ids,
    stable_logit,
    transformed_features,
    validate_manifest_mass_window,
    weight_measure_contract,
)


class CommonUtilitiesTest(unittest.TestCase):
    def test_known_nominal_weight_measures_are_compatible(self):
        expected_events = weight_measure_contract("weight", luminosity_fb=312.0)
        cross_section_pb = weight_measure_contract(
            "weight_nominal_pb", luminosity_fb=312.0
        )

        self.assertTrue(
            known_weight_measures_compatible(expected_events, cross_section_pb)
        )
        self.assertTrue(
            known_weight_measures_compatible(cross_section_pb, expected_events)
        )
        self.assertFalse(
            known_weight_measures_compatible(
                weight_measure_contract("custom_weight", luminosity_fb=312.0),
                cross_section_pb,
            )
        )

    def test_file_provenance_sha_detects_same_size_rewrite_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "input.root"
            path.write_bytes(b"original")
            provenance = capture_file_provenance(path, include_sha256=True)
            self.assertIsNotNone(provenance["sha256"])

            path.write_bytes(b"modified")
            self.assertEqual(path.stat().st_size, provenance["size_bytes"])
            os.utime(
                path,
                ns=(path.stat().st_atime_ns, int(provenance["mtime_ns"])),
            )
            self.assertEqual(path.stat().st_mtime_ns, provenance["mtime_ns"])

            with self.assertRaisesRegex(RuntimeError, "input changed"):
                assert_file_unchanged(path, provenance)

    def test_periodic_encoding_has_no_boundary_discontinuity(self):
        raw = np.zeros((2, len(FEATURES)), dtype=np.float32)
        raw[0, :3] = -math.pi + 1.0e-6
        raw[1, :3] = math.pi - 1.0e-6
        transformed = transformed_features(raw)
        np.testing.assert_allclose(transformed[0, :6], transformed[1, :6], atol=3.0e-6)
        self.assertEqual(transformed.shape, (2, 11))

    def test_four_lepton_mass_is_selection_only(self):
        self.assertNotIn("reco_m_ZZ", FEATURES)

    def test_manifest_mass_window_contract_is_strict(self):
        manifest = {
            "selections": {
                "mass_window_gev": {
                    "branch": "reco_m_ZZ",
                    "low_exclusive": 130.0,
                    "high_exclusive": 160.0,
                }
            }
        }
        validate_manifest_mass_window(
            manifest,
            CORRECTION_MASS_WINDOW,
            artifact_label="Correction artifact",
        )
        manifest["selections"]["mass_window_gev"]["low_exclusive"] = 105.0
        with self.assertRaisesRegex(ValueError, "required strict"):
            validate_manifest_mass_window(
                manifest,
                CORRECTION_MASS_WINDOW,
                artifact_label="Correction artifact",
            )

    def test_split_is_deterministic_and_source_salted(self):
        event_ids = np.arange(10_000, dtype=np.uint64)
        first = deterministic_split(event_ids, source="gg_H", seed=123)
        second = deterministic_split(event_ids, source="gg_H", seed=123)
        zz = deterministic_split(event_ids, source="ZZ", seed=123)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, zz))
        fractions = [np.mean(first == code) for code in (0, 1, 2)]
        np.testing.assert_allclose(fractions, [0.60, 0.15, 0.25], atol=0.02)

    def test_row_fingerprints_group_exact_bootstrap_duplicates(self):
        rows = np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.1]],
            dtype=np.float32,
        )
        identities = row_fingerprint_ids(rows)
        self.assertEqual(identities[0], identities[1])
        self.assertNotEqual(identities[0], identities[2])
        splits = deterministic_split(identities, source="data", seed=123)
        self.assertEqual(splits[0], splits[1])

    def test_ratio_and_background_removal_formula(self):
        # Four identical members make the aggregation transparent.
        member_scores = np.full((3, 4), [[0.2], [0.5], [0.8]])
        outputs = ratio_outputs(
            member_scores,
            calibration_scale=1.0,
            calibration_bias=0.0,
            yield_ratio=0.25,
            logit_clip=30.0,
        )
        expected_shape = np.array([0.25, 1.0, 4.0])
        expected_physical = 0.25 * expected_shape
        np.testing.assert_allclose(outputs["background_shape_ratio"], expected_shape)
        np.testing.assert_allclose(outputs["shape_ratio"], expected_shape)
        np.testing.assert_allclose(
            outputs["signal_to_background_ratio"], expected_physical
        )
        np.testing.assert_allclose(outputs["physical_ratio"], expected_physical)
        np.testing.assert_allclose(
            outputs["background_removal_weight"],
            expected_physical / (1.0 + expected_physical),
        )
        np.testing.assert_allclose(
            outputs["target_purity"],
            expected_physical / (1.0 + expected_physical),
        )
        np.testing.assert_allclose(outputs["signal_score_ensemble_std"], 0.0)

    def test_shape_ratio_is_distinct_from_purity_weight(self):
        scores = np.full((1, 4), 0.75)
        outputs = ratio_outputs(
            scores,
            calibration_scale=1.0,
            calibration_bias=0.0,
            yield_ratio=2.0,
            logit_clip=30.0,
        )
        self.assertAlmostEqual(outputs["background_shape_ratio"][0], 3.0)
        self.assertAlmostEqual(outputs["signal_to_background_ratio"][0], 6.0)
        self.assertAlmostEqual(outputs["background_removal_weight"][0], 6.0 / 7.0)

    def test_exact_saturated_scores_have_finite_logits(self):
        logits = stable_logit(np.array([0.0, 1.0]), logit_clip=100.0)
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertLess(logits[0], 0.0)
        self.assertGreater(logits[1], 0.0)


if __name__ == "__main__":
    unittest.main()
