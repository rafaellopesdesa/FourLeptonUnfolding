from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import uproot

from Analysis.merge_analysis_outputs import (
    PseudoDataComponent,
    SignedExpectation,
    _draw_signed_counts,
    _effective_luminosity_fb,
    _recommended_ensemble_count,
    _sample_reconstructed,
    _signed_expectation,
    build_pseudo_data,
    build_pseudo_data_ensembles,
    merge_directory,
    scan_files,
)


def write_sample(
    path: Path,
    cross_section_pb: float,
    *,
    weights: np.ndarray | None = None,
    reconstructed: np.ndarray | None = None,
    event_start: int = 10,
) -> None:
    if weights is None:
        weights = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    size = len(weights)
    if reconstructed is None:
        reconstructed = np.resize(
            np.array([True, False, True, True], dtype=np.bool_), size
        )
    else:
        reconstructed = np.asarray(reconstructed, dtype=np.bool_)
    if len(reconstructed) != size:
        raise ValueError("weights and reconstructed must have the same length")

    index = np.arange(size)
    reco_type = np.where(reconstructed, index % 4, -1).astype(np.int8)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = {
            "event_id": index.astype(np.uint64),
            "event_number": np.arange(
                event_start, event_start + size, dtype=np.int64
            ),
            "weight": weights,
            "cross_section_pb": np.full(
                size, cross_section_pb, dtype=np.float64
            ),
            "fiducial": (index % 2 == 0),
            "reconstructed": reconstructed,
            "truth_type": (index % 4).astype(np.int8),
            "reco_type": reco_type,
            "type": reco_type,
            "reco_m_ZZ": (120.0 + index / 10.0).astype(np.float32),
        }


class SignedStatisticsTest(unittest.TestCase):
    def test_signed_count_moments_follow_skellam_expectation(self):
        expectation = SignedExpectation(positive=12.0, negative=4.0)
        positive_rng = np.random.default_rng(2026)
        negative_rng = np.random.default_rng(2027)
        draws = np.asarray(
            [
                _draw_signed_counts(expectation, positive_rng, negative_rng)
                for _ in range(50_000)
            ]
        )
        signed = draws[:, 0] - draws[:, 1]

        self.assertAlmostEqual(float(np.mean(signed)), expectation.net, delta=0.1)
        self.assertAlmostEqual(
            float(np.var(signed)),
            expectation.positive + expectation.negative,
            delta=0.3,
        )
        self.assertAlmostEqual(
            float(np.cov(draws[:, 0], draws[:, 1], ddof=0)[0, 1]),
            0.0,
            delta=0.15,
        )

    def test_signed_sums_expectations_and_effective_luminosity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "signed.root"
            write_sample(
                path,
                0.01,
                weights=np.array([2.0, -1.0, 0.5, -0.5]),
                reconstructed=np.array([True, True, False, True]),
            )
            stats, _ = scan_files([path], step_size="1 MB")

            self.assertAlmostEqual(stats.sum_weights, 1.0)
            self.assertAlmostEqual(stats.sum_squared_weights, 5.5)
            self.assertEqual(stats.reconstructed_positive_entries, 1)
            self.assertEqual(stats.reconstructed_negative_entries, 2)
            self.assertAlmostEqual(stats.reconstructed_positive_sum_weights, 2.0)
            self.assertAlmostEqual(
                stats.reconstructed_negative_sum_abs_weights, 1.5
            )
            self.assertAlmostEqual(
                stats.reconstructed_positive_sum_squared_weights, 4.0
            )
            self.assertAlmostEqual(
                stats.reconstructed_negative_sum_squared_weights, 1.25
            )

            expectation = _signed_expectation(stats, luminosity_fb=2.0)
            self.assertAlmostEqual(expectation.positive, 40.0)
            self.assertAlmostEqual(expectation.negative, 30.0)
            self.assertAlmostEqual(expectation.net, 10.0)
            self.assertAlmostEqual(_effective_luminosity_fb(stats, +1), 0.05)
            self.assertAlmostEqual(_effective_luminosity_fb(stats, -1), 0.12)

    def test_nonpositive_reconstructed_signed_cross_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "negative_reco.root"
            write_sample(
                path,
                0.01,
                weights=np.array([3.0, -1.0, -1.0]),
                reconstructed=np.array([False, True, True]),
            )
            stats, _ = scan_files([path], step_size="1 MB")
            with self.assertRaisesRegex(
                ValueError, "reconstructed signed cross section"
            ):
                _signed_expectation(stats, luminosity_fb=1.0)

    def test_signed_bootstrap_samples_with_replacement_and_excludes_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "signed.root"
            weights = np.array([4.0, 0.0, -3.0, 1.0, -1.0])
            write_sample(
                path,
                0.01,
                weights=weights,
                reconstructed=np.ones(len(weights), dtype=np.bool_),
            )
            stats, _ = scan_files([path], step_size="1 MB")

            positive = _sample_reconstructed(
                path,
                5_000,
                sign=+1,
                total_abs_weight=stats.reconstructed_positive_sum_weights,
                rng=np.random.default_rng(12),
                step_size="1 MB",
            )
            negative = _sample_reconstructed(
                path,
                5_000,
                sign=-1,
                total_abs_weight=stats.reconstructed_negative_sum_abs_weights,
                rng=np.random.default_rng(13),
                step_size="1 MB",
            )

            self.assertTrue(np.all(positive["weight"] > 0.0))
            self.assertTrue(np.all(negative["weight"] < 0.0))
            self.assertNotIn(11, positive["event_number"])
            self.assertNotIn(11, negative["event_number"])
            self.assertLess(np.unique(positive["event_number"]).size, 5_000)
            self.assertLess(np.unique(negative["event_number"]).size, 5_000)
            self.assertAlmostEqual(
                float(np.mean(positive["event_number"] == 10)), 0.8, delta=0.03
            )
            self.assertAlmostEqual(
                float(np.mean(negative["event_number"] == 12)), 0.75, delta=0.03
            )


class MergeAnalysisOutputsTest(unittest.TestCase):
    def test_positive_only_auto_ensembles_preserve_legacy_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for sample_index, sample in enumerate(
                ("ZZ_pythia", "ZZ_herwig", "gg_H_pythia", "gg_H_herwig")
            ):
                cross_section = 1.0e-5
                write_sample(
                    directory / f"{sample}_seed101.root",
                    cross_section,
                    event_start=1000 * sample_index,
                )
                write_sample(
                    directory / f"{sample}_seed102.root",
                    cross_section,
                    event_start=1000 * sample_index + 100,
                )

            merge_directory(
                directory,
                directory,
                luminosity_fb=300.0,
                seed=7,
                step_size="1 MB",
                zz_cross_section_pb=None,
                higgs_cross_section_pb=None,
                overwrite=False,
            )

            for sample in (
                "ZZ_pythia",
                "ZZ_herwig",
                "gg_H_pythia",
                "gg_H_herwig",
            ):
                with uproot.open(directory / f"{sample}.root") as root_file:
                    tree = root_file["Analysis"]
                    arrays = tree.arrays(library="np")
                    self.assertEqual(tree.num_entries, 8)
                    self.assertAlmostEqual(float(np.sum(arrays["weight"])), 8.0)
                    self.assertEqual(
                        arrays["event_id"].tolist(), list(range(8))
                    )

            manifest = json.loads(
                (directory / "pseudo_data_manifest.json").read_text()
            )
            self.assertEqual(manifest["recommended_ensemble_count"], 2)
            self.assertEqual(manifest["generated_ensemble_count"], 2)
            self.assertTrue((directory / "data.root").exists())
            self.assertTrue((directory / "data_0001.root").exists())

            with uproot.open(directory / "data.root") as root_file:
                tree = root_file["Analysis"]
                arrays = tree.arrays(library="np")
                self.assertGreater(tree.num_entries, 0)
                self.assertTrue(np.all(arrays["reconstructed"]))
                self.assertTrue(np.all(arrays["weight"] == 1.0))
                self.assertTrue(np.all(arrays["type"] == arrays["reco_type"]))
                self.assertEqual(
                    arrays["event_id"].tolist(), list(range(tree.num_entries))
                )
                metadata = json.loads(str(root_file["merge_metadata"]))
                self.assertEqual(metadata["total_observed"], tree.num_entries)
                self.assertEqual(metadata["total_entries"], tree.num_entries)

    def test_signed_pseudo_data_metadata_and_determinism(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pattern = np.tile(np.array([1.0, 1.0, 1.0, -1.0]), 25)
            reconstructed = np.ones(pattern.size, dtype=np.bool_)
            zz = directory / "ZZ.root"
            higgs = directory / "gg_H.root"
            write_sample(
                zz,
                1.0e-4,
                weights=pattern,
                reconstructed=reconstructed,
                event_start=1000,
            )
            write_sample(
                higgs,
                1.0e-4,
                weights=pattern,
                reconstructed=reconstructed,
                event_start=2000,
            )

            first = directory / "first.root"
            second = directory / "second.root"
            build_pseudo_data(
                zz,
                higgs,
                first,
                luminosity_fb=100.0,
                seed=123,
                step_size="1 MB",
                overwrite=False,
            )
            build_pseudo_data(
                zz,
                higgs,
                second,
                luminosity_fb=100.0,
                seed=123,
                step_size="1 MB",
                overwrite=False,
            )

            with uproot.open(first) as first_file, uproot.open(second) as second_file:
                first_arrays = first_file["Analysis"].arrays(library="np")
                second_arrays = second_file["Analysis"].arrays(library="np")
                np.testing.assert_array_equal(
                    first_arrays["event_number"], second_arrays["event_number"]
                )
                np.testing.assert_array_equal(
                    first_arrays["weight"], second_arrays["weight"]
                )
                self.assertEqual(
                    set(np.unique(first_arrays["weight"]).tolist()), {-1.0, 1.0}
                )
                self.assertTrue(np.all(first_arrays["reconstructed"]))
                metadata = json.loads(str(first_file["merge_metadata"]))

            self.assertEqual(
                metadata["total_entries"],
                metadata["total_sum_abs_weights"],
            )
            self.assertEqual(
                metadata["total_observed"],
                int(np.sum(first_arrays["weight"])),
            )
            self.assertEqual(
                metadata["total_entries"],
                metadata["total_observed_positive"]
                + metadata["total_observed_negative"],
            )
            self.assertEqual(
                metadata["total_observed"],
                metadata["total_observed_positive"]
                - metadata["total_observed_negative"],
            )
            for process in ("ZZ", "gg_H"):
                details = metadata["components"][process]
                self.assertAlmostEqual(
                    details["expected_net"],
                    details["expected_positive"] - details["expected_negative"],
                )
                self.assertEqual(
                    details["observed_net"],
                    details["observed_positive"] - details["observed_negative"],
                )

    def test_explicit_count_requires_override_above_recommendation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pattern = np.tile(np.array([1.0, 1.0, 1.0, -1.0]), 5)
            reconstructed = np.ones(pattern.size, dtype=np.bool_)
            zz = directory / "ZZ.root"
            higgs = directory / "gg_H.root"
            write_sample(
                zz, 1.0e-5, weights=pattern, reconstructed=reconstructed
            )
            write_sample(
                higgs,
                1.0e-5,
                weights=pattern,
                reconstructed=reconstructed,
                event_start=100,
            )
            zz_stats, _ = scan_files([zz], step_size="1 MB")
            higgs_stats, _ = scan_files([higgs], step_size="1 MB")
            components = [
                PseudoDataComponent(
                    "ZZ", zz, zz_stats, _signed_expectation(zz_stats, 100.0)
                ),
                PseudoDataComponent(
                    "gg_H",
                    higgs,
                    higgs_stats,
                    _signed_expectation(higgs_stats, 100.0),
                ),
            ]
            recommended, _ = _recommended_ensemble_count(components, 100.0)
            self.assertGreater(recommended, 0)

            with self.assertRaisesRegex(ValueError, "recommendation"):
                build_pseudo_data_ensembles(
                    zz,
                    higgs,
                    directory / "rejected",
                    luminosity_fb=100.0,
                    seed=4,
                    ensemble_count=recommended + 1,
                    allow_ensemble_oversubscription=False,
                    step_size="1 MB",
                    overwrite=False,
                )

            manifest = build_pseudo_data_ensembles(
                zz,
                higgs,
                directory / "allowed",
                luminosity_fb=100.0,
                seed=4,
                ensemble_count=recommended + 1,
                allow_ensemble_oversubscription=True,
                step_size="1 MB",
                overwrite=False,
            )
            self.assertEqual(
                manifest["generated_ensemble_count"], recommended + 1
            )
            self.assertTrue(
                (directory / "allowed" / "pseudo_data_manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
