from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

import Analysis.merge_analysis_outputs as merge_module
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
    merge_sample,
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
    def test_merged_luminosity_weights_preserve_signs_and_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "signed.root"
            destination = directory / "merged.root"
            write_sample(
                source,
                0.01,
                weights=np.array([2.0, -1.0, 1.0]),
                reconstructed=np.ones(3, dtype=np.bool_),
            )
            stats = merge_sample(
                [source],
                destination,
                step_size="1 MB",
                cross_section_override_pb=None,
                luminosity_fb=10.0,
                overwrite=False,
            )
            with uproot.open(destination) as root_file:
                arrays = root_file["Analysis"].arrays(library="np")
                metadata = json.loads(str(root_file["merge_metadata"]))
            self.assertAlmostEqual(float(np.sum(arrays["weight_shape"])), 3.0)
            self.assertAlmostEqual(
                float(np.sum(arrays["weight_nominal_pb"])), 0.01
            )
            self.assertAlmostEqual(float(np.sum(arrays["weight"])), 100.0)
            self.assertAlmostEqual(
                float(np.sum(arrays["weight"] ** 2)),
                stats.sum_squared_weights,
            )
            self.assertAlmostEqual(stats.sum_weights, 100.0)
            self.assertEqual(np.sign(arrays["weight"]).tolist(), [1.0, -1.0, 1.0])
            np.testing.assert_allclose(
                arrays["weight"], arrays["weight_nominal_pb"] * arrays["lumi"]
            )
            self.assertEqual(metadata["format_version"], 2)
            self.assertEqual(
                metadata["weight_semantics"], "expected_events_at_luminosity"
            )
            self.assertEqual(metadata["weight_units"], "events")
            self.assertAlmostEqual(metadata["input_sum_squared_weights"], 6.0)
            self.assertEqual(len(metadata["input_sources"]), 1)
            self.assertEqual(
                metadata["input_sources"][0]["sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertAlmostEqual(
                metadata["output_sum_squared_weights"],
                float(np.sum(arrays["weight"] ** 2)),
            )

    def test_merged_luminosity_weights_require_a_cross_section(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "missing_xsec.root"
            write_sample(source, float("nan"))
            with self.assertRaisesRegex(ValueError, "positive cross section"):
                merge_sample(
                    [source],
                    directory / "merged.root",
                    step_size="1 MB",
                    cross_section_override_pb=None,
                    luminosity_fb=312.0,
                    overwrite=False,
                )

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
                luminosity_fb=312.0,
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
                    self.assertAlmostEqual(
                        float(np.sum(arrays["weight_shape"])), 8.0
                    )
                    self.assertAlmostEqual(
                        float(np.sum(arrays["weight_nominal_pb"])), 1.0e-5
                    )
                    self.assertAlmostEqual(
                        float(np.sum(arrays["weight"])), 3.12
                    )
                    self.assertTrue(np.all(arrays["lumi"] == 312_000.0))
                    self.assertTrue(np.all(arrays["luminosity_fb"] == 312.0))
                    np.testing.assert_allclose(
                        arrays["weight"],
                        arrays["weight_nominal_pb"] * arrays["lumi"],
                    )
                    self.assertEqual(
                        arrays["event_id"].tolist(), list(range(8))
                    )

            manifest = json.loads(
                (directory / "pseudo_data_manifest.json").read_text()
            )
            self.assertEqual(manifest["recommended_ensemble_count"], 2)
            self.assertEqual(manifest["generated_ensemble_count"], 2)
            self.assertEqual(manifest["luminosity_fb"], 312.0)
            self.assertTrue((directory / "data.root").exists())
            self.assertTrue((directory / "data_0001.root").exists())
            for entry in manifest["files"]:
                pseudo_path = directory / entry["path"]
                self.assertEqual(
                    entry["sha256"],
                    hashlib.sha256(pseudo_path.read_bytes()).hexdigest(),
                )

            with uproot.open(directory / "data.root") as root_file:
                tree = root_file["Analysis"]
                arrays = tree.arrays(library="np")
                self.assertGreater(tree.num_entries, 0)
                self.assertTrue(np.all(arrays["reconstructed"]))
                self.assertTrue(np.all(arrays["weight"] == 1.0))
                self.assertTrue(np.all(arrays["weight_shape"] == 1.0))
                self.assertTrue(np.all(arrays["lumi"] == 312_000.0))
                self.assertTrue(np.all(arrays["luminosity_fb"] == 312.0))
                np.testing.assert_allclose(
                    arrays["weight_nominal_pb"],
                    arrays["weight"] / arrays["lumi"],
                )
                self.assertTrue(np.all(np.isnan(arrays["cross_section_pb"])))
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
                np.testing.assert_allclose(
                    first_arrays["weight"],
                    first_arrays["weight_nominal_pb"] * first_arrays["lumi"],
                )
                self.assertTrue(np.all(first_arrays["lumi"] == 100_000.0))
                self.assertTrue(np.all(first_arrays["luminosity_fb"] == 100.0))
                metadata = json.loads(str(first_file["merge_metadata"]))

            self.assertEqual(
                metadata["total_entries"],
                metadata["total_sum_abs_weights"],
            )
            self.assertEqual(
                metadata["total_sum_squared_weights"],
                int(np.sum(first_arrays["weight"] ** 2)),
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

    def test_merge_rejects_same_schema_source_replacement_after_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.root"
            replacement = directory / "replacement.root"
            output = directory / "merged.root"
            write_sample(source, 0.01, event_start=10)
            write_sample(
                replacement,
                0.01,
                weights=np.array([4.0, -1.0, 2.0, 1.0]),
                event_start=100,
            )
            real_tree_schema = merge_module._tree_schema
            replaced = False

            def replace_after_schema(*args, **kwargs):
                nonlocal replaced
                schema = real_tree_schema(*args, **kwargs)
                if not replaced:
                    os.replace(replacement, source)
                    replaced = True
                return schema

            with patch.object(
                merge_module, "_tree_schema", side_effect=replace_after_schema
            ):
                with self.assertRaisesRegex(RuntimeError, "source file changed"):
                    merge_sample(
                        [source],
                        output,
                        step_size="1 MB",
                        cross_section_override_pb=None,
                        luminosity_fb=10.0,
                        overwrite=False,
                    )
            self.assertFalse(output.exists())

    def test_merge_reads_snapshotted_target_if_input_symlink_is_retargeted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / "original.root"
            replacement = directory / "replacement.root"
            source_link = directory / "source.root"
            output = directory / "merged.root"
            write_sample(original, 0.01, event_start=10)
            write_sample(replacement, 0.01, event_start=100)
            source_link.symlink_to(original)
            real_tree_schema = merge_module._tree_schema
            retargeted = False

            def retarget_before_schema(*args, **kwargs):
                nonlocal retargeted
                if not retargeted:
                    source_link.unlink()
                    source_link.symlink_to(replacement)
                    retargeted = True
                return real_tree_schema(*args, **kwargs)

            with patch.object(
                merge_module, "_tree_schema", side_effect=retarget_before_schema
            ):
                merge_sample(
                    [source_link],
                    output,
                    step_size="1 MB",
                    cross_section_override_pb=None,
                    luminosity_fb=10.0,
                    overwrite=False,
                )

            with uproot.open(output) as root_file:
                arrays = root_file["Analysis"].arrays(
                    expressions=["event_number"], library="np"
                )
            self.assertEqual(arrays["event_number"].tolist(), [10, 11, 12, 13])

    def test_pseudo_data_rejects_source_replacement_between_sampling_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            zz = directory / "ZZ.root"
            higgs = directory / "gg_H.root"
            replacement = directory / "ZZ_replacement.root"
            output = directory / "data.root"
            weights = np.tile(np.array([1.0, 1.0, 1.0, -1.0]), 10)
            reconstructed = np.ones(weights.size, dtype=np.bool_)
            write_sample(zz, 1.0e-4, weights=weights, reconstructed=reconstructed)
            write_sample(
                higgs,
                1.0e-4,
                weights=weights,
                reconstructed=reconstructed,
                event_start=1000,
            )
            write_sample(
                replacement,
                1.0e-4,
                weights=np.roll(weights, 1),
                reconstructed=reconstructed,
                event_start=2000,
            )
            real_tree_schema = merge_module._tree_schema
            replaced = False

            def replace_after_schema(*args, **kwargs):
                nonlocal replaced
                schema = real_tree_schema(*args, **kwargs)
                if not replaced:
                    os.replace(replacement, zz)
                    replaced = True
                return schema

            with patch.object(
                merge_module, "_tree_schema", side_effect=replace_after_schema
            ):
                with self.assertRaisesRegex(RuntimeError, "source file changed"):
                    build_pseudo_data(
                        zz,
                        higgs,
                        output,
                        luminosity_fb=100.0,
                        seed=17,
                        step_size="1 MB",
                        overwrite=False,
                    )
            self.assertFalse(output.exists())

    def test_pseudo_campaign_rechecks_sources_between_ensembles(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            zz = directory / "ZZ.root"
            higgs = directory / "gg_H.root"
            replacement = directory / "ZZ_replacement.root"
            output_directory = directory / "campaign"
            weights = np.tile(np.array([1.0, 1.0, 1.0, -1.0]), 10)
            reconstructed = np.ones(weights.size, dtype=np.bool_)
            write_sample(zz, 1.0e-4, weights=weights, reconstructed=reconstructed)
            write_sample(
                higgs,
                1.0e-4,
                weights=weights,
                reconstructed=reconstructed,
                event_start=1000,
            )
            write_sample(
                replacement,
                1.0e-4,
                weights=np.roll(weights, 1),
                reconstructed=reconstructed,
                event_start=2000,
            )
            real_build = merge_module._build_pseudo_data_file
            calls = 0

            def replace_after_first_ensemble(*args, **kwargs):
                nonlocal calls
                result = real_build(*args, **kwargs)
                calls += 1
                if calls == 1:
                    os.replace(replacement, zz)
                return result

            with patch.object(
                merge_module,
                "_build_pseudo_data_file",
                side_effect=replace_after_first_ensemble,
            ):
                with self.assertRaisesRegex(RuntimeError, "source file changed"):
                    build_pseudo_data_ensembles(
                        zz,
                        higgs,
                        output_directory,
                        luminosity_fb=100.0,
                        seed=4,
                        ensemble_count=2,
                        allow_ensemble_oversubscription=True,
                        step_size="1 MB",
                        overwrite=False,
                    )
            self.assertFalse((output_directory / "data.root").exists())
            self.assertFalse((output_directory / "data_0001.root").exists())
            self.assertFalse(
                (output_directory / "pseudo_data_manifest.json").exists()
            )

    def test_failed_overwrite_campaign_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            zz = directory / "ZZ.root"
            higgs = directory / "gg_H.root"
            replacement = directory / "ZZ_replacement.root"
            output_directory = directory / "campaign"
            output_directory.mkdir()
            prior_files = {
                "data.root": b"prior primary pseudo-data\n",
                "data_0001.root": b"prior second pseudo-data\n",
                "pseudo_data_manifest.json": b'{"prior": true}\n',
            }
            for name, contents in prior_files.items():
                (output_directory / name).write_bytes(contents)

            weights = np.tile(np.array([1.0, 1.0, 1.0, -1.0]), 10)
            reconstructed = np.ones(weights.size, dtype=np.bool_)
            write_sample(zz, 1.0e-4, weights=weights, reconstructed=reconstructed)
            write_sample(
                higgs,
                1.0e-4,
                weights=weights,
                reconstructed=reconstructed,
                event_start=1000,
            )
            write_sample(
                replacement,
                1.0e-4,
                weights=np.roll(weights, 1),
                reconstructed=reconstructed,
                event_start=2000,
            )
            real_build = merge_module._build_pseudo_data_file
            calls = 0

            def replace_after_first_ensemble(*args, **kwargs):
                nonlocal calls
                result = real_build(*args, **kwargs)
                calls += 1
                if calls == 1:
                    os.replace(replacement, zz)
                return result

            with patch.object(
                merge_module,
                "_build_pseudo_data_file",
                side_effect=replace_after_first_ensemble,
            ):
                with self.assertRaisesRegex(RuntimeError, "source file changed"):
                    build_pseudo_data_ensembles(
                        zz,
                        higgs,
                        output_directory,
                        luminosity_fb=100.0,
                        seed=4,
                        ensemble_count=2,
                        allow_ensemble_oversubscription=True,
                        step_size="1 MB",
                        overwrite=True,
                    )
            for name, contents in prior_files.items():
                self.assertEqual((output_directory / name).read_bytes(), contents)

    def test_campaign_publish_rolls_back_a_mid_publish_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_outputs = [staging / "data.root", staging / "data_0001.root"]
            final_outputs = [directory / path.name for path in staged_outputs]
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            new_contents = [b"new data zero", b"new data one"]
            old_contents = [b"old data zero", b"old data one"]
            for staged, final, new, old in zip(
                staged_outputs,
                final_outputs,
                new_contents,
                old_contents,
                strict=True,
            ):
                staged.write_bytes(new)
                final.write_bytes(old)
            staged_manifest.write_bytes(b"new manifest")
            final_manifest.write_bytes(b"old manifest")
            expected_sha256 = {
                path: hashlib.sha256(contents).hexdigest()
                for path, contents in zip(
                    final_outputs, new_contents, strict=True
                )
            }
            real_link = os.link
            failed = False

            def fail_second_publish(source, destination, **kwargs):
                nonlocal failed
                if (
                    Path(source) == staged_outputs[1]
                    and Path(destination) == final_outputs[1]
                    and not failed
                ):
                    failed = True
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            with patch.object(
                merge_module.os, "link", side_effect=fail_second_publish
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "retained safety copies",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        list(zip(staged_outputs, final_outputs, strict=True)),
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=True,
                        expected_sha256=expected_sha256,
                    )

            self.assertIsInstance(raised.exception.__cause__, OSError)
            for final, old in zip(final_outputs, old_contents, strict=True):
                self.assertEqual(final.read_bytes(), old)
            self.assertEqual(final_manifest.read_bytes(), b"old manifest")

    def test_no_clobber_rollback_preserves_a_concurrent_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_outputs = [staging / "data.root", staging / "data_0001.root"]
            final_outputs = [directory / path.name for path in staged_outputs]
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            new_contents = [b"new data zero", b"new data one"]
            for staged, contents in zip(
                staged_outputs, new_contents, strict=True
            ):
                staged.write_bytes(contents)
            staged_manifest.write_bytes(b"new manifest")
            expected_sha256 = {
                path: hashlib.sha256(contents).hexdigest()
                for path, contents in zip(
                    final_outputs, new_contents, strict=True
                )
            }
            replacement_contents = b"concurrent data zero"
            real_link = os.link
            failed = False

            def replace_first_then_fail(source, destination, **kwargs):
                nonlocal failed
                if (
                    Path(source) == staged_outputs[1]
                    and Path(destination) == final_outputs[1]
                    and not failed
                ):
                    failed = True
                    replacement = directory / "concurrent.tmp"
                    replacement.write_bytes(replacement_contents)
                    os.replace(replacement, final_outputs[0])
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            with patch.object(
                merge_module.os, "link", side_effect=replace_first_then_fail
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rollback detected concurrent changes.*concurrent replacement",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        list(zip(staged_outputs, final_outputs, strict=True)),
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=False,
                        expected_sha256=expected_sha256,
                    )

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(final_outputs[0].read_bytes(), replacement_contents)
            self.assertFalse(final_outputs[1].exists())
            self.assertFalse(final_manifest.exists())

    def test_rollback_claim_prevents_check_then_unlink_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_outputs = [staging / "data.root", staging / "data_0001.root"]
            final_outputs = [directory / path.name for path in staged_outputs]
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            new_contents = [b"new data zero", b"new data one"]
            for staged, contents in zip(
                staged_outputs, new_contents, strict=True
            ):
                staged.write_bytes(contents)
            staged_manifest.write_bytes(b"new manifest")
            expected_sha256 = {
                path: hashlib.sha256(contents).hexdigest()
                for path, contents in zip(
                    final_outputs, new_contents, strict=True
                )
            }
            replacement_contents = b"replacement installed after identity check"
            real_link = os.link
            real_identity_check = merge_module._is_unchanged_published_path
            publication_failed = False
            replacement_installed = False

            def fail_second_publish(source, destination, **kwargs):
                nonlocal publication_failed
                if (
                    Path(source) == staged_outputs[1]
                    and Path(destination) == final_outputs[1]
                    and not publication_failed
                ):
                    publication_failed = True
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            def install_replacement_after_check(publication, path=None):
                nonlocal replacement_installed
                result = real_identity_check(publication, path)
                if result and not replacement_installed:
                    replacement_installed = True
                    replacement = directory / "concurrent.tmp"
                    replacement.write_bytes(replacement_contents)
                    os.replace(replacement, publication.path)
                return result

            with (
                patch.object(
                    merge_module.os,
                    "link",
                    side_effect=fail_second_publish,
                ),
                patch.object(
                    merge_module,
                    "_is_unchanged_published_path",
                    side_effect=install_replacement_after_check,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rollback detected concurrent changes.*concurrent replacement preserved",
                ):
                    merge_module._publish_staged_campaign(
                        list(zip(staged_outputs, final_outputs, strict=True)),
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=False,
                        expected_sha256=expected_sha256,
                    )

            self.assertTrue(replacement_installed)
            self.assertEqual(final_outputs[0].read_bytes(), replacement_contents)
            self.assertFalse(final_outputs[1].exists())
            self.assertFalse(final_manifest.exists())

    def test_overwrite_rollback_preserves_replacement_and_old_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_outputs = [staging / "data.root", staging / "data_0001.root"]
            final_outputs = [directory / path.name for path in staged_outputs]
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            new_contents = [b"new data zero", b"new data one"]
            old_contents = [b"old data zero", b"old data one"]
            for staged, final, new, old in zip(
                staged_outputs,
                final_outputs,
                new_contents,
                old_contents,
                strict=True,
            ):
                staged.write_bytes(new)
                final.write_bytes(old)
            staged_manifest.write_bytes(b"new manifest")
            final_manifest.write_bytes(b"old manifest")
            expected_sha256 = {
                path: hashlib.sha256(contents).hexdigest()
                for path, contents in zip(
                    final_outputs, new_contents, strict=True
                )
            }
            replacement_contents = b"concurrent data zero"
            real_link = os.link
            failed = False

            def replace_first_then_fail(source, destination, **kwargs):
                nonlocal failed
                if (
                    Path(source) == staged_outputs[1]
                    and Path(destination) == final_outputs[1]
                    and not failed
                ):
                    failed = True
                    replacement = directory / "concurrent.tmp"
                    replacement.write_bytes(replacement_contents)
                    os.replace(replacement, final_outputs[0])
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            with patch.object(
                merge_module.os, "link", side_effect=replace_first_then_fail
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "pre-campaign output preserved at",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        list(zip(staged_outputs, final_outputs, strict=True)),
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=True,
                        expected_sha256=expected_sha256,
                    )

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(final_outputs[0].read_bytes(), replacement_contents)
            self.assertEqual(final_outputs[1].read_bytes(), old_contents[1])
            self.assertEqual(final_manifest.read_bytes(), b"old manifest")
            recovery_paths = list(
                directory.glob(".data.root.pre-campaign-*.recovery")
            )
            self.assertEqual(len(recovery_paths), 1)
            self.assertEqual(
                (recovery_paths[0] / "data.root").read_bytes(),
                old_contents[0],
            )

    def test_overwrite_claim_rejects_destination_swapped_after_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_output = staging / "data.root"
            final_output = directory / "data.root"
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            staged_output.write_bytes(b"new data")
            final_output.write_bytes(b"old data")
            staged_manifest.write_bytes(b"new manifest")
            final_manifest.write_bytes(b"old manifest")
            external_old = directory / "old retained by concurrent writer"
            replacement_contents = b"concurrent replacement"
            real_replace = os.replace
            swapped = False

            def swap_before_claim(source, destination):
                nonlocal swapped
                if Path(source) == final_output and not swapped:
                    swapped = True
                    real_replace(final_output, external_old)
                    replacement = directory / "concurrent.tmp"
                    replacement.write_bytes(replacement_contents)
                    real_replace(replacement, final_output)
                return real_replace(source, destination)

            with patch.object(
                merge_module.os,
                "replace",
                side_effect=swap_before_claim,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "retained safety copies",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        [(staged_output, final_output)],
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=True,
                        expected_sha256={
                            final_output: hashlib.sha256(b"new data").hexdigest()
                        },
                    )

            self.assertTrue(swapped)
            self.assertIsInstance(raised.exception.__cause__, RuntimeError)
            self.assertIn(
                "output changed while it was being claimed",
                str(raised.exception.__cause__),
            )
            self.assertEqual(external_old.read_bytes(), b"old data")
            self.assertEqual(final_output.read_bytes(), replacement_contents)
            self.assertEqual(final_manifest.read_bytes(), b"old manifest")
            recovery_directories = list(
                directory.glob(".data.root.pre-campaign-*.recovery")
            )
            self.assertEqual(len(recovery_directories), 1)
            self.assertEqual(
                (recovery_directories[0] / "data.root").read_bytes(),
                replacement_contents,
            )

    def test_backup_survives_replacement_immediately_after_restore_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_outputs = [staging / "data.root", staging / "data_0001.root"]
            final_outputs = [directory / path.name for path in staged_outputs]
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            new_contents = [b"new data zero", b"new data one"]
            old_contents = [b"old data zero", b"old data one"]
            for staged, final, new, old in zip(
                staged_outputs,
                final_outputs,
                new_contents,
                old_contents,
                strict=True,
            ):
                staged.write_bytes(new)
                final.write_bytes(old)
            staged_manifest.write_bytes(b"new manifest")
            final_manifest.write_bytes(b"old manifest")
            expected_sha256 = {
                path: hashlib.sha256(contents).hexdigest()
                for path, contents in zip(
                    final_outputs, new_contents, strict=True
                )
            }
            replacement_contents = b"replacement immediately after restore link"
            real_link = os.link
            publication_failed = False
            replacement_installed = False

            def fail_and_replace_after_restore(source, destination, **kwargs):
                nonlocal publication_failed, replacement_installed
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path == staged_outputs[1]
                    and destination_path == final_outputs[1]
                    and not publication_failed
                ):
                    publication_failed = True
                    raise OSError("injected publication failure")
                result = real_link(source, destination, **kwargs)
                if (
                    destination_path == final_outputs[0]
                    and ".data.root.pre-campaign-" in source_path.parent.name
                    and not replacement_installed
                ):
                    replacement_installed = True
                    replacement = directory / "concurrent.tmp"
                    replacement.write_bytes(replacement_contents)
                    os.replace(replacement, final_outputs[0])
                return result

            with patch.object(
                merge_module.os,
                "link",
                side_effect=fail_and_replace_after_restore,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "safety copy preserved",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        list(zip(staged_outputs, final_outputs, strict=True)),
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=True,
                        expected_sha256=expected_sha256,
                    )

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertTrue(replacement_installed)
            self.assertEqual(final_outputs[0].read_bytes(), replacement_contents)
            self.assertEqual(final_outputs[1].read_bytes(), old_contents[1])
            recovery_directories = list(
                directory.glob(".data.root.pre-campaign-*.recovery")
            )
            self.assertEqual(len(recovery_directories), 1)
            self.assertEqual(
                (recovery_directories[0] / "data.root").read_bytes(),
                old_contents[0],
            )

    def test_rollback_preserves_a_concurrent_directory_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_outputs = [staging / "data.root", staging / "data_0001.root"]
            final_outputs = [directory / path.name for path in staged_outputs]
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            new_contents = [b"new data zero", b"new data one"]
            for staged, contents in zip(
                staged_outputs, new_contents, strict=True
            ):
                staged.write_bytes(contents)
            staged_manifest.write_bytes(b"new manifest")
            expected_sha256 = {
                path: hashlib.sha256(contents).hexdigest()
                for path, contents in zip(
                    final_outputs, new_contents, strict=True
                )
            }
            real_link = os.link
            failed = False

            def install_directory_then_fail(source, destination, **kwargs):
                nonlocal failed
                if (
                    Path(source) == staged_outputs[1]
                    and Path(destination) == final_outputs[1]
                    and not failed
                ):
                    failed = True
                    final_outputs[0].unlink()
                    final_outputs[0].mkdir()
                    (final_outputs[0] / "marker.txt").write_bytes(b"keep me")
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            with patch.object(
                merge_module.os,
                "link",
                side_effect=install_directory_then_fail,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "concurrent replacement preserved",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        list(zip(staged_outputs, final_outputs, strict=True)),
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=False,
                        expected_sha256=expected_sha256,
                    )

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertFalse(final_outputs[0].exists())
            recovery_directories = list(
                directory.glob(
                    ".data.root.concurrent-replacement-*.recovery"
                )
            )
            self.assertEqual(len(recovery_directories), 1)
            recovered_directory = recovery_directories[0] / "data.root"
            self.assertTrue(recovered_directory.is_dir())
            self.assertEqual(
                (recovered_directory / "marker.txt").read_bytes(),
                b"keep me",
            )

    def test_interrupt_after_publish_link_rolls_back_registered_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_output = staging / "data.root"
            final_output = directory / "data.root"
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            staged_output.write_bytes(b"new data")
            staged_manifest.write_bytes(b"new manifest")
            real_link = os.link
            interrupted = False

            def interrupt_after_link(source, destination, **kwargs):
                nonlocal interrupted
                result = real_link(source, destination, **kwargs)
                if Path(source) == staged_output and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                return result

            with patch.object(
                merge_module.os,
                "link",
                side_effect=interrupt_after_link,
            ):
                with self.assertRaisesRegex(
                    merge_module.CampaignPublicationError,
                    "rollback completed without retained claims",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        [(staged_output, final_output)],
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=False,
                        expected_sha256={
                            final_output: hashlib.sha256(b"new data").hexdigest()
                        },
                    )

            self.assertTrue(interrupted)
            self.assertIsInstance(raised.exception.__cause__, KeyboardInterrupt)
            self.assertFalse(final_output.exists())
            self.assertFalse(final_manifest.exists())

    def test_interrupt_after_backup_claim_preserves_registered_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_output = staging / "data.root"
            final_output = directory / "data.root"
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            staged_output.write_bytes(b"new data")
            final_output.write_bytes(b"old data")
            staged_manifest.write_bytes(b"new manifest")
            final_manifest.write_bytes(b"old manifest")
            real_replace = os.replace
            interrupted = False

            def interrupt_after_claim(source, destination):
                nonlocal interrupted
                result = real_replace(source, destination)
                if Path(source) == final_output and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                return result

            with patch.object(
                merge_module.os,
                "replace",
                side_effect=interrupt_after_claim,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "safety copy preserved",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        [(staged_output, final_output)],
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=True,
                        expected_sha256={
                            final_output: hashlib.sha256(b"new data").hexdigest()
                        },
                    )

            self.assertTrue(interrupted)
            self.assertIsInstance(raised.exception.__cause__, KeyboardInterrupt)
            self.assertEqual(final_output.read_bytes(), b"old data")
            self.assertEqual(final_manifest.read_bytes(), b"old manifest")
            recovery_directories = list(
                directory.glob(".data.root.pre-campaign-*.recovery")
            )
            self.assertEqual(len(recovery_directories), 1)
            self.assertEqual(
                (recovery_directories[0] / "data.root").read_bytes(),
                b"old data",
            )

    def test_failed_durable_recovery_retains_staging_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            retained_staging = None
            real_link = os.link
            publication_failed = False

            def fail_second_publish(source, destination, **kwargs):
                nonlocal publication_failed
                source_path = Path(source)
                if source_path.name == "data_0001.root" and not publication_failed:
                    publication_failed = True
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            with self.assertRaisesRegex(
                merge_module.CampaignPublicationError,
                "automatic cleanup disabled",
            ) as raised:
                with merge_module._campaign_staging_directory(
                    directory
                ) as staging:
                    retained_staging = staging
                    staged_outputs = [
                        staging / "data.root",
                        staging / "data_0001.root",
                    ]
                    final_outputs = [directory / path.name for path in staged_outputs]
                    staged_manifest = staging / "pseudo_data_manifest.json"
                    final_manifest = directory / staged_manifest.name
                    new_contents = [b"new data zero", b"new data one"]
                    old_contents = [b"old data zero", b"old data one"]
                    for staged, final, new, old in zip(
                        staged_outputs,
                        final_outputs,
                        new_contents,
                        old_contents,
                        strict=True,
                    ):
                        staged.write_bytes(new)
                        final.write_bytes(old)
                    staged_manifest.write_bytes(b"new manifest")
                    final_manifest.write_bytes(b"old manifest")
                    expected_sha256 = {
                        path: hashlib.sha256(contents).hexdigest()
                        for path, contents in zip(
                            final_outputs, new_contents, strict=True
                        )
                    }

                    with (
                        patch.object(
                            merge_module.os,
                            "link",
                            side_effect=fail_second_publish,
                        ),
                        patch.object(
                            merge_module,
                            "_preserve_rollback_path",
                            side_effect=OSError("injected recovery failure"),
                        ),
                    ):
                        merge_module._publish_staged_campaign(
                            list(zip(staged_outputs, final_outputs, strict=True)),
                            staged_manifest,
                            final_manifest,
                            staging_directory=staging,
                            overwrite=True,
                            expected_sha256=expected_sha256,
                        )

            self.assertIsNotNone(retained_staging)
            self.assertEqual(
                raised.exception.retained_staging_directory,
                retained_staging,
            )
            self.assertTrue(retained_staging.is_dir())
            self.assertEqual(
                (retained_staging / "backups" / "0000-data.root").read_bytes(),
                b"old data zero",
            )
            self.assertEqual(
                (
                    retained_staging / "backups" / "0001-data_0001.root"
                ).read_bytes(),
                b"old data one",
            )

    def test_manifest_publish_is_followed_by_output_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_output = staging / "data.root"
            final_output = directory / "data.root"
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            staged_output.write_bytes(b"new data")
            staged_manifest.write_bytes(b"new manifest")
            replacement_contents = b"replacement during manifest publication"
            real_link = os.link
            replacement_installed = False

            def replace_during_manifest_link(source, destination, **kwargs):
                nonlocal replacement_installed
                result = real_link(source, destination, **kwargs)
                if Path(source) == staged_manifest and not replacement_installed:
                    replacement_installed = True
                    replacement = directory / "concurrent.tmp"
                    replacement.write_bytes(replacement_contents)
                    os.replace(replacement, final_output)
                return result

            with patch.object(
                merge_module.os,
                "link",
                side_effect=replace_during_manifest_link,
            ):
                with self.assertRaisesRegex(
                    merge_module.CampaignPublicationError,
                    "concurrent replacement",
                ) as raised:
                    merge_module._publish_staged_campaign(
                        [(staged_output, final_output)],
                        staged_manifest,
                        final_manifest,
                        staging_directory=staging,
                        overwrite=False,
                        expected_sha256={
                            final_output: hashlib.sha256(b"new data").hexdigest()
                        },
                    )

            self.assertTrue(replacement_installed)
            self.assertIsInstance(raised.exception.__cause__, RuntimeError)
            self.assertEqual(final_output.read_bytes(), replacement_contents)
            self.assertFalse(final_manifest.exists())

    def test_interrupted_rollback_retains_publication_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            retained_staging = None
            real_link = os.link
            real_identity_check = merge_module._is_unchanged_published_path
            publication_failed = False
            rollback_interrupted = False

            def fail_second_publish(source, destination, **kwargs):
                nonlocal publication_failed
                if Path(source).name == "data_0001.root" and not publication_failed:
                    publication_failed = True
                    raise OSError("injected publication failure")
                return real_link(source, destination, **kwargs)

            def interrupt_after_rollback_claim(publication, path=None):
                nonlocal rollback_interrupted
                if (
                    path is not None
                    and Path(path).name.startswith("rollback-")
                    and not rollback_interrupted
                ):
                    rollback_interrupted = True
                    raise KeyboardInterrupt
                return real_identity_check(publication, path)

            with self.assertRaisesRegex(
                merge_module.CampaignPublicationError,
                "rollback was interrupted",
            ) as raised:
                with merge_module._campaign_staging_directory(
                    directory
                ) as staging:
                    retained_staging = staging
                    staged_outputs = [
                        staging / "data.root",
                        staging / "data_0001.root",
                    ]
                    final_outputs = [directory / path.name for path in staged_outputs]
                    staged_manifest = staging / "pseudo_data_manifest.json"
                    final_manifest = directory / staged_manifest.name
                    new_contents = [b"new data zero", b"new data one"]
                    old_contents = [b"old data zero", b"old data one"]
                    for staged, final, new, old in zip(
                        staged_outputs,
                        final_outputs,
                        new_contents,
                        old_contents,
                        strict=True,
                    ):
                        staged.write_bytes(new)
                        final.write_bytes(old)
                    staged_manifest.write_bytes(b"new manifest")
                    final_manifest.write_bytes(b"old manifest")
                    expected_sha256 = {
                        path: hashlib.sha256(contents).hexdigest()
                        for path, contents in zip(
                            final_outputs, new_contents, strict=True
                        )
                    }

                    with (
                        patch.object(
                            merge_module.os,
                            "link",
                            side_effect=fail_second_publish,
                        ),
                        patch.object(
                            merge_module,
                            "_is_unchanged_published_path",
                            side_effect=interrupt_after_rollback_claim,
                        ),
                    ):
                        merge_module._publish_staged_campaign(
                            list(zip(staged_outputs, final_outputs, strict=True)),
                            staged_manifest,
                            final_manifest,
                            staging_directory=staging,
                            overwrite=True,
                            expected_sha256=expected_sha256,
                        )

            self.assertTrue(publication_failed)
            self.assertTrue(rollback_interrupted)
            self.assertIsInstance(raised.exception.__cause__, KeyboardInterrupt)
            self.assertEqual(
                raised.exception.retained_staging_directory,
                retained_staging,
            )
            self.assertTrue(retained_staging.is_dir())
            self.assertEqual(
                (retained_staging / "backups" / "0000-data.root").read_bytes(),
                b"old data zero",
            )
            self.assertEqual(
                (
                    retained_staging / "backups" / "0001-data_0001.root"
                ).read_bytes(),
                b"old data one",
            )

    def test_campaign_publish_successfully_replaces_existing_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staging = directory / "staging"
            staging.mkdir()
            staged_output = staging / "data.root"
            final_output = directory / "data.root"
            staged_manifest = staging / "pseudo_data_manifest.json"
            final_manifest = directory / staged_manifest.name
            staged_output.write_bytes(b"new data")
            final_output.write_bytes(b"old data")
            staged_manifest.write_bytes(b"new manifest")
            final_manifest.write_bytes(b"old manifest")

            merge_module._publish_staged_campaign(
                [(staged_output, final_output)],
                staged_manifest,
                final_manifest,
                staging_directory=staging,
                overwrite=True,
                expected_sha256={
                    final_output: hashlib.sha256(b"new data").hexdigest()
                },
            )

            self.assertEqual(final_output.read_bytes(), b"new data")
            self.assertEqual(final_manifest.read_bytes(), b"new manifest")


if __name__ == "__main__":
    unittest.main()
