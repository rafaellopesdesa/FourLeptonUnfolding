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

from BackgroundRemoval.Application.apply_background_ratio import ApplicationModels
from BackgroundRemoval.Application.apply_merged_directory import (
    SUMMARY_NAME,
    _commit_new_directory_no_clobber,
    _commit_staged_directory,
    apply_directory,
    discover_inputs,
)


class _Bundle:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = {
            "inputs": {
                "data": {"sha256": None},
                "gg_H_pythia": {"sha256": None},
                "ZZ_pythia": {"sha256": None},
            }
        }


def _write_pseudo_root(
    path: Path, *, ensemble_index: int, ensemble_count: int
) -> dict[str, object]:
    entry = {
        "ensemble_index": ensemble_index,
        "path": path.name,
        "total_entries": 1,
        "total_observed_positive": 1,
        "total_observed_negative": 0,
        "total_observed": 1,
        "seed_spawn_key": [ensemble_index],
    }
    metadata = {
        "format_version": 2,
        "ensemble_index": ensemble_index,
        "ensemble_count": ensemble_count,
        "seed": 42,
        "luminosity_fb": 312.0,
        "seed_spawn_key": entry["seed_spawn_key"],
        "total_entries": entry["total_entries"],
        "total_observed_positive": entry["total_observed_positive"],
        "total_observed_negative": entry["total_observed_negative"],
        "total_observed": entry["total_observed"],
    }
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = {"weight": np.ones(1, dtype=np.float64)}
        root_file["merge_metadata"] = json.dumps(metadata)
    return entry


class DirectoryApplicationTest(unittest.TestCase):
    def _inputs(self, directory: Path) -> None:
        for name in (
            "data.root",
            "data_0002.root",
            "data_0001.root",
            "data_001.root",
            "data_00001.root",
            "data_10000.root",
            "data_background_removed.root",
            "ZZ_pythia.root",
            "gg_H_pythia.root",
            "ZZ_herwig.root",
            "gg_H_herwig.root",
        ):
            (directory / name).write_bytes(name.encode())

    def test_discovery_is_sorted_and_never_includes_herwig(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            self._inputs(directory)
            found = [(path.name, kind) for path, kind in discover_inputs(directory)]
            self.assertEqual(
                found,
                [
                    ("data.root", "data"),
                    ("data_0001.root", "data"),
                    ("data_0002.root", "data"),
                    ("data_10000.root", "data"),
                    ("ZZ_pythia.root", "zz"),
                    ("gg_H_pythia.root", "gg-h"),
                ],
            )

    def test_directory_commit_restores_previous_campaign_if_publish_rename_fails(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            destination = directory / "decorated"
            staging = directory / "staging"
            destination.mkdir()
            staging.mkdir()
            (destination / "data.root").write_bytes(b"old")
            (staging / "data.root").write_bytes(b"new")
            real_replace = os.replace
            calls = 0

            def fail_publish(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated publish failure")
                return real_replace(source, target)

            with patch(
                "BackgroundRemoval.Application.apply_merged_directory.os.replace",
                side_effect=fail_publish,
            ):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    _commit_staged_directory(staging, destination)
            self.assertEqual((destination / "data.root").read_bytes(), b"old")

    def test_new_directory_commit_never_replaces_a_late_arrival(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            destination = directory / "decorated"
            staging = directory / "staging"
            staging.mkdir()
            (staging / "data.root").write_bytes(b"ours")
            # Simulate another publisher winning the race immediately before
            # the no-clobber commit path reserves the final name.
            destination.mkdir()
            (destination / "data.root").write_bytes(b"theirs")

            with self.assertRaisesRegex(FileExistsError, "appeared while decorating"):
                _commit_new_directory_no_clobber(staging, destination)
            self.assertEqual((destination / "data.root").read_bytes(), b"theirs")
            self.assertEqual((staging / "data.root").read_bytes(), b"ours")

    def test_output_must_not_overlap_inputs_or_model_artifacts(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            merged = directory / "merged"
            background = directory / "background"
            correction = directory / "correction"
            merged.mkdir()
            background.mkdir()
            correction.mkdir()
            for output in (
                merged,
                merged / "decorated",
                background / "decorated",
                correction / "decorated",
                directory,
            ):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "must not equal"):
                        apply_directory(
                            merged,
                            output,
                            background,
                            correction,
                            step_size="1 MB",
                            inference_batch_size=10,
                            device="cpu",
                            overwrite=True,
                            replace_existing_branches=False,
                            write_diagnostic_branches=False,
                        )

    def test_one_loaded_model_pair_is_reused_and_summary_is_written(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            merged = directory / "merged"
            output = directory / "decorated"
            merged.mkdir()
            output.mkdir()
            self._inputs(merged)
            (output / "data_9999.root").write_bytes(b"stale")
            (output / "keep.txt").write_bytes(b"unrelated")
            background = directory / "background"
            correction = directory / "correction"
            background.mkdir()
            correction.mkdir()
            models = ApplicationModels(
                background=_Bundle(background),  # type: ignore[arg-type]
                correction=_Bundle(correction),  # type: ignore[arg-type]
                background_manifest_sha256="a" * 64,
                correction_manifest_sha256="b" * 64,
            )
            used_models: list[ApplicationModels] = []

            def fake_decorate(source: Path, destination: Path, *args, **kwargs):
                used_models.append(kwargs["models"])
                destination.write_bytes(b"decorated-" + source.name.encode())
                return {
                    "entries": 10,
                    "analysis_region_entries": 2,
                    "predicted_entries": 2,
                }

            with (
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.load_application_models",
                    return_value=models,
                ) as loader,
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.decorate",
                    side_effect=fake_decorate,
                ),
            ):
                summary = apply_directory(
                    merged,
                    output,
                    background,
                    correction,
                    step_size="1 MB",
                    inference_batch_size=10,
                    device="cpu",
                    overwrite=True,
                    replace_existing_branches=False,
                    write_diagnostic_branches=False,
                )

            loader.assert_called_once()
            self.assertEqual(len(used_models), 6)
            self.assertTrue(all(item is models for item in used_models))
            self.assertFalse((output / "ZZ_herwig.root").exists())
            self.assertFalse((output / "gg_H_herwig.root").exists())
            self.assertFalse((output / "data_9999.root").exists())
            self.assertEqual((output / "keep.txt").read_bytes(), b"unrelated")
            stored = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(stored, summary)
            self.assertEqual(stored["correction_model_manifest_sha256"], "b" * 64)
            self.assertEqual(
                [Path(item["source"]).name for item in stored["files"][:3]],
                ["data.root", "data_0001.root", "data_0002.root"],
            )

    def test_pseudo_manifest_is_authoritative_and_copied(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            merged = directory / "merged"
            output = directory / "decorated"
            merged.mkdir()
            self._inputs(merged)
            pseudo_entries = [
                _write_pseudo_root(
                    merged / ("data.root" if index == 0 else f"data_{index:04d}.root"),
                    ensemble_index=index,
                    ensemble_count=3,
                )
                for index in range(3)
            ]
            manifest = {
                "format_version": 2,
                "generated_ensemble_count": 3,
                "seed": 42,
                "luminosity_fb": 312.0,
                "files": pseudo_entries,
            }
            source_manifest = merged / "pseudo_data_manifest.json"
            source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            found = [(path.name, kind) for path, kind in discover_inputs(merged)]
            self.assertEqual(
                found,
                [
                    ("data.root", "data"),
                    ("data_0001.root", "data"),
                    ("data_0002.root", "data"),
                    ("ZZ_pythia.root", "zz"),
                    ("gg_H_pythia.root", "gg-h"),
                ],
            )

            background = directory / "background"
            correction = directory / "correction"
            background.mkdir()
            correction.mkdir()
            models = ApplicationModels(
                background=_Bundle(background),  # type: ignore[arg-type]
                correction=_Bundle(correction),  # type: ignore[arg-type]
                background_manifest_sha256="a" * 64,
                correction_manifest_sha256="b" * 64,
            )

            def fake_decorate(source: Path, destination: Path, *args, **kwargs):
                destination.write_bytes(b"decorated-" + source.name.encode())
                return {
                    "entries": 1,
                    "analysis_region_entries": 1,
                    "predicted_entries": 1,
                }

            with (
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.load_application_models",
                    return_value=models,
                ),
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.decorate",
                    side_effect=fake_decorate,
                ),
            ):
                summary = apply_directory(
                    merged,
                    output,
                    background,
                    correction,
                    step_size="1 MB",
                    inference_batch_size=10,
                    device="cpu",
                    overwrite=False,
                    replace_existing_branches=False,
                    write_diagnostic_branches=False,
                )
            self.assertEqual(
                (output / "pseudo_data_manifest.json").read_bytes(),
                source_manifest.read_bytes(),
            )
            self.assertEqual(
                summary["upstream_pseudo_data_manifest"]["sha256"],
                hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                summary["upstream_pseudo_data_manifest"]["copied_sha256"],
                summary["upstream_pseudo_data_manifest"]["sha256"],
            )

    def test_truncated_pseudo_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            self._inputs(directory)
            (directory / "pseudo_data_manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "generated_ensemble_count": 2,
                        "files": [
                            {"ensemble_index": 0, "path": "data.root"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "campaign contract"):
                discover_inputs(directory)

    def test_pseudo_manifest_root_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            self._inputs(directory)
            entry = _write_pseudo_root(
                directory / "data.root",
                ensemble_index=0,
                ensemble_count=1,
            )
            # Keep the campaign contract structurally valid while making its
            # recorded observation disagree with the ROOT merge metadata.
            entry["total_observed"] = 2
            (directory / "pseudo_data_manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "generated_ensemble_count": 1,
                        "seed": 42,
                        "luminosity_fb": 312.0,
                        "files": [entry],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "ROOT metadata does not match its manifest entry"
            ):
                discover_inputs(directory)

    def test_late_failure_preserves_complete_previous_campaign(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            merged = directory / "merged"
            output = directory / "decorated"
            merged.mkdir()
            output.mkdir()
            self._inputs(merged)
            old_data = b"old-data"
            old_summary = b"old-summary"
            (output / "data.root").write_bytes(old_data)
            (output / SUMMARY_NAME).write_bytes(old_summary)
            (output / "data_9999.root").write_bytes(b"old-stale")
            (output / "keep.txt").write_bytes(b"unrelated")
            background = directory / "background"
            correction = directory / "correction"
            background.mkdir()
            correction.mkdir()
            models = ApplicationModels(
                background=_Bundle(background),  # type: ignore[arg-type]
                correction=_Bundle(correction),  # type: ignore[arg-type]
                background_manifest_sha256="a" * 64,
                correction_manifest_sha256="b" * 64,
            )
            calls = 0

            def failing_decorate(source: Path, destination: Path, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("late inference failure")
                destination.write_bytes(b"new-" + source.name.encode())
                return {
                    "entries": 10,
                    "analysis_region_entries": 2,
                    "predicted_entries": 2,
                }

            with (
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.load_application_models",
                    return_value=models,
                ),
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.decorate",
                    side_effect=failing_decorate,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "late inference failure"):
                    apply_directory(
                        merged,
                        output,
                        background,
                        correction,
                        step_size="1 MB",
                        inference_batch_size=10,
                        device="cpu",
                        overwrite=True,
                        replace_existing_branches=False,
                        write_diagnostic_branches=False,
                    )

            self.assertEqual((output / "data.root").read_bytes(), old_data)
            self.assertEqual((output / SUMMARY_NAME).read_bytes(), old_summary)
            self.assertEqual((output / "data_9999.root").read_bytes(), b"old-stale")
            self.assertEqual((output / "keep.txt").read_bytes(), b"unrelated")
            self.assertFalse((output / "data_0001.root").exists())

    def test_campaign_checksum_mismatch_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            merged = directory / "merged"
            output = directory / "decorated"
            background = directory / "background"
            correction = directory / "correction"
            merged.mkdir()
            background.mkdir()
            correction.mkdir()
            self._inputs(merged)
            background_bundle = _Bundle(background)
            correction_bundle = _Bundle(correction)
            correction_bundle.manifest["inputs"]["data"]["sha256"] = "0" * 64
            models = ApplicationModels(
                background=background_bundle,  # type: ignore[arg-type]
                correction=correction_bundle,  # type: ignore[arg-type]
                background_manifest_sha256="a" * 64,
                correction_manifest_sha256="b" * 64,
            )
            with patch(
                "BackgroundRemoval.Application.apply_merged_directory.load_application_models",
                return_value=models,
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    apply_directory(
                        merged,
                        output,
                        background,
                        correction,
                        step_size="1 MB",
                        inference_batch_size=10,
                        device="cpu",
                        overwrite=False,
                        replace_existing_branches=False,
                        write_diagnostic_branches=False,
                    )
            self.assertFalse(output.exists())

            def fake_decorate(source: Path, destination: Path, *args, **kwargs):
                destination.write_bytes(b"decorated-" + source.name.encode())
                return {
                    "entries": 1,
                    "analysis_region_entries": 0,
                    "predicted_entries": 0,
                }

            with (
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.load_application_models",
                    return_value=models,
                ),
                patch(
                    "BackgroundRemoval.Application.apply_merged_directory.decorate",
                    side_effect=fake_decorate,
                ),
            ):
                summary = apply_directory(
                    merged,
                    output,
                    background,
                    correction,
                    step_size="1 MB",
                    inference_batch_size=10,
                    device="cpu",
                    overwrite=False,
                    replace_existing_branches=False,
                    write_diagnostic_branches=False,
                    allow_input_mismatch=True,
                )
            mismatch = next(
                item
                for item in summary["input_provenance_checks"]
                if item["artifact"] == "correction" and item["input"] == "data"
            )
            self.assertFalse(mismatch["matched"])
            self.assertTrue(summary["allow_input_mismatch"])
            self.assertTrue(summary["options"]["allow_input_mismatch"])


if __name__ == "__main__":
    unittest.main()
