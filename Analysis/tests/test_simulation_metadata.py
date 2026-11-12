"""Muon-scale variations stay distinct through reduction and pseudo-data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import uproot

from Analysis.build_analysis_tree import INPUT_BRANCHES, main as reduce_main
from Analysis.merge_analysis_outputs import build_pseudo_data, merge_sample
from Analysis.simulation_metadata import (
    MUON_SCALE_FACTORS, MUON_SCALE_MODEL, collect_delphes_metadata,
    merge_simulation_metadata,
)
from Analysis.tests.test_merge_analysis_outputs import write_sample


def metadata(variation):
    return {
        "schema_version": 1,
        "muon_scale_variation": variation,
        "muon_scale_factor": MUON_SCALE_FACTORS[variation],
        "muon_scale_one_sigma": 0.01,
        "muon_scale_model": MUON_SCALE_MODEL,
        "legacy_nominal_inputs": [],
    }


def write_sidecar(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "simulation-metadata.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in settings.items() if key.startswith("muon_scale_"))
    )


def compact(path, variation=None):
    write_sample(path, 0.1)
    if variation is not None:
        with uproot.update(path) as root_file:
            root_file["simulation_metadata"] = json.dumps(metadata(variation))


def merge(files, output):
    return merge_sample(files, output, step_size="1 kB", cross_section_override_pb=None,
                        luminosity_fb=1.0, overwrite=False)


class SimulationMetadataTest(unittest.TestCase):
    def test_all_variations_are_read_from_delphes_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            for variation in MUON_SCALE_FACTORS:
                with self.subTest(variation=variation):
                    path = Path(directory) / variation / "delphes.root"
                    write_sidecar(path, metadata(variation))
                    self.assertEqual(collect_delphes_metadata([path]), metadata(variation))

    def test_old_sidecars_and_missing_sidecars_are_explicitly_assumed_nominal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "old" / "delphes.root", root / "missing" / "delphes.root"]
            write_sidecar(paths[0], {})
            (paths[0].parent / "simulation-metadata.txt").write_text("process=gg_H\n")
            expected = metadata("nominal") | {"legacy_nominal_inputs": [str(path) for path in paths]}
            self.assertEqual(collect_delphes_metadata(paths), expected)

    def test_incomplete_or_inconsistent_sidecars_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delphes.root"
            broken = [
                {"muon_scale_variation": "up"},
                metadata("up") | {"muon_scale_variation": "bad"},
                metadata("up") | {"muon_scale_factor": 0.99},
                metadata("up") | {"muon_scale_factor": "nan"},
                metadata("up") | {"muon_scale_one_sigma": 0.02},
                metadata("up") | {"muon_scale_model": "different"},
            ]
            for settings in broken:
                with self.subTest(settings=settings):
                    write_sidecar(path, settings)
                    with self.assertRaises(ValueError):
                        collect_delphes_metadata([path])

    def test_recursive_reducer_rejects_mixed_variations_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variation in ("nominal", "up"):
                path = root / variation / "delphes.root"
                write_sidecar(path, metadata(variation))
                path.touch()
            output = root / "analysis.root"
            with patch("sys.argv", ["build_analysis_tree.py", str(root), "-o", str(output)]):
                with self.assertRaisesRegex(ValueError, "different muon-scale variations"):
                    reduce_main()
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_reducer_embeds_scale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "delphes.root", root / "analysis.root"
            write_sidecar(path, metadata("down"))
            with uproot.recreate(path) as root_file:
                root_file.mktree("Delphes", {name: "var * float64" for name in INPUT_BRANCHES})
            with patch("sys.argv", ["build_analysis_tree.py", str(path), "-o", str(output)]):
                with redirect_stdout(io.StringIO()):
                    reduce_main()
            with uproot.open(output) as root_file:
                self.assertEqual(json.loads(str(root_file["simulation_metadata"])), metadata("down"))

    def test_merges_and_pseudo_data_preserve_up_variation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.root", root / "second.root"
            compact(first, "up")
            compact(second, "up")
            zz, higgs, data = root / "ZZ.root", root / "gg_H.root", root / "data.root"
            merge([first, second], zz)
            merge([first], higgs)
            build_pseudo_data(zz, higgs, data, luminosity_fb=1.0, seed=42,
                              step_size="1 kB", overwrite=False)
            for output in (zz, higgs, data):
                with self.subTest(output=output), uproot.open(output) as root_file:
                    self.assertEqual(json.loads(str(root_file["simulation_metadata"])), metadata("up"))

    def test_merger_rejects_mixed_and_legacy_plus_varied_inputs_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            varied = root / "up.root"
            compact(varied, "up")
            for variation in (None, "nominal", "down"):
                with self.subTest(variation=variation):
                    other, output = root / "other.root", root / "merged.root"
                    compact(other, variation)
                    with self.assertRaisesRegex(ValueError, "different muon-scale variations"):
                        merge([varied, other], output)
                    self.assertFalse(output.exists())

    def test_pseudo_data_rejects_mixed_components_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "up.root", root / "nominal.root"
            compact(first, "up")
            compact(second, "nominal")
            zz, higgs, output = root / "ZZ.root", root / "gg_H.root", root / "data.root"
            merge([first], zz)
            merge([second], higgs)
            with self.assertRaisesRegex(ValueError, "different muon-scale variations"):
                build_pseudo_data(zz, higgs, output, luminosity_fb=1.0, seed=42,
                                  step_size="1 kB", overwrite=False)
            self.assertFalse(output.exists())

    def test_legacy_nominal_assumptions_survive_multiple_merges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy, nominal, merged = root / "old.root", root / "nominal.root", root / "merged.root"
            compact(legacy)
            compact(nominal, "nominal")
            merge([legacy, nominal], merged)
            self.assertEqual(merge_simulation_metadata([merged]),
                             metadata("nominal") | {"legacy_nominal_inputs": [str(legacy)]})

    def test_malformed_compact_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.root"
            cases = [
                "invalid-json", [], {"schema_version": 1},
                metadata("up") | {"schema_version": 2},
                metadata("up") | {"schema_version": True},
                metadata("up") | {"schema_version": 1.0},
                metadata("up") | {"muon_scale_factor": 1.0},
                metadata("up") | {"legacy_nominal_inputs": ["legacy.root"]},
            ]
            for values in cases:
                with self.subTest(values=values):
                    with uproot.recreate(path) as root_file:
                        root_file["simulation_metadata"] = (
                            values if isinstance(values, str) else json.dumps(values)
                        )
                    with self.assertRaises(ValueError):
                        merge_simulation_metadata([path])


if __name__ == "__main__":
    unittest.main()
