from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot
import vector

from Analysis.build_analysis_tree import INPUT_BRANCHES
from Analysis.lhe import event_observables, iter_lhe_events
from Analysis.lhe_matching import EVENT_NUMBER_CONVENTION, file_sha256, load_lhe_source
from Analysis.merge_analysis_outputs import build_pseudo_data, merge_sample
from Analysis.tests.test_lhe import _event_text, _particle
from Analysis.validate_analysis_output import validate_analysis_output


REPO = Path(__file__).resolve().parents[2]


def write_inputs(directory, numbers=(4, 2, 2)):
    lhe = directory / "events.lhe"
    events = [_event_text([_particle(25, vector.obj(px=10. * i, py=0., pz=0., mass=125.))],
                          header=f"1 10001 {(-1) ** i * i} 125 .007 .118")
              for i in range(1, 5)]
    lhe.write_text('<LesHouchesEvents version="3.0">\n<header/>\n<init/>\n'
                   + "".join(events) + '</LesHouchesEvents>\n')
    root = directory / "delphes.root"
    arrays = {name: ak.Array([[] for _ in numbers]) for name in INPUT_BRANCHES}
    arrays["Event.Number"] = ak.Array([[number] for number in numbers])
    arrays["Event.Weight"] = ak.Array([[.01 * (i + 1)] for i in range(len(numbers))])
    arrays["Event.CrossSection"] = ak.Array([[.002] for _ in numbers])
    integral_numbers = all(isinstance(number, int) for number in numbers)
    schema = {name: ("var * int64" if name == "Event.Number" and integral_numbers else "var * float64")
              for name in INPUT_BRANCHES}
    with uproot.recreate(root) as output:
        output.mktree("Delphes", schema)
        output["Delphes"].extend(arrays)
    manifest = {
        "schema_version": 1, "lhe_file": str(lhe), "lhe_sha256": file_sha256(lhe),
        "lhe_event_count": 4, "delphes_sha256": file_sha256(root),
        "delphes_event_count": len(numbers), "event_number": EVENT_NUMBER_CONVENTION,
    }
    path = directory / "lhe-provenance.json"
    path.write_text(json.dumps(manifest))
    return lhe, root, path, manifest


class LHEMatchingTest(unittest.TestCase):
    def test_skipped_reordered_and_repeated_source_events_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, root, _, _ = write_inputs(base)
            output = base / "analysis.root"
            result = subprocess.run(
                [sys.executable, str(REPO / "Analysis/build_analysis_tree.py"),
                 str(root), "-o", str(output), "--require-lhe", "--step-size", "1 kB"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = validate_analysis_output(output, require_lhe=True)
            self.assertEqual(summary.entries, 3)
            with uproot.open(output) as file:
                arrays = file["Analysis"].arrays(library="np")
                np.testing.assert_array_equal(arrays["lhe_event_index"], [4, 2, 2])
                np.testing.assert_allclose(arrays["lhe_higgs_pT"], [40., 20., 20.])
                np.testing.assert_allclose(arrays["lhe_weight"], [4., 2., 2.])
                np.testing.assert_allclose(arrays["weight"], [.01, .02, .03])
                self.assertTrue(np.all(arrays["lhe_available"]))
                self.assertFalse(np.any(arrays["lhe_has_four_leptons"]))
                self.assertTrue(np.all(np.isnan(arrays["lhe_m_ZZ"])))
                self.assertEqual(len(json.loads(str(file["lhe_metadata"]))["sources"]), 1)

    def test_requires_certified_mapping_and_verifies_both_hashes(self):
        for mutation in ("missing", "convention", "root_hash", "lhe_hash", "count"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                _, root, manifest_path, manifest = write_inputs(Path(directory))
                if mutation == "missing":
                    manifest_path.unlink()
                else:
                    key, value = {
                        "convention": ("event_number", "accepted_ordinal"),
                        "root_hash": ("delphes_sha256", "0" * 64),
                        "lhe_hash": ("lhe_sha256", "0" * 64),
                        "count": ("lhe_event_count", 5),
                    }[mutation]
                    manifest[key] = value
                    manifest_path.write_text(json.dumps(manifest))
                with uproot.open(root) as file, self.assertRaises((ValueError, FileNotFoundError)):
                    load_lhe_source(root, file["Delphes"], require_lhe=True)

    def test_invalid_event_numbers_do_not_fall_back_to_row_order(self):
        for numbers in ((0, 1), (1, 5), (1.5, 2)):
            with self.subTest(numbers=numbers), tempfile.TemporaryDirectory() as directory:
                _, root, _, _ = write_inputs(Path(directory), numbers=numbers)
                with uproot.open(root) as file, self.assertRaises(ValueError):
                    load_lhe_source(root, file["Delphes"], require_lhe=True)

    def test_relocated_original_requires_identical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lhe, root, manifest_path, _ = write_inputs(base)
            moved = base / "moved.lhe"
            lhe.rename(moved)
            with uproot.open(root) as file:
                source = load_lhe_source(root, file["Delphes"], lhe_path=moved, manifest_path=manifest_path)
                self.assertEqual(set(source.observations), {2, 4})

    def test_failed_match_preserves_existing_analysis_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, root, manifest_path, _ = write_inputs(base)
            manifest_path.unlink()
            output = base / "analysis.root"
            output.write_bytes(b"previous good result")
            result = subprocess.run(
                [sys.executable, str(REPO / "Analysis/build_analysis_tree.py"),
                 str(root), "-o", str(output), "--require-lhe"], capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), b"previous good result")
            self.assertEqual(list(base.glob("*.partial")), [])

    def test_legacy_input_has_explicit_unavailable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            _, root, manifest_path, _ = write_inputs(Path(directory))
            manifest_path.unlink()
            with uproot.open(root) as file:
                self.assertIsNone(load_lhe_source(root, file["Delphes"]))

    def test_output_cannot_overwrite_original_lhe_or_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            lhe, root, manifest_path, _ = write_inputs(Path(directory))
            for target in (lhe, manifest_path):
                before = target.read_bytes()
                result = subprocess.run(
                    [sys.executable, str(REPO / "Analysis/build_analysis_tree.py"),
                     str(root), "-o", str(target), "--require-lhe"], capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"must not overwrite", result.stderr)
                self.assertEqual(target.read_bytes(), before)

    def test_merge_and_pseudo_data_preserve_lhe_payload_and_source_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lhe, root, _, _ = write_inputs(base)
            with uproot.open(root) as file:
                metadata = load_lhe_source(root, file["Delphes"]).metadata
            observations = [event_observables(event) for event in iter_lhe_events(lhe)]
            source_id = int(file_sha256(lhe)[:16], 16)
            payload = {name: np.array([row[name] for row in observations]) for name in observations[0]}
            payload["lhe_source_id"] = np.full(4, source_id, dtype=np.uint64)
            payload.update(event_id=np.arange(4, dtype=np.uint64), weight=np.array([2., -1., 3., 1.]),
                           reconstructed=np.ones(4, dtype=bool), cross_section_pb=np.full(4, .01))
            shard = base / "shard.root"
            with uproot.recreate(shard) as file:
                file.mktree("Analysis", payload)
                file["lhe_metadata"] = json.dumps({"schema_version": 1, "sources": [metadata]})
            merged = base / "merged.root"
            merge_sample([shard], merged, step_size="1 MB", cross_section_override_pb=None,
                         luminosity_fb=1., overwrite=False)
            with uproot.open(merged) as file:
                arrays = file["Analysis"].arrays(library="np")
                for name in observations[0]:
                    np.testing.assert_equal(arrays[name], payload[name])
                inventory = json.loads(str(file["lhe_metadata"]))
                self.assertEqual(inventory["sources"][0]["lhe_sha256"], file_sha256(lhe))
            pseudo = base / "data.root"
            build_pseudo_data(merged, merged, pseudo, luminosity_fb=1., seed=10,
                              step_size="1 MB", overwrite=False)
            with uproot.open(pseudo) as file:
                arrays = file["Analysis"].arrays(library="np")
                self.assertGreater(len(arrays["weight"]), 0)
                np.testing.assert_allclose(arrays["lhe_higgs_pT"], 10. * arrays["lhe_event_index"])
                self.assertTrue(np.all(arrays["lhe_source_id"] == source_id))
                self.assertEqual(json.loads(str(file["lhe_metadata"])), inventory)


if __name__ == "__main__":
    unittest.main()
