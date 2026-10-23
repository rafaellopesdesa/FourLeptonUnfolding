"""Born payload/provenance survive merging and signed pseudo-data sampling."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import uproot
import vector

from Analysis.born import BORN_BRANCH_TYPES, born_observables
from Analysis.lhe import LHE_BRANCH_TYPES, LHEEvent, event_observables
from Analysis.lhe_matching import EVENT_NUMBER_CONVENTION, merge_born_metadata
from Analysis.merge_analysis_outputs import build_pseudo_data, merge_sample
from Analysis.tests.test_born import _lhe_particle, _quartet
from Analysis.tests.test_validate_analysis_output import write_output
from Analysis.validate_analysis_output import validate_analysis_output
from Tools.born_projection import BORN_PROJECTION_METADATA


def _metadata(sources):
    return {"schema_version": 1, "projection": deepcopy(BORN_PROJECTION_METADATA),
            "sources": sources}


def _write_metadata(path: Path, metadata=None):
    with uproot.recreate(path) as root_file:
        root_file["placeholder"] = "metadata-only fixture"
        if metadata is not None:
            root_file["born_metadata"] = json.dumps(metadata)


def _write_shard(path: Path, *, identity: int, source: int):
    """Create mixed e/mu/tau/unavailable states with signed event weights."""
    weights = np.array([2.0, -1.0, 3.0, 1.0], dtype=np.float64)
    quartets = (_quartet(), _quartet(first=15), (), _quartet(first=15, second=15))
    born_rows, lhe_rows = [], []
    for index, quartet in enumerate(quartets, start=1):
        born_rows.append(born_observables(
            quartet, source=source, shower_event_index=index if source in (2, 3) else -1,
        ))
        if source == 1:
            particles = tuple(_lhe_particle(pid, p4) for pid, p4 in quartet)
        else:
            total = (sum((p4 for _, p4 in quartet[1:]), quartet[0][1]) if quartet
                     else vector.obj(px=0.0, py=0.0, pz=0.0, mass=125.0))
            particles = (_lhe_particle(25, total, status=2),)
        event = LHEEvent(index, 10001, float(weights[index - 1]), 125.0,
                         0.007297, 0.118, particles)
        observation = event_observables(event)
        observation["lhe_source_id"] = identity
        lhe_rows.append(observation)
    born = {name: np.asarray([row[name] for row in born_rows], dtype=dtype)
            for name, dtype in BORN_BRANCH_TYPES.items()}
    lhe = {name: np.asarray([row[name] for row in lhe_rows], dtype=dtype)
           for name, dtype in LHE_BRANCH_TYPES.items()}
    identifier = f"{identity:016x}"
    record = {
        "lhe_source_id": identifier, "lhe_sha256": identifier + "0" * 48,
        "lhe_event_count": 4, "lhe_file": str(path.with_suffix(".lhe")),
        "delphes_file": str(path.with_suffix(".delphes.root")),
        "delphes_sha256": identifier + "1" * 48,
        "born_source": source,
        "hard_decay_definition": "original LHE quartet" if source == 1 else "native hard pre-FSR leptons",
        "born_available_entries": int(np.count_nonzero(born["born_available"])),
        "born_kinematics_valid_entries": int(np.count_nonzero(born["born_kinematics_valid"])),
        "decay_status_counts": {"ok": 3, "unsupported": 1},
    }
    if source in (2, 3):
        record["hepmc_file"] = str(path.with_suffix(".hepmc"))
    born_metadata = _metadata([record])
    lhe_metadata = {"schema_version": 1, "event_number": EVENT_NUMBER_CONVENTION,
                    "sources": [record]}
    write_output(path, weights=weights, cross_section=0.1, born_columns=born,
                 lhe_columns=lhe, born_metadata=born_metadata, lhe_metadata=lhe_metadata)
    return {**lhe, **born}, born_metadata


def _merge(shards, destination, *, overwrite=False):
    return merge_sample(shards, destination, step_size="1 kB", cross_section_override_pb=None,
                        luminosity_fb=1.0, overwrite=overwrite)


class BornMetadataMergeTest(unittest.TestCase):
    def test_legacy_absence_and_explicit_empty_inventory_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy, empty = root / "legacy.root", root / "empty.root"
            _write_metadata(legacy)
            _write_metadata(empty, _metadata([]))
            self.assertIsNone(merge_born_metadata([legacy]))
            self.assertEqual(merge_born_metadata([legacy, empty]), _metadata([]))

    def test_union_deduplicates_identical_records_but_retains_different_input_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.root", root / "second.root"
            common = {"lhe_source_id": "0000000000000001", "lhe_sha256": "0" * 15 + "1" + "0" * 48}
            pythia = {**common, "born_source": 2, "hepmc_file": "pythia.hepmc"}
            herwig = {**common, "born_source": 3, "hepmc_file": "herwig.hepmc"}
            _write_metadata(first, _metadata([pythia]))
            _write_metadata(second, _metadata([pythia, herwig]))
            self.assertEqual(merge_born_metadata([first, second, first]), _metadata([pythia, herwig]))

    def test_incompatible_projection_or_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "incompatible.root"
            for changed in (_metadata([]) | {"schema_version": 2},
                            _metadata([]) | {"projection": {**BORN_PROJECTION_METADATA, "equations": [99]}}):
                with self.subTest(metadata=changed):
                    _write_metadata(path, changed)
                    with self.assertRaisesRegex(ValueError, "Incompatible Born projection metadata"):
                        merge_born_metadata([path])


class BornPayloadMergeTest(unittest.TestCase):
    def test_actual_merge_preserves_all_born_values_dtypes_and_unavailable_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "first.root", root / "second.root"]
            fixtures = [_write_shard(paths[0], identity=101, source=1),
                        _write_shard(paths[1], identity=102, source=1)]
            destination = root / "merged.root"
            _merge(paths, destination)
            with uproot.open(destination) as root_file:
                arrays = root_file["Analysis"].arrays(library="np")
                for name, dtype in BORN_BRANCH_TYPES.items():
                    expected = np.concatenate([fixture[0][name] for fixture in fixtures])
                    np.testing.assert_equal(arrays[name], expected, err_msg=name)
                    self.assertEqual(arrays[name].dtype, dtype)
                self.assertEqual(json.loads(str(root_file["born_metadata"])), merge_born_metadata(paths))
                np.testing.assert_array_equal(arrays["lhe_source_id"], [101] * 4 + [102] * 4)
                np.testing.assert_array_equal(arrays["event_id"], np.arange(8))
                self.assertEqual(np.count_nonzero(~arrays["born_available"]), 2)
            summary = validate_analysis_output(destination, require_lhe=True, require_born=True)
            self.assertEqual(summary.born_source_entries, 8)
            self.assertEqual(summary.born_available_entries, 6)

    def test_signed_pseudo_data_preserves_each_bootstrapped_rows_born_association(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zz_shard, higgs_shard = root / "zz.root", root / "higgs.root"
            fixtures = [_write_shard(zz_shard, identity=201, source=1),
                        _write_shard(higgs_shard, identity=202, source=3)]
            zz_merged, higgs_merged = root / "ZZ_herwig.root", root / "gg_H_herwig.root"
            _merge([zz_shard], zz_merged)
            _merge([higgs_shard], higgs_merged)
            output = root / "data.root"
            build_pseudo_data(zz_merged, higgs_merged, output, luminosity_fb=1.0,
                              seed=10, step_size="1 kB", overwrite=False)
            expected_rows = {}
            for payload, _ in fixtures:
                for row in range(4):
                    key = (int(payload["lhe_source_id"][row]), int(payload["lhe_event_index"][row]))
                    expected_rows[key] = {name: payload[name][row] for name in BORN_BRANCH_TYPES}
            with uproot.open(output) as root_file:
                arrays = root_file["Analysis"].arrays(library="np")
                keys = list(zip(arrays["lhe_source_id"], arrays["lhe_event_index"]))
                self.assertGreater(len(keys), len(expected_rows))
                self.assertLess(len(set(keys)), len(keys))
                self.assertTrue(np.any(arrays["weight"] < 0.0))
                self.assertTrue(np.all(np.isin(arrays["weight"], [-1.0, 1.0])))
                for name, dtype in BORN_BRANCH_TYPES.items():
                    expected = np.asarray([expected_rows[tuple(map(int, key))][name] for key in keys], dtype=dtype)
                    np.testing.assert_equal(arrays[name], expected, err_msg=name)
                    self.assertEqual(arrays[name].dtype, dtype)
                self.assertEqual(json.loads(str(root_file["born_metadata"])),
                                 merge_born_metadata([zz_merged, higgs_merged]))
                self.assertTrue(np.any(~arrays["born_available"]))
                np.testing.assert_array_equal(arrays["event_id"], np.arange(len(keys)))
            # Pseudo-data deliberately have no generator cross_section_pb, so
            # the batch-shard validator is not applicable to this mixed data
            # sample. Exact association checks above cover its Born payload.
            self.assertEqual(np.count_nonzero(arrays["born_source"]), len(keys))

    def test_incompatible_projection_preserves_existing_merge_and_pseudo_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good, bad = root / "good.root", root / "bad.root"
            _write_shard(good, identity=301, source=1)
            _, metadata = _write_shard(bad, identity=302, source=3)
            metadata["projection"]["method"] = "different_projection"
            with uproot.update(bad) as root_file:
                root_file["born_metadata"] = json.dumps(metadata)
            merged, pseudo = root / "merged.root", root / "data.root"
            for path in (merged, pseudo):
                path.write_bytes(b"previous result must remain intact")
            with self.assertRaisesRegex(ValueError, "Incompatible Born projection metadata"):
                _merge([good, bad], merged, overwrite=True)
            with self.assertRaisesRegex(ValueError, "Incompatible Born projection metadata"):
                build_pseudo_data(good, bad, pseudo, luminosity_fb=1.0, seed=10,
                                  step_size="1 kB", overwrite=True)
            for path in (merged, pseudo):
                self.assertEqual(path.read_bytes(), b"previous result must remain intact")
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
