from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Simulation.lhe_provenance import (
    bind_output,
    read_manifest,
    sha256_file,
    verify_hepmc,
    verify_output,
)


class SimulationLHEProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hepmc = self.root / "events.hepmc3"
        # An accepted-event sequence with a skipped LHE event. The provenance
        # must retain source identity, rather than renumbering accepted events.
        self.hepmc.write_text("E 1 0 0\nE 3 0 0\n", encoding="utf-8")
        self.delphes = self.root / "delphes.root"
        self.delphes.write_bytes(b"ROOT fixture with source events 1, 3")
        self.source = self.root / "source.json"
        self.bound = self.root / "simulation" / "lhe-provenance.json"
        self.manifest = {
            "schema_version": 1,
            "event_number": "lhe_event_index_1based",
            "lhe_file": "/original/location/events.lhe",
            "lhe_sha256": "a" * 64,
            "lhe_event_count": 3,
            "hepmc_sha256": sha256_file(self.hepmc),
            "hepmc_event_count": 2,
            "shower": "pythia",
        }
        self.source.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_binds_hash_and_retains_skipped_source_identity(self):
        verified = verify_hepmc(self.source, self.hepmc)
        bind_output(self.source, self.delphes, self.bound, event_count=2)
        bound = read_manifest(self.bound)
        self.assertEqual(bound["lhe_event_count"], 3)
        self.assertEqual(bound["delphes_event_count"], 2)
        self.assertEqual(bound["delphes_sha256"], sha256_file(self.delphes))
        self.assertEqual(bound["lhe_sha256"], verified["lhe_sha256"])
        self.assertNotIn("delphes_sha256", read_manifest(self.source))
        verify_output(self.source, self.bound, self.delphes, event_count=2)

    def test_rejects_changed_hepmc_before_simulation(self):
        self.hepmc.write_text("E 2 0 0\nE 3 0 0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "HepMC hash"):
            verify_hepmc(self.source, self.hepmc)

    def test_rejects_changed_delphes_on_resume(self):
        bind_output(self.source, self.delphes, self.bound, event_count=2)
        self.delphes.write_bytes(b"different ROOT file with same event count")
        with self.assertRaisesRegex(ValueError, "Delphes hash"):
            verify_output(self.source, self.bound, self.delphes, event_count=2)

    def test_rejects_changed_source_identity_on_resume(self):
        bind_output(self.source, self.delphes, self.bound, event_count=2)
        self.manifest["lhe_sha256"] = "b" * 64
        self.source.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source provenance changed"):
            verify_output(self.source, self.bound, self.delphes, event_count=2)

    def test_resume_requires_same_max_event_count(self):
        bind_output(self.source, self.delphes, self.bound, event_count=1)
        with self.assertRaisesRegex(ValueError, "event count"):
            verify_output(self.source, self.bound, self.delphes, event_count=2)

    def test_relocated_paths_do_not_change_identity(self):
        bind_output(self.source, self.delphes, self.bound, event_count=2)
        self.manifest["lhe_file"] = "/relocated/events.lhe"
        self.source.write_text(json.dumps(self.manifest), encoding="utf-8")
        moved = self.root / "moved.root"
        self.delphes.rename(moved)
        verify_output(self.source, self.bound, moved, event_count=2)

    def test_rejects_legacy_numbering_and_invalid_schema(self):
        for key, value in (("event_number", "accepted_ordinal"), ("schema_version", 2)):
            with self.subTest(key=key):
                manifest = {**self.manifest, key: value}
                self.source.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_hepmc(self.source, self.hepmc)

    def test_cannot_overwrite_original_manifest(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            bind_output(self.source, self.delphes, self.source, event_count=2)

    def test_binding_rejects_manifest_replaced_after_simulation_check(self):
        expected_sha256 = sha256_file(self.source)
        self.manifest["lhe_sha256"] = "b" * 64
        self.source.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            bind_output(
                self.source, self.delphes, self.bound, event_count=2,
                expected_manifest_sha256=expected_sha256,
            )
        self.assertFalse(self.bound.exists())


if __name__ == "__main__":
    unittest.main()
