from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

from Simulation.lhe_provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]


class SimulationLHEChainTest(unittest.TestCase):
    """Exercise the shell integration with the external Delphes process stubbed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.simulation = self.root / "Simulation"
        self.simulation.mkdir()
        for name in ("run_simulation.sh", "lhe_provenance.py"):
            shutil.copy2(REPO_ROOT / "Simulation" / name, self.simulation / name)
        (self.simulation / "prepare_fiducial_card.py").write_text(
            "import argparse\nfrom pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('source')\nparser.add_argument('destination')\n"
            "parser.add_argument('--muon-scale', choices=('nominal', 'up', 'down'), required=True)\n"
            "args = parser.parse_args()\n"
            "Path(args.destination).write_text(Path(args.source).read_text() + "
            "'# muon_scale=' + args.muon_scale + '\\n')\n",
            encoding="utf-8",
        )
        self.executables = self.root / "bin"
        self.executables.mkdir()
        self.delphes = self.root / "delphes"
        self.delphes.mkdir()
        card = self.delphes / "card.tcl"
        card.write_text("# test card\n", encoding="utf-8")
        self.call_log = self.root / "delphes-calls.txt"
        reader = (
            "#!/bin/bash\nset -eu\n"
            f"echo called >> {shlex.quote(str(self.call_log))}\n"
            "printf 'Delphes placeholder from test process\\n' > \"$2\"\n"
        )
        for name in ("DelphesHepMC2", "DelphesHepMC3"):
            self.write_executable(self.delphes / name, reader)
        self.write_executable(self.executables / "root", "#!/bin/bash\nexit 0\n")
        self.write_executable(self.executables / "git", "#!/bin/bash\necho test-commit\n")
        (self.simulation / "env.sh").write_text(
            f"DELPHES_ROOT={shlex.quote(str(self.delphes))}\n"
            f"DELPHES_ATLAS_CARD={shlex.quote(str(card))}\n",
            encoding="utf-8",
        )
        self.generation = self.root / "generation"
        self.generation.mkdir()
        self.hepmc = self.generation / "events.hepmc3"
        self.hepmc.write_text(
            "HepMC::Version 3.0.0\nHepMC::Asciiv3-START_EVENT_LISTING\n"
            "E 1 0 0\nE 3 0 0\nHepMC::Asciiv3-END_EVENT_LISTING\n",
            encoding="utf-8",
        )
        (self.generation / "run-metadata.txt").write_text(
            "process=gg_H\nseed=1001\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "event_number": "lhe_event_index_1based",
            "lhe_file": "/retained/events.lhe",
            "lhe_sha256": "a" * 64,
            "lhe_event_count": 3,
            "hepmc_sha256": sha256_file(self.hepmc),
            "hepmc_event_count": 2,
        }
        (self.generation / "lhe-provenance.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.output = self.root / "outputs" / "generation"

    @staticmethod
    def write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def run_simulation(self, *extra: str, output_root: bool = True):
        environment = os.environ.copy()
        environment["PATH"] = str(self.executables) + os.pathsep + environment["PATH"]
        output_options = ["--output-root", str(self.root / "outputs")] if output_root else []
        return subprocess.run(
            [
                "bash", str(self.simulation / "run_simulation.sh"),
                str(self.generation), *output_options, *extra,
            ],
            env=environment, capture_output=True, text=True, check=False,
        )

    def test_runner_binds_manifest_and_reuses_only_matching_output(self):
        first = self.run_simulation()
        self.assertEqual(first.returncode, 0, first.stderr)
        bound = json.loads((self.output / "lhe-provenance.json").read_text())
        self.assertEqual(bound["lhe_event_count"], 3)
        self.assertEqual(bound["delphes_event_count"], 2)
        self.assertEqual(bound["delphes_sha256"], sha256_file(self.output / "delphes.root"))
        metadata = (self.output / "simulation-metadata.txt").read_text()
        self.assertIn("event_number=lhe_event_index_1based", metadata)
        self.assertIn("lhe_provenance_sha256=", metadata)
        self.assertIn("muon_scale_variation=nominal", metadata)
        resumed = self.run_simulation()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("Already complete", resumed.stdout)
        self.assertEqual(self.call_log.read_text().splitlines(), ["called"])

    def test_muon_scale_variants_coexist_with_same_seed_and_source(self):
        original_hepmc = self.hepmc.read_bytes()
        nominal_snapshot = None
        for variation, factor in (("nominal", "1.0"), ("up", "1.01"), ("down", "0.99")):
            with self.subTest(variation=variation):
                result = self.run_simulation("--muon-scale", variation)
                self.assertEqual(result.returncode, 0, result.stderr)
                suffix = "" if variation == "nominal" else f"_muon_scale_{variation}"
                output = self.output.with_name(self.output.name + suffix)
                self.assertTrue((output / "SUCCESS").exists())
                metadata = dict(line.split("=", 1) for line in (output / "simulation-metadata.txt").read_text().splitlines())
                self.assertEqual(metadata["muon_scale_variation"], variation)
                self.assertEqual(metadata["muon_scale_factor"], factor)
                self.assertEqual(metadata["muon_scale_one_sigma"], "0.01")
                self.assertEqual(metadata["muon_scale_model"], "uniform_relative_momentum_scale_v1")
                self.assertEqual(metadata["muon_scale_affected_collections"], "RecoMuon,RecoMuonNoIso")
                self.assertEqual(metadata["random_seed"], "1001")
                self.assertEqual(metadata["reco_muon_resolution_model"], "atlas_run2_momentum_proxy")
                card = (output / "delphes_card_ATLAS_resolved.tcl").read_text()
                self.assertIn(f"# muon_scale={variation}\n", card)
                self.assertIn("set RandomSeed 1001\n", card)
                self.assertEqual(metadata["resolved_card_sha256"], sha256_file(output / "delphes_card_ATLAS_resolved.tcl"))
                if nominal_snapshot is None:
                    nominal_snapshot = (self.output / "simulation-metadata.txt").read_bytes()
                self.assertEqual((self.output / "simulation-metadata.txt").read_bytes(), nominal_snapshot)
                resumed = self.run_simulation("--muon-scale", variation)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertIn("Already complete", resumed.stdout)
        self.assertEqual(self.call_log.read_text().splitlines(), ["called"] * 3)
        self.assertEqual(self.hepmc.read_bytes(), original_hepmc)

    def test_default_output_location_keeps_nominal_and_separates_variations(self):
        for variation in ("nominal", "up", "down"):
            with self.subTest(variation=variation):
                result = self.run_simulation("--muon-scale", variation, output_root=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                suffix = "" if variation == "nominal" else f"_muon_scale_{variation}"
                self.assertTrue((self.generation / f"delphes_ATLAS{suffix}" / "SUCCESS").exists())

    def test_non_nominal_reuse_requires_complete_matching_scale_metadata(self):
        first = self.run_simulation("--muon-scale", "up")
        self.assertEqual(first.returncode, 0, first.stderr)
        status = self.output.with_name("generation_muon_scale_up") / "simulation-metadata.txt"
        original = status.read_text()
        cases = [
            None,
            "\n".join(line for line in original.splitlines() if not line.startswith("muon_scale_")) + "\n",
            original.replace("muon_scale_variation=up", "muon_scale_variation=down"),
            original.replace("muon_scale_factor=1.01", "muon_scale_factor=1.02"),
            original.replace("muon_scale_one_sigma=0.01", "muon_scale_one_sigma=0.02"),
            original.replace("muon_scale_model=uniform_relative_momentum_scale_v1", "muon_scale_model=unknown"),
        ]
        for changed in cases:
            with self.subTest(metadata=changed):
                if changed is None:
                    status.unlink()
                else:
                    status.write_text(changed)
                result = self.run_simulation("--muon-scale", "up")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("muon-scale metadata", result.stderr)
                status.write_text(original)
        self.assertEqual(self.call_log.read_text().splitlines(), ["called"])

    def test_legacy_nominal_metadata_remains_usable_but_card_hash_is_verified(self):
        result = self.run_simulation()
        self.assertEqual(result.returncode, 0, result.stderr)
        status = self.output / "simulation-metadata.txt"
        original = status.read_text()
        status.write_text("\n".join(line for line in original.splitlines() if not line.startswith("muon_scale_")) + "\n")
        legacy = self.run_simulation()
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        card = self.output / "delphes_card_ATLAS_resolved.tcl"
        card.write_text(card.read_text() + "# changed\n")
        mismatch = self.run_simulation()
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("resolved card checksum", mismatch.stderr)
        status.unlink()
        absent = self.run_simulation()
        self.assertEqual(absent.returncode, 0, absent.stderr)
        self.assertEqual(self.call_log.read_text().splitlines(), ["called"])

    def test_invalid_scale_fails_before_dependency_checks(self):
        (self.simulation / "env.sh").unlink()
        result = self.run_simulation("--muon-scale", "sideways")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--muon-scale must be nominal, up, or down", result.stderr)
        self.assertNotIn("Missing", result.stderr)
        self.assertFalse(self.call_log.exists())

    def test_changed_hepmc_never_reaches_delphes(self):
        self.hepmc.write_text(self.hepmc.read_text().replace("E 1", "E 2"))
        result = self.run_simulation()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HepMC hash", result.stderr)
        self.assertFalse(self.call_log.exists())

    def test_changed_root_is_rejected_on_resume(self):
        first = self.run_simulation()
        self.assertEqual(first.returncode, 0, first.stderr)
        (self.output / "delphes.root").write_bytes(b"different ROOT output")
        resumed = self.run_simulation()
        self.assertNotEqual(resumed.returncode, 0)
        self.assertIn("Delphes hash", resumed.stderr)
        self.assertEqual(self.call_log.read_text().splitlines(), ["called"])

    def test_overwrite_cannot_delete_generation_manifest(self):
        manifest_path = self.generation / "lhe-provenance.json"
        original_manifest = manifest_path.read_bytes()
        original_hepmc = self.hepmc.read_bytes()
        result = self.run_simulation("--output-root", str(self.root), "--overwrite")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must differ from the generation", result.stderr)
        self.assertEqual(manifest_path.read_bytes(), original_manifest)
        self.assertEqual(self.hepmc.read_bytes(), original_hepmc)
        self.assertFalse(self.call_log.exists())

    def test_inputs_replaced_together_during_delphes_are_not_certified(self):
        # Replacing both files produces a valid new manifest/HepMC pair.
        # Merely verifying them again after the run would therefore be wrong.
        mutator = self.root / "replace_inputs.py"
        mutator.write_text(
            "import hashlib, json\nfrom pathlib import Path\n"
            f"hepmc = Path({str(self.hepmc)!r})\n"
            "hepmc.write_text(hepmc.read_text().replace('E 1', 'E 2'))\n"
            f"manifest_path = Path({str(self.generation / 'lhe-provenance.json')!r})\n"
            "manifest = json.loads(manifest_path.read_text())\n"
            "manifest['hepmc_sha256'] = hashlib.sha256(hepmc.read_bytes()).hexdigest()\n"
            "manifest_path.write_text(json.dumps(manifest))\n",
            encoding="utf-8",
        )
        reader = self.delphes / "DelphesHepMC3"
        reader.write_text(reader.read_text() + f"python3 {shlex.quote(str(mutator))}\n")
        result = self.run_simulation()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inputs changed during simulation", result.stderr)
        self.assertTrue((self.output / "FAILED").exists())
        self.assertFalse((self.output / "SUCCESS").exists())
        self.assertFalse((self.output / "lhe-provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
