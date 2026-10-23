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
            "import shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[2])\n",
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

    def run_simulation(self, *extra: str):
        environment = os.environ.copy()
        environment["PATH"] = str(self.executables) + os.pathsep + environment["PATH"]
        return subprocess.run(
            [
                "bash", str(self.simulation / "run_simulation.sh"),
                str(self.generation), "--output-root", str(self.root / "outputs"),
                *extra,
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
        resumed = self.run_simulation()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("Already complete", resumed.stdout)
        self.assertEqual(self.call_log.read_text().splitlines(), ["called"])

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
