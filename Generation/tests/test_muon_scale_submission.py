"""Dry-run submission checks without a generator installation or Slurm."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class MuonScaleSubmissionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        for name in ("Generation", "Simulation", "BatchSubmit", "Analysis", "PowhegCards"):
            (self.repo / name).mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "Generation" / "submit_generation.sh",
                     self.repo / "Generation" / "submit_generation.sh")
        for name in ("Generation", "Simulation"):
            (self.repo / name / "env.sh").write_text("# Stub installation environment\n")
        self.executable(self.repo / "BatchSubmit" / "unity_generation_job.sh", "#!/bin/bash\nexit 0\n")
        (self.repo / "PowhegCards" / "gg_H.powheg.input").write_text("ebeam1 6800d0\nebeam2 6800d0\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        python = self.bin / "analysis-python"
        self.executable(python, "#!/bin/bash\nexit 0\n")
        self.executable(self.bin / "pixi", "#!/bin/bash\nset -eu\n"
                        f"if [[ $1 == run ]]; then printf '%s\\n' {shlex.quote(str(python))}; fi\n")
        self.outputs = self.root / "outputs"

    @staticmethod
    def executable(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    def submit(self, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PATH"] = str(self.bin) + os.pathsep + environment["PATH"]
        return subprocess.run(
            ["bash", self.repo / "Generation" / "submit_generation.sh", "gg_H", "pythia",
             "--campaign", "study", "--output-root", str(self.outputs), "--jobs", "1",
             "--events-per-job", "1000", "--dry-run", *extra],
            env=environment, capture_output=True, text=True, check=False,
        )

    def test_variants_have_separate_campaigns_and_saved_worker_settings(self):
        for variation in ("nominal", "up", "down"):
            with self.subTest(variation=variation):
                options = [] if variation == "nominal" else ["--muon-scale", variation]
                result = self.submit(*options)
                self.assertEqual(result.returncode, 0, result.stderr)
                suffix = "" if variation == "nominal" else f"_muon_scale_{variation}"
                campaign = self.outputs / f"study{suffix}"
                self.assertTrue(campaign.is_dir())
                config = campaign / "campaign.env"
                settings = subprocess.run(
                    ["bash", "-c", 'source "$1"; printf "%s\\n" "$CAMPAIGN_NAME" "$MUON_SCALE" "$BASE_SEED" "$ANALYSIS_OUTPUT_ROOT"',
                     "config-reader", str(config)],
                    capture_output=True, text=True, check=True,
                ).stdout.splitlines()
                self.assertEqual(settings, ["study", variation, "1001", str(self.outputs / "analysis")])
                self.assertIn(str(campaign), result.stdout)
        self.assertEqual(len(list(self.outputs.glob("study*"))), 3)

    def test_invalid_variation_is_rejected_before_dependency_checks(self):
        (self.repo / "Generation" / "env.sh").unlink()
        result = self.submit("--muon-scale", "sideways")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--muon-scale", result.stderr)
        self.assertNotIn("Missing", result.stderr)
        self.assertFalse(self.outputs.exists())


if __name__ == "__main__":
    unittest.main()
