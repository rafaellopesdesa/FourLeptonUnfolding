from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest

import uproot


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "BatchSubmit" / "unity_generation_job.sh"


def write_executable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(payload).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def shell_assignment(name: str, value: object) -> str:
    return f"{name}={shlex.quote(str(value))}\n"


class FullChainWorkerTest(unittest.TestCase):
    def make_campaign(self, root: Path, *, fail_simulation: bool = False) -> tuple[Path, Path]:
        fake_repo = root / "repo"
        campaign = root / "campaign"
        grid = campaign / "grid"
        outputs = root / "analysis"
        scratch = root / "scratch"
        for directory in (grid, outputs, scratch):
            directory.mkdir(parents=True, exist_ok=True)
        (grid / "GRID_READY").touch()
        (grid / "pwg-test-grid.dat").write_text("grid\n", encoding="utf-8")
        (grid / "pwg-test-ubound.dat").write_text("ubound\n", encoding="utf-8")
        run_card = root / "powheg.input"
        run_card.write_text("ebeam1 6800d0\nebeam2 6800d0\n", encoding="utf-8")

        write_executable(
            fake_repo / "Generation" / "run_generation.sh",
            """
            #!/bin/bash
            set -euo pipefail
            output=""
            while (($#)); do
              if [[ "$1" == --output-dir ]]; then output="$2"; shift 2; else shift; fi
            done
            mkdir -p "$output"
            printf 'HepMC::Version 3.0.0\nHepMC::Asciiv3-START_EVENT_LISTING\nE 1\n' >"$output/events.hepmc3"
            printf 'process=gg_H\nseed=1001\n' >"$output/run-metadata.txt"
            printf 'generation log\n' >"$output/powheg.log"
            """,
        )
        simulation_body = "exit 17" if fail_simulation else """
            input="$1"; shift
            output_root=""
            while (($#)); do
              if [[ "$1" == --output-root ]]; then output_root="$2"; shift 2; else shift; fi
            done
            output="$output_root/$(basename "$input")"
            mkdir -p "$output"
            printf 'delphes placeholder\n' >"$output/delphes.root"
            printf 'simulation log\n' >"$output/delphes.log"
            printf 'process=gg_H\n' >"$output/simulation-metadata.txt"
        """
        write_executable(
            fake_repo / "Simulation" / "run_simulation.sh",
            f"""
            #!/bin/bash
            set -euo pipefail
            {simulation_body}
            """,
        )
        write_executable(
            fake_repo / "Analysis" / "build_analysis_tree.py",
            """
            #!/usr/bin/env python3
            import argparse
            import numpy as np
            import uproot
            parser = argparse.ArgumentParser()
            parser.add_argument("input")
            parser.add_argument("--output", required=True)
            parser.add_argument("--mass-region")
            args = parser.parse_args()
            size = 3
            index = np.arange(size)
            with uproot.recreate(args.output) as root_file:
                root_file["Analysis"] = {
                    "event_id": index.astype(np.uint64),
                    "event_number": index.astype(np.int64),
                    "weight": np.ones(size),
                    "cross_section_pb": np.full(size, 0.2),
                    "fiducial": np.ones(size, dtype=np.bool_),
                    "reconstructed": np.ones(size, dtype=np.bool_),
                    "truth_type": np.zeros(size, dtype=np.int8),
                    "reco_type": np.zeros(size, dtype=np.int8),
                }
            """,
        )
        write_executable(
            fake_repo / "Analysis" / "validate_analysis_output.py",
            """
            #!/usr/bin/env python3
            import argparse
            import uproot
            parser = argparse.ArgumentParser()
            parser.add_argument("path")
            parser.add_argument("--expected-entries", type=int, required=True)
            args = parser.parse_args()
            with uproot.open(args.path) as root_file:
                assert root_file["Analysis"].num_entries == args.expected_entries
            """,
        )

        config = campaign / "campaign.env"
        settings = {
            "REPO_ROOT": fake_repo,
            "CAMPAIGN_NAME": "campaign",
            "CAMPAIGN_DIR": campaign,
            "PROCESS": "gg_H",
            "SHOWER": "pythia",
            "EVENTS_PER_JOB": 3,
            "BASE_SEED": 1001,
            "GRID_DIR": grid,
            "REUSED_GRID": 0,
            "RUN_CARD": run_card,
            "PDF_ID": 303400,
            "ANALYSIS_OUTPUT_ROOT": outputs,
            "ANALYSIS_PYTHON": sys.executable,
            "MASS_REGION": "extended",
            "SCRATCH_ROOT": scratch,
            "DELPHES_CARD": "",
            "HIGGS_BR": "2.771E-04",
            "KEEP_INTERMEDIATES": 0,
        }
        config.write_text(
            "".join(shell_assignment(name, value) for name, value in settings.items()),
            encoding="utf-8",
        )
        return config, outputs

    def run_worker(self, config: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_ARRAY_JOB_ID": "700",
                "SLURM_JOB_ID": "700_0",
            }
        )
        return subprocess.run(
            ["bash", WORKER, "events", config, "0"],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_runs_full_chain_publishes_compact_shard_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, outputs = self.make_campaign(root)
            result = self.run_worker(config)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = outputs / "gg_H_pythia_campaign_job_000000_seed1001.root"
            self.assertTrue(output.is_file())
            with uproot.open(output) as root_file:
                self.assertEqual(root_file["Analysis"].num_entries, 3)
            job = root / "campaign" / "jobs" / "job_000000_seed1001"
            self.assertTrue((job / "SUCCESS").is_file())
            self.assertFalse((job / "FAILED").exists())
            self.assertTrue((job / "diagnostics" / "generation-powheg.log").is_file())
            self.assertFalse(any((root / "scratch").iterdir()))
            self.assertEqual(
                (root / "campaign" / "grid" / "pwg-test-grid.dat").read_text(),
                "grid\n",
            )

    def test_failure_never_exposes_a_final_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, outputs = self.make_campaign(root, fail_simulation=True)
            result = self.run_worker(config)
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertFalse(any(outputs.iterdir()))
            job = root / "campaign" / "jobs" / "job_000000_seed1001"
            self.assertTrue((job / "FAILED").is_file())
            status = (job / "slurm-status.txt").read_text(encoding="utf-8")
            self.assertIn("stage=simulation", status)
            self.assertFalse(any((root / "scratch").iterdir()))

    def test_requeue_accepts_only_the_same_recorded_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, outputs = self.make_campaign(root)
            first = self.run_worker(config)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self.run_worker(config)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("Already complete", second.stdout)

            metadata = (
                root
                / "campaign"
                / "jobs"
                / "job_000000_seed1001"
                / "output-metadata.txt"
            )
            metadata.write_text(
                metadata.read_text(encoding="utf-8").replace(
                    "campaign_config_sha256=", "campaign_config_sha256=wrong-"
                ),
                encoding="utf-8",
            )
            rejected = self.run_worker(config)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("campaign provenance", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
