"""CLI integration with real sessions/ROOT files and exact histogram ratios."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import uproot

from BackgroundRemoval.common import TREE_NAME, sha256_file
from omnifold import OmniFoldSession, RatioConfig
from omnifold import cli
from omnifold.tests.test_workflow import _HistogramRatio, _expected_updates, _inputs, _write


class OmniFoldCLITest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-omnifold-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.data, self.mc, self.truth, self.counts = _inputs(self.root)
        _HistogramRatio.calls = []
        _HistogramRatio.fail_stage = None
        self.addCleanup(setattr, _HistogramRatio, "fail_stage", None)
        estimator = patch("omnifold.workflow.RatioEstimator", _HistogramRatio)
        estimator.start()
        self.addCleanup(estimator.stop)
        reports = patch("omnifold.cli.save_reports", return_value=[])
        self.reports = reports.start()
        self.addCleanup(reports.stop)
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([str(argument) for argument in arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def success(self, *arguments):
        code, stdout, stderr = self.invoke(*arguments)
        self.assertEqual(code, 0, stderr + stdout)
        return stdout

    def train(self, *arguments):
        return self.success("train", "--run-dir", self.run_dir, "--data-root", self.data,
                            "--gg-h-root", self.mc, "--truth-root", self.truth, *arguments)

    def create_from_api(self, *, iterations=1):
        session = OmniFoldSession.create(data_path=self.data, mc_path=self.mc,
                                         truth_path=self.truth, run_dir=self.run_dir,
                                         config=RatioConfig(device="cpu"))
        with redirect_stdout(io.StringIO()):
            for _ in range(iterations):
                session.step()
        return session

    @staticmethod
    def report_iteration(call):
        return call.kwargs.get("iteration", call.args[1] if len(call.args) > 1 else None)

    def test_cli_train_loads_from_notebook_api_and_reports_each_completed_iteration(self):
        self.train("--iterations", 2, "--device", "cpu", "--shape-only")
        session = OmniFoldSession.load(self.run_dir)
        self.assertEqual([result.index for result in session.iterations], [1, 2])
        self.assertEqual(len(_HistogramRatio.calls), 6)
        self.assertEqual(self.reports.call_count, 2)
        self.assertEqual([self.report_iteration(call) for call in self.reports.call_args_list], [1, 2])
        for call in self.reports.call_args_list:
            self.assertIsInstance(call.args[0], OmniFoldSession)
            self.assertTrue(call.kwargs["shape_only"])
        categories = session._mc.truth[:, 0].astype(int)
        for result, (expected, _) in zip(session.iterations, _expected_updates(self.counts)):
            np.testing.assert_allclose(result.truth_weights, expected[categories])
        np.testing.assert_array_equal(session._data.weights, [81., -1., 48.])
        self.assertEqual(session.config, RatioConfig(device="cpu"))

    def test_notebook_campaign_resumes_with_one_additional_iteration_by_default(self):
        original = self.create_from_api()
        original_manifest = (self.run_dir / "session.json").read_bytes()
        first_checksum = sha256_file(original.iterations[0].root / "iteration.json")
        self.success("resume", "--run-dir", self.run_dir, "--device", "cpu", "--no-reports")
        resumed = OmniFoldSession.load(self.run_dir)
        self.assertEqual(len(resumed.iterations), 2)
        self.assertEqual((self.run_dir / "session.json").read_bytes(), original_manifest)
        self.assertEqual(sha256_file(resumed.iterations[0].root / "iteration.json"), first_checksum)
        expected = _expected_updates(self.counts)[1][0][resumed._mc.truth[:, 0].astype(int)]
        np.testing.assert_allclose(resumed.iterations[-1].truth_weights, expected)
        self.reports.assert_not_called()

    def test_train_default_is_one_iteration_and_all_ratio_settings_are_recorded(self):
        settings = dict(ensemble_size=5, hidden_layers=2, neurons=16, batch_size=64,
                        maximum_epochs=3, patience=2, learning_rate=.0007,
                        learning_rate_decay=.9, max_retries=1, seed=91, logit_clip=12.,
                        weight_decay=0., validation_interval_steps=7,
                        learning_rate_decay_steps=40, validation_min_delta=3e-5,
                        baseline_tolerance=.015,
                        inference_batch_size=128, steps_per_epoch=2, device="cpu")
        flags = [value for name, setting in settings.items()
                 for value in ("--" + name.replace("_", "-"), setting)]
        self.train(*flags, "--split-seed", 319, "--data-weight-branch", "weight",
                   "--mc-weight-branch", "weight", "--truth-weight-branch", "weight",
                   "--no-reports")
        session = OmniFoldSession.load(self.run_dir)
        self.assertEqual(len(session.iterations), 1)
        self.assertEqual(asdict(session.config), asdict(RatioConfig(**settings)))
        self.assertEqual(session.manifest["settings"], {
            "seed": 319, "data_weight_branch": "weight", "mc_weight_branch": "weight",
            "truth_weight_branch": "weight"})
        np.testing.assert_array_equal(session._data.weights, [162., -2., 96.])
        self.reports.assert_not_called()

    def test_stage_override_reaches_only_pull_and_survives_resume(self):
        path = self.root / "stages.json"
        path.write_text(json.dumps({"reco_pull": {"hidden_layers": 2, "neurons": 32,
                                                "batch_size": 64, "seed": 42}}))
        self.train("--stage-config-json", path, "--device", "cpu", "--no-reports")
        first = _HistogramRatio.calls[:3]
        self.assertEqual([call["config"]["neurons"] for call in first], [32, 1024, 1024])
        self.assertEqual([call["config"]["seed"] for call in first], [10042, 19120908, 19120909])
        self.success("resume", "--run-dir", self.run_dir, "--device", "cpu", "--no-reports")
        self.assertEqual([call["config"]["neurons"] for call in _HistogramRatio.calls[3:]], [32, 1024, 1024])
        session = OmniFoldSession.load(self.run_dir, device="cpu")
        session.stage_configs["reco_pull"] = replace(session.stage_configs["reco_pull"], neurons=64)
        with self.assertRaisesRegex(ValueError, "Training settings changed"):
            session.step()
        self.assertFalse((self.run_dir / ".training.lock").exists())

    def test_invalid_stage_overrides_fail_without_creating_campaign(self):
        path = self.root / "stages.json"
        for values in ({"unknown": {}}, {"reco_pull": {"dropout": .1}},
                       {"reco_pull": {"weight_decay": .1}}, {"reco_pull": {"neurons": 0}},
                       {"reco_pull": {"device": "cuda"}}, []):
            with self.subTest(values=values):
                path.write_text(json.dumps(values))
                code, _, _ = self.invoke("train", "--run-dir", self.run_dir,
                    "--data-root", self.data, "--gg-h-root", self.mc,
                    "--stage-config-json", path, "--no-reports")
                self.assertEqual(code, 1)
                self.assertFalse(self.run_dir.exists())
        self.assertEqual(_HistogramRatio.calls, [])

    def test_previous_raw_policy_can_report_but_cli_resume_requires_new_campaign(self):
        session = self.create_from_api()
        path = self.run_dir / "session.json"
        manifest = json.loads(path.read_text())
        manifest.pop("training_policy_version")
        path.write_text(json.dumps(manifest))
        record_path = session.iterations[0].root / "iteration.json"
        record = json.loads(record_path.read_text())
        record["session_sha256"] = sha256_file(path)
        record_path.write_text(json.dumps(record))
        self.success("report", "--run-dir", self.run_dir)
        self.success("likelihood", "--run-dir", self.run_dir)
        code, _, stderr = self.invoke("resume", "--run-dir", self.run_dir)
        self.assertEqual(code, 1)
        self.assertIn("start a new campaign", stderr)
        self.assertEqual(len(_HistogramRatio.calls), 3)

    def test_failed_second_stage_preserves_checkpoint_and_cli_resume_recovers(self):
        session = self.create_from_api()
        checkpoint = session.iterations[0].root / "truth_weights.npy"
        original_sha = sha256_file(checkpoint)
        _HistogramRatio.fail_stage = "truth_push"
        code, _, stderr = self.invoke("resume", "--run-dir", self.run_dir)
        self.assertEqual(code, 1)
        self.assertIn("injected histogram fitting failure", stderr)
        self.assertEqual(sha256_file(checkpoint), original_sha)
        self.assertFalse((self.run_dir / ".training.lock").exists())
        self.assertEqual([path.name for path in (self.run_dir / "iterations").iterdir()],
                         ["iteration_0001"])
        self.assertEqual(len(OmniFoldSession.load(self.run_dir).iterations), 1)
        self.reports.assert_not_called()
        _HistogramRatio.fail_stage = None
        self.success("resume", "--run-dir", self.run_dir)
        self.assertEqual(len(OmniFoldSession.load(self.run_dir).iterations), 2)
        self.assertEqual(self.reports.call_count, 1)
        self.assertEqual(self.report_iteration(self.reports.call_args), 2)

    def test_train_cannot_overwrite_an_existing_campaign(self):
        self.create_from_api()
        snapshot = (self.run_dir / "session.json").read_bytes()
        previous_calls = len(_HistogramRatio.calls)
        code, _, stderr = self.invoke("train", "--run-dir", self.run_dir,
                                      "--data-root", self.data, "--gg-h-root", self.mc)
        self.assertEqual(code, 1)
        self.assertIn("exists", stderr)
        self.assertEqual((self.run_dir / "session.json").read_bytes(), snapshot)
        self.assertEqual(len(_HistogramRatio.calls), previous_calls)

    def test_report_failure_preserves_completed_fit_and_report_retry_never_refits(self):
        self.reports.side_effect = OSError("injected report write failure")
        code, _, stderr = self.invoke("train", "--run-dir", self.run_dir,
                                      "--data-root", self.data, "--gg-h-root", self.mc,
                                      "--iterations", 2)
        self.assertEqual(code, 1)
        self.assertIn("already checkpointed", stderr)
        self.assertIn("injected report write failure", stderr)
        self.assertEqual(len(OmniFoldSession.load(self.run_dir).iterations), 1)
        self.assertEqual(len(_HistogramRatio.calls), 3)
        self.reports.side_effect = None
        self.success("report", "--run-dir", self.run_dir, "--iteration", 1)
        self.assertEqual(len(_HistogramRatio.calls), 3)
        self.assertEqual(len(OmniFoldSession.load(self.run_dir).iterations), 1)

    def test_report_latest_selected_and_all_iterations_without_training(self):
        self.create_from_api(iterations=2)
        previous_calls = len(_HistogramRatio.calls)
        self.success("report", "--run-dir", self.run_dir, "--device", "cpu")
        self.assertIn(self.report_iteration(self.reports.call_args), (None, 2))
        self.reports.reset_mock()
        self.success("report", "--run-dir", self.run_dir, "--iteration", 1, "--shape-only")
        self.assertEqual(self.report_iteration(self.reports.call_args), 1)
        self.assertTrue(self.reports.call_args.kwargs["shape_only"])
        self.reports.reset_mock()
        self.success("report", "--run-dir", self.run_dir, "--all-iterations")
        self.assertEqual([self.report_iteration(call) for call in self.reports.call_args_list], [1, 2])
        self.assertEqual(len(_HistogramRatio.calls), previous_calls)

    def test_history_and_likelihood_json_match_api_and_do_not_refit(self):
        session = self.create_from_api(iterations=2)
        previous_calls = len(_HistogramRatio.calls)
        output = self.root / "history.json"
        stdout = self.success("history", "--run-dir", self.run_dir, "--output", output)
        self.assertEqual(json.loads(stdout), session.history())
        self.assertEqual(json.loads(output.read_text()), session.history())
        stdout = self.success("likelihood", "--run-dir", self.run_dir, "--device", "cpu")
        self.assertEqual(json.loads(stdout), session.evaluate_likelihood())
        with uproot.open(self.data) as root_file:
            arrays = root_file[TREE_NAME].arrays(library="np")
        arrays["weight"] = np.array([20., 0., 40.])
        toy = self.root / "data_0001.root"
        _write(toy, arrays)
        toy_output = self.root / "likelihood.json"
        stdout = self.success("likelihood", "--run-dir", self.run_dir,
                              "--data-root", toy, "--output", toy_output)
        expected = session.evaluate_likelihood(toy)
        self.assertEqual(json.loads(stdout), expected)
        self.assertEqual(json.loads(toy_output.read_text()), expected)
        self.assertNotEqual(expected, session.evaluate_likelihood())
        self.assertEqual(len(_HistogramRatio.calls), previous_calls)
        self.reports.assert_not_called()
        before = output.read_bytes()
        code, _, stderr = self.invoke("history", "--run-dir", self.run_dir, "--output", output)
        self.assertEqual(code, 1)
        self.assertIn("exist", stderr.lower())
        self.assertEqual(output.read_bytes(), before)

    def test_apply_expands_directories_globs_and_files_preserving_physics_outputs(self):
        session = self.create_from_api(iterations=2)
        toy = self.root / "data_0001.root"
        shutil.copyfile(self.data, toy)
        source_hashes = {path: sha256_file(path) for path in (self.data, self.mc, toy)}
        destination = self.root / "applied"
        # All three forms overlap; deduplication must retain each input once.
        self.success("apply", "--run-dir", self.run_dir, "--inputs", self.root,
                     str(self.root / "data*.root"), self.data, "--output-dir", destination)
        self.assertEqual(sorted(path.name for path in destination.glob("*.root")),
                         ["data.root", "data_0001.root", "gg_H_pythia.root"])
        with uproot.open(destination / "data.root") as root_file:
            data = root_file[TREE_NAME].arrays(library="np")
        np.testing.assert_allclose(data["omnifold_weight"], [[81., 81.], [-1., -1.], [48., 48.]])
        self.assertFalse(data["omnifold_truth_push_valid"].any())
        self.assertTrue(np.isnan(data["omnifold_truth_push"]).all())
        with uproot.open(destination / "gg_H_pythia.root") as root_file:
            mc = root_file[TREE_NAME].arrays(library="np")
        expected = np.column_stack([result[0] for result in _expected_updates(self.counts)])
        np.testing.assert_allclose(mc["omnifold_truth_push"][[3, 6]], expected)
        np.testing.assert_allclose(mc["omnifold_weight"][[3, 6]], expected * [[20.], [60.]])
        self.assertTrue(np.isnan(mc["omnifold_reco_pull"][[3, 6]]).all())
        self.assertFalse(mc["omnifold_weight_valid"][-1])
        np.testing.assert_array_equal(mc["weight"], [61., -1., 20., 20., 10., 30., 60., 7.])
        for source, digest in source_hashes.items():
            self.assertEqual(sha256_file(source), digest)
        self.assertEqual(len(_HistogramRatio.calls), 3 * len(session.iterations))
        self.reports.assert_not_called()

    def test_apply_no_mc_overwrite_and_unmatched_pattern(self):
        session = self.create_from_api()
        destination = self.root / "data_only"
        arguments = ("apply", "--run-dir", self.run_dir, "--inputs", self.data,
                     "--output-dir", destination, "--no-mc", "--device", "cpu")
        self.success(*arguments)
        self.assertEqual([path.name for path in destination.glob("*.root")], ["data.root"])
        code, _, stderr = self.invoke(*arguments)
        self.assertEqual(code, 1)
        self.assertIn("exists", stderr)
        self.success(*arguments, "--overwrite")
        with uproot.open(destination / "data.root") as root_file:
            self.assertEqual(root_file[TREE_NAME]["omnifold_weight"].array(library="np").shape, (3, 1))
        # --no-mc suppresses automatic inclusion. An explicitly supplied
        # nominal MC still needs MC weighting, including reconstruction losses.
        explicit_mc = self.root / "explicit_mc"
        self.success("apply", "--run-dir", self.run_dir, "--inputs", self.data, self.mc,
                     "--output-dir", explicit_mc, "--no-mc")
        self.assertEqual(sorted(path.name for path in explicit_mc.glob("*.root")),
                         ["data.root", "gg_H_pythia.root"])
        with uproot.open(explicit_mc / self.mc.name) as root_file:
            arrays = root_file[TREE_NAME].arrays(library="np")
        np.testing.assert_allclose(arrays["omnifold_weight"][:7, 0],
                                   session._mc.weights * session.iterations[0].truth_weights)
        self.assertTrue(arrays["omnifold_weight_valid"][[3, 6]].all())
        self.assertFalse(arrays["omnifold_weight_valid"][-1])
        missing_destination = self.root / "missing"
        code, _, stderr = self.invoke("apply", "--run-dir", self.run_dir, "--inputs",
                                      str(self.root / "not_present*.root"), "--output-dir", missing_destination)
        self.assertEqual(code, 1)
        self.assertTrue(stderr)
        self.assertFalse(missing_destination.exists())

    def test_exports_cannot_create_checkpoints_or_locks_through_direct_or_symlink_paths(self):
        self.create_from_api()
        original_manifest = (self.run_dir / "session.json").read_bytes()
        iterations = self.run_dir / "iterations"
        alias = self.root / "checkpoint_alias"
        alias.symlink_to(iterations, target_is_directory=True)
        protected_outputs = [
            iterations,
            iterations / "iteration_0002" / "history.json",
            self.run_dir / ".training.lock",
            self.run_dir / ".training.lock" / "history.json",
            self.run_dir / "session.json",
            self.run_dir / "session.json" / "history.json",
            alias,
            alias / "iteration_0002" / "history.json",
        ]
        for command in ("history", "likelihood", "apply"):
            for destination in protected_outputs:
                arguments = [command, "--run-dir", self.run_dir]
                if command == "apply":
                    arguments.extend(["--inputs", self.data, "--output-dir", destination])
                else:
                    arguments.extend(["--output", destination])
                with self.subTest(command=command, destination=destination):
                    code, _, stderr = self.invoke(*arguments)
                    self.assertEqual(code, 1)
                    self.assertIn("reserved", stderr)
                    self.assertFalse((iterations / "iteration_0002").exists())
                    self.assertFalse((self.run_dir / ".training.lock").exists())
                    self.assertEqual((self.run_dir / "session.json").read_bytes(), original_manifest)
                    self.assertEqual(len(OmniFoldSession.load(self.run_dir).iterations), 1)
        self.assertEqual([path.name for path in iterations.iterdir()], ["iteration_0001"])
        self.success("resume", "--run-dir", self.run_dir, "--no-reports")
        self.assertEqual(len(OmniFoldSession.load(self.run_dir).iterations), 2)

    def test_parser_rejects_nonpositive_counts_and_resume_training_overrides(self):
        commands = [
            ["train", "--run-dir", self.run_dir, "--data-root", self.data, "--gg-h-root", self.mc],
            ["resume", "--run-dir", self.run_dir],
        ]
        bad_arguments = [command + ["--iterations", count] for command in commands for count in (0, -1)]
        bad_arguments.extend([
            commands[0] + ["--calibration-scale-min", .1],
            commands[0] + ["--calibration-scale-max", 5.],
            ["resume", "--run-dir", self.run_dir, "--neurons", 32],
            ["resume", "--run-dir", self.run_dir, "--split-seed", 44],
            ["report", "--run-dir", self.run_dir, "--iteration", 0],
            ["report", "--run-dir", self.run_dir, "--iteration", 1, "--all-iterations"],
            ["history", "--run-dir", self.run_dir, "--cpu-threads", 0],
            ["history", "--run-dir", self.run_dir, "--cpu-threads", -2],
        ])
        for arguments in bad_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as exception:
                self.invoke(*arguments)
            self.assertEqual(exception.exception.code, 2)
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(_HistogramRatio.calls, [])

    def test_cpu_threads_explicit_slurm_and_local_default(self):
        self.create_from_api()
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "3"}):
            self.success("history", "--run-dir", self.run_dir)
            self.assertEqual(torch.get_num_threads(), 3)
            self.success("history", "--run-dir", self.run_dir, "--cpu-threads", 2)
            self.assertEqual(torch.get_num_threads(), 2)
        with patch.dict(os.environ):
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
            self.success("history", "--run-dir", self.run_dir)
            self.assertEqual(torch.get_num_threads(), 4)

    def test_python_module_help_is_available_in_a_fresh_process(self):
        completed = subprocess.run([sys.executable, "-m", "omnifold", "--help"],
                                   cwd=Path(__file__).resolve().parents[2], capture_output=True,
                                   text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for command in ("train", "resume", "report", "apply", "history", "likelihood"):
            self.assertIn(command, completed.stdout)


if __name__ == "__main__":
    unittest.main()
