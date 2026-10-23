"""Analytical workflow checks independent of neural fitting fluctuations."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from BackgroundRemoval.common import FEATURES, TREE_NAME
from omnifold.workflow import OmniFoldSession, TRUTH_FEATURES


class _HistogramRatio:
    """Exact physical ratio on two categories, persisted across session reloads."""

    calls = []
    fail_stage = None

    def __init__(self, manifest):
        self.manifest = manifest
        self.yield_ratio = manifest["yield_ratio"]

    @classmethod
    def fit(cls, tx, tw, rx, rw, *, target_splits, reference_splits,
            output_dir, config, metadata):
        call = {
            "target": np.array(tx, copy=True),
            "target_weights": np.array(tw, copy=True),
            "reference": np.array(rx, copy=True),
            "reference_weights": np.array(rw, copy=True),
            "target_splits": np.array(target_splits, copy=True),
            "reference_splits": np.array(reference_splits, copy=True),
            "metadata": dict(metadata),
        }
        cls.calls.append(call)
        if metadata["stage"] == cls.fail_stage:
            raise RuntimeError("injected histogram fitting failure")
        ratios = {}
        for category in np.unique(np.concatenate([tx[:, 0], rx[:, 0]])):
            numerator = float(np.sum(tw[tx[:, 0] == category]))
            denominator = float(np.sum(rw[rx[:, 0] == category]))
            if numerator <= 0 or denominator <= 0:
                raise ValueError("test histogram category has no positive measure")
            ratios[str(int(category))] = numerator / denominator
        manifest = {
            "ratios": ratios,
            "yield_ratio": float(np.sum(tw) / np.sum(rw)),
            "metadata": dict(metadata),
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return cls(manifest)

    @classmethod
    def load(cls, path, *, device=None):
        return cls(json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8")))

    def predict(self, features, batch_size=None):
        full = np.array([self.manifest["ratios"][str(int(value))]
                         for value in np.asarray(features)[:, 0]])
        return {"full_ratio": full, "shape_ratio": full / self.yield_ratio}


def _write(path, arrays):
    with uproot.recreate(path) as root_file:
        root_file.mktree(TREE_NAME, arrays)


def _feature_columns(prefix, categories):
    categories = np.asarray(categories, dtype=np.float32)
    names = FEATURES if prefix == "reco" else TRUTH_FEATURES
    return {name: categories.copy() for name in names}


def _inputs(directory, *, identity=False, direct_data_weight=False):
    """Response counts [[60, 10], [20, 30]], plus 20/60 missed events."""
    mc = directory / "gg_H_pythia.root"
    data = directory / "data.root"
    truth = directory / "gg_H_herwig.root"
    # The first two signed rows cancel to a net 60. The last row is a fake
    # outside the fiducial truth region and must never enter unfolding fits.
    mc_arrays = {
        "event_id": np.arange(8, dtype=np.uint64),
        "weight": np.array([61., -1., 20., 20., 10., 30., 60., 7.]),
        "fiducial": np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
        "reconstructed": np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=bool),
        "reco_m_ZZ": np.array([125., 125., 125., np.nan, 125., 125., np.nan, 125.]),
        "luminosity_fb": np.full(8, 312.),
        **_feature_columns("truth", [0, 0, 0, 0, 1, 1, 1, np.nan]),
        **_feature_columns("reco", [0, 0, 1, np.nan, 0, 1, np.nan, 0]),
    }
    # Known truth distortion (1.2, 0.8) gives reco counts (80, 48).
    counts = np.array([70., 50.]) if identity else np.array([80., 48.])
    observation_weights = np.array([counts[0] + 1, -1., counts[1]])
    data_arrays = {
        "event_id": np.arange(3, dtype=np.uint64),
        "weight": observation_weights * 2,
        "background_removal_weight": np.full(3, .5),
        "reconstructed": np.ones(3, dtype=bool),
        "reco_m_ZZ": np.full(3, 125.),
        "luminosity_fb": np.full(3, 312.),
        **_feature_columns("reco", [0, 0, 1]),
    }
    if direct_data_weight:
        data_arrays["weight_background_removed"] = observation_weights
    truth_arrays = {
        "event_id": np.arange(2, dtype=np.uint64),
        "weight": np.array([120., 80.]),
        "fiducial": np.ones(2, dtype=bool),
        "luminosity_fb": np.full(2, 312.),
        **_feature_columns("truth", [0, 1]),
    }
    _write(mc, mc_arrays)
    _write(data, data_arrays)
    _write(truth, truth_arrays)
    return data, mc, truth, counts


def _expected_updates(counts):
    # Independently written binned inefficient-response EM expression.
    response = np.array([[.6, .1], [.2, .3]])
    original = np.array([100., 100.])
    unfolded = original.copy()
    outputs = []
    for _ in range(2):
        unfolded *= (response.T @ (counts / (response @ unfolded))) / response.sum(axis=0)
        outputs.append((unfolded / original, response @ unfolded))
    return outputs


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _HistogramRatio.calls = []
        _HistogramRatio.fail_stage = None
        estimator_patch = patch("omnifold.workflow.RatioEstimator", _HistogramRatio)
        estimator_patch.start()
        self.addCleanup(estimator_patch.stop)

    def create(self, **input_options):
        self.data, self.mc, self.truth, self.counts = _inputs(self.root, **input_options)
        return OmniFoldSession.create(data_path=self.data, mc_path=self.mc,
                                      truth_path=self.truth, run_dir=self.root / "run")

    def test_inefficient_response_two_updates_resume_and_fixed_reference(self):
        session = self.create()
        first = session.step()
        session = OmniFoldSession.load(session.run_dir)
        second = session.step()
        self.assertEqual([item.index for item in session.iterations], [1, 2])
        self.assertEqual(len(_HistogramRatio.calls), 6)
        np.testing.assert_array_equal(session._data.weights, [81., -1., 48.])
        self.assertIsNone(session._data.truth)
        self.assertEqual(session._mc.weights.size, 7)
        self.assertIsNone(session._truth)  # Closure target has not been read.
        expected = _expected_updates(self.counts)
        original_reco = np.array([70., 50.])
        z = session._mc.truth[:, 0].astype(int)
        matched = session._mc.matched
        xs = np.array([[0.] * len(FEATURES), [1.] * len(FEATURES)])
        base = session._mc.weights[matched]
        for index, (result, (weights, folded)) in enumerate(zip((first, second), expected)):
            np.testing.assert_allclose(result.truth_weights, weights[z])
            np.testing.assert_allclose(result.truth_weights[~matched], weights)
            np.testing.assert_allclose(result.reco_likelihood.predict(xs)["full_ratio"],
                                       folded / original_reco)
            pull, push, fold = _HistogramRatio.calls[3 * index:3 * index + 3]
            np.testing.assert_array_equal(push["reference_weights"], base)
            np.testing.assert_array_equal(fold["reference_weights"], base)
            np.testing.assert_array_equal(push["target"], push["reference"])
            np.testing.assert_array_equal(push["target_splits"], push["reference_splits"])
            np.testing.assert_array_equal(pull["reference_splits"], push["target_splits"])
            np.testing.assert_array_equal(fold["target_splits"], push["target_splits"])
            self.assertLess(push["target_weights"][1], 0)
            self.assertLess(fold["target_weights"][1], 0)
            self.assertAlmostEqual(result.manifest["reference_reco_yield"], 120.)
            self.assertAlmostEqual(result.manifest["updated_reco_yield"], 128.)
            self.assertAlmostEqual(result.manifest["reference_truth_yield"], 200.)
            self.assertAlmostEqual(result.manifest["updated_truth_yield"], 100 * weights.sum())
            self.assertAlmostEqual(result.reco_likelihood.yield_ratio, 128 / 120)
        # Each bootstrap copy stays in the same data split, even with a new ID.
        self.assertEqual(session._data.splits[0], session._data.splits[1])
        np.testing.assert_allclose(_HistogramRatio.calls[3]["reference_weights"],
                                   base * expected[0][0][z[matched]])
        self.assertFalse(np.allclose(second.truth_weights,
                                     expected[0][0][z] * expected[1][0][z]))
        reloaded = OmniFoldSession.load(session.run_dir)
        self.assertEqual(reloaded.history(), session.history())
        for item in reloaded.iterations:
            self.assertTrue((item.root / "reco_pull" / "manifest.json").is_file())

    def test_identity_closure_and_direct_data_weight(self):
        session = self.create(identity=True, direct_data_weight=True)
        for _ in range(2):
            result = session.step()
            np.testing.assert_allclose(result.truth_weights, 1.)
            self.assertEqual(result.manifest["yield_ratios"],
                             {"reco_pull": 1., "truth_push": 1., "reco_likelihood": 1.})
            self.assertAlmostEqual(result.manifest["weighted_extended_log_ratio"], 0.)

    def test_fixed_reference_likelihood_and_frozen_toy_evaluation(self):
        session = self.create()
        session.step()
        session.step()
        nominal = session.evaluate_likelihood()
        toy_path = self.root / "data_0001.root"
        with uproot.open(self.data) as file:
            toy_arrays = file[TREE_NAME].arrays(library="np")
        toy_arrays["weight"] = np.array([20., 0., 40.])
        _write(toy_path, toy_arrays)
        toy = session.evaluate_likelihood(toy_path)
        self.assertEqual(len(_HistogramRatio.calls), 6)  # No refitting.
        for index, (_, folded) in enumerate(_expected_updates(self.counts)):
            full = folded / [70., 50.]
            shape = full / (128 / 120)
            self.assertAlmostEqual(nominal[index]["weighted_extended_log_ratio"],
                                   self.counts @ np.log(full) - 8)
            self.assertAlmostEqual(nominal[index]["weighted_shape_log_ratio"],
                                   self.counts @ np.log(shape))
            self.assertAlmostEqual(toy[index]["weighted_extended_log_ratio"],
                                   np.array([10., 20.]) @ np.log(full) - 8)

    def test_export_preserves_misses_and_truthless_data(self):
        session = self.create()
        session.step()
        session.step()
        files = session.decorate([self.data], output_dir=self.root / "decorated")
        self.assertEqual([path.name for path in files], [self.data.name, self.mc.name])
        with uproot.open(files[0]) as file:
            data = file[TREE_NAME].arrays(library="np")
        self.assertFalse(data["omnifold_truth_push_valid"].any())
        self.assertTrue(np.isnan(data["omnifold_truth_push"]).all())
        np.testing.assert_allclose(data["omnifold_weight"], [[81., 81.], [-1., -1.], [48., 48.]])
        with uproot.open(files[1]) as file:
            mc = file[TREE_NAME].arrays(library="np")
        expected = np.column_stack([item[0] for item in _expected_updates(self.counts)])
        np.testing.assert_allclose(mc["omnifold_truth_push"][[3, 6]], expected)
        np.testing.assert_allclose(mc["omnifold_weight"][[3, 6]], expected * [[20.], [60.]])
        self.assertTrue(np.isnan(mc["omnifold_reco_pull"][[3, 6]]).all())
        self.assertFalse(mc["omnifold_weight_valid"][-1])

    def test_failed_step_does_not_change_previous_checkpoint_and_can_retry(self):
        session = self.create()
        first = session.step()
        original = first.truth_weights.copy()
        _HistogramRatio.fail_stage = "truth_push"
        with self.assertRaisesRegex(RuntimeError, "injected"):
            session.step()
        self.assertEqual(len(session.iterations), 1)
        self.assertEqual([path.name for path in (session.run_dir / "iterations").iterdir()],
                         ["iteration_0001"])
        self.assertFalse((session.run_dir / ".training.lock").exists())
        np.testing.assert_array_equal(first.truth_weights, original)
        _HistogramRatio.fail_stage = None
        second = session.step()
        np.testing.assert_allclose(second.truth_weights,
                                   _expected_updates(self.counts)[1][0][session._mc.truth[:, 0].astype(int)])

    def test_changed_input_is_rejected_before_training_and_on_reload(self):
        session = self.create()
        with self.data.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            session.step()
        self.assertEqual(_HistogramRatio.calls, [])
        self.assertFalse((session.run_dir / ".training.lock").exists())
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            OmniFoldSession.load(session.run_dir)

    def test_checkpoint_corruption_is_rejected_live_and_on_reload(self):
        session = self.create()
        result = session.step()
        np.save(result.root / "truth_weights.npy", np.ones(7))
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            session.step()
        self.assertEqual(len(_HistogramRatio.calls), 3)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            OmniFoldSession.load(session.run_dir)

    def test_model_manifest_corruption_is_rejected_on_reload(self):
        session = self.create()
        result = session.step()
        path = result.root / "reco_likelihood" / "manifest.json"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Ratio-manifest checksum"):
            OmniFoldSession.load(session.run_dir)

    def test_live_session_and_iteration_manifest_changes_are_rejected(self):
        session = self.create()
        result = session.step()
        for path, message in ((session.run_dir / "session.json", "Session settings changed"),
                              (result.root / "iteration.json", "checkpoint changed")):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    session.step()
                self.assertFalse((session.run_dir / ".training.lock").exists())
                path.write_bytes(original)
        self.assertEqual(len(_HistogramRatio.calls), 3)

    def test_closure_cannot_use_selected_mixed_data_as_truth(self):
        data, mc, _, _ = _inputs(self.root)
        with self.assertRaisesRegex(ValueError, "separate full signal-only"):
            OmniFoldSession.create(data_path=data, mc_path=mc, truth_path=data,
                                   run_dir=self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
