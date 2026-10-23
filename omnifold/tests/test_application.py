from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import uproot

from BackgroundRemoval.common import FEATURES, sha256_file
from omnifold.application import MANIFEST_NAME, TRUTH_FEATURES, decorate_files


class _Ratio:
    def __init__(self, value: float, yield_ratio: float = 2.0):
        self.value = value
        self.yield_ratio = yield_ratio
        self.manifest = {"test_ratio": value, "yield_ratio": yield_ratio}
        self.calls: list[np.ndarray] = []

    def predict(self, features, batch_size=65536):
        self.calls.append(features.copy())
        return {
            "full_ratio": np.full(len(features), self.value),
            "shape_ratio": np.full(len(features), self.value / self.yield_ratio),
        }


def _iterations():
    return [
        SimpleNamespace(
            index=1,
            reco_pull=_Ratio(1.2), truth_push=_Ratio(1.5), reco_likelihood=_Ratio(1.4),
        ),
        SimpleNamespace(
            index=2,
            reco_pull=_Ratio(1.1), truth_push=_Ratio(1.8), reco_likelihood=_Ratio(1.7),
        ),
    ]


def _sample(path: Path, *, with_truth=True, with_weight=True, fallback=False, overrides=None):
    arrays = {
        "event_id": np.arange(5, dtype=np.uint64),
        "weight": np.array([2.0, -0.5, 3.0, 4.0, 5.0]),
        "reconstructed": np.array([True, True, False, True, True]),
        "reco_m_ZZ": np.array([122.0, 122.0, np.nan, 115.0, 122.0], np.float32),
        "existing_vector": np.arange(15, dtype=np.float32).reshape(5, 3),
    }
    if with_weight:
        arrays["background_removal_weight"] = np.full(5, 0.25)
        if not fallback:
            arrays["weight_background_removed"] = arrays["weight"] * 0.25
    for i, name in enumerate(FEATURES):
        arrays[name] = np.full(5, 0.1 * i, dtype=np.float32)
        arrays[name][2] = np.nan
    if with_truth:
        arrays["fiducial"] = np.array([True, True, True, True, False])
        for i, name in enumerate(TRUTH_FEATURES):
            arrays[name] = np.full(5, 0.2 * i, dtype=np.float32)
    arrays.update(overrides or {})
    with uproot.recreate(path) as root_file:
        root_file.mktree("Analysis", arrays)
        root_file["background_removal_metadata"] = json.dumps({"keep_me": True})
    return arrays


class ApplicationTest(unittest.TestCase):
    def test_three_cumulative_vectors_preserve_signed_weights_and_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.root"
            mc = root / "gg_H_pythia.root"
            original = _sample(data)
            _sample(mc, with_weight=False)
            before = sha256_file(data)
            iterations = _iterations()
            paths = decorate_files([data], root / "decorated", iterations, mc_path=mc, step_size=2)
            self.assertEqual([p.name for p in paths], [data.name, mc.name])
            with uproot.open(paths[0]) as file:
                arrays = file["Analysis"].arrays(library="np")
                for name, values in original.items():
                    np.testing.assert_equal(arrays[name], values)
                np.testing.assert_allclose(arrays["omnifold_reco_pull"][[0, 1, 4]], [[1.2, 1.1]] * 3)
                np.testing.assert_allclose(arrays["omnifold_reco_likelihood"][[0, 1, 4]], [[1.4, 1.7]] * 3)
                np.testing.assert_allclose(arrays["omnifold_weight"][1], [-0.125, -0.125])
                np.testing.assert_array_equal(arrays["omnifold_reco_pull_valid"], [1, 1, 0, 0, 1])
                self.assertTrue(np.isnan(arrays["omnifold_reco_pull"][2:4]).all())
                self.assertEqual(json.loads(str(file["background_removal_metadata"])), {"keep_me": True})
                metadata = json.loads(str(file["omnifold_metadata"]))
                self.assertTrue(metadata["data_truth_is_diagnostic_only"])
                self.assertEqual(metadata["source_provenance"]["sha256"], before)
            with uproot.open(paths[1]) as file:
                arrays = file["Analysis"].arrays(library="np")
                np.testing.assert_allclose(arrays["omnifold_truth_push"][:4], [[1.5, 1.8]] * 4)
                # Index 1 is negative; second column is cumulative, not 1.5*1.8.
                np.testing.assert_allclose(arrays["omnifold_weight"][1], [-0.75, -0.9])
                # The missed reco event still receives its truth weight.
                np.testing.assert_allclose(arrays["omnifold_weight"][2], [4.5, 5.4])
                self.assertTrue(np.isnan(arrays["omnifold_weight"][4]).all())
                self.assertFalse(arrays["omnifold_reco_pull_valid"][4])
                np.testing.assert_allclose(arrays["omnifold_truth_push_yield_ratio"], 2.0)
                np.testing.assert_allclose(
                    arrays["omnifold_truth_push"][:4],
                    arrays["omnifold_truth_push_shape"][:4] * arrays["omnifold_truth_push_yield_ratio"][:4],
                )
            self.assertEqual(sha256_file(data), before)
            manifest = json.loads((root / "decorated" / MANIFEST_NAME).read_text())
            self.assertEqual(manifest["iterations"], [1, 2])
            self.assertEqual(len(manifest["files"]), 2)

    def test_real_data_has_no_fabricated_truth_and_supports_primary_purity_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.root"
            _sample(data, with_truth=False, fallback=True)
            paths = decorate_files([data], root / "decorated", _iterations())
            with uproot.open(paths[0]) as file:
                arrays = file["Analysis"].arrays(library="np")
                self.assertFalse(np.any(arrays["omnifold_truth_push_valid"]))
                self.assertTrue(np.isnan(arrays["omnifold_truth_push"]).all())
                np.testing.assert_allclose(arrays["omnifold_weight"][1], [-0.125, -0.125])
                metadata = json.loads(str(file["omnifold_metadata"]))
                self.assertEqual(metadata["weight_expression"], "weight * background_removal_weight")

    def test_failed_campaign_leaves_no_partial_output_and_overwrite_is_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad = root / "data.root", root / "data_0001.root"
            _sample(good)
            _sample(bad, with_weight=False)
            destination = root / "decorated"
            with self.assertRaises(KeyError):
                decorate_files([good, bad], destination, _iterations())
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".decorated.*")), [])
            decorate_files([good], destination, _iterations())
            with self.assertRaises(FileExistsError):
                decorate_files([good], destination, _iterations())
            decorate_files([good], destination, _iterations(), overwrite=True)
            (destination / "unrelated.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "unrecognized"):
                decorate_files([good], destination, _iterations(), overwrite=True)
            self.assertEqual((destination / "unrelated.txt").read_text(), "keep")

    def test_input_collision_extra_objects_and_invalid_iterations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.root"
            _sample(data)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                decorate_files([data, data], root / "duplicate", _iterations())
            with self.assertRaisesRegex(ValueError, "contain an input"):
                decorate_files([data], root, _iterations(), overwrite=True)
            with self.assertRaisesRegex(ValueError, "ordered sequence"):
                decorate_files([data], root / "bad_order", _iterations()[1:])
            with uproot.update(data) as file:
                file.mktree("Additional", {"x": np.array([1])})
            with self.assertRaisesRegex(ValueError, "refusing to drop"):
                decorate_files([data], root / "objects", _iterations())

    def test_invalid_model_predictions_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.root"
            _sample(data)
            iterations = _iterations()
            iterations[1].reco_likelihood.value = np.inf
            with self.assertRaisesRegex(ValueError, "non-positive/non-finite"):
                decorate_files([data], root / "bad", iterations)
            self.assertFalse((root / "bad").exists())

    def test_campaign_luminosity_and_selected_features_cannot_be_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.root"
            _sample(source, overrides={"luminosity_fb": np.full(5, 100.0)})
            with self.assertRaisesRegex(ValueError, "luminosity_fb does not match"):
                decorate_files(
                    [source], root / "wrong_lumi", _iterations(),
                    metadata={"luminosity_fb": 312.0},
                )
            self.assertFalse((root / "wrong_lumi").exists())
            _sample(source, overrides={"luminosity_fb": np.full(5, np.nan)})
            with self.assertRaisesRegex(ValueError, "luminosity_fb does not match"):
                decorate_files(
                    [source], root / "nan_lumi", _iterations(),
                    metadata={"luminosity_fb": 312.0},
                )
            _sample(source, overrides={"luminosity_fb": np.full(5, 312.0)})
            decorate_files(
                [source], root / "matching_lumi", _iterations(),
                metadata={"luminosity_fb": 312.0},
            )
            invalid_reco = np.zeros(5, dtype=np.float32)
            invalid_reco[0] = np.nan
            _sample(source, overrides={FEATURES[0]: invalid_reco})
            with self.assertRaisesRegex(ValueError, "non-finite reconstructed features"):
                decorate_files([source], root / "bad_reco", _iterations())
            invalid_truth = np.zeros(5, dtype=np.float32)
            invalid_truth[2] = np.nan  # Fiducial event failing reconstruction.
            _sample(source, overrides={TRUTH_FEATURES[0]: invalid_truth})
            with self.assertRaisesRegex(ValueError, "non-finite truth features"):
                decorate_files([], root / "bad_truth", _iterations(), mc_path=source)
            # The same truth information is optional diagnostics on pseudo-data.
            paths = decorate_files([source], root / "optional_truth", _iterations())
            with uproot.open(paths[0]) as file:
                truth_valid = file["Analysis"]["omnifold_truth_push_valid"].array(library="np")
                self.assertFalse(truth_valid[2])
            # Real-data files may legitimately have neither truth nor luminosity.
            _sample(source, with_truth=False)
            decorate_files(
                [source], root / "real_data", _iterations(),
                metadata={"luminosity_fb": 312.0},
            )


if __name__ == "__main__":
    unittest.main()
