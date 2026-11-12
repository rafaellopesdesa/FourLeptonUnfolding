"""Signed raw-odds semantics, diagnostics and artifact round trips."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from omnifold import ratios


def _sample(seed=19, size=96):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(size, len(ratios.FEATURE_ORDER))).astype(np.float32)
    features[:, -2:] += (80.0, 30.0)
    splits = (np.arange(size) % 3).astype(np.uint8)
    weights = rng.uniform(0.8, 1.2, size)
    weights[:3] = -0.2
    return features, weights, splits


class RatioContractTests(unittest.TestCase):
    def test_negative_events_retained_and_each_split_requires_positive_net_measure(self):
        features, weights, splits = _sample()
        cached, stats = ratios._checked_class(features, weights, splits, "target")
        np.testing.assert_array_equal(cached.weights, weights)
        self.assertEqual(stats.negative_weight_entries, 3)
        self.assertEqual(stats.split_negative_weight_entries, {"fit": 1, "validation": 1, "closure": 1})
        invalid = weights.copy()
        invalid[splits == 1] = -1.0
        with self.assertRaisesRegex(ValueError, "validation split has non-positive"):
            ratios._checked_class(features, invalid, splits, "target")

    def test_invalid_features_splits_and_nonfinite_weights_are_rejected(self):
        features, weights, splits = _sample()
        with self.assertRaisesRegex(ValueError, "shape"):
            ratios._checked_class(features[:, :-1], weights, splits, "target")
        bad_splits = splits.copy()
        bad_splits[0] = 3
        with self.assertRaisesRegex(ValueError, "splits must contain"):
            ratios._checked_class(features, weights, bad_splits, "target")
        weights[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            ratios._checked_class(features, weights, splits, "target")

    def test_scaler_uses_only_fit_bank_and_accepts_signed_weights(self):
        features, weights, splits = _sample()
        bank, _ = ratios._checked_class(features, weights, splits, "target")
        mean, scale = ratios._scaler({"signal": bank, "background": bank})
        changed = features.copy()
        changed[splits != 0] *= 100.0
        other_bank, _ = ratios._checked_class(changed, weights, splits, "target")
        other_mean, other_scale = ratios._scaler({"signal": other_bank, "background": other_bank})
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(scale, other_scale)
        self.assertTrue(np.all(scale > 0))

    def test_full_ratio_is_exact_yield_times_shape_even_for_large_yield(self):
        predictions = ratios._outputs(np.array([[0.5] * 4, [0.8] * 4]), 2.0, 1.0e8)
        np.testing.assert_allclose(predictions["shape_ratio"], [1.0, 4.0])
        np.testing.assert_allclose(predictions["full_ratio"], 1.0e8 * predictions["shape_ratio"])
        self.assertEqual(set(predictions), {"shape_ratio", "full_ratio", "score_raw", "ensemble_std"})

    def test_raw_odds_use_mean_probability_without_rescaling_or_batch_dependence(self):
        scores = np.array([[.1, .2, .3, .4], [.7, .8, .9, .8]])
        outputs = ratios._outputs(scores, 30.0, 2.0)
        raw = scores.mean(axis=1)
        np.testing.assert_allclose(outputs["score_raw"], raw)
        np.testing.assert_allclose(outputs["shape_ratio"], raw / (1.0 - raw))
        self.assertNotAlmostEqual(outputs["shape_ratio"].mean(), 1.0)
        for index in range(len(scores)):
            alone = ratios._outputs(scores[index:index + 1], 30.0, 2.0)
            self.assertEqual(outputs["shape_ratio"][index], alone["shape_ratio"][0])
        self.assertNotAlmostEqual(outputs["shape_ratio"][0], np.mean(scores[0] / (1.0 - scores[0])))

    def test_configuration_uses_compact_regularized_training_defaults(self):
        config = ratios.RatioConfig()
        config.validate()
        self.assertEqual((config.ensemble_size, config.hidden_layers, config.neurons, config.batch_size), (4, 2, 64, 512))
        self.assertEqual((config.weight_decay, config.validation_interval_steps,
                          config.learning_rate_decay_steps, config.validation_min_delta,
                          config.baseline_tolerance), (1e-4, 25, 100, 1e-5, .01))
        replace(config, hidden_layers=3, neurons=128).validate()
        for replacement in ({"ensemble_size": 3}, {"batch_size": 7}, {"logit_clip": np.inf},
                            {"learning_rate": 0.0}, {"steps_per_epoch": 0}, {"max_retries": -1},
                            {"weight_decay": -1e-4}, {"weight_decay": np.inf},
                            {"validation_interval_steps": 0}, {"validation_interval_steps": 2.5},
                            {"learning_rate_decay_steps": 0}, {"validation_min_delta": -1e-5},
                            {"baseline_tolerance": np.nan}, {"baseline_tolerance": True}):
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    replace(config, **replacement).validate()


@unittest.skipUnless(importlib.util.find_spec("torch"), "signed ratio smoke test requires PyTorch")
class RatioTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.features, cls.weights, cls.splits = _sample()
        cls.config = ratios.RatioConfig(hidden_layers=1, neurons=8, batch_size=16,
                                       maximum_epochs=2, patience=2, steps_per_epoch=2,
                                       inference_batch_size=32, max_retries=1, device="cpu")

        def tiny_model(architecture):
            return torch.nn.Sequential(
                torch.nn.Linear(len(ratios.MODEL_FEATURE_ORDER), architecture["neurons"]),
                torch.nn.SiLU(), torch.nn.Linear(architecture["neurons"], 1),
            )

        cls.patches = [patch.object(ratios.common, "build_toolkit_model", side_effect=tiny_model),
                       patch.object(ratios.training, "build_toolkit_model", side_effect=tiny_model)]
        for item in cls.patches:
            item.start()
        try:
            cls.model = ratios.RatioEstimator.fit(
                cls.features, 3.0 * cls.weights, cls.features, cls.weights,
                target_splits=cls.splits, reference_splits=cls.splits,
                output_dir=cls.root / "ratio", config=cls.config,
                metadata={"stage": "truth", "iteration": 1},
            )
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        import torch

        for item in reversed(cls.patches):
            item.stop()
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)

    def test_real_signed_training_raw_odds_diagnostics_and_round_trip(self):
        prediction = self.model.predict(self.features)
        validation = self.splits == 1
        normalization = np.sum(self.weights[validation] * prediction["shape_ratio"][validation]) / np.sum(self.weights[validation])
        raw_odds = prediction["score_raw"] / (1.0 - prediction["score_raw"])
        np.testing.assert_allclose(prediction["shape_ratio"], raw_odds, rtol=1e-12)
        self.assertNotAlmostEqual(normalization, 1.0, places=5)
        self.assertAlmostEqual(self.model.manifest["validation_metrics"]["reference_shape_mean"], normalization)
        self.assertNotIn("calibration", self.model.manifest)
        self.assertEqual(self.model.manifest["ratio_estimator"], ratios.common.RAW_RATIO_ESTIMATOR)
        self.assertAlmostEqual(self.model.yield_ratio, 3.0)
        self.assertEqual(self.model.manifest["classes"]["target"]["negative_weight_entries"], 3)
        self.assertEqual(self.model.manifest["metadata"], {"stage": "truth", "iteration": 1})
        self.assertEqual(len(self.model.history), 4)
        self.assertEqual(self.model.manifest["training_policy_version"], ratios.TRAINING_POLICY_VERSION)
        self.assertTrue(self.model.manifest["ensemble_validation"]["within_baseline_tolerance"])
        for member in self.model.history:
            summary = member["training_summary"]
            self.assertIn(summary["selected_optimizer_step"], member["history"]["optimizer_step"])
            self.assertEqual(summary["weight_decay"], self.config.weight_decay)
            self.assertEqual(summary["learning_rate_decay_steps"], self.config.learning_rate_decay_steps)
            self.assertEqual(member["history"]["optimizer_step"][:2], [0, 1])
        np.testing.assert_allclose(prediction["full_ratio"], 3.0 * prediction["shape_ratio"])
        restored = ratios.RatioEstimator.load(self.model.root, device="cpu")
        other = restored.predict(self.features, batch_size=7)
        for key in prediction:
            np.testing.assert_allclose(prediction[key], other[key], rtol=1e-6, atol=1e-8)
        empty = restored.predict(self.features[:0])
        self.assertTrue(all(value.shape == (0,) for value in empty.values()))

    def test_closure_features_do_not_change_fitted_network_or_raw_odds(self):
        changed = self.features.copy()
        changed[self.splits == 2, 6:] += 400.0
        second = ratios.RatioEstimator.fit(
            changed, 3.0 * self.weights, changed, self.weights,
            target_splits=self.splits, reference_splits=self.splits,
            output_dir=self.root / "altered-closure", config=self.config,
        )
        self.assertEqual(self.model.manifest["validation_metrics"], second.manifest["validation_metrics"])
        self.assertEqual([entry["history"] for entry in self.model.history],
                         [entry["history"] for entry in second.history])
        np.testing.assert_array_equal(self.model.predict(self.features)["shape_ratio"],
                                      second.predict(self.features)["shape_ratio"])

    def test_previous_raw_artifact_keeps_serialized_topology_and_inference(self):
        destination = self.root / "previous-raw-policy"
        shutil.copytree(self.model.root, destination)
        path = destination / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("training_policy_version")
        manifest.pop("ensemble_validation")
        for name in ("weight_decay", "validation_interval_steps", "learning_rate_decay_steps",
                     "validation_min_delta", "baseline_tolerance"):
            manifest["config"].pop(name)
        for member in manifest["members"]:
            member.pop("training_summary")
        path.write_text(json.dumps(manifest))
        restored = ratios.RatioEstimator.load(destination, device="cpu")
        self.assertEqual(restored.manifest["architecture"]["hidden_layers"], self.config.hidden_layers)
        self.assertEqual(restored.manifest["architecture"]["neurons"], self.config.neurons)
        np.testing.assert_array_equal(restored.predict(self.features)["shape_ratio"],
                                      self.model.predict(self.features)["shape_ratio"])

    def test_new_policy_training_controls_must_match_serialized_architecture(self):
        destination = self.root / "inconsistent-policy"
        shutil.copytree(self.model.root, destination)
        path = destination / "manifest.json"
        original = json.loads(path.read_text())
        for name in ("weight_decay", "validation_interval_steps", "learning_rate_decay_steps",
                     "validation_min_delta", "baseline_tolerance"):
            with self.subTest(name=name):
                changed = json.loads(json.dumps(original))
                changed["architecture"][name] *= 2
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "training policy mismatch"):
                    ratios.RatioEstimator.load(destination, device="cpu")
                changed = json.loads(json.dumps(original))
                changed["config"].pop(name)
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "training policy mismatch"):
                    ratios.RatioEstimator.load(destination, device="cpu")

    def test_same_events_cannot_leak_between_target_and_reference_banks(self):
        changed_splits = (self.splits + 1) % 3
        with self.assertRaisesRegex(ValueError, "identical target/reference splits"):
            ratios.RatioEstimator.fit(
                self.features, self.weights, self.features, self.weights,
                target_splits=self.splits, reference_splits=changed_splits,
                output_dir=self.root / "bad-splits", config=self.config,
            )

    def test_default_prediction_batches_use_saved_config_and_models_finish_on_cpu(self):
        batch_sizes = []
        hook = self.model.models[0].register_forward_pre_hook(
            lambda _model, inputs: batch_sizes.append(len(inputs[0]))
        )
        try:
            self.model.predict(self.features)
        finally:
            hook.remove()
        self.assertEqual(batch_sizes, [self.config.inference_batch_size] * 3)
        self.assertTrue(all(parameter.device.type == "cpu"
                            for model in self.model.models for parameter in model.parameters()))
        bad = self.features.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.model.predict(bad)
        self.assertTrue(all(parameter.device.type == "cpu"
                            for model in self.model.models for parameter in model.parameters()))

    def test_loading_detects_modified_weights_and_invalid_feature_contract(self):
        corrupt = self.root / "corrupt"
        shutil.copytree(self.model.root, corrupt)
        with (corrupt / "member_000.pt").open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            ratios.RatioEstimator.load(corrupt, device="cpu")
        manifest = json.loads((corrupt / "manifest.json").read_text())
        manifest["feature_order"] = list(reversed(manifest["feature_order"]))
        (corrupt / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "feature order"):
            ratios.RatioEstimator.load(corrupt, device="cpu")

    def test_all_predictions_use_seven_inputs_and_are_even_in_phi_and_psi(self):
        self.assertEqual(ratios.MODEL_FEATURE_ORDER,
                         ("cos(Phi)", "cos(Psi)", "cos_theta1", "cos_theta2",
                          "cos_theta_star", "m_Z1", "m_Z2"))
        self.assertTrue(all(model[0].in_features == 7 for model in self.model.models))
        expected = self.model.predict(self.features)
        for indices in ([0], [1], [0, 1]):
            with self.subTest(flipped=indices):
                reflected = self.features.copy()
                reflected[:, indices] *= -1
                prediction = self.model.predict(reflected)
                for name in expected:
                    np.testing.assert_array_equal(prediction[name], expected[name])

    def test_legacy_calibrated_artifacts_and_injected_corrections_are_rejected(self):
        destination = self.root / "legacy-or-tampered"
        shutil.copytree(self.model.root, destination)
        manifest_path = destination / "manifest.json"
        original = json.loads(manifest_path.read_text())
        cases = [
            ({**original, "format_version": 1, "calibration": {"scale": .5, "bias": .2}}, "Legacy calibrated"),
            ({**original, "format_version": 2}, "seven cosine-only"),
            ({**original, "model_features": ["sin(Phi)", "cos(Phi)", "sin(Phi1)",
                "cos(Phi1)", "sin(Psi)", "cos(Psi)", "cos_theta1", "cos_theta2",
                "cos_theta_star", "m_Z1", "m_Z2"]}, "feature order"),
            ({**original, "calibration": {"scale": 1., "bias": 0.}}, "without calibration"),
            ({**original, "ratio_estimator": {**original["ratio_estimator"], "empirical_normalization": True}}, "without calibration"),
        ]
        for manifest, message in cases:
            with self.subTest(manifest=manifest["ratio_estimator"]):
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    ratios.RatioEstimator.load(destination, device="cpu")

    def test_overwrite_requires_recognized_artifact_and_failed_publication_restores_old(self):
        with self.assertRaises(FileExistsError):
            ratios.RatioEstimator.fit(
                self.features, self.weights, self.features, self.weights,
                target_splits=self.splits, reference_splits=self.splits,
                output_dir=self.model.root, config=self.config,
            )
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("keep")
        with self.assertRaises(FileNotFoundError):
            ratios.RatioEstimator.fit(
                self.features, self.weights, self.features, self.weights,
                target_splits=self.splits, reference_splits=self.splits,
                output_dir=unrelated, config=replace(self.config, overwrite=True),
            )
        self.assertEqual((unrelated / "keep.txt").read_text(), "keep")
        staging = self.root / "staging"
        staging.mkdir()
        before = (self.model.root / "manifest.json").read_bytes()
        actual_replace = ratios.os.replace

        def fail_final(source, destination):
            if Path(source) == staging:
                raise OSError("publication failure")
            return actual_replace(source, destination)

        with patch.object(ratios.os, "replace", side_effect=fail_final):
            with self.assertRaisesRegex(OSError, "publication failure"):
                ratios._publish(staging, self.model.root, overwrite=True)
        self.assertEqual((self.model.root / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
