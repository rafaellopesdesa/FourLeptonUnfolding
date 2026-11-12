from __future__ import annotations

import importlib.util
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from BackgroundRemoval.common import FEATURES, FIT_SPLIT, MODEL_FEATURES, VALIDATION_SPLIT
from BackgroundRemoval.Training import train_background_ratio as training


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _cache(target_entries=80, reference_entries=80):
    result = {}
    for name, count, offset in (
        ("signal", target_entries, 0.3),
        ("background", reference_entries, -0.3),
    ):
        raw = np.full((count + 2, len(FEATURES)), offset, dtype=np.float32)
        result[name] = training.CachedClass(
            features=raw,
            weights=np.ones(count + 2, dtype=np.float64),
            splits=np.array([FIT_SPLIT] * count + [VALIDATION_SPLIT] * 2),
        )
    return result


def _architecture(**updates):
    result = {
        "loss_logit_clip": 15.0,
        "learning_rate": 0.01,
        "learning_rate_decay": 0.8,
        "learning_rate_decay_steps": 4,
        "weight_decay": 1.0e-4,
        "validation_interval_steps": 5,
        "validation_min_delta": 1.0e-5,
        "baseline_tolerance": 0.01,
    }
    result.update(updates)
    return result


@unittest.skipUnless(TORCH_AVAILABLE, "training-control tests require PyTorch")
class TrainingControlTest(unittest.TestCase):
    def _run(
        self,
        *,
        loss_at_step,
        architecture=None,
        cache=None,
        maximum_epochs=1,
        steps_per_epoch=20,
        patience=20,
        batch_size=4,
    ):
        """Use real gradient updates with a deterministic held-out-loss oracle."""
        import torch

        observations = []
        update_lrs = []
        optimizers = []
        completed = 0
        actual_optimizer = torch.optim.NAdam
        actual_evaluator = training._evaluate_member

        class TrackingNAdam(actual_optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                optimizers.append(self)

            def step(self, *args, **kwargs):
                nonlocal completed
                update_lrs.append(float(self.param_groups[0]["lr"]))
                output = super().step(*args, **kwargs)
                completed += 1
                return output

        def build_model(_architecture):
            model = torch.nn.Linear(len(MODEL_FEATURES), 1)
            with torch.no_grad():
                model.weight.fill_(0.01)
                model.bias.zero_()
            return model

        def evaluate(model, *_args, **_kwargs):
            if loss_at_step is None:
                return actual_evaluator(model, *_args, **_kwargs)
            model.eval()
            state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            # The final evaluation must measure the restored checkpoint,
            # which can precede the last optimizer update.
            prior = next(
                (entry for entry in observations if all(torch.equal(state[key], entry["state"][key]) for key in state)),
                None,
            )
            loss = prior["loss"] if prior is not None else loss_at_step(completed)
            observations.append({"step": completed, "state": state, "loss": loss})
            return loss, 0.5, 0.1, 0.0

        with (
            patch.object(training, "build_toolkit_model", side_effect=build_model),
            patch.object(training, "_evaluate_member", side_effect=evaluate),
            patch.object(torch.optim, "NAdam", TrackingNAdam),
        ):
            result = training._train_member(
                0,
                0,
                _cache() if cache is None else cache,
                architecture=_architecture() if architecture is None else architecture,
                mean=np.zeros(len(MODEL_FEATURES)),
                scale=np.ones(len(MODEL_FEATURES)),
                device=torch.device("cpu"),
                batch_size=batch_size,
                maximum_epochs=maximum_epochs,
                patience=patience,
                steps_per_epoch=steps_per_epoch,
                inference_batch_size=256,
                base_seed=123,
            )
        self.last_optimizers = optimizers
        return result, observations, update_lrs

    def test_imbalanced_epoch_checks_initial_first_update_and_restores_earlier_best(self):
        import torch

        result, evaluations, lrs = self._run(
            cache=_cache(target_entries=10, reference_entries=400),
            architecture=_architecture(validation_interval_steps=25),
            steps_per_epoch=None,
            patience=2,
            loss_at_step=lambda step: {0: 0.69, 1: 0.66, 5: 0.67, 10: 0.68}[step],
        )
        # Ten target rows at two draws per update define a five-step cadence;
        # 400 reference rows define only the budget, so early overfit is visible.
        self.assertEqual([entry["step"] for entry in evaluations], [0, 1, 5, 10, 10])
        self.assertEqual(result.history["optimizer_step"], [0, 1, 5, 10])
        self.assertEqual(len(lrs), 10)
        self.assertEqual(result.training_summary["planned_optimizer_steps"], 200)
        self.assertEqual(result.training_summary["effective_validation_interval_steps"], 5)
        self.assertEqual(result.training_summary["selected_optimizer_step"], 1)
        self.assertAlmostEqual(result.validation_loss, 0.66)
        for key, tensor in result.state_dict.items():
            self.assertTrue(torch.equal(tensor, evaluations[1]["state"][key]))
        self.assertIsNone(result.history["training_loss"][0])
        self.assertIsNone(result.history["objective_loss"][0])
        # No NaN placeholder is allowed in persisted training histories.
        json.dumps(result.history, allow_nan=False)

    def test_weight_decay_penalizes_only_matrix_weights_and_stays_out_of_validation(self):
        cache = _cache()
        for arrays in cache.values():
            arrays.features.fill(0.0)
        result, _, _ = self._run(
            loss_at_step=None,
            architecture=_architecture(weight_decay=0.2),
            cache=cache,
            steps_per_epoch=1,
        )
        parameter_groups = self.last_optimizers[0].param_groups
        decays = [(parameter.ndim, group["weight_decay"]) for group in parameter_groups for parameter in group["params"]]
        self.assertIn((2, 0.2), decays)
        self.assertIn((1, 0.0), decays)
        self.assertTrue(all(decay == 0.0 for ndim, decay in decays if ndim < 2))
        initial_penalty = 0.5 * 0.2 * len(MODEL_FEATURES) * 0.01 ** 2
        # The three cosine inputs are one, so initial logits are exactly .03.
        expected_bce = 0.5 * (np.logaddexp(0.0, -0.03) + np.logaddexp(0.0, 0.03))
        self.assertAlmostEqual(result.history["validation_loss"][0], expected_bce, places=6)
        self.assertAlmostEqual(result.history["regularization_loss"][0], initial_penalty, places=9)
        self.assertAlmostEqual(result.history["objective_loss"][1] - result.history["training_loss"][1], initial_penalty, places=9)

    def test_initial_checkpoint_is_kept_if_the_first_update_overfits(self):
        import torch

        result, evaluations, lrs = self._run(
            loss_at_step=lambda step: 0.68 if step == 0 else 0.70,
            patience=1,
        )
        self.assertEqual([entry["step"] for entry in evaluations], [0, 1, 1])
        self.assertEqual(len(lrs), 1)
        self.assertTrue(result.training_summary["initial_checkpoint_selected"])
        self.assertEqual(result.training_summary["selected_optimizer_step"], 0)
        for key, tensor in result.state_dict.items():
            self.assertTrue(torch.equal(tensor, evaluations[0]["state"][key]))

    def test_patience_counts_validation_checks_and_respects_minimum_improvement(self):
        result, evaluations, lrs = self._run(
            architecture=_architecture(validation_interval_steps=3, validation_min_delta=0.001),
            loss_at_step=lambda step: 0.69 if step == 0 else 0.67 - step * 0.00001,
            patience=2,
        )
        self.assertEqual([entry["step"] for entry in evaluations], [0, 1, 3, 6, 6])
        self.assertEqual(len(lrs), 6)
        self.assertEqual(result.training_summary["selected_optimizer_step"], 6)

    def test_learning_rate_and_updates_do_not_depend_on_epoch_partition_or_validation_cadence(self):
        import torch

        runs = []
        for epochs, steps, interval in ((1, 6, 1), (3, 2, 1), (1, 6, 4)):
            runs.append(self._run(
                architecture=_architecture(validation_interval_steps=interval),
                maximum_epochs=epochs,
                steps_per_epoch=steps,
                loss_at_step=lambda step: 0.69 - step * 0.001,
            ))
        expected_update_lrs = 0.01 * 0.8 ** (np.arange(6) / 4.0)
        for result, _, lrs in runs:
            np.testing.assert_allclose(lrs, expected_update_lrs, rtol=1.0e-12)
            self.assertEqual(result.training_summary["selected_optimizer_step"], 6)
            for key, tensor in result.state_dict.items():
                self.assertTrue(torch.equal(tensor, runs[0][0].state_dict[key]))
        self.assertEqual(runs[2][0].history["optimizer_step"], [0, 1, 4, 6])

    def test_baseline_gate_rejects_an_entire_consistently_bad_ensemble(self):
        results = [self._member(slot, math.log(2.0) + 0.02) for slot in range(4)]
        self.assertEqual(training._bad_members(results), [0, 1, 2, 3])

    def test_near_null_fit_is_allowed_and_flagged_without_inventing_improvement(self):
        baseline = math.log(2.0)
        result, _, _ = self._run(loss_at_step=lambda _step: baseline + 0.005, patience=1)
        self.assertEqual(training._bad_members([result]), [])
        self.assertFalse(result.training_summary["improved_over_constant"])
        self.assertTrue(result.training_summary["within_baseline_tolerance"])

    def test_raw_ensemble_gate_rejects_consistent_bad_scores(self):
        cache = _cache()
        validation = {
            name: training.ClassPredictions(
                indices=np.flatnonzero(arrays.splits == VALIDATION_SPLIT),
                member_scores=np.full((2, 4), 0.9),
            )
            for name, arrays in cache.items()
        }
        with self.assertRaisesRegex(RuntimeError, "baseline|constant"):
            training._ensemble_validation_summary(
                validation,
                cache,
                logit_clip=15.0,
                baseline_tolerance=0.01,
                validation_min_delta=1.0e-5,
            )

    def test_raw_ensemble_gate_uses_signed_weights_and_accepts_near_null_scores(self):
        cache = _cache()
        target_scores = np.array([[0.45, 0.55, 0.52, 0.4808], [0.45, 0.55, 0.52, 0.4816]])
        validation = {}
        for name, arrays in cache.items():
            indices = np.flatnonzero(arrays.splits == VALIDATION_SPLIT)
            arrays.weights[indices] = [2.0, -1.0]
            validation[name] = training.ClassPredictions(
                indices=indices,
                member_scores=target_scores if name == "signal" else 1.0 - target_scores,
            )
        with self.assertWarns(RuntimeWarning):
            summary = training._ensemble_validation_summary(
                validation,
                cache,
                logit_clip=15.0,
                baseline_tolerance=0.01,
                validation_min_delta=1.0e-5,
            )
        means = target_scores.mean(axis=1)
        expected_signed_bce = -2.0 * np.log(means[0]) + np.log(means[1])
        self.assertAlmostEqual(summary["validation_loss"], expected_signed_bce, places=12)
        self.assertFalse(summary["improved_over_constant"])
        self.assertTrue(summary["within_baseline_tolerance"])

    def test_signed_ensemble_gate_can_fail_even_when_each_member_passes(self):
        # Signed BCE is not convex under probability averaging. Here q=2*p^2
        # makes every member's signed class loss exactly log(2), while averaging
        # the two different classifiers makes that signed loss much worse.
        cache = _cache()
        scores = np.array([[0.05, 0.7, 0.05, 0.7], [0.005, 0.98, 0.005, 0.98]])
        member_losses = -2.0 * np.log(scores[0]) + np.log(scores[1])
        np.testing.assert_allclose(member_losses, math.log(2.0))
        members = [self._member(slot, loss) for slot, loss in enumerate(member_losses)]
        self.assertEqual(training._bad_members(members), [])
        validation = {}
        for name, arrays in cache.items():
            indices = np.flatnonzero(arrays.splits == VALIDATION_SPLIT)
            arrays.weights[indices] = [2.0, -1.0]
            validation[name] = training.ClassPredictions(
                indices=indices,
                member_scores=scores if name == "signal" else 1.0 - scores,
            )
        with self.assertRaisesRegex(RuntimeError, "baseline|constant"):
            training._ensemble_validation_summary(
                validation,
                cache,
                logit_clip=15.0,
                baseline_tolerance=0.01,
                validation_min_delta=1.0e-5,
            )

    @staticmethod
    def _member(slot, loss):
        return training.MemberResult(
            slot=slot,
            attempt=0,
            seed=slot,
            state_dict={},
            history={},
            validation_loss=loss,
            validation_score_mean=0.5,
            validation_score_std=0.1,
            validation_saturated_fraction=0.0,
        )


if __name__ == "__main__":
    unittest.main()
