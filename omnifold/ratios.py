"""Calibrated target/reference density ratios for either OmniFold level.

The neural training and signed-weight calibration are shared with the existing
background-removal workflow.  Artifacts here have a generic ratio contract:
``full_ratio = yield_ratio * shape_ratio``.  No purity factor is introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

import numpy as np

from BackgroundRemoval import common
from BackgroundRemoval.Training import train_background_ratio as training


FORMAT_VERSION = 1
MODEL_KIND = "omnifold_density_ratio"
FEATURE_ORDER = tuple(name.removeprefix("reco_") for name in common.FEATURES)
MODEL_FEATURE_ORDER = tuple(name.replace("reco_", "") for name in common.MODEL_FEATURES)


@dataclass(frozen=True)
class RatioConfig:
    """Defaults match the established background-removal ensemble."""

    ensemble_size: int = 4
    hidden_layers: int = 4
    neurons: int = 1024
    batch_size: int = 8192
    maximum_epochs: int = 100
    patience: int = 12
    learning_rate: float = 3.0e-4
    learning_rate_decay: float = 0.98
    max_retries: int = 3
    seed: int = 19110907
    logit_clip: float = 30.0
    calibration_scale_min: float = 0.1
    calibration_scale_max: float = 5.0
    inference_batch_size: int = 65536
    steps_per_epoch: int | None = None
    device: str = "auto"
    overwrite: bool = False

    def validate(self) -> None:
        for name in ("ensemble_size", "hidden_layers", "neurons", "batch_size",
                     "maximum_epochs", "patience", "inference_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ensemble_size < 4:
            raise ValueError("ensemble_size must be at least four")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch_size must be even and at least two")
        for name in ("max_retries", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.steps_per_epoch is not None and (
            isinstance(self.steps_per_epoch, bool)
            or not isinstance(self.steps_per_epoch, (int, np.integer))
            or self.steps_per_epoch < 1
        ):
            raise ValueError("steps_per_epoch must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.learning_rate_decay) or not 0.0 < self.learning_rate_decay <= 1.0:
            raise ValueError("learning_rate_decay must lie in (0, 1]")
        if not np.isfinite(self.logit_clip) or not 0.0 < self.logit_clip <= 100.0:
            raise ValueError("logit_clip must lie in (0, 100]")
        if not (
            np.isfinite(self.calibration_scale_min)
            and np.isfinite(self.calibration_scale_max)
            and 0.0 < self.calibration_scale_min <= 1.0 <= self.calibration_scale_max
        ):
            raise ValueError("calibration scale bounds must be finite, positive and contain one")


def _device(name: str):
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def _checked_class(features, weights, splits, name: str):
    features = np.asarray(features)
    weights = np.asarray(weights, dtype=np.float64)
    splits = np.asarray(splits)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
        raise ValueError(f"{name} features must have shape (events, {len(FEATURE_ORDER)})")
    if weights.shape != (features.shape[0],) or splits.shape != weights.shape:
        raise ValueError(f"{name} weights and splits must match the feature row count")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} features and weights must be finite")
    if not np.all(np.isin(splits, tuple(common.SPLIT_NAMES))):
        raise ValueError(f"{name} splits must contain only 0 (fit), 1 (validation), 2 (closure)")
    nonzero = weights != 0.0
    # Most production inputs have no zero rows.  Keep their arrays as views;
    # in the truth push both classes can share the same large feature bank.
    all_nonzero = bool(np.all(nonzero))
    selected_features = features if all_nonzero else features[nonzero]
    selected_weights = weights if all_nonzero else weights[nonzero]
    selected_splits = splits if all_nonzero else splits[nonzero]
    statistics = training.WeightedStatistics()
    statistics.add(selected_weights, selected_splits, zero_count=int(np.count_nonzero(~nonzero)))
    if not np.isfinite(statistics.sum_weights) or statistics.sum_weights <= 0.0:
        raise ValueError(f"{name} has non-positive net signed yield")
    for split_name in common.SPLIT_NAMES.values():
        if not np.isfinite(statistics.split_sum_weights[split_name]) or statistics.split_sum_weights[split_name] <= 0.0:
            raise ValueError(f"{name} {split_name} split has non-positive net signed yield")
    if not np.isfinite(statistics.sum_squared_weights) or not np.isfinite(statistics.sum_absolute_weights):
        raise ValueError(f"{name} weighted statistics overflow")
    # CachedClass is the shared trainer's input contract.  It uses the words
    # signal/background internally; only target/reference are exposed here.
    cache = training.CachedClass(
        np.asarray(selected_features, dtype=np.float32),
        selected_weights, selected_splits.astype(np.uint8, copy=False),
    )
    if not np.all(np.isfinite(cache.features)):
        raise ValueError(f"{name} transformed features overflow")
    return cache, statistics


def _scaler(cache):
    moments, seconds = [], []
    for arrays in cache.values():
        first = np.zeros(len(MODEL_FEATURE_ORDER), dtype=np.float64)
        second = np.zeros_like(first)
        total = 0.0
        # Bound temporary arrays even for samples with tens of millions of
        # events.  Only fit-bank moments determine preprocessing.
        for start in range(0, len(arrays.weights), 65536):
            stop = min(start + 65536, len(arrays.weights))
            fit = arrays.splits[start:stop] == common.FIT_SPLIT
            values = common.transformed_features(arrays.features[start:stop][fit]).astype(np.float64)
            weights = np.abs(arrays.weights[start:stop][fit])
            total += float(np.sum(weights, dtype=np.float64))
            first += np.sum(values * weights[:, None], axis=0)
            second += np.sum(values * values * weights[:, None], axis=0)
        moments.append(first / total)
        seconds.append(second / total)
    mean = 0.5 * (moments[0] + moments[1])
    variance = np.maximum(0.5 * (seconds[0] + seconds[1]) - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-7, scale, 1.0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("feature scaler is not finite")
    return mean, scale


def _outputs(member_scores, calibration, yield_ratio):
    scores = np.asarray(member_scores, dtype=np.float64)
    raw = np.mean(scores, axis=1, dtype=np.float64)
    bound = float(calibration["logit_clip"])
    log_ratio = np.clip(
        float(calibration["scale"]) * common.stable_logit(raw, logit_clip=bound)
        + float(calibration["bias"]), -bound, bound,
    )
    shape = np.exp(log_ratio)
    # Do not clip the physical ratio: the event yield is a separate factor.
    full = float(yield_ratio) * shape
    if not np.all(np.isfinite(full)):
        raise RuntimeError("full density ratio is not finite")
    return {
        "shape_ratio": shape,
        "full_ratio": full,
        "score_raw": raw,
        "score_calibrated": common.stable_sigmoid(log_ratio),
        "ensemble_std": np.std(scores, axis=1),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("model_kind") != MODEL_KIND or result.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"not a supported OmniFold ratio artifact: {path}")
    return result


def _publish(staging: Path, destination: Path, overwrite: bool):
    backup = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"ratio output exists: {destination}; enable overwrite to replace it")
        _read_manifest(destination)
        backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.previous-", dir=destination.parent))
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as error:
            warnings.warn(f"ratio saved but old artifact backup remains at {backup}: {error}", RuntimeWarning)


class RatioEstimator:
    """A persisted, calibrated ensemble with an explicit physical yield ratio."""

    def __init__(self, root, manifest, models, device):
        self.root = Path(root)
        self.manifest = manifest
        self.models = models
        self.device = device
        self.yield_ratio = float(manifest["yield_ratio"])
        self.history = manifest["members"]
        self.mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float64)
        self.scale = np.asarray(manifest["scaler"]["scale"], dtype=np.float64)

    @classmethod
    def fit(cls, target_features, target_weights, reference_features, reference_weights, *,
            target_splits, reference_splits, output_dir, config: RatioConfig, metadata=None):
        """Fit on split 0, calibrate/normalize on split 1, report split 2.

        The caller owns event identity and supplies stable split codes.  When
        target and reference are the same ordered events (the truth update),
        their split arrays must match, including zero-weight rows.
        """
        import torch

        config.validate()
        metadata = {} if metadata is None else metadata
        json.dumps(metadata, allow_nan=False)  # Fail before expensive training.
        target, target_stats = _checked_class(target_features, target_weights, target_splits, "target")
        reference, reference_stats = _checked_class(reference_features, reference_weights, reference_splits, "reference")
        if (np.shape(target_features) == np.shape(reference_features)
                and np.array_equal(target_features, reference_features)
                and not np.array_equal(target_splits, reference_splits)):
            raise ValueError("the same ordered events must use identical target/reference splits")
        cache = {"signal": target, "background": reference}
        mean, scale = _scaler(cache)
        yield_ratio = target_stats.sum_weights / reference_stats.sum_weights
        if not np.isfinite(yield_ratio) or yield_ratio <= 0.0:
            raise ValueError("target/reference yield ratio must be finite and positive")
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists():
            if not config.overwrite:
                raise FileExistsError(f"ratio output exists: {destination}; enable overwrite to replace it")
            _read_manifest(destination)
        device = _device(config.device)
        effective_steps = config.steps_per_epoch or math.ceil(
            max(np.count_nonzero(target.splits == common.FIT_SPLIT),
                np.count_nonzero(reference.splits == common.FIT_SPLIT)) / (config.batch_size // 2)
        )
        architecture = {
            "implementation": "nsbi_common_utils.lightning_tools.DensityRatioLightning",
            "hidden_layers": config.hidden_layers, "neurons": config.neurons,
            "activation": "SiLU", "output": "logit",
            "loss": "balanced signed-weight logit-clipped BCEWithLogitsLoss",
            "loss_logit_clip": config.logit_clip,
            "class_risk": "sum(w * BCE) / sum(w)",
            "sampling": "with_replacement_proportional_to_absolute_weight",
            "sampling_coefficient": "sign(w) * sum(abs(w)) / sum(w)",
            "dropout": 0.0, "weight_decay": 0.0, "optimizer": "NAdam",
            "learning_rate": config.learning_rate,
            "learning_rate_schedule": "ExponentialLR", "learning_rate_decay": config.learning_rate_decay,
            "batch_size": config.batch_size, "steps_per_epoch": effective_steps,
            "maximum_epochs": config.maximum_epochs, "early_stopping_patience": config.patience,
        }
        shared = dict(architecture=architecture, mean=mean, scale=scale, device=device,
                      inference_batch_size=config.inference_batch_size)
        train_kwargs = dict(shared, batch_size=config.batch_size, maximum_epochs=config.maximum_epochs,
                            patience=config.patience, steps_per_epoch=config.steps_per_epoch, base_seed=config.seed)
        runtime = common.toolkit_runtime_provenance()
        results = [training._train_member(slot, 0, cache, **train_kwargs) for slot in range(config.ensemble_size)]
        for retry_round in range(1, config.max_retries + 1):
            bad = training._bad_members(results)
            if not bad:
                break
            for slot in bad:
                results[slot] = training._train_member(slot, retry_round, cache, **train_kwargs)
        remaining = training._bad_members(results)
        if remaining:
            raise RuntimeError(f"ensemble slots {remaining} remain discrepant after {config.max_retries} retries")
        validation = training._predict_split(cache, results, split_code=common.VALIDATION_SPLIT, **shared)
        scores, labels, weights = training._balanced_calibration_arrays(validation, cache)
        calibration_scale, original_bias = training.fit_logit_calibrator(
            scores, labels, weights, logit_clip=config.logit_clip,
            minimum_scale=config.calibration_scale_min, maximum_scale=config.calibration_scale_max,
        )
        reference_validation = validation["background"]
        reference_weights_validation = reference.weights[reference_validation.indices]
        bias, before = training._normalization_bias(
            reference_validation.ensemble_score, reference_weights_validation,
            scale=calibration_scale, initial_bias=original_bias, logit_clip=config.logit_clip,
        )
        calibration = {
            "scale": calibration_scale, "bias": bias,
            "bias_before_normalization": original_bias, "logit_clip": config.logit_clip,
            "reference_shape_mean_before_normalization": before,
            "reference_shape_mean_after_normalization": 1.0, "split": "validation",
        }
        normalized = _outputs(reference_validation.member_scores, calibration, yield_ratio)["shape_ratio"]
        checked_mean = float(np.sum(reference_weights_validation * normalized, dtype=np.float64)
                             / np.sum(reference_weights_validation, dtype=np.float64))
        if not np.isclose(checked_mean, 1.0, rtol=1.0e-9, atol=1.0e-11):
            raise RuntimeError("serialized inference rule does not preserve validation ratio normalization")
        calibration["reference_shape_mean_after_normalization"] = checked_mean
        closure = training._predict_split(cache, results, split_code=common.CLOSURE_SPLIT, **shared)
        closure_reference_weights = reference.weights[closure["background"].indices]
        closure_outputs = _outputs(closure["background"].member_scores, calibration, yield_ratio)
        closure_metrics = {
            "reference_shape_mean": float(np.sum(closure_reference_weights * closure_outputs["shape_ratio"])
                                          / np.sum(closure_reference_weights)),
            "reference_full_ratio_weighted_sum": float(np.sum(closure_reference_weights * closure_outputs["full_ratio"])),
            "target_sum_weights": float(np.sum(target.weights[closure["signal"].indices])),
            "uses_full_sample_yield_ratio": True,
        }
        # Training and calibration finish before opening the publication stage;
        # a failed run cannot modify an existing artifact.
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.training-", dir=destination.parent))
        try:
            members = []
            for result in results:
                filename = f"member_{result.slot:03d}.pt"
                torch.save(result.state_dict, staging / filename)
                members.append({
                    "slot": result.slot, "attempt": result.attempt, "seed": result.seed,
                    "file": filename, "sha256": common.sha256_file(staging / filename),
                    "validation_loss": result.validation_loss,
                    "validation_score_mean": result.validation_score_mean,
                    "validation_score_std": result.validation_score_std,
                    "validation_saturated_fraction": result.validation_saturated_fraction,
                    "history": result.history,
                })
            manifest = {
                "format_version": FORMAT_VERSION, "model_kind": MODEL_KIND,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "feature_order": list(FEATURE_ORDER), "model_features": list(MODEL_FEATURE_ORDER),
                "architecture": architecture, "config": asdict(config), "toolkit": runtime,
                "classes": {"target": target_stats.serializable(), "reference": reference_stats.serializable()},
                "yield_ratio": yield_ratio, "yield_ratio_definition": "sum(target weights) / sum(reference weights)",
                "ratio_definition": "full_ratio = yield_ratio * shape_ratio",
                "ensemble_aggregation": "arithmetic_mean_of_member_probabilities",
                "scaler": {"mean": mean.tolist(), "scale": scale.tolist(), "weight_measure": "class-balanced absolute fit weights"},
                "calibration": calibration, "closure_metrics": closure_metrics,
                "split_usage": {"fit": "network optimization and feature scaling",
                                "validation": "early stopping, retries, calibration and shape normalization",
                                "closure": "diagnostics only; full-sample scalar yield ratio is shared"},
                "members": members, "metadata": metadata,
            }
            with (staging / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            # Validate the serialized artifact before replacing its destination.
            estimator = cls.load(staging, device=config.device)
            _publish(staging, destination, config.overwrite)
            estimator.root = destination
            return estimator
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @classmethod
    def load(cls, path, device="auto"):
        import torch

        root = Path(path).expanduser().resolve()
        manifest = _read_manifest(root)
        if manifest.get("feature_order") != list(FEATURE_ORDER) or manifest.get("model_features") != list(MODEL_FEATURE_ORDER):
            raise ValueError("ratio artifact feature order is incompatible")
        config = RatioConfig(**manifest["config"])
        config.validate()
        members = manifest.get("members", [])
        if len(members) != config.ensemble_size:
            raise ValueError("ratio artifact ensemble size is inconsistent")
        mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float64)
        scale = np.asarray(manifest["scaler"]["scale"], dtype=np.float64)
        if (mean.shape != (len(MODEL_FEATURE_ORDER),) or scale.shape != mean.shape
                or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0)):
            raise ValueError("ratio artifact scaler is invalid")
        calibration = manifest["calibration"]
        if (not np.isfinite(calibration["scale"]) or calibration["scale"] <= 0.0
                or not np.isfinite(calibration["bias"]) or calibration["logit_clip"] != config.logit_clip
                or not np.isfinite(manifest["yield_ratio"]) or manifest["yield_ratio"] <= 0.0):
            raise ValueError("ratio artifact calibration/yield contract is invalid")
        architecture = manifest["architecture"]
        for field, expected in (("hidden_layers", config.hidden_layers), ("neurons", config.neurons),
                                ("loss_logit_clip", config.logit_clip), ("activation", "SiLU"), ("output", "logit")):
            if architecture.get(field) != expected:
                raise ValueError(f"ratio artifact architecture mismatch: {field}")
        actual_device = _device(device)
        models, seen = [], set()
        for slot, member in enumerate(members):
            filename = member.get("file", "")
            if (not filename or Path(filename).name != filename or filename in seen
                    or member.get("slot") != slot or (root / filename).resolve().parent != root):
                raise ValueError("ratio artifact contains an invalid member file or slot")
            seen.add(filename)
            if common.sha256_file(root / filename) != member.get("sha256"):
                raise ValueError(f"ratio member checksum mismatch: {filename}")
            model = common.build_toolkit_model(architecture)
            model.load_state_dict(torch.load(root / filename, map_location="cpu", weights_only=True))
            # Completed iterations retain CPU parameters only.  A notebook
            # may keep many iterations; their GPU memory must not accumulate.
            model.cpu()
            model.eval()
            models.append(model)
        return cls(root, manifest, models, actual_device)

    def predict(self, features, batch_size=None):
        """Predict in row order and release GPU parameters after inference.

        A missing batch size uses this artifact's inference_batch_size setting.
        Models reside on CPU between calls, while ``self.device`` retains the
        requested compute device.
        """
        import torch

        if batch_size is None:
            batch_size = self.manifest["config"]["inference_batch_size"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
            raise ValueError(f"features must have shape (events, {len(FEATURE_ORDER)})")
        bound = float(self.manifest["architecture"]["loss_logit_clip"])
        scores = np.empty((features.shape[0], len(self.models)), dtype=np.float32)
        try:
            for model in self.models:
                model.to(self.device)
            with torch.inference_mode():
                for start in range(0, len(features), batch_size):
                    stop = min(start + batch_size, len(features))
                    raw = np.asarray(features[start:stop], dtype=np.float32)
                    transformed = common.transformed_features(raw)
                    if not np.all(np.isfinite(transformed)):
                        raise ValueError("transformed prediction features overflow")
                    tensor = training._scaled_tensor(raw, self.mean, self.scale, self.device)
                    for member, model in enumerate(self.models):
                        logits = torch.clamp(model(tensor).reshape(-1), min=-bound, max=bound)
                        scores[start:stop, member] = torch.sigmoid(logits).cpu().numpy()
        finally:
            for model in self.models:
                model.cpu()
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("ratio model predictions are not finite")
        return _outputs(scores, self.manifest["calibration"], self.yield_ratio)
