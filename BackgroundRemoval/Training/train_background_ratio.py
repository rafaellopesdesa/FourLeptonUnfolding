#!/usr/bin/env python3
"""Train and validate the decay-only signal/background density ratio."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

import numpy as np
from numpy.lib.format import open_memmap
import uproot

from BackgroundRemoval.common import (
    ARTIFACT_FORMAT_VERSION,
    CLOSURE_SPLIT,
    FEATURES,
    FIT_SPLIT,
    MODEL_FEATURES,
    SPLIT_NAMES,
    TREE_NAME,
    VALIDATION_SPLIT,
    build_toolkit_model,
    deterministic_split,
    sha256_file,
    stable_logit,
    stable_sigmoid,
    toolkit_runtime_provenance,
    transformed_features,
)


DEFAULT_SEED = 19110907
DEFAULT_WEIGHT_BRANCH = "weight_nominal_pb"
DEFAULT_LUMINOSITY_FB = 312.0


@dataclass
class WeightedStatistics:
    entries: int = 0
    zero_weight_entries: int = 0
    sum_weights: float = 0.0
    sum_squared_weights: float = 0.0
    split_entries: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES.values()}
    )
    split_sum_weights: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in SPLIT_NAMES.values()}
    )
    split_sum_squared_weights: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in SPLIT_NAMES.values()}
    )

    def add(self, weights: np.ndarray, splits: np.ndarray, *, zero_count: int) -> None:
        self.zero_weight_entries += int(zero_count)
        self.entries += int(weights.size)
        self.sum_weights += float(np.sum(weights, dtype=np.float64))
        self.sum_squared_weights += float(np.sum(weights * weights, dtype=np.float64))
        for code, name in SPLIT_NAMES.items():
            selected = splits == code
            selected_weights = weights[selected]
            self.split_entries[name] += int(selected_weights.size)
            self.split_sum_weights[name] += float(
                np.sum(selected_weights, dtype=np.float64)
            )
            self.split_sum_squared_weights[name] += float(
                np.sum(selected_weights * selected_weights, dtype=np.float64)
            )

    def effective_entries(self) -> float:
        if self.sum_squared_weights <= 0.0:
            return 0.0
        return self.sum_weights * self.sum_weights / self.sum_squared_weights

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["effective_entries"] = self.effective_entries()
        return result


@dataclass(frozen=True)
class SelectedChunk:
    class_name: str
    component_name: str
    raw_features: np.ndarray
    weights: np.ndarray
    splits: np.ndarray
    zero_count: int


@dataclass(frozen=True)
class CachePaths:
    features: Path
    weights: Path
    splits: Path


@dataclass
class CachedClass:
    features: np.ndarray
    weights: np.ndarray
    splits: np.ndarray


@dataclass
class MemberResult:
    slot: int
    attempt: int
    seed: int
    state_dict: dict[str, Any]
    history: dict[str, list[float]]
    validation_loss: float
    validation_score_mean: float
    validation_score_std: float
    validation_saturated_fraction: float


@dataclass
class ClassPredictions:
    indices: np.ndarray
    member_scores: np.ndarray

    @property
    def ensemble_score(self) -> np.ndarray:
        return np.mean(self.member_scores, axis=1, dtype=np.float64)


def _branch_names(tree: object) -> set[str]:
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def _selected_chunks(
    gg_h_path: Path,
    zz_path: Path,
    *,
    weight_branch: str,
    split_seed: int,
    expected_luminosity_fb: float,
    step_size: str,
) -> Iterator[SelectedChunk]:
    required = {
        *FEATURES,
        "event_id",
        "fiducial",
        "reconstructed",
        "luminosity_fb",
        weight_branch,
    }
    for source, path in (("gg_H", gg_h_path), ("ZZ", zz_path)):
        with uproot.open(path) as root_file:
            if TREE_NAME not in root_file:
                raise KeyError(f"{path} does not contain the {TREE_NAME} tree")
            tree = root_file[TREE_NAME]
            missing = sorted(required.difference(_branch_names(tree)))
            if missing:
                raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
            for arrays in tree.iterate(
                expressions=sorted(required),
                step_size=step_size,
                library="np",
                how=dict,
            ):
                luminosities = np.asarray(arrays["luminosity_fb"], dtype=np.float64)
                if not np.all(
                    np.isfinite(luminosities)
                    & np.isclose(
                        luminosities,
                        expected_luminosity_fb,
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                ):
                    raise ValueError(
                        f"{path} is not normalized to {expected_luminosity_fb:g} fb^-1"
                    )
                reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
                fiducial = np.asarray(arrays["fiducial"], dtype=np.bool_)
                event_ids = np.asarray(arrays["event_id"], dtype=np.uint64)
                all_weights = np.asarray(arrays[weight_branch], dtype=np.float64)
                raw = np.column_stack(
                    [np.asarray(arrays[name], dtype=np.float32) for name in FEATURES]
                )
                splits = deterministic_split(
                    event_ids, source=source, seed=split_seed
                )
                masks = (
                    (
                        "signal",
                        "gg_H_reconstructed_and_fiducial",
                        reconstructed & fiducial,
                    ),
                    (
                        "background",
                        "gg_H_reconstructed_not_fiducial",
                        reconstructed & ~fiducial,
                    ),
                ) if source == "gg_H" else (
                    ("background", "ZZ_reconstructed", reconstructed),
                )
                for class_name, component_name, mask in masks:
                    selected_weights = all_weights[mask]
                    selected_raw = raw[mask]
                    selected_splits = splits[mask]
                    if not np.all(np.isfinite(selected_weights)):
                        raise ValueError(
                            f"{component_name} contains non-finite {weight_branch} values"
                        )
                    negative_count = int(np.count_nonzero(selected_weights < 0.0))
                    if negative_count:
                        raise ValueError(
                            f"{component_name} contains {negative_count} selected negative "
                            f"{weight_branch} values. A signed weighted BCE is not a density "
                            "ratio and can be unbounded; this training intentionally stops "
                            "instead of taking absolute values or dropping negative events."
                        )
                    if not np.all(np.isfinite(selected_raw)):
                        bad = int(np.count_nonzero(~np.all(np.isfinite(selected_raw), axis=1)))
                        raise ValueError(
                            f"{component_name} contains {bad} selected events with non-finite "
                            "model inputs"
                        )
                    positive = selected_weights > 0.0
                    yield SelectedChunk(
                        class_name=class_name,
                        component_name=component_name,
                        raw_features=selected_raw[positive],
                        weights=selected_weights[positive],
                        splits=selected_splits[positive],
                        zero_count=int(np.count_nonzero(selected_weights == 0.0)),
                    )


def _scan_inputs(
    gg_h_path: Path,
    zz_path: Path,
    *,
    weight_branch: str,
    split_seed: int,
    expected_luminosity_fb: float,
    step_size: str,
) -> tuple[dict[str, WeightedStatistics], dict[str, WeightedStatistics], np.ndarray, np.ndarray]:
    components = {
        "gg_H_reconstructed_and_fiducial": WeightedStatistics(),
        "gg_H_reconstructed_not_fiducial": WeightedStatistics(),
        "ZZ_reconstructed": WeightedStatistics(),
    }
    classes = {"signal": WeightedStatistics(), "background": WeightedStatistics()}
    moment_sum = {
        name: np.zeros(len(MODEL_FEATURES), dtype=np.float64) for name in classes
    }
    moment_square_sum = {
        name: np.zeros(len(MODEL_FEATURES), dtype=np.float64) for name in classes
    }

    for chunk in _selected_chunks(
        gg_h_path,
        zz_path,
        weight_branch=weight_branch,
        split_seed=split_seed,
        expected_luminosity_fb=expected_luminosity_fb,
        step_size=step_size,
    ):
        components[chunk.component_name].add(
            chunk.weights, chunk.splits, zero_count=chunk.zero_count
        )
        classes[chunk.class_name].add(
            chunk.weights, chunk.splits, zero_count=chunk.zero_count
        )
        fit = chunk.splits == FIT_SPLIT
        if np.any(fit):
            transformed = transformed_features(chunk.raw_features[fit]).astype(
                np.float64, copy=False
            )
            weights = chunk.weights[fit]
            moment_sum[chunk.class_name] += np.sum(
                transformed * weights[:, np.newaxis], axis=0, dtype=np.float64
            )
            moment_square_sum[chunk.class_name] += np.sum(
                transformed * transformed * weights[:, np.newaxis],
                axis=0,
                dtype=np.float64,
            )

    for class_name, statistics in classes.items():
        if statistics.entries == 0 or statistics.sum_weights <= 0.0:
            raise ValueError(f"the selected {class_name} class is empty")
        for split_name in SPLIT_NAMES.values():
            if statistics.split_entries[split_name] == 0:
                raise ValueError(f"the {class_name} {split_name} split is empty")
            if statistics.split_sum_weights[split_name] <= 0.0:
                raise ValueError(
                    f"the {class_name} {split_name} split has non-positive total weight"
                )

    class_means: list[np.ndarray] = []
    class_seconds: list[np.ndarray] = []
    for class_name in ("signal", "background"):
        fit_sum = classes[class_name].split_sum_weights["fit"]
        class_means.append(moment_sum[class_name] / fit_sum)
        class_seconds.append(moment_square_sum[class_name] / fit_sum)
    mean = 0.5 * (class_means[0] + class_means[1])
    second = 0.5 * (class_seconds[0] + class_seconds[1])
    variance = np.maximum(second - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-7, scale, 1.0)
    return components, classes, mean, scale


def _make_cache(
    cache_directory: Path,
    gg_h_path: Path,
    zz_path: Path,
    *,
    classes: dict[str, WeightedStatistics],
    weight_branch: str,
    split_seed: int,
    expected_luminosity_fb: float,
    step_size: str,
) -> dict[str, CachePaths]:
    paths: dict[str, CachePaths] = {}
    writers: dict[str, tuple[np.memmap, np.memmap, np.memmap]] = {}
    positions = {"signal": 0, "background": 0}
    for class_name, statistics in classes.items():
        class_paths = CachePaths(
            features=cache_directory / f"{class_name}_features.npy",
            weights=cache_directory / f"{class_name}_weights.npy",
            splits=cache_directory / f"{class_name}_splits.npy",
        )
        paths[class_name] = class_paths
        writers[class_name] = (
            open_memmap(
                class_paths.features,
                mode="w+",
                dtype=np.float32,
                shape=(statistics.entries, len(FEATURES)),
            ),
            open_memmap(
                class_paths.weights,
                mode="w+",
                dtype=np.float64,
                shape=(statistics.entries,),
            ),
            open_memmap(
                class_paths.splits,
                mode="w+",
                dtype=np.uint8,
                shape=(statistics.entries,),
            ),
        )

    for chunk in _selected_chunks(
        gg_h_path,
        zz_path,
        weight_branch=weight_branch,
        split_seed=split_seed,
        expected_luminosity_fb=expected_luminosity_fb,
        step_size=step_size,
    ):
        start = positions[chunk.class_name]
        stop = start + chunk.weights.size
        feature_writer, weight_writer, split_writer = writers[chunk.class_name]
        feature_writer[start:stop] = chunk.raw_features
        weight_writer[start:stop] = chunk.weights
        split_writer[start:stop] = chunk.splits
        positions[chunk.class_name] = stop

    for class_name, expected in classes.items():
        if positions[class_name] != expected.entries:
            raise RuntimeError(
                f"cache size changed while reading {class_name}: expected "
                f"{expected.entries}, wrote {positions[class_name]}"
            )
    for arrays in writers.values():
        for array in arrays:
            array.flush()
    del writers
    return paths


def _load_cache(paths: dict[str, CachePaths]) -> dict[str, CachedClass]:
    return {
        name: CachedClass(
            features=np.load(item.features, mmap_mode="r"),
            weights=np.load(item.weights, mmap_mode="r"),
            splits=np.load(item.splits, mmap_mode="r"),
        )
        for name, item in paths.items()
    }


def _scaled_tensor(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: Any):
    import torch

    transformed = transformed_features(raw)
    standardized = (transformed - mean.astype(np.float32)) / scale.astype(np.float32)
    return torch.from_numpy(standardized).to(device)


def _evaluate_member(
    model: Any,
    cache: dict[str, CachedClass],
    *,
    split_code: np.uint8,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    inference_batch_size: int,
) -> tuple[float, float, float, float]:
    import torch
    import torch.nn.functional as functional

    model.eval()
    class_losses: list[float] = []
    score_sum = 0.0
    score_square_sum = 0.0
    score_count = 0
    saturated = 0
    with torch.inference_mode():
        for class_name, label in (("signal", 1.0), ("background", 0.0)):
            arrays = cache[class_name]
            weighted_loss = 0.0
            total_weight = 0.0
            for start in range(0, len(arrays.weights), inference_batch_size):
                stop = min(start + inference_batch_size, len(arrays.weights))
                selected = np.asarray(arrays.splits[start:stop]) == split_code
                if not np.any(selected):
                    continue
                raw = np.asarray(arrays.features[start:stop][selected])
                weights = np.asarray(arrays.weights[start:stop][selected], dtype=np.float64)
                tensor = _scaled_tensor(raw, mean, scale, device)
                logits = model(tensor).reshape(-1)
                labels = torch.full_like(logits, label)
                losses = functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                ).detach().cpu().numpy()
                scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
                weighted_loss += float(np.sum(weights * losses, dtype=np.float64))
                total_weight += float(np.sum(weights, dtype=np.float64))
                score_sum += float(np.sum(scores, dtype=np.float64))
                score_square_sum += float(np.sum(scores * scores, dtype=np.float64))
                score_count += scores.size
                saturated += int(np.count_nonzero((scores < 1.0e-6) | (scores > 1.0 - 1.0e-6)))
            if total_weight <= 0.0:
                raise RuntimeError(f"empty {class_name} evaluation split")
            class_losses.append(weighted_loss / total_weight)
    mean_score = score_sum / score_count
    variance = max(score_square_sum / score_count - mean_score * mean_score, 0.0)
    return (
        0.5 * (class_losses[0] + class_losses[1]),
        mean_score,
        math.sqrt(variance),
        saturated / score_count,
    )


def _train_member(
    slot: int,
    attempt: int,
    cache: dict[str, CachedClass],
    *,
    architecture: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    steps_per_epoch: int | None,
    inference_batch_size: int,
    base_seed: int,
) -> MemberResult:
    import torch
    import torch.nn.functional as functional

    seed = (base_seed + 1009 * slot + 100_003 * attempt) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_toolkit_model(architecture).to(device)
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=float(architecture["learning_rate"]),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(architecture["learning_rate_decay"])
    )
    rng = np.random.default_rng(seed)
    signal_indices = np.flatnonzero(cache["signal"].splits == FIT_SPLIT)
    background_indices = np.flatnonzero(cache["background"].splits == FIT_SPLIT)
    half_batch = batch_size // 2
    if steps_per_epoch is None:
        epoch_steps = math.ceil(
            max(signal_indices.size, background_indices.size) / half_batch
        )
    else:
        epoch_steps = steps_per_epoch
    signal_mean_weight = float(
        np.sum(cache["signal"].weights[signal_indices], dtype=np.float64)
        / signal_indices.size
    )
    background_mean_weight = float(
        np.sum(cache["background"].weights[background_indices], dtype=np.float64)
        / background_indices.size
    )

    history = {"training_loss": [], "validation_loss": [], "learning_rate": []}
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(maximum_epochs):
        model.train()
        accumulated = 0.0
        for _ in range(epoch_steps):
            signal_draw = signal_indices[
                rng.integers(0, signal_indices.size, size=half_batch)
            ]
            background_draw = background_indices[
                rng.integers(0, background_indices.size, size=half_batch)
            ]
            # Sorted indices turn random memmap access into mostly sequential reads.
            signal_draw.sort()
            background_draw.sort()
            signal_tensor = _scaled_tensor(
                np.asarray(cache["signal"].features[signal_draw]), mean, scale, device
            )
            background_tensor = _scaled_tensor(
                np.asarray(cache["background"].features[background_draw]), mean, scale, device
            )
            signal_coefficients = torch.from_numpy(
                np.asarray(
                    cache["signal"].weights[signal_draw] / signal_mean_weight,
                    dtype=np.float32,
                )
            ).to(device)
            background_coefficients = torch.from_numpy(
                np.asarray(
                    cache["background"].weights[background_draw]
                    / background_mean_weight,
                    dtype=np.float32,
                )
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            signal_logits = model(signal_tensor).reshape(-1)
            background_logits = model(background_tensor).reshape(-1)
            signal_loss = functional.binary_cross_entropy_with_logits(
                signal_logits, torch.ones_like(signal_logits), reduction="none"
            )
            background_loss = functional.binary_cross_entropy_with_logits(
                background_logits, torch.zeros_like(background_logits), reduction="none"
            )
            loss = 0.5 * (
                torch.mean(signal_loss * signal_coefficients)
                + torch.mean(background_loss * background_coefficients)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"member {slot} produced a non-finite training loss")
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach().cpu())

        validation_loss, score_mean, score_std, saturated_fraction = _evaluate_member(
            model,
            cache,
            split_code=VALIDATION_SPLIT,
            mean=mean,
            scale=scale,
            device=device,
            inference_batch_size=inference_batch_size,
        )
        history["training_loss"].append(accumulated / epoch_steps)
        history["validation_loss"].append(validation_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        print(
            f"member={slot} attempt={attempt} epoch={epoch + 1}/{maximum_epochs} "
            f"train={history['training_loss'][-1]:.7g} validation={validation_loss:.7g} "
            f"lr={history['learning_rate'][-1]:.3g}",
            flush=True,
        )
        if validation_loss < best_loss - 1.0e-7:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"member {slot} did not produce a usable checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_loss, score_mean, score_std, saturated_fraction = _evaluate_member(
        model,
        cache,
        split_code=VALIDATION_SPLIT,
        mean=mean,
        scale=scale,
        device=device,
        inference_batch_size=inference_batch_size,
    )
    return MemberResult(
        slot=slot,
        attempt=attempt,
        seed=seed,
        state_dict=best_state,
        history=history,
        validation_loss=validation_loss,
        validation_score_mean=score_mean,
        validation_score_std=score_std,
        validation_saturated_fraction=saturated_fraction,
    )


def _bad_members(results: list[MemberResult]) -> list[int]:
    losses = np.asarray([item.validation_loss for item in results], dtype=np.float64)
    median = float(np.median(losses))
    mad = float(np.median(np.abs(losses - median)))
    bad: list[int] = []
    for index, item in enumerate(results):
        invalid = not np.isfinite(item.validation_loss)
        relative_outlier = item.validation_loss > 1.05 * median
        mad_outlier = item.validation_loss > median + 5.0 * max(mad, 1.0e-8)
        if invalid or (relative_outlier and mad_outlier):
            bad.append(index)
    return bad


def _predict_split(
    cache: dict[str, CachedClass],
    results: list[MemberResult],
    *,
    split_code: np.uint8,
    architecture: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    inference_batch_size: int,
) -> dict[str, ClassPredictions]:
    import torch

    models = []
    for result in results:
        model = build_toolkit_model(architecture)
        model.load_state_dict(result.state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    output: dict[str, ClassPredictions] = {}
    with torch.inference_mode():
        for class_name, arrays in cache.items():
            indices = np.flatnonzero(arrays.splits == split_code)
            predictions = np.empty((indices.size, len(models)), dtype=np.float32)
            for start in range(0, indices.size, inference_batch_size):
                stop = min(start + inference_batch_size, indices.size)
                selected = indices[start:stop]
                tensor = _scaled_tensor(
                    np.asarray(arrays.features[selected]), mean, scale, device
                )
                for member_index, model in enumerate(models):
                    predictions[start:stop, member_index] = (
                        torch.sigmoid(model(tensor).reshape(-1))
                        .detach()
                        .cpu()
                        .numpy()
                    )
            output[class_name] = ClassPredictions(indices, predictions)
    return output


def _balanced_calibration_arrays(
    predictions: dict[str, ClassPredictions], cache: dict[str, CachedClass]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = predictions["signal"]
    background = predictions["background"]
    signal_weights = np.asarray(
        cache["signal"].weights[signal.indices], dtype=np.float64
    )
    background_weights = np.asarray(
        cache["background"].weights[background.indices], dtype=np.float64
    )
    weights = np.concatenate(
        [
            0.5 * signal_weights / np.sum(signal_weights, dtype=np.float64),
            0.5 * background_weights / np.sum(background_weights, dtype=np.float64),
        ]
    )
    scores = np.concatenate([signal.ensemble_score, background.ensemble_score])
    labels = np.concatenate(
        [np.ones(signal.indices.size), np.zeros(background.indices.size)]
    )
    return scores, labels, weights


def fit_logit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    logit_clip: float,
    minimum_scale: float = 0.1,
    maximum_scale: float = 5.0,
    maximum_iterations: int = 100,
) -> tuple[float, float]:
    """Fit a two-parameter affine logit calibrator with weighted BCE."""

    logits = stable_logit(scores, logit_clip=logit_clip)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if logits.shape != labels.shape or labels.shape != weights.shape:
        raise ValueError("calibration arrays must have identical shapes")
    if not (0.0 < minimum_scale <= 1.0 <= maximum_scale):
        raise ValueError("calibration scale bounds must be positive and contain one")
    design = np.column_stack([logits, np.ones_like(logits)])
    parameters = np.array([1.0, 0.0], dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        prediction = design @ candidate
        return float(
            np.sum(
                weights
                * (
                    np.maximum(prediction, 0.0)
                    - labels * prediction
                    + np.log1p(np.exp(-np.abs(prediction)))
                ),
                dtype=np.float64,
            )
        )

    for _ in range(maximum_iterations):
        prediction = design @ parameters
        probability = stable_sigmoid(prediction)
        gradient = design.T @ (weights * (probability - labels))
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, np.newaxis])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            # Perfectly separated validation samples have no finite logistic
            # MLE.  A pseudoinverse gives the last identifiable Newton step;
            # clipping below supplies the documented finite ratio boundary.
            step = np.linalg.pinv(hessian, rcond=1.0e-12) @ gradient
            if not np.all(np.isfinite(step)) or float(np.max(np.abs(step))) == 0.0:
                break
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
        old_objective = objective(parameters)
        fraction = 1.0
        accepted = False
        while fraction >= 1.0e-8:
            candidate = parameters - fraction * step
            candidate[0] = np.clip(candidate[0], minimum_scale, maximum_scale)
            candidate[1] = np.clip(candidate[1], -logit_clip, logit_clip)
            if objective(candidate) <= old_objective:
                unchanged = np.allclose(candidate, parameters, rtol=0.0, atol=1.0e-12)
                parameters = candidate
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            break
        if unchanged:
            break
    if not np.all(np.isfinite(parameters)) or parameters[0] <= 0.0:
        raise RuntimeError("calibration produced an invalid affine transform")
    return float(parameters[0]), float(parameters[1])


def _normalization_bias(
    background_scores: np.ndarray,
    background_weights: np.ndarray,
    *,
    scale: float,
    initial_bias: float,
    logit_clip: float,
) -> tuple[float, float]:
    logits = stable_logit(background_scores, logit_clip=logit_clip)
    scaled_logits = scale * logits
    if not np.all(np.isfinite(scaled_logits)):
        raise RuntimeError("calibration logits overflow during ratio normalization")

    def mean_ratio(bias: float) -> float:
        ratios = np.exp(np.clip(scale * logits + bias, -logit_clip, logit_clip))
        return float(
            np.sum(background_weights * ratios, dtype=np.float64)
            / np.sum(background_weights, dtype=np.float64)
        )

    before = mean_ratio(initial_bias)
    # At -max(t), every t+b is <= 0 and the mean ratio is <= 1.  At
    # -min(t), every t+b is >= 0 and the mean is >= 1.  This is a guaranteed
    # bracket even when an affine calibrator reaches its scale bound.
    low = -float(np.max(scaled_logits))
    high = -float(np.min(scaled_logits))
    if high == low:
        return low, before
    for _ in range(120):
        middle = 0.5 * (low + high)
        if mean_ratio(middle) > 1.0:
            high = middle
        else:
            low = middle
    normalized_bias = 0.5 * (low + high)
    normalized_mean = mean_ratio(normalized_bias)
    if not np.isclose(normalized_mean, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError(
            "failed to normalize the calibrated density ratio: "
            f"E_B[r]={normalized_mean:.12g}"
        )
    return normalized_bias, before


def _install_artifact(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output directory exists: {destination}; pass --overwrite to replace it"
        )
    if destination.exists():
        _validate_existing_artifact(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.previous-", dir=destination.parent
            )
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _validate_existing_artifact(path: Path) -> None:
    """Restrict destructive replacement to a directory made by this workflow."""

    if not path.is_dir():
        raise NotADirectoryError(f"model output is not a directory: {path}")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"refusing to overwrite non-artifact directory without manifest: {path}"
        )
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"refusing to overwrite invalid model artifact: {path}") from error
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"refusing to overwrite model artifact with unsupported format: {path}"
        )


def train(args: argparse.Namespace) -> Path:
    import torch

    from BackgroundRemoval.Training.diagnostics import make_diagnostics_pdf

    gg_h_path = args.gg_h_root.expanduser().resolve()
    zz_path = args.zz_root.expanduser().resolve()
    for path in (gg_h_path, zz_path):
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    if gg_h_path == zz_path:
        raise ValueError("gg_H and ZZ training inputs must be different files")
    if args.ensemble_size < 4:
        raise ValueError("the ensemble must contain at least four members")
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("batch size must be an even integer of at least four")
    if args.maximum_epochs < 1 or args.patience < 1:
        raise ValueError("maximum epochs and patience must be positive")
    if args.neurons < 1 or args.hidden_layers < 1:
        raise ValueError("the network architecture must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not np.isfinite(args.learning_rate_decay) or not (
        0.0 < args.learning_rate_decay <= 1.0
    ):
        raise ValueError("learning-rate decay must be finite and lie in (0, 1]")
    if args.max_retries < 0:
        raise ValueError("maximum retries cannot be negative")
    maximum_seed = int(np.iinfo(np.uint32).max)
    if not (0 <= args.seed <= maximum_seed) or not (
        0 <= args.split_seed <= maximum_seed
    ):
        raise ValueError(f"training and split seeds must lie in [0, {maximum_seed}]")
    if not np.isfinite(args.expected_luminosity_fb) or args.expected_luminosity_fb <= 0.0:
        raise ValueError("expected luminosity must be finite and positive")
    if not np.isfinite(args.logit_clip) or not (0.0 < args.logit_clip <= 100.0):
        raise ValueError("logit clip must be finite and lie in (0, 100]")
    if not np.isfinite(args.calibration_scale_min) or not np.isfinite(
        args.calibration_scale_max
    ) or not (
        0.0 < args.calibration_scale_min <= 1.0 <= args.calibration_scale_max
    ):
        raise ValueError("calibration scale bounds must be positive and contain one")
    if args.inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError("steps per epoch must be positive when supplied")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    toolkit_provenance = toolkit_runtime_provenance()
    if not toolkit_provenance["runtime_commit_verified"]:
        print(
            "Warning: the nsbi-common-utils runtime commit could not be verified from "
            "package metadata; use the pinned Pixi environment for production.",
            flush=True,
        )

    output_directory = args.output_dir.expanduser().resolve()
    if output_directory == gg_h_path or output_directory == zz_path:
        raise ValueError("model output directory cannot be a training input")
    if output_directory in gg_h_path.parents or output_directory in zz_path.parents:
        raise ValueError("model output directory cannot contain a training input")
    if output_directory.exists() and not args.overwrite:
        raise FileExistsError(
            f"output directory exists: {output_directory}; pass --overwrite to replace it"
        )
    if output_directory.exists():
        _validate_existing_artifact(output_directory)
    cache_parent = (
        args.cache_directory.expanduser().resolve()
        if args.cache_directory is not None
        else None
    )
    if cache_parent is not None and (
        cache_parent == output_directory or output_directory in cache_parent.parents
    ):
        raise ValueError("cache directory cannot be inside the model output directory")
    if cache_parent is not None:
        cache_parent.mkdir(parents=True, exist_ok=True)

    print("Scanning selections, yields, and deterministic splits...", flush=True)
    components, classes, scaler_mean, scaler_scale = _scan_inputs(
        gg_h_path,
        zz_path,
        weight_branch=args.weight_branch,
        split_seed=args.split_seed,
        expected_luminosity_fb=args.expected_luminosity_fb,
        step_size=args.step_size,
    )
    signal_yield = classes["signal"].sum_weights
    background_yield = classes["background"].sum_weights
    yield_ratio = signal_yield / background_yield
    print(
        f"Selected yields ({args.weight_branch}): S={signal_yield:.9g}, "
        f"B={background_yield:.9g}, S/B={yield_ratio:.9g}",
        flush=True,
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.training-", dir=output_directory.parent
        )
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="fourlepton-background-cache-", dir=cache_parent
        ) as cache_name:
            cache_directory = Path(cache_name)
            print(f"Building bounded-memory cache in {cache_directory}...", flush=True)
            cache_paths = _make_cache(
                cache_directory,
                gg_h_path,
                zz_path,
                classes=classes,
                weight_branch=args.weight_branch,
                split_seed=args.split_seed,
                expected_luminosity_fb=args.expected_luminosity_fb,
                step_size=args.step_size,
            )
            cache = _load_cache(cache_paths)
            architecture = {
                "implementation": "nsbi_common_utils.lightning_tools.DensityRatioLightning",
                "hidden_layers": args.hidden_layers,
                "neurons": args.neurons,
                "activation": "SiLU",
                "output": "logit",
                "loss": "balanced weighted BCEWithLogitsLoss",
                "dropout": 0.0,
                "weight_decay": 0.0,
                "optimizer": "NAdam",
                "learning_rate": args.learning_rate,
                "learning_rate_schedule": "ExponentialLR",
                "learning_rate_decay": args.learning_rate_decay,
                "batch_size": args.batch_size,
                "maximum_epochs": args.maximum_epochs,
                "early_stopping_patience": args.patience,
            }

            results: list[MemberResult] = []
            for slot in range(args.ensemble_size):
                results.append(
                    _train_member(
                        slot,
                        0,
                        cache,
                        architecture=architecture,
                        mean=scaler_mean,
                        scale=scaler_scale,
                        device=device,
                        batch_size=args.batch_size,
                        maximum_epochs=args.maximum_epochs,
                        patience=args.patience,
                        steps_per_epoch=args.steps_per_epoch,
                        inference_batch_size=args.inference_batch_size,
                        base_seed=args.seed,
                    )
                )
            for retry_round in range(1, args.max_retries + 1):
                bad = _bad_members(results)
                if not bad:
                    break
                print(
                    f"Retrying discrepant ensemble slots {bad} (round {retry_round})...",
                    flush=True,
                )
                for index in bad:
                    results[index] = _train_member(
                        index,
                        retry_round,
                        cache,
                        architecture=architecture,
                        mean=scaler_mean,
                        scale=scaler_scale,
                        device=device,
                        batch_size=args.batch_size,
                        maximum_epochs=args.maximum_epochs,
                        patience=args.patience,
                        steps_per_epoch=args.steps_per_epoch,
                        inference_batch_size=args.inference_batch_size,
                        base_seed=args.seed,
                    )
            remaining_bad = _bad_members(results)
            if remaining_bad:
                raise RuntimeError(
                    f"ensemble slots {remaining_bad} remain discrepant after "
                    f"{args.max_retries} retries"
                )

            print("Predicting the held-out validation/calibration bank...", flush=True)
            validation = _predict_split(
                cache,
                results,
                split_code=VALIDATION_SPLIT,
                architecture=architecture,
                mean=scaler_mean,
                scale=scaler_scale,
                device=device,
                inference_batch_size=args.inference_batch_size,
            )
            calibration_scores, calibration_labels, calibration_weights = (
                _balanced_calibration_arrays(validation, cache)
            )
            calibration_scale, calibration_bias = fit_logit_calibrator(
                calibration_scores,
                calibration_labels,
                calibration_weights,
                logit_clip=args.logit_clip,
                minimum_scale=args.calibration_scale_min,
                maximum_scale=args.calibration_scale_max,
            )
            background_validation_weights = np.asarray(
                cache["background"].weights[validation["background"].indices],
                dtype=np.float64,
            )
            normalized_bias, normalization_before = _normalization_bias(
                validation["background"].ensemble_score,
                background_validation_weights,
                scale=calibration_scale,
                initial_bias=calibration_bias,
                logit_clip=args.logit_clip,
            )

            print("Predicting the untouched closure bank...", flush=True)
            closure = _predict_split(
                cache,
                results,
                split_code=CLOSURE_SPLIT,
                architecture=architecture,
                mean=scaler_mean,
                scale=scaler_scale,
                device=device,
                inference_batch_size=args.inference_batch_size,
            )

            member_entries: list[dict[str, Any]] = []
            for result in results:
                member_name = f"member_{result.slot:03d}.pt"
                member_path = temporary_artifact / member_name
                torch.save(result.state_dict, member_path)
                member_entries.append(
                    {
                        "slot": result.slot,
                        "attempt": result.attempt,
                        "seed": result.seed,
                        "file": member_name,
                        "sha256": sha256_file(member_path),
                        "validation_loss": result.validation_loss,
                        "validation_score_mean": result.validation_score_mean,
                        "validation_score_std": result.validation_score_std,
                        "validation_saturated_fraction": (
                            result.validation_saturated_fraction
                        ),
                        "history": result.history,
                    }
                )

            input_provenance = {}
            for name, path in (("gg_H_pythia", gg_h_path), ("ZZ_pythia", zz_path)):
                stat = path.stat()
                input_provenance[name] = {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": None if args.skip_input_checksums else sha256_file(path),
                }
            manifest: dict[str, Any] = {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "tree_name": TREE_NAME,
                "features": list(FEATURES),
                "model_features": list(MODEL_FEATURES),
                "preprocessing": {
                    "periodic_variables": list(FEATURES[:3]),
                    "periodic_encoding": "sin_cos",
                    "remaining_variables": "identity",
                },
                "selections": {
                    "positive_label_1": "gg_H_pythia: reconstructed && fiducial",
                    "negative_label_0": (
                        "ZZ_pythia: reconstructed; plus gg_H_pythia: "
                        "reconstructed && !fiducial"
                    ),
                },
                "weight_branch": args.weight_branch,
                "luminosity_fb": args.expected_luminosity_fb,
                "yields": {
                    "signal": signal_yield,
                    "background": background_yield,
                    "signal_to_background": yield_ratio,
                    "units": "pb" if args.weight_branch == "weight_nominal_pb" else "events",
                },
                "ratio_convention": {
                    "shape": "p(signal)/p(background) = score/(1-score)",
                    "physical_odds": "(signal_yield/background_yield) * shape",
                    "background_removal_weight": "physical_odds/(1+physical_odds)",
                },
                "split": {
                    "method": "splitmix64(source,event_id,seed)",
                    "seed": args.split_seed,
                    "fit_fraction": 0.60,
                    "validation_fraction": 0.15,
                    "closure_fraction": 0.25,
                },
                "components": {
                    name: statistics.serializable()
                    for name, statistics in components.items()
                },
                "classes": {
                    name: statistics.serializable()
                    for name, statistics in classes.items()
                },
                "scaler": {
                    "type": "balanced_weighted_standard",
                    "fit_split_only": True,
                    "mean": scaler_mean.tolist(),
                    "scale": scaler_scale.tolist(),
                },
                "architecture": architecture,
                "ensemble": {
                    "aggregation": "arithmetic_mean_probability",
                    "size": len(results),
                    "minimum_size": 4,
                    "outlier_rule": (
                        "validation loss is nonfinite, or worse than the median by "
                        "both >5% and >5 MAD"
                    ),
                    "maximum_retries_per_slot": args.max_retries,
                },
                "members": member_entries,
                "calibration": {
                    "method": "affine_logit_then_density_normalization",
                    "fit_split": "validation",
                    "scale": calibration_scale,
                    "scale_bounds": [
                        args.calibration_scale_min,
                        args.calibration_scale_max,
                    ],
                    "bias_before_normalization": calibration_bias,
                    "normalization_before": normalization_before,
                    "bias_after_normalization": normalized_bias,
                    "logit_clip": args.logit_clip,
                },
                "toolkit": {
                    "repository": "https://github.com/iris-hep/nsbi-lhc-toolkit",
                    **toolkit_provenance,
                    "model_class": (
                        "nsbi_common_utils.lightning_tools.density_ratio_model."
                        "DensityRatioLightning"
                    ),
                },
                "inputs": input_provenance,
            }

            diagnostics_path = temporary_artifact / "diagnostics.pdf"
            manifest["closure_metrics"] = make_diagnostics_pdf(
                diagnostics_path,
                manifest=manifest,
                cache=cache,
                validation=validation,
                closure=closure,
            )
            manifest["diagnostics"] = diagnostics_path.name
            with (temporary_artifact / "manifest.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")

        _install_artifact(
            temporary_artifact, output_directory, overwrite=args.overwrite
        )
    except Exception:
        if temporary_artifact.exists():
            shutil.rmtree(temporary_artifact)
        raise
    print(f"Background-removal model written to {output_directory}", flush=True)
    return output_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a calibrated NSBI-toolkit BCE ensemble for reconstructed "
            "fiducial ggH versus reconstructed contamination"
        )
    )
    parser.add_argument("--gg-h-root", type=Path, required=True)
    parser.add_argument("--zz-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-branch", default=DEFAULT_WEIGHT_BRANCH)
    parser.add_argument(
        "--expected-luminosity-fb", type=float, default=DEFAULT_LUMINOSITY_FB
    )
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--learning-rate-decay", type=float, default=0.98)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED + 1)
    parser.add_argument("--logit-clip", type=float, default=30.0)
    parser.add_argument("--calibration-scale-min", type=float, default=0.1)
    parser.add_argument("--calibration-scale-max", type=float, default=5.0)
    parser.add_argument("--step-size", default="200 MB")
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        help="override the full-coverage number of balanced batches (primarily for tests)",
    )
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--skip-input-checksums", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    train(_parser().parse_args())


if __name__ == "__main__":
    main()
