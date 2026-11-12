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
import warnings
from typing import Any, Iterator

import numpy as np
from numpy.lib.format import open_memmap
import uproot

from BackgroundRemoval.common import (
    ANALYSIS_MASS_WINDOW,
    ARTIFACT_FORMAT_VERSION,
    CLOSURE_SPLIT,
    CORRECTION_MASS_WINDOW,
    FEATURES,
    FIT_SPLIT,
    MASS_BRANCH,
    MODEL_KIND_BACKGROUND_REMOVAL,
    MODEL_KIND_DATA_MC_CORRECTION,
    MODEL_FEATURES,
    RAW_RATIO_ESTIMATOR,
    SPLIT_NAMES,
    SUPPORTED_ARTIFACT_FORMAT_VERSIONS,
    TREE_NAME,
    VALIDATION_SPLIT,
    ModelBundle,
    assert_file_unchanged,
    build_toolkit_model,
    capture_file_provenance,
    deterministic_split,
    known_weight_measures_compatible,
    sha256_file,
    stable_logit,
    toolkit_runtime_provenance,
    transformed_features,
    validate_manifest_mass_window,
    weight_measure_contract,
)


DEFAULT_SEED = 19110907
DEFAULT_WEIGHT_BRANCH = "weight_nominal_pb"
DEFAULT_LUMINOSITY_FB = 312.0


@dataclass
class WeightedStatistics:
    entries: int = 0
    zero_weight_entries: int = 0
    negative_weight_entries: int = 0
    sum_weights: float = 0.0
    sum_absolute_weights: float = 0.0
    sum_squared_weights: float = 0.0
    split_entries: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES.values()}
    )
    split_sum_weights: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in SPLIT_NAMES.values()}
    )
    split_sum_absolute_weights: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in SPLIT_NAMES.values()}
    )
    split_sum_squared_weights: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in SPLIT_NAMES.values()}
    )
    split_negative_weight_entries: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES.values()}
    )

    def add(self, weights: np.ndarray, splits: np.ndarray, *, zero_count: int) -> None:
        self.zero_weight_entries += int(zero_count)
        self.entries += int(weights.size)
        self.negative_weight_entries += int(np.count_nonzero(weights < 0.0))
        self.sum_weights += float(np.sum(weights, dtype=np.float64))
        self.sum_absolute_weights += float(
            np.sum(np.abs(weights), dtype=np.float64)
        )
        self.sum_squared_weights += float(np.sum(weights * weights, dtype=np.float64))
        for code, name in SPLIT_NAMES.items():
            selected = splits == code
            selected_weights = weights[selected]
            self.split_entries[name] += int(selected_weights.size)
            self.split_negative_weight_entries[name] += int(
                np.count_nonzero(selected_weights < 0.0)
            )
            self.split_sum_weights[name] += float(
                np.sum(selected_weights, dtype=np.float64)
            )
            self.split_sum_absolute_weights[name] += float(
                np.sum(np.abs(selected_weights), dtype=np.float64)
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
        result["net_to_absolute_weight_ratio"] = (
            self.sum_weights / self.sum_absolute_weights
            if self.sum_absolute_weights > 0.0
            else 0.0
        )
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
    history: dict[str, list[float | int | None]]
    validation_loss: float
    validation_score_mean: float
    validation_score_std: float
    validation_saturated_fraction: float
    training_summary: dict[str, Any] = field(default_factory=dict)


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
    correction_bundle: ModelBundle,
    inference_batch_size: int,
) -> Iterator[SelectedChunk]:
    required = {
        *FEATURES,
        "event_id",
        "fiducial",
        "reconstructed",
        "luminosity_fb",
        MASS_BRANCH,
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
                masses = np.asarray(arrays[MASS_BRANCH], dtype=np.float64)
                if np.any(reconstructed & ~np.isfinite(masses)):
                    raise ValueError(
                        f"{path} contains reconstructed events with non-finite "
                        f"{MASS_BRANCH}"
                    )
                low_mass, high_mass = ANALYSIS_MASS_WINDOW
                in_analysis_window = (
                    np.isfinite(masses) & (masses > low_mass) & (masses < high_mass)
                )
                splits = deterministic_split(
                    event_ids, source=source, seed=split_seed
                )
                masks = (
                    (
                        "signal",
                        "gg_H_reconstructed_and_fiducial",
                        reconstructed & fiducial & in_analysis_window,
                    ),
                    (
                        "background",
                        "gg_H_reconstructed_not_fiducial",
                        reconstructed & ~fiducial & in_analysis_window,
                    ),
                ) if source == "gg_H" else (
                    (
                        "background",
                        "ZZ_reconstructed_correction_weighted",
                        reconstructed & in_analysis_window,
                    ),
                )
                for class_name, component_name, mask in masks:
                    selected_weights = all_weights[mask]
                    selected_raw = raw[mask]
                    selected_splits = splits[mask]
                    if not np.all(np.isfinite(selected_weights)):
                        raise ValueError(
                            f"{component_name} contains non-finite {weight_branch} values"
                        )
                    if not np.all(np.isfinite(selected_raw)):
                        bad = int(np.count_nonzero(~np.all(np.isfinite(selected_raw), axis=1)))
                        raise ValueError(
                            f"{component_name} contains {bad} selected events with non-finite "
                            "model inputs"
                        )
                    if source == "ZZ" and selected_weights.size:
                        correction = np.asarray(
                            correction_bundle.predict(
                                selected_raw,
                                batch_size=inference_batch_size,
                            )["physical_ratio"],
                            dtype=np.float64,
                        )
                        if not np.all(np.isfinite(correction) & (correction > 0.0)):
                            raise ValueError(
                                "the data/MC Correction model produced a non-finite or "
                                "non-positive multiplicative correction factor"
                            )
                        selected_weights = selected_weights * correction
                    nonzero = selected_weights != 0.0
                    yield SelectedChunk(
                        class_name=class_name,
                        component_name=component_name,
                        raw_features=selected_raw[nonzero],
                        weights=selected_weights[nonzero],
                        splits=selected_splits[nonzero],
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
    correction_bundle: ModelBundle,
    inference_batch_size: int,
) -> tuple[dict[str, WeightedStatistics], dict[str, WeightedStatistics], np.ndarray, np.ndarray]:
    components = {
        "gg_H_reconstructed_and_fiducial": WeightedStatistics(),
        "gg_H_reconstructed_not_fiducial": WeightedStatistics(),
        "ZZ_reconstructed_correction_weighted": WeightedStatistics(),
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
        correction_bundle=correction_bundle,
        inference_batch_size=inference_batch_size,
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
            # The scaler is only a numerical preconditioner.  Absolute weights
            # give it a positive measure even when generator events carry a
            # sign; the signed weights themselves are retained everywhere in
            # the physics loss, yields, and diagnostics.
            weights = np.abs(chunk.weights[fit])
            moment_sum[chunk.class_name] += np.sum(
                transformed * weights[:, np.newaxis], axis=0, dtype=np.float64
            )
            moment_square_sum[chunk.class_name] += np.sum(
                transformed * transformed * weights[:, np.newaxis],
                axis=0,
                dtype=np.float64,
            )

    for class_name, statistics in classes.items():
        if statistics.entries == 0:
            raise ValueError(f"the selected {class_name} class is empty")
        if not np.isfinite(statistics.sum_weights) or statistics.sum_weights <= 0.0:
            raise ValueError(
                f"the selected {class_name} class has non-positive net signed yield"
            )
        for split_name in SPLIT_NAMES.values():
            if statistics.split_entries[split_name] == 0:
                raise ValueError(f"the {class_name} {split_name} split is empty")
            if (
                not np.isfinite(statistics.split_sum_weights[split_name])
                or statistics.split_sum_weights[split_name] <= 0.0
            ):
                raise ValueError(
                    f"the {class_name} {split_name} split has non-positive net signed yield"
                )

    class_means: list[np.ndarray] = []
    class_seconds: list[np.ndarray] = []
    for class_name in ("signal", "background"):
        fit_sum = classes[class_name].split_sum_absolute_weights["fit"]
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
    correction_bundle: ModelBundle,
    inference_batch_size: int,
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
        correction_bundle=correction_bundle,
        inference_batch_size=inference_batch_size,
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
    logit_clip: float,
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
                logits = torch.clamp(
                    model(tensor).reshape(-1),
                    min=-abs(logit_clip),
                    max=abs(logit_clip),
                )
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
                raise RuntimeError(
                    f"{class_name} evaluation split has non-positive net signed yield"
                )
            class_losses.append(weighted_loss / total_weight)
    mean_score = score_sum / score_count
    variance = max(score_square_sum / score_count - mean_score * mean_score, 0.0)
    return (
        0.5 * (class_losses[0] + class_losses[1]),
        mean_score,
        math.sqrt(variance),
        saturated / score_count,
    )


def _absolute_weight_sampling_plan(
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    class_name: str,
) -> tuple[np.ndarray, float]:
    """Build an |w|-importance sampler and its signed-risk coefficient.

    Drawing event ``i`` with probability ``|w_i| / sum_j |w_j|`` and using
    ``sign(w_i) * sum_j|w_j| / sum_j w_j`` in the minibatch gives an unbiased
    estimator of ``sum_i w_i loss_i / sum_i w_i``.  This avoids letting a few
    large-magnitude signed events dominate the variance of uniform sampling,
    while never replacing the signed physics measure by an absolute one.
    """

    selected = np.asarray(weights[indices], dtype=np.float64)
    absolute = np.abs(selected)
    cumulative = np.cumsum(absolute, dtype=np.float64)
    absolute_sum = float(cumulative[-1]) if cumulative.size else 0.0
    net_sum = float(np.sum(selected, dtype=np.float64))
    if not np.isfinite(absolute_sum) or absolute_sum <= 0.0:
        raise RuntimeError(f"{class_name} fit split has no finite nonzero weights")
    if not np.isfinite(net_sum) or net_sum <= 0.0:
        raise RuntimeError(
            f"{class_name} fit split has non-positive net signed yield"
        )
    return cumulative, absolute_sum / net_sum


def _draw_absolute_weighted(
    rng: np.random.Generator,
    indices: np.ndarray,
    cumulative_absolute_weights: np.ndarray,
    *,
    size: int,
) -> np.ndarray:
    """Sample cached event indices with replacement in proportion to |w|."""

    draws = rng.random(size) * cumulative_absolute_weights[-1]
    positions = np.searchsorted(
        cumulative_absolute_weights, draws, side="right"
    )
    # rng.random is specified on [0, 1), but guard the final lookup against a
    # platform-level rounding to the cumulative endpoint.
    np.minimum(positions, indices.size - 1, out=positions)
    return indices[positions]


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

    def positive_integer(name: str, default: int) -> int:
        value = architecture.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    validation_cap = positive_integer("validation_interval_steps", 25)
    decay_steps = positive_integer("learning_rate_decay_steps", 100)
    weight_decay = float(architecture.get("weight_decay", 1.0e-4))
    minimum_delta = float(architecture.get("validation_min_delta", 1.0e-5))
    baseline_tolerance = float(architecture.get("baseline_tolerance", 0.01))
    for name, value in (("weight_decay", weight_decay), ("validation_min_delta", minimum_delta),
                        ("baseline_tolerance", baseline_tolerance)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    decay_factor = float(architecture["learning_rate_decay"])
    if not np.isfinite(decay_factor) or not 0.0 < decay_factor <= 1.0:
        raise ValueError("learning_rate_decay must lie in (0, 1]")

    seed = (base_seed + 1009 * slot + 100_003 * attempt) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_toolkit_model(architecture).to(device)
    loss_logit_clip = float(architecture["loss_logit_clip"])
    matrix_parameters = [parameter for parameter in model.parameters()
                         if parameter.requires_grad and parameter.ndim > 1]
    other_parameters = [parameter for parameter in model.parameters()
                        if parameter.requires_grad and parameter.ndim <= 1]
    # NAdam's coupled decay is the gradient of wd/2 * sum(W**2).
    # Biases and other scalar/vector parameters are deliberately not penalized.
    optimizer = torch.optim.NAdam(
        [{"params": matrix_parameters, "weight_decay": weight_decay},
         {"params": other_parameters, "weight_decay": 0.0}],
        lr=float(architecture["learning_rate"]),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_factor ** (1.0 / decay_steps)
    )
    rng = np.random.default_rng(seed)
    signal_indices = np.flatnonzero((cache["signal"].splits == FIT_SPLIT)
                                    & (cache["signal"].weights != 0.0))
    background_indices = np.flatnonzero((cache["background"].splits == FIT_SPLIT)
                                        & (cache["background"].weights != 0.0))
    half_batch = batch_size // 2
    if half_batch < 1 or batch_size % 2:
        raise ValueError("batch_size must be even and at least two")
    if not signal_indices.size or not background_indices.size:
        raise ValueError("each class must have nonzero-weight fit events")
    epoch_steps = (math.ceil(max(signal_indices.size, background_indices.size) / half_batch)
                   if steps_per_epoch is None else steps_per_epoch)
    if epoch_steps < 1 or maximum_epochs < 1 or patience < 1:
        raise ValueError("epoch steps, maximum epochs and validation patience must be positive")
    planned_steps = int(maximum_epochs * epoch_steps)
    validation_interval = min(validation_cap, max(1, math.ceil(
        min(signal_indices.size, background_indices.size) / half_batch)))
    signal_cumulative, signal_coefficient_scale = _absolute_weight_sampling_plan(
        cache["signal"].weights, signal_indices, class_name="signal")
    background_cumulative, background_coefficient_scale = _absolute_weight_sampling_plan(
        cache["background"].weights, background_indices, class_name="background")

    def regularization_value() -> float:
        if weight_decay == 0.0 or not matrix_parameters:
            return 0.0
        with torch.no_grad():
            squared_norm = sum(torch.sum(parameter.detach() ** 2)
                               for parameter in matrix_parameters)
            return 0.5 * weight_decay * float(squared_norm.item())

    def evaluate() -> tuple[float, float, float, float]:
        result = _evaluate_member(
            model, cache, split_code=VALIDATION_SPLIT, mean=mean, scale=scale,
            device=device, inference_batch_size=inference_batch_size,
            logit_clip=loss_logit_clip)
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f"member {slot} produced non-finite validation diagnostics")
        return result

    def snapshot() -> dict[str, Any]:
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    initial_validation_loss, _, _, _ = evaluate()
    best_loss = initial_validation_loss
    patience_reference_loss = initial_validation_loss
    best_state = snapshot()  # The actual initialized network, never a fabricated unity ratio.
    best_step = 0
    checks_without_improvement = 0
    history = {
        "optimizer_step": [0], "epoch": [0.0], "training_loss": [None],
        "validation_loss": [initial_validation_loss],
        "regularization_loss": [regularization_value()], "objective_loss": [None],
        "learning_rate": [float(optimizer.param_groups[0]["lr"])],
    }
    print(f"member={slot} attempt={attempt} step=0/{planned_steps} epoch=0 "
          f"validation={initial_validation_loss:.7g} baseline={math.log(2.0):.7g} "
          f"validation_interval_steps={validation_interval}", flush=True)
    accumulated_bce = accumulated_regularization = 0.0
    accumulated_steps = 0
    completed_steps = 0
    stop_reason = "maximum_optimizer_steps"
    for step in range(1, planned_steps + 1):
        model.train()
        signal_draw = _draw_absolute_weighted(
            rng, signal_indices, signal_cumulative, size=half_batch)
        background_draw = _draw_absolute_weighted(
            rng, background_indices, background_cumulative, size=half_batch)
        signal_draw.sort()
        background_draw.sort()
        signal_tensor = _scaled_tensor(
            np.asarray(cache["signal"].features[signal_draw]), mean, scale, device)
        background_tensor = _scaled_tensor(
            np.asarray(cache["background"].features[background_draw]), mean, scale, device)
        signal_coefficients = torch.from_numpy(np.asarray(
            np.sign(cache["signal"].weights[signal_draw]) * signal_coefficient_scale,
            dtype=np.float32)).to(device)
        background_coefficients = torch.from_numpy(np.asarray(
            np.sign(cache["background"].weights[background_draw]) * background_coefficient_scale,
            dtype=np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        # Preserve the signed empirical BCE and its existing finite logit domain.
        signal_logits = torch.clamp(model(signal_tensor).reshape(-1),
                                     min=-loss_logit_clip, max=loss_logit_clip)
        background_logits = torch.clamp(model(background_tensor).reshape(-1),
                                         min=-loss_logit_clip, max=loss_logit_clip)
        signal_loss = functional.binary_cross_entropy_with_logits(
            signal_logits, torch.ones_like(signal_logits), reduction="none")
        background_loss = functional.binary_cross_entropy_with_logits(
            background_logits, torch.zeros_like(background_logits), reduction="none")
        loss = 0.5 * (torch.mean(signal_loss * signal_coefficients)
                      + torch.mean(background_loss * background_coefficients))
        regularization = regularization_value()
        if not torch.isfinite(loss) or not math.isfinite(regularization):
            raise RuntimeError(f"member {slot} produced a non-finite training objective")
        # Coupled L2 is applied by NAdam; do not add its gradient a second time.
        loss.backward()
        optimizer.step()
        scheduler.step()
        accumulated_bce += float(loss.detach().cpu())
        accumulated_regularization += regularization
        accumulated_steps += 1
        completed_steps = step
        if step != 1 and step % validation_interval != 0 and step != planned_steps:
            continue

        validation_loss, _, _, _ = evaluate()
        training_loss = accumulated_bce / accumulated_steps
        regularization_loss = accumulated_regularization / accumulated_steps
        history["optimizer_step"].append(step)
        history["epoch"].append(step / epoch_steps)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        history["regularization_loss"].append(regularization_loss)
        history["objective_loss"].append(training_loss + regularization_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        print(f"member={slot} attempt={attempt} step={step}/{planned_steps} "
              f"epoch={step / epoch_steps:.6g} train={training_loss:.7g} "
              f"regularization={regularization_loss:.7g} validation={validation_loss:.7g} "
              f"lr={history['learning_rate'][-1]:.3g}", flush=True)
        # Always retain the true best checkpoint. A separate reference makes
        # patience insensitive to a series of numerically tiny improvements.
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = snapshot()
            best_step = step
        if validation_loss < patience_reference_loss - minimum_delta:
            patience_reference_loss = validation_loss
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        accumulated_bce = accumulated_regularization = 0.0
        accumulated_steps = 0
        if checks_without_improvement >= patience:
            stop_reason = "validation_patience"
            break

    model.load_state_dict(best_state)
    model.to(device)
    validation_loss, score_mean, score_std, saturated_fraction = evaluate()
    baseline = math.log(2.0)
    summary = {
        "initial_validation_loss": initial_validation_loss,
        "constant_baseline_loss": baseline,
        "selected_optimizer_step": best_step,
        "selected_epoch": best_step / epoch_steps,
        "selected_validation_loss": validation_loss,
        "stop_reason": stop_reason,
        "optimizer_steps_completed": completed_steps,
        "planned_optimizer_steps": planned_steps,
        "epoch_steps": int(epoch_steps),
        "effective_validation_interval_steps": validation_interval,
        "requested_validation_interval_steps": validation_cap,
        "validation_min_delta": minimum_delta,
        "baseline_tolerance": baseline_tolerance,
        "improved_over_constant": bool(validation_loss < baseline - minimum_delta),
        "within_baseline_tolerance": bool(validation_loss <= baseline + baseline_tolerance),
        "initial_checkpoint_selected": best_step == 0,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                     if parameter.requires_grad),
        "target_fit_entries": int(signal_indices.size),
        "reference_fit_entries": int(background_indices.size),
        "target_draws_per_fit_event": completed_steps * half_batch / signal_indices.size,
        "reference_draws_per_fit_event": completed_steps * half_batch / background_indices.size,
        "draw_reuse_definition": "mean draws per nonzero fit row; individual reuse follows absolute weights",
        "weight_decay": weight_decay,
        "regularization_definition": "weight_decay / 2 * sum(matrix weights squared); biases excluded",
        "learning_rate_decay_steps": decay_steps,
        "learning_rate_history": "rate for the next optimizer update after the recorded completed step",
        "training_loss_history": "mean sampled signed BCE since previous validation; unavailable at step zero",
        "patience_validation_checks": patience,
    }
    print(f"member={slot} selected_step={best_step} validation={validation_loss:.7g} "
          f"improved_over_constant={summary['improved_over_constant']} "
          f"within_baseline_tolerance={summary['within_baseline_tolerance']} "
          f"stop_reason={stop_reason}", flush=True)
    return MemberResult(
        slot=slot, attempt=attempt, seed=seed, state_dict=best_state, history=history,
        validation_loss=validation_loss, validation_score_mean=score_mean,
        validation_score_std=score_std, validation_saturated_fraction=saturated_fraction,
        training_summary=summary,
    )


def _bad_members(results: list[MemberResult]) -> list[int]:
    losses = np.asarray([item.validation_loss for item in results], dtype=np.float64)
    finite_losses = losses[np.isfinite(losses)]
    median = float(np.median(finite_losses)) if finite_losses.size else 0.0
    mad = float(np.median(np.abs(finite_losses - median))) if finite_losses.size else 0.0
    degradation = np.abs(losses - median)
    relative_threshold = 0.05 * max(abs(median), 1.0e-8)
    mad_threshold = 5.0 * max(mad, 1.0e-8)
    bad: list[int] = []
    for index, item in enumerate(results):
        tolerance = float(item.training_summary.get("baseline_tolerance", 0.01))
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("baseline_tolerance must be finite and nonnegative")
        invalid = not np.isfinite(item.validation_loss)
        worse_than_baseline = item.validation_loss > math.log(2.0) + tolerance
        relative_outlier = degradation[index] > relative_threshold
        mad_outlier = degradation[index] > mad_threshold
        # Signed losses may be negative. Keep the existing two-sided relative
        # check, plus an absolute check that catches an equally poor ensemble.
        if invalid or worse_than_baseline or (relative_outlier and mad_outlier):
            bad.append(index)
    return bad


def _ensemble_validation_summary(
    predictions: dict[str, ClassPredictions],
    cache: dict[str, CachedClass],
    *,
    logit_clip: float,
    baseline_tolerance: float,
    validation_min_delta: float,
) -> dict[str, Any]:
    """Check the actual mean-score ensemble against the constant classifier.

    This is an acceptance diagnostic, never a calibration or ratio rescaling.
    Signed empirical risk need not preserve a convex-ensemble inequality, so
    checking each member alone is insufficient.
    """
    for name, value in (("baseline_tolerance", baseline_tolerance),
                        ("validation_min_delta", validation_min_delta)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    class_losses = []
    for name, label in (("signal", 1), ("background", 0)):
        prediction = predictions[name]
        scores = np.asarray(prediction.ensemble_score, dtype=np.float64)
        weights = np.asarray(cache[name].weights[prediction.indices], dtype=np.float64)
        if (scores.shape != weights.shape or np.any(~np.isfinite(scores))
                or np.any((scores < 0.0) | (scores > 1.0))
                or np.any(~np.isfinite(weights))):
            raise ValueError("ensemble validation predictions/weights must be finite and aligned")
        total = float(np.sum(weights, dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("ensemble validation class must have positive net signed yield")
        logits = np.clip(stable_logit(scores, logit_clip=logit_clip),
                         -abs(logit_clip), abs(logit_clip))
        loss = np.logaddexp(0.0, -logits if label else logits)
        class_losses.append(float(np.sum(weights * loss, dtype=np.float64) / total))
    validation_loss = 0.5 * sum(class_losses)
    baseline = math.log(2.0)
    if not np.isfinite(validation_loss) or validation_loss > baseline + baseline_tolerance:
        raise RuntimeError(
            "raw mean-score ensemble fails the constant-classifier validation baseline: "
            f"signed BCE={validation_loss:.8g}, limit={baseline + baseline_tolerance:.8g}. "
            "Inspect data reuse and training histories or reduce capacity/learning rate; "
            "no output calibration or unity fallback was applied.")
    improved = validation_loss < baseline - validation_min_delta
    if not improved:
        warnings.warn(
            "The raw mean-score ensemble is within the constant-classifier tolerance but "
            "does not meaningfully improve its validation BCE. This can indicate a "
            "near-unity ratio or insufficient learned discrimination; inspect held-out "
            "reliability and closure. No unity ratio was substituted.",
            RuntimeWarning, stacklevel=2)
    return {
        "validation_loss": validation_loss, "constant_baseline_loss": baseline,
        "baseline_tolerance": baseline_tolerance, "validation_min_delta": validation_min_delta,
        "improved_over_constant": bool(improved), "within_baseline_tolerance": True,
    }


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
                    logits = torch.clamp(
                        model(tensor).reshape(-1),
                        min=-float(architecture["loss_logit_clip"]),
                        max=float(architecture["loss_logit_clip"]),
                    )
                    predictions[start:stop, member_index] = (
                        torch.sigmoid(logits)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            output[class_name] = ClassPredictions(indices, predictions)
    return output


def _install_artifact(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
    expected_model_kind: str,
) -> None:
    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output directory exists: {destination}; pass --overwrite to replace it"
        )
    if destination.exists():
        _validate_existing_artifact(
            destination, expected_model_kind=expected_model_kind
        )
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
        try:
            shutil.rmtree(backup)
        except OSError as error:
            warnings.warn(
                f"new artifact was published, but its old backup could not be "
                f"removed: {backup}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


def _validate_existing_artifact(path: Path, *, expected_model_kind: str) -> None:
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
    if manifest.get("format_version") not in SUPPORTED_ARTIFACT_FORMAT_VERSIONS:
        raise ValueError(
            f"refusing to overwrite model artifact with unsupported format: {path}"
        )
    if manifest.get("model_kind") != expected_model_kind:
        raise ValueError(
            "refusing to overwrite an artifact of a different model kind: "
            f"{manifest.get('model_kind')!r} != {expected_model_kind!r}"
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
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and nonnegative")
    for name in ("validation_interval_steps", "learning_rate_decay_steps"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(args.validation_min_delta) or args.validation_min_delta < 0.0:
        raise ValueError("validation minimum improvement must be finite and nonnegative")
    if not np.isfinite(args.baseline_tolerance) or args.baseline_tolerance < 0.0:
        raise ValueError("baseline tolerance must be finite and nonnegative")
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
    if args.inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError("steps per epoch must be positive when supplied")
    input_provenance = {
        "gg_H_pythia": capture_file_provenance(
            gg_h_path, include_sha256=not args.skip_input_checksums
        ),
        "ZZ_pythia": capture_file_provenance(
            zz_path, include_sha256=not args.skip_input_checksums
        ),
    }
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

    correction_bundle = ModelBundle.load(
        args.correction_model_dir,
        device_name=device_name,
        expected_model_kind=MODEL_KIND_DATA_MC_CORRECTION,
    )
    validate_manifest_mass_window(
        correction_bundle.manifest,
        CORRECTION_MASS_WINDOW,
        artifact_label="Correction artifact",
    )
    correction_luminosity = float(correction_bundle.manifest["luminosity_fb"])
    if not np.isclose(
        correction_luminosity,
        args.expected_luminosity_fb,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "Correction model luminosity does not match Training: "
            f"{correction_luminosity:g} != {args.expected_luminosity_fb:g} fb^-1"
        )
    training_weight_contract = weight_measure_contract(
        args.weight_branch, luminosity_fb=args.expected_luminosity_fb
    )
    correction_weight_branch = correction_bundle.manifest.get("weight_branch")
    if not isinstance(correction_weight_branch, str) or not correction_weight_branch:
        raise ValueError("Correction artifact has no valid weight-branch contract")
    correction_weight_contract = weight_measure_contract(
        correction_weight_branch, luminosity_fb=correction_luminosity
    )
    recorded_correction_contract = correction_bundle.manifest.get("weight_measure")
    if not isinstance(recorded_correction_contract, dict) or (
        recorded_correction_contract != correction_weight_contract
    ):
        raise ValueError("Correction artifact has an inconsistent weight-measure contract")
    weight_measures_compatible = known_weight_measures_compatible(
        training_weight_contract, correction_weight_contract
    )
    weight_measure_override = bool(
        getattr(args, "allow_weight_measure_mismatch", False)
    )
    if not weight_measures_compatible and not weight_measure_override:
        raise ValueError(
            "Training and Correction use a custom or unknown weight-measure "
            "combination. Use the known-compatible weight/weight_nominal_pb pair, "
            "or pass --allow-weight-measure-mismatch after verifying the semantics."
        )
    correction_manifest_path = correction_bundle.root / "manifest.json"
    correction_manifest_sha256 = sha256_file(correction_manifest_path)
    correction_inputs = correction_bundle.manifest.get("inputs")
    correction_input_matches: dict[str, bool] = {}
    for name in ("gg_H_pythia", "ZZ_pythia"):
        expected = (
            correction_inputs.get(name, {}).get("sha256")
            if isinstance(correction_inputs, dict)
            and isinstance(correction_inputs.get(name), dict)
            else None
        )
        observed = input_provenance[name]["sha256"]
        if expected is not None and observed is not None and expected != observed:
            raise ValueError(
                f"{name} does not match the input used to derive the Correction model"
            )
        correction_input_matches[name] = bool(
            expected is not None and observed is not None and expected == observed
        )

    output_directory = args.output_dir.expanduser().resolve()
    if output_directory in {gg_h_path, zz_path, correction_bundle.root}:
        raise ValueError("model output directory cannot be a training input or model")
    if output_directory in gg_h_path.parents or output_directory in zz_path.parents:
        raise ValueError("model output directory cannot contain a training input")
    if (
        output_directory in correction_bundle.root.parents
        or correction_bundle.root in output_directory.parents
    ):
        raise ValueError(
            "Training output and Correction model directories cannot contain one another"
        )
    if output_directory.exists() and not args.overwrite:
        raise FileExistsError(
            f"output directory exists: {output_directory}; pass --overwrite to replace it"
        )
    if output_directory.exists():
        _validate_existing_artifact(
            output_directory,
            expected_model_kind=MODEL_KIND_BACKGROUND_REMOVAL,
        )
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
        correction_bundle=correction_bundle,
        inference_batch_size=args.inference_batch_size,
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
                correction_bundle=correction_bundle,
                inference_batch_size=args.inference_batch_size,
            )
            assert_file_unchanged(gg_h_path, input_provenance["gg_H_pythia"])
            assert_file_unchanged(zz_path, input_provenance["ZZ_pythia"])
            cache = _load_cache(cache_paths)
            effective_steps_per_epoch = (
                args.steps_per_epoch
                if args.steps_per_epoch is not None
                else math.ceil(
                    max(
                        np.count_nonzero(cache["signal"].splits == FIT_SPLIT),
                        np.count_nonzero(cache["background"].splits == FIT_SPLIT),
                    )
                    / (args.batch_size // 2)
                )
            )
            architecture = {
                "implementation": "nsbi_common_utils.lightning_tools.DensityRatioLightning",
                "hidden_layers": args.hidden_layers,
                "neurons": args.neurons,
                "activation": "SiLU",
                "output": "logit",
                "loss": "balanced signed-weight logit-clipped BCEWithLogitsLoss",
                "loss_logit_clip": args.logit_clip,
                "class_risk": "sum(w * BCE) / sum(w)",
                "sampling": "with_replacement_proportional_to_absolute_weight",
                "sampling_coefficient": "sign(w) * sum(abs(w)) / sum(w)",
                "dropout": 0.0,
                "weight_decay": args.weight_decay,
                "regularization": "coupled L2 on weight matrices; biases excluded",
                "optimizer": "NAdam",
                "learning_rate": args.learning_rate,
                "learning_rate_schedule": "exponential_per_optimizer_step",
                "learning_rate_decay": args.learning_rate_decay,
                "learning_rate_decay_steps": args.learning_rate_decay_steps,
                "validation_interval_steps": args.validation_interval_steps,
                "validation_interval_policy": (
                    "min(requested, ceil(smaller nonzero fit class / half batch))"
                ),
                "validation_min_delta": args.validation_min_delta,
                "baseline_tolerance": args.baseline_tolerance,
                "validation_loss": "unpenalized balanced signed-weight BCE",
                "checkpoint_policy": "best validation loss including untrained step zero",
                "batch_size": args.batch_size,
                "steps_per_epoch": effective_steps_per_epoch,
                "steps_per_epoch_policy": (
                    "explicit_override"
                    if args.steps_per_epoch is not None
                    else "full_fit_split_coverage"
                ),
                "maximum_epochs": args.maximum_epochs,
                "early_stopping_patience": args.patience,
                "early_stopping_patience_unit": "validation_checks",
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
                    f"Retrying ensemble slots {bad} that failed baseline or "
                    f"ensemble-quality checks (round {retry_round})...",
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
                    f"ensemble slots {remaining_bad} fail baseline or "
                    f"ensemble-quality checks after "
                    f"{args.max_retries} retries"
                )

            print("Predicting the held-out validation bank...", flush=True)
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
            ensemble_validation = _ensemble_validation_summary(
                validation,
                cache,
                logit_clip=args.logit_clip,
                baseline_tolerance=args.baseline_tolerance,
                validation_min_delta=args.validation_min_delta,
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
                        "training_summary": result.training_summary,
                    }
                )

            manifest: dict[str, Any] = {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "training_policy_version": 1,
                "model_kind": MODEL_KIND_BACKGROUND_REMOVAL,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "tree_name": TREE_NAME,
                "features": list(FEATURES),
                "model_features": list(MODEL_FEATURES),
                "preprocessing": {
                    "periodic_variables": list(FEATURES[:2]),
                    "periodic_encoding": "identity",
                    "remaining_variables": "identity",
                },
                "selections": {
                    "mass_window_gev": {
                        "branch": MASS_BRANCH,
                        "low_exclusive": ANALYSIS_MASS_WINDOW[0],
                        "high_exclusive": ANALYSIS_MASS_WINDOW[1],
                    },
                    "positive_label_1": (
                        "gg_H_pythia: reconstructed && fiducial && "
                        "115 < reco_m_ZZ < 130 GeV"
                    ),
                    "negative_label_0": (
                        "Correction-weighted ZZ_pythia: reconstructed && "
                        "115 < reco_m_ZZ < 130 GeV; plus gg_H_pythia: "
                        "reconstructed && !fiducial && 115 < reco_m_ZZ < 130 GeV"
                    ),
                },
                "weight_branch": args.weight_branch,
                "weight_measure": training_weight_contract,
                "signed_weight_policy": {
                    "event_weights": "finite nonzero weights retain their sign",
                    "class_normalization": "net signed sum",
                    "yield_central_values": "sum(w)",
                    "statistical_variances": "sum(w^2)",
                    "zero_weight_events": "omitted from cache and counted",
                },
                "luminosity_fb": args.expected_luminosity_fb,
                "yields": {
                    "signal": signal_yield,
                    "background": background_yield,
                    "signal_to_background": yield_ratio,
                    "target": signal_yield,
                    "reference": background_yield,
                    "target_to_reference": yield_ratio,
                    "units": training_weight_contract["units"],
                },
                "ratio_convention": {
                    "shape": "p(signal)/p(background) = score/(1-score)",
                    "physical_odds": "(signal_yield/background_yield) * shape",
                    "background_removal_weight": "physical_odds/(1+physical_odds)",
                    "orientation": "target_to_reference",
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
                    "type": "balanced_absolute_weight_standard",
                    "fit_split_only": True,
                    "weight_semantics": (
                        "absolute weights are used only for numerical preprocessing"
                    ),
                    "mean": scaler_mean.tolist(),
                    "scale": scaler_scale.tolist(),
                },
                "architecture": architecture,
                "ensemble": {
                    "aggregation": "arithmetic_mean_probability",
                    "size": len(results),
                    "minimum_size": 4,
                    "outlier_rule": (
                        "validation loss is nonfinite, exceeds log(2) plus "
                        "baseline_tolerance, or differs from the median by "
                        "both >5% and >5 MAD"
                    ),
                    "baseline_tolerance": args.baseline_tolerance,
                    "maximum_retries_per_slot": args.max_retries,
                },
                "ensemble_validation": ensemble_validation,
                "members": member_entries,
                "ratio_estimator": dict(RAW_RATIO_ESTIMATOR),
                "toolkit": {
                    "repository": "https://github.com/iris-hep/nsbi-lhc-toolkit",
                    **toolkit_provenance,
                    "model_class": (
                        "nsbi_common_utils.lightning_tools.density_ratio_model."
                        "DensityRatioLightning"
                    ),
                },
                "correction_model": {
                    "path": str(correction_bundle.root),
                    "manifest_sha256": correction_manifest_sha256,
                    "created_utc": correction_bundle.manifest.get("created_utc"),
                    "model_kind": correction_bundle.manifest.get("model_kind"),
                    "weight_branch": correction_weight_branch,
                    "weight_measure": correction_weight_contract,
                    "weight_measures_known_compatible": weight_measures_compatible,
                    "weight_measure_override": (
                        weight_measure_override and not weight_measures_compatible
                    ),
                    "sideband_mass_window_gev": correction_bundle.manifest.get(
                        "selections", {}
                    ).get("mass_window_gev"),
                    "applied_to": "ZZ_pythia only, using full yield-times-shape ratio",
                    "input_checksum_matches": correction_input_matches,
                    "all_input_checksums_verified": all(
                        correction_input_matches.values()
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
            temporary_artifact,
            output_directory,
            overwrite=args.overwrite,
            expected_model_kind=MODEL_KIND_BACKGROUND_REMOVAL,
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
            "Train a raw-odds NSBI-toolkit BCE ensemble for reconstructed "
            "fiducial ggH versus reconstructed contamination"
        )
    )
    parser.add_argument("--gg-h-root", type=Path, required=True)
    parser.add_argument("--zz-root", type=Path, required=True)
    parser.add_argument(
        "--correction-model-dir",
        type=Path,
        required=True,
        help=(
            "data/MC sideband Correction artifact; its full physical ratio is "
            "multiplied into every selected ZZ event"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-branch", default=DEFAULT_WEIGHT_BRANCH)
    parser.add_argument(
        "--allow-weight-measure-mismatch",
        action="store_true",
        help=(
            "allow a custom/unknown Training and Correction weight-measure pair; "
            "the standard weight and weight_nominal_pb pair needs no override"
        ),
    )
    parser.add_argument(
        "--expected-luminosity-fb", type=float, default=DEFAULT_LUMINOSITY_FB
    )
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12,
                        help="Validation checks without a meaningful improvement")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--learning-rate-decay", type=float, default=0.98)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4,
                        help="Coupled L2 on weight matrices; validation uses unpenalized signed BCE")
    parser.add_argument("--validation-interval-steps", type=int, default=25,
                        help="Maximum optimizer steps between validation checks")
    parser.add_argument("--learning-rate-decay-steps", type=int, default=100,
                        help="Optimizer steps per learning-rate decay factor")
    parser.add_argument("--validation-min-delta", type=float, default=1.0e-5,
                        help="Minimum validation improvement that resets patience")
    parser.add_argument("--baseline-tolerance", type=float, default=0.01,
                        help="Maximum accepted validation BCE excess above log(2)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED + 1)
    parser.add_argument("--logit-clip", type=float, default=30.0)
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
