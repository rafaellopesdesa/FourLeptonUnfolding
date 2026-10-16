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
    stable_sigmoid,
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
            # the physics loss, yields, calibration, and diagnostics.
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

    seed = (base_seed + 1009 * slot + 100_003 * attempt) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_toolkit_model(architecture).to(device)
    loss_logit_clip = float(architecture["loss_logit_clip"])
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
    signal_cumulative, signal_coefficient_scale = _absolute_weight_sampling_plan(
        cache["signal"].weights,
        signal_indices,
        class_name="signal",
    )
    background_cumulative, background_coefficient_scale = (
        _absolute_weight_sampling_plan(
            cache["background"].weights,
            background_indices,
            class_name="background",
        )
    )

    history = {"training_loss": [], "validation_loss": [], "learning_rate": []}
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(maximum_epochs):
        model.train()
        accumulated = 0.0
        for _ in range(epoch_steps):
            signal_draw = _draw_absolute_weighted(
                rng,
                signal_indices,
                signal_cumulative,
                size=half_batch,
            )
            background_draw = _draw_absolute_weighted(
                rng,
                background_indices,
                background_cumulative,
                size=half_batch,
            )
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
                    np.sign(cache["signal"].weights[signal_draw])
                    * signal_coefficient_scale,
                    dtype=np.float32,
                )
            ).to(device)
            background_coefficients = torch.from_numpy(
                np.asarray(
                    np.sign(cache["background"].weights[background_draw])
                    * background_coefficient_scale,
                    dtype=np.float32,
                )
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            # A finite logit domain is essential for a signed empirical BCE:
            # without it, a negative-weight event can drive its contribution
            # to minus infinity.  Clipping bounds the objective without
            # changing, dropping, or taking the absolute value of any event
            # weight.  The same bound is used for validation and inference.
            signal_logits = torch.clamp(
                model(signal_tensor).reshape(-1),
                min=-loss_logit_clip,
                max=loss_logit_clip,
            )
            background_logits = torch.clamp(
                model(background_tensor).reshape(-1),
                min=-loss_logit_clip,
                max=loss_logit_clip,
            )
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
            logit_clip=loss_logit_clip,
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
        logit_clip=loss_logit_clip,
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
    # Signed BCE values can legitimately be negative.  Either tail can signal
    # a discrepant member: in particular, an anomalously *low* signed loss can
    # indicate that subtractive events have driven the member to the bounded
    # logit edge.  Use a two-sided additive deviation from the robust centre.
    degradation = np.abs(losses - median)
    relative_threshold = 0.05 * max(abs(median), 1.0e-8)
    mad_threshold = 5.0 * max(mad, 1.0e-8)
    bad: list[int] = []
    for index, item in enumerate(results):
        invalid = not np.isfinite(item.validation_loss)
        relative_outlier = degradation[index] > relative_threshold
        mad_outlier = degradation[index] > mad_threshold
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
    signal_sum = float(np.sum(signal_weights, dtype=np.float64))
    background_sum = float(np.sum(background_weights, dtype=np.float64))
    if not np.isfinite(signal_sum) or signal_sum <= 0.0:
        raise ValueError("signal calibration split has non-positive net signed yield")
    if not np.isfinite(background_sum) or background_sum <= 0.0:
        raise ValueError("background calibration split has non-positive net signed yield")
    weights = np.concatenate(
        [
            0.5 * signal_weights / signal_sum,
            0.5 * background_weights / background_sum,
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
    maximum_iterations: int = 1000,
) -> tuple[float, float]:
    """Fit a bounded affine-logit calibrator with signed weighted BCE.

    With signed weights the BCE Hessian need not be positive semidefinite, so
    the Newton solve appropriate for nonnegative weights is not reliable.  We
    instead use an absolute-weight Fisher matrix only as a local
    preconditioner, with projected Armijo steps evaluated against the actual
    signed objective.  A deterministic box-spanning multistart is used for
    signed samples because their objective can have saddles and distinct local
    minima.  This retains every weight sign while avoiding both saddle-point
    acceptance and the very slow global step size that broad classifier logits
    would otherwise impose.
    """

    logits = stable_logit(scores, logit_clip=logit_clip)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if logits.shape != labels.shape or labels.shape != weights.shape:
        raise ValueError("calibration arrays must have identical shapes")
    if logits.size == 0:
        raise ValueError("calibration arrays cannot be empty")
    if not (
        np.all(np.isfinite(logits))
        and np.all(np.isfinite(labels))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("calibration arrays must be finite")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("calibration labels must be zero or one")
    if float(np.sum(weights, dtype=np.float64)) <= 0.0:
        raise ValueError("calibration weights must have positive net signed sum")
    if not (0.0 < minimum_scale <= 1.0 <= maximum_scale):
        raise ValueError("calibration scale bounds must be positive and contain one")
    if maximum_iterations < 1:
        raise ValueError("maximum calibration iterations must be positive")
    design = np.column_stack([logits, np.ones_like(logits)])
    lower_bounds = np.array([minimum_scale, -logit_clip], dtype=np.float64)
    upper_bounds = np.array([maximum_scale, logit_clip], dtype=np.float64)
    gradient_tolerance = 1.0e-7

    def objective_and_gradient(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        prediction = design @ candidate
        objective = float(
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
        probability = stable_sigmoid(prediction)
        gradient = design.T @ (weights * (probability - labels))
        return objective, gradient

    def optimize_from(initial: np.ndarray) -> tuple[np.ndarray, float, float]:
        parameters = np.clip(
            np.asarray(initial, dtype=np.float64), lower_bounds, upper_bounds
        )
        for _ in range(maximum_iterations):
            prediction = design @ parameters
            probability = stable_sigmoid(prediction)
            old_objective, gradient = objective_and_gradient(parameters)

            # This projected-gradient mapping is the KKT stationarity measure
            # for the box and remains well behaved near a bound.  It avoids
            # relying on a brittle exact-distance active-set test.
            projected_gradient = parameters - np.clip(
                parameters - gradient, lower_bounds, upper_bounds
            )
            free = np.abs(projected_gradient) > gradient_tolerance
            if not np.any(free):
                break

            # Positive absolute-weight curvature is deliberately *not* the
            # objective Hessian; it is a well-conditioned metric for a descent
            # direction of the signed gradient.
            curvature = np.abs(weights) * probability * (1.0 - probability)
            fisher = design.T @ (design * curvature[:, np.newaxis])
            ridge = 1.0e-10 * max(float(np.trace(fisher)), 1.0)
            fisher.flat[::3] += ridge
            direction = np.zeros_like(parameters)
            try:
                direction[free] = -np.linalg.solve(
                    fisher[np.ix_(free, free)], gradient[free]
                )
            except np.linalg.LinAlgError:
                direction[free] = -gradient[free]

            # Check the local direction itself, not only a fully projected
            # unit step: clipping can make a unit step look descending even
            # when every sufficiently small step is ascending.
            if float(gradient @ direction) >= 0.0:
                diagonal = np.maximum(np.diag(fisher), ridge)
                direction.fill(0.0)
                direction[free] = -gradient[free] / diagonal[free]

            fraction = 1.0
            accepted = False
            while fraction >= 1.0e-12:
                candidate = np.clip(
                    parameters + fraction * direction,
                    lower_bounds,
                    upper_bounds,
                )
                displacement = candidate - parameters
                if float(np.max(np.abs(displacement))) < 1.0e-12:
                    break
                # Armijo descent remains valid for a non-convex signed
                # objective.
                candidate_objective, _ = objective_and_gradient(candidate)
                if candidate_objective <= old_objective + 1.0e-4 * float(
                    gradient @ displacement
                ) + 1.0e-14 * max(1.0, abs(old_objective)):
                    parameters = candidate
                    accepted = True
                    break
                fraction *= 0.5
            if not accepted:
                break

        if not np.all(np.isfinite(parameters)) or parameters[0] <= 0.0:
            return parameters, math.inf, math.inf
        final_objective, final_gradient = objective_and_gradient(parameters)
        final_projected_gradient = parameters - np.clip(
            parameters - final_gradient, lower_bounds, upper_bounds
        )
        maximum_projected_gradient = float(
            np.max(np.abs(final_projected_gradient))
        )
        return parameters, final_objective, maximum_projected_gradient

    # A nonnegative-weight logistic objective is convex, so the identity start
    # is sufficient.  Signed BCE is a difference of convex functions: a point
    # with zero gradient can be a saddle, and different starts can reach
    # distinct local minima.  Cover the signed parameter box deterministically
    # and retain the converged solution with the lowest *signed* objective.
    starts = [np.array([1.0, 0.0], dtype=np.float64)]
    if np.any(weights < 0.0):
        middle_scale = 0.5 * (minimum_scale + maximum_scale)
        for start_scale in (minimum_scale, middle_scale, maximum_scale):
            for start_bias in (-logit_clip, 0.0, logit_clip):
                candidate = np.array([start_scale, start_bias], dtype=np.float64)
                if not any(np.array_equal(candidate, start) for start in starts):
                    starts.append(candidate)

    solutions = [optimize_from(start) for start in starts]
    converged = [
        solution
        for solution in solutions
        if np.isfinite(solution[1])
        and solution[2] <= gradient_tolerance
    ]
    if not converged:
        best_projected_gradient = min(solution[2] for solution in solutions)
        raise RuntimeError(
            "signed affine-logit calibration did not converge from any "
            "deterministic start; minimum maximum projected gradient="
            f"{best_projected_gradient:.6g}"
        )
    parameters, _, _ = min(converged, key=lambda solution: solution[1])
    return float(parameters[0]), float(parameters[1])


def _normalization_bias(
    background_scores: np.ndarray,
    background_weights: np.ndarray,
    *,
    scale: float,
    initial_bias: float,
    logit_clip: float,
) -> tuple[float, float]:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization scale must be finite and positive")
    if not np.isfinite(initial_bias):
        raise ValueError("normalization initial bias must be finite")
    if not np.isfinite(logit_clip) or not (0.0 < logit_clip <= 100.0):
        raise ValueError("normalization logit clip must be finite and lie in (0, 100]")
    logits = stable_logit(background_scores, logit_clip=logit_clip)
    scaled_logits = scale * logits
    if not np.all(np.isfinite(scaled_logits)):
        raise RuntimeError("calibration logits overflow during ratio normalization")

    background_weights = np.asarray(background_weights, dtype=np.float64)
    if background_weights.shape != logits.shape:
        raise ValueError("normalization scores and weights must have identical shapes")
    if not np.all(np.isfinite(background_weights)):
        raise ValueError("normalization weights must be finite")
    absolute_sum = float(np.sum(np.abs(background_weights), dtype=np.float64))
    if not np.isfinite(absolute_sum) or absolute_sum <= 0.0:
        raise ValueError("normalization weights contain no nonzero measure")
    normalized_weights = background_weights / absolute_sum
    net_fraction = float(np.sum(normalized_weights, dtype=np.float64))
    if not np.isfinite(net_fraction) or net_fraction <= 0.0:
        raise ValueError("normalization weights have non-positive net signed yield")

    # A density-ratio normalization is one common positive multiplier.  Work
    # in a shifted exponential basis so that signed cancellation is evaluated
    # without overflow.  If the signed moment is non-positive, no such
    # multiplier exists; finding a crossing by clipping different events at
    # different bounds would manufacture a shape-changing "normalization".
    maximum_logit = float(np.max(scaled_logits))
    shifted_ratios = np.exp(scaled_logits - maximum_logit)
    signed_shifted_moment = float(
        np.sum(normalized_weights * shifted_ratios, dtype=np.float64)
    )
    absolute_shifted_moment = float(
        np.sum(np.abs(normalized_weights) * shifted_ratios, dtype=np.float64)
    )
    if not np.isfinite(absolute_shifted_moment) or absolute_shifted_moment <= 0.0:
        raise RuntimeError("ratio normalization has no finite weighted measure")
    cancellation_ratio = signed_shifted_moment / absolute_shifted_moment
    if not np.isfinite(cancellation_ratio) or cancellation_ratio <= 1.0e-12:
        raise RuntimeError(
            "the signed reference ratio moment is non-positive or numerically "
            "cancelled; no stable positive density normalization exists"
        )

    log_mean_ratio = (
        maximum_logit
        + math.log(signed_shifted_moment)
        - math.log(net_fraction)
    )
    normalized_bias = -log_mean_ratio
    before_log = log_mean_ratio + initial_bias
    try:
        before = math.exp(before_log)
    except OverflowError as error:
        raise RuntimeError(
            "the pre-normalization signed ratio moment overflows"
        ) from error
    if not np.isfinite(before) or before <= 0.0:
        raise RuntimeError("the pre-normalization signed ratio moment is invalid")

    normalized_logits = scaled_logits + normalized_bias
    clip_tolerance = 1.0e-12 * max(1.0, abs(logit_clip))
    if np.any(normalized_logits < -logit_clip - clip_tolerance) or np.any(
        normalized_logits > logit_clip + clip_tolerance
    ):
        raise RuntimeError(
            "density normalization would require clipping calibrated ratios and "
            "therefore change their shape; increase --logit-clip or inspect the "
            "signed-weight cancellation"
        )
    normalized_ratios = np.exp(
        np.clip(normalized_logits, -logit_clip, logit_clip)
    )
    normalized_mean = float(
        np.sum(normalized_weights * normalized_ratios, dtype=np.float64)
        / net_fraction
    )
    if not np.isclose(normalized_mean, 1.0, rtol=1.0e-9, atol=1.0e-11):
        raise RuntimeError(
            "failed to normalize the calibrated density ratio: "
            f"E_B[r]={normalized_mean:.12g}"
        )
    return normalized_bias, before


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
                "weight_decay": 0.0,
                "optimizer": "NAdam",
                "learning_rate": args.learning_rate,
                "learning_rate_schedule": "ExponentialLR",
                "learning_rate_decay": args.learning_rate_decay,
                "batch_size": args.batch_size,
                "steps_per_epoch": effective_steps_per_epoch,
                "steps_per_epoch_policy": (
                    "explicit_override"
                    if args.steps_per_epoch is not None
                    else "full_fit_split_coverage"
                ),
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

            manifest: dict[str, Any] = {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "model_kind": MODEL_KIND_BACKGROUND_REMOVAL,
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
                        "validation loss is nonfinite, or differs from the median by "
                        "both >5% and >5 MAD"
                    ),
                    "maximum_retries_per_slot": args.max_retries,
                },
                "members": member_entries,
                "calibration": {
                    "method": (
                        "bounded_multistart_fisher_preconditioned_affine_logit_then_"
                        "shape_preserving_signed_density_normalization"
                    ),
                    "fit_split": "validation",
                    "weight_semantics": "signed class-balanced weights",
                    "scale": calibration_scale,
                    "scale_bounds": [
                        args.calibration_scale_min,
                        args.calibration_scale_max,
                    ],
                    "bias_before_normalization": calibration_bias,
                    "normalization_before": normalization_before,
                    "bias_after_normalization": normalized_bias,
                    "normalization_condition": (
                        "signed E_background[r_shape] = 1"
                    ),
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
            "Train a calibrated NSBI-toolkit BCE ensemble for reconstructed "
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
