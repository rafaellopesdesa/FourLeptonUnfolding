#!/usr/bin/env python3
"""Train the calibrated sideband data-to-Pythia correction ensemble."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
    CORRECTION_MASS_WINDOW,
    FEATURES,
    FIT_SPLIT,
    MASS_BRANCH,
    MODEL_FEATURES,
    MODEL_KIND_DATA_MC_CORRECTION,
    SPLIT_NAMES,
    TREE_NAME,
    VALIDATION_SPLIT,
    assert_file_unchanged,
    capture_file_provenance,
    deterministic_split,
    row_fingerprint_ids,
    sha256_file,
    toolkit_runtime_provenance,
    transformed_features,
    weight_measure_contract,
)
from BackgroundRemoval.Training.train_background_ratio import (
    CachePaths,
    CachedClass,
    MemberResult,
    SelectedChunk,
    WeightedStatistics,
    _bad_members,
    _balanced_calibration_arrays,
    _install_artifact,
    _load_cache,
    _normalization_bias,
    _predict_split,
    _train_member,
    fit_logit_calibrator,
)


DEFAULT_SEED = 19110908
DEFAULT_WEIGHT_BRANCH = "weight"
DEFAULT_LUMINOSITY_FB = 312.0

_TARGET_KEY = "signal"
_REFERENCE_KEY = "background"
_COMPONENT_DATA = "data_reconstructed_correction_sideband"
_COMPONENT_ZZ = "ZZ_reconstructed_correction_sideband"
_COMPONENT_GGH = "gg_H_reconstructed_correction_sideband"
_REFERENCE_COMPONENT_CODE = {_COMPONENT_ZZ: np.uint8(0), _COMPONENT_GGH: np.uint8(1)}


def _branch_names(tree: object) -> set[str]:
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def _selected_chunks(
    data_path: Path,
    gg_h_path: Path,
    zz_path: Path,
    *,
    weight_branch: str,
    split_seed: int,
    expected_luminosity_fb: float,
    step_size: str,
) -> Iterator[SelectedChunk]:
    """Yield positive-weight events in the strict correction sideband."""

    required = {
        *FEATURES,
        MASS_BRANCH,
        "event_id",
        "reconstructed",
        "luminosity_fb",
        weight_branch,
    }
    sources = (
        ("data", data_path, _TARGET_KEY, _COMPONENT_DATA),
        ("gg_H", gg_h_path, _REFERENCE_KEY, _COMPONENT_GGH),
        ("ZZ", zz_path, _REFERENCE_KEY, _COMPONENT_ZZ),
    )
    lower, upper = CORRECTION_MASS_WINDOW
    for source, path, class_name, component_name in sources:
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
                masses = np.asarray(arrays[MASS_BRANCH], dtype=np.float64)
                if np.any(reconstructed & ~np.isfinite(masses)):
                    raise ValueError(
                        f"{component_name} contains reconstructed events with non-finite "
                        f"{MASS_BRANCH}"
                    )
                selected = reconstructed & (masses > lower) & (masses < upper)
                all_weights = np.asarray(arrays[weight_branch], dtype=np.float64)
                selected_weights = all_weights[selected]
                if not np.all(np.isfinite(selected_weights)):
                    raise ValueError(
                        f"{component_name} contains non-finite selected {weight_branch} values"
                    )
                negative_count = int(np.count_nonzero(selected_weights < 0.0))
                if negative_count:
                    raise ValueError(
                        f"{component_name} contains {negative_count} selected negative "
                        f"{weight_branch} values. A signed weighted BCE is not a density "
                        "ratio and can be unbounded; this training intentionally stops "
                        "instead of taking absolute values or dropping negative events."
                    )
                raw = np.column_stack(
                    [np.asarray(arrays[name], dtype=np.float32) for name in FEATURES]
                )[selected]
                if not np.all(np.isfinite(raw)):
                    bad = int(np.count_nonzero(~np.all(np.isfinite(raw), axis=1)))
                    raise ValueError(
                        f"{component_name} contains {bad} selected events with non-finite "
                        "model inputs"
                    )
                if source == "data":
                    split_ids = row_fingerprint_ids(
                        np.column_stack(
                            [raw, np.asarray(masses[selected], dtype=np.float32)]
                        )
                    )
                    splits = deterministic_split(
                        split_ids, source=source, seed=split_seed
                    )
                else:
                    event_ids = np.asarray(arrays["event_id"], dtype=np.uint64)
                    splits = deterministic_split(
                        event_ids, source=source, seed=split_seed
                    )[selected]
                positive = selected_weights > 0.0
                yield SelectedChunk(
                    class_name=class_name,
                    component_name=component_name,
                    raw_features=raw[positive],
                    weights=selected_weights[positive],
                    splits=splits[positive],
                    zero_count=int(np.count_nonzero(selected_weights == 0.0)),
                )


def _scan_inputs(
    data_path: Path,
    gg_h_path: Path,
    zz_path: Path,
    *,
    weight_branch: str,
    split_seed: int,
    expected_luminosity_fb: float,
    step_size: str,
) -> tuple[
    dict[str, WeightedStatistics],
    dict[str, WeightedStatistics],
    np.ndarray,
    np.ndarray,
]:
    components = {
        _COMPONENT_DATA: WeightedStatistics(),
        _COMPONENT_ZZ: WeightedStatistics(),
        _COMPONENT_GGH: WeightedStatistics(),
    }
    classes = {
        _TARGET_KEY: WeightedStatistics(),
        _REFERENCE_KEY: WeightedStatistics(),
    }
    moment_sum = {
        name: np.zeros(len(MODEL_FEATURES), dtype=np.float64) for name in classes
    }
    moment_square_sum = {
        name: np.zeros(len(MODEL_FEATURES), dtype=np.float64) for name in classes
    }

    for chunk in _selected_chunks(
        data_path,
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
            label = "target data" if class_name == _TARGET_KEY else "reference MC"
            raise ValueError(f"the selected {label} class is empty")
        for split_name in SPLIT_NAMES.values():
            if statistics.split_entries[split_name] == 0:
                raise ValueError(f"the {class_name} {split_name} split is empty")
            if statistics.split_sum_weights[split_name] <= 0.0:
                raise ValueError(
                    f"the {class_name} {split_name} split has non-positive total weight"
                )

    means: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    for class_name in (_TARGET_KEY, _REFERENCE_KEY):
        fit_sum = classes[class_name].split_sum_weights["fit"]
        means.append(moment_sum[class_name] / fit_sum)
        seconds.append(moment_square_sum[class_name] / fit_sum)
    mean = 0.5 * (means[0] + means[1])
    second = 0.5 * (seconds[0] + seconds[1])
    variance = np.maximum(second - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-7, scale, 1.0)
    return components, classes, mean, scale


def _make_cache(
    cache_directory: Path,
    data_path: Path,
    gg_h_path: Path,
    zz_path: Path,
    *,
    classes: dict[str, WeightedStatistics],
    weight_branch: str,
    split_seed: int,
    expected_luminosity_fb: float,
    step_size: str,
) -> tuple[dict[str, CachePaths], Path]:
    paths: dict[str, CachePaths] = {}
    writers: dict[str, tuple[np.memmap, np.memmap, np.memmap]] = {}
    positions = {_TARGET_KEY: 0, _REFERENCE_KEY: 0}
    reference_component_path = cache_directory / "reference_components.npy"
    reference_component_writer = open_memmap(
        reference_component_path,
        mode="w+",
        dtype=np.uint8,
        shape=(classes[_REFERENCE_KEY].entries,),
    )
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
        data_path,
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
        if chunk.class_name == _REFERENCE_KEY:
            reference_component_writer[start:stop] = _REFERENCE_COMPONENT_CODE[
                chunk.component_name
            ]
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
    reference_component_writer.flush()
    del writers
    del reference_component_writer
    return paths, reference_component_path


def _validate_existing_correction_artifact(path: Path) -> None:
    manifest_path = path / "manifest.json"
    if not path.is_dir() or not manifest_path.is_file():
        raise ValueError(
            f"refusing to overwrite a directory that is not a correction artifact: {path}"
        )
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid correction artifact manifest: {manifest_path}") from error
    if (
        manifest.get("format_version") != ARTIFACT_FORMAT_VERSION
        or manifest.get("model_kind") != MODEL_KIND_DATA_MC_CORRECTION
    ):
        raise ValueError(f"refusing to overwrite a non-correction artifact: {path}")


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Any]:
    import torch

    data_path = args.data_root.expanduser().resolve()
    gg_h_path = args.gg_h_root.expanduser().resolve()
    zz_path = args.zz_root.expanduser().resolve()
    paths = (data_path, gg_h_path, zz_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
    if len(set(paths)) != len(paths):
        raise ValueError("data, gg_H, and ZZ correction inputs must be different files")
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

    output_directory = args.output_dir.expanduser().resolve()
    for path in paths:
        if output_directory == path or output_directory in path.parents:
            raise ValueError("model output directory cannot be or contain a training input")
    if output_directory.exists() and not args.overwrite:
        raise FileExistsError(
            f"output directory exists: {output_directory}; pass --overwrite to replace it"
        )
    if output_directory.exists():
        _validate_existing_correction_artifact(output_directory)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    return data_path, gg_h_path, zz_path, output_directory, torch.device(device_name)


def train(args: argparse.Namespace) -> Path:
    import torch

    # Training.diagnostics is intentionally model-kind aware.  Keeping this
    # call at one boundary lets the correction reuse the identical reliability
    # and closure machinery without duplicating the plotting implementation.
    from BackgroundRemoval.Training.diagnostics import make_diagnostics_pdf

    data_path, gg_h_path, zz_path, output_directory, device = _validate_args(args)
    weight_contract = weight_measure_contract(
        args.weight_branch, luminosity_fb=args.expected_luminosity_fb
    )
    input_provenance = {
        name: capture_file_provenance(
            path, include_sha256=not args.skip_input_checksums
        )
        for name, path in (
            ("data", data_path),
            ("gg_H_pythia", gg_h_path),
            ("ZZ_pythia", zz_path),
        )
    }
    toolkit_provenance = toolkit_runtime_provenance()
    if not toolkit_provenance["runtime_commit_verified"]:
        print(
            "Warning: the nsbi-common-utils runtime commit could not be verified from "
            "package metadata; use the pinned Pixi environment for production.",
            flush=True,
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

    print("Scanning correction-sideband selections and yields...", flush=True)
    components, classes, scaler_mean, scaler_scale = _scan_inputs(
        data_path,
        gg_h_path,
        zz_path,
        weight_branch=args.weight_branch,
        split_seed=args.split_seed,
        expected_luminosity_fb=args.expected_luminosity_fb,
        step_size=args.step_size,
    )
    target_yield = classes[_TARGET_KEY].sum_weights
    reference_yield = classes[_REFERENCE_KEY].sum_weights
    yield_ratio = target_yield / reference_yield
    gg_h_yield = components[_COMPONENT_GGH].sum_weights
    zz_yield = components[_COMPONENT_ZZ].sum_weights
    gg_h_fraction = gg_h_yield / reference_yield
    print(
        f"Selected yields ({args.weight_branch}): data={target_yield:.9g}, "
        f"MC={reference_yield:.9g}, data/MC={yield_ratio:.9g}; "
        f"reference ggH fraction={gg_h_fraction:.5%}",
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
            prefix="fourlepton-correction-cache-", dir=cache_parent
        ) as cache_name:
            cache_directory = Path(cache_name)
            cache_paths, reference_component_path = _make_cache(
                cache_directory,
                data_path,
                gg_h_path,
                zz_path,
                classes=classes,
                weight_branch=args.weight_branch,
                split_seed=args.split_seed,
                expected_luminosity_fb=args.expected_luminosity_fb,
                step_size=args.step_size,
            )
            for name, path in (
                ("data", data_path),
                ("gg_H_pythia", gg_h_path),
                ("ZZ_pythia", zz_path),
            ):
                assert_file_unchanged(path, input_provenance[name])
            cache: dict[str, CachedClass] = _load_cache(cache_paths)
            reference_components = np.load(
                reference_component_path, mmap_mode="r"
            )
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
                "loss": "balanced weighted BCEWithLogitsLoss",
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
            reference_validation_weights = np.asarray(
                cache[_REFERENCE_KEY].weights[validation[_REFERENCE_KEY].indices],
                dtype=np.float64,
            )
            normalized_bias, normalization_before = _normalization_bias(
                validation[_REFERENCE_KEY].ensemble_score,
                reference_validation_weights,
                scale=calibration_scale,
                initial_bias=calibration_bias,
                logit_clip=args.logit_clip,
            )
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
                "model_kind": MODEL_KIND_DATA_MC_CORRECTION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "tree_name": TREE_NAME,
                "features": list(FEATURES),
                "model_features": list(MODEL_FEATURES),
                "preprocessing": {
                    "periodic_variables": list(FEATURES[:3]),
                    "periodic_encoding": "sin_cos",
                    "remaining_variables": "identity",
                    "mass_is_selection_only": MASS_BRANCH,
                },
                "selections": {
                    "mass_window_gev": {
                        "branch": MASS_BRANCH,
                        "low_exclusive": CORRECTION_MASS_WINDOW[0],
                        "high_exclusive": CORRECTION_MASS_WINDOW[1],
                    },
                    "positive_label_1": (
                        "data: reconstructed && 130 < reco_m_ZZ < 160 GeV"
                    ),
                    "negative_label_0": (
                        "ZZ_pythia + gg_H_pythia: reconstructed && "
                        "130 < reco_m_ZZ < 160 GeV"
                    ),
                },
                "weight_branch": args.weight_branch,
                "weight_measure": weight_contract,
                "luminosity_fb": args.expected_luminosity_fb,
                "yields": {
                    "target": target_yield,
                    "reference": reference_yield,
                    "target_data": target_yield,
                    "reference_mc": reference_yield,
                    "target_to_reference": yield_ratio,
                    # Compatibility aliases used by the common calibrated-ratio
                    # diagnostic machinery.  Their target/reference semantics
                    # are made explicit by model_kind and ratio_convention.
                    "signal": target_yield,
                    "background": reference_yield,
                    "signal_to_background": yield_ratio,
                    "reference_gg_H": gg_h_yield,
                    "reference_ZZ": zz_yield,
                    "gg_H_fraction_of_reference": gg_h_fraction,
                    "units": weight_contract["units"],
                },
                "ratio_convention": {
                    "shape": "p(data)/p(Pythia MC) = score/(1-score)",
                    "yield": "target_data_yield/reference_mc_yield",
                    "correction": "yield * shape",
                    "orientation": "target_to_reference",
                    "downstream_application": "multiply reconstructed ZZ_pythia weights",
                    "application_assumption": (
                        "the correction is learned against ZZ+gg_H but applied only to "
                        "ZZ; this uses qqZZ dominance in the upper sideband"
                    ),
                },
                "split": {
                    "method": (
                        "splitmix64(source,row_fingerprint,seed) for sampled data; "
                        "splitmix64(source,event_id,seed) for Pythia"
                    ),
                    "data_grouping": (
                        "exact duplicates in the eight inputs plus reco_m_ZZ share a split"
                    ),
                    "seed": args.split_seed,
                    "fit_fraction": 0.60,
                    "validation_fraction": 0.15,
                    "closure_fraction": 0.25,
                },
                "components": {
                    name: statistics.serializable()
                    for name, statistics in components.items()
                },
                "reference_component_codes": {
                    "ZZ_pythia": int(_REFERENCE_COMPONENT_CODE[_COMPONENT_ZZ]),
                    "gg_H_pythia": int(_REFERENCE_COMPONENT_CODE[_COMPONENT_GGH]),
                },
                "classes": {
                    "target_data": classes[_TARGET_KEY].serializable(),
                    "reference_mc": classes[_REFERENCE_KEY].serializable(),
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
                    "normalization_condition": "E_reference_MC[r_shape] = 1",
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
                reference_is_zz=(
                    np.asarray(reference_components)
                    == _REFERENCE_COMPONENT_CODE[_COMPONENT_ZZ]
                ),
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
            expected_model_kind=MODEL_KIND_DATA_MC_CORRECTION,
        )
    except Exception:
        if temporary_artifact.exists():
            shutil.rmtree(temporary_artifact)
        raise
    print(f"Data-to-MC correction model written to {output_directory}", flush=True)
    return output_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a calibrated data-to-Pythia density ratio in the strict "
            "130 < reco_m_ZZ < 160 GeV correction sideband"
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
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
        help="override full-coverage balanced batches (primarily for tests)",
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
