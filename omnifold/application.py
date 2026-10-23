"""Decorate nominal data, pseudo-experiments, and MC with frozen iterations.

All vector columns are indexed by iteration, beginning with iteration 1 at
index 0.  In particular, ``omnifold_truth_push`` is cumulative relative to
the original MC.  Do not multiply its entries across iterations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import uproot

from BackgroundRemoval.common import (
    ANALYSIS_MASS_WINDOW,
    FEATURES,
    TREE_NAME,
    assert_file_unchanged,
    capture_file_provenance,
)


TRUTH_FEATURES = tuple(name.replace("reco_", "truth_", 1) for name in FEATURES)
STAGES = ("reco_pull", "truth_push", "reco_likelihood")
MANIFEST_NAME = "omnifold_application_manifest.json"
ARTIFACT_TYPE = "omnifold_decorated_campaign"


def _as_dict(arrays: Any) -> dict[str, np.ndarray]:
    if isinstance(arrays, Mapping):
        return {name: np.asarray(value) for name, value in arrays.items()}
    if getattr(arrays.dtype, "names", None):
        return {name: np.asarray(arrays[name]) for name in arrays.dtype.names}
    raise TypeError("the Analysis tree did not return named arrays")


def _schema(tree: Any) -> dict[str, np.dtype]:
    result: dict[str, np.dtype] = {}
    for name in tree.keys():
        value = np.asarray(tree[name].array(entry_start=0, entry_stop=1, library="np"))
        if value.dtype.hasobject:
            raise TypeError(f"cannot preserve variable-length/object branch {name!r}")
        result[name] = (
            np.dtype((value.dtype, value.shape[1:])) if value.ndim > 1 else value.dtype
        )
    return result


def _root_metadata(root_file: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, kind in root_file.classnames(cycle=False).items():
        if name == TREE_NAME:
            continue
        if kind != "TObjString":
            raise ValueError(
                f"cannot preserve ROOT object {name!r} ({kind}); refusing to drop it"
            )
        if name == "omnifold_metadata":
            raise ValueError("input already contains omnifold_metadata")
        result[name] = str(root_file[name])
    return result


def _model_record(model: Any) -> dict[str, Any]:
    yield_ratio = float(model.yield_ratio)
    if not np.isfinite(yield_ratio) or yield_ratio <= 0:
        raise ValueError("every ratio model must have a finite positive yield ratio")
    manifest = getattr(model, "manifest", {})
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    return {
        "yield_ratio": yield_ratio,
        "model_manifest_digest": hashlib.sha256(encoded).hexdigest(),
        "model_manifest": manifest,
    }


def _validate_existing(destination: Path) -> None:
    manifest_path = destination / MANIFEST_NAME
    if not destination.is_dir() or not manifest_path.is_file():
        raise ValueError("overwrite requires a recognized OmniFold decoration campaign")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != ARTIFACT_TYPE or manifest.get("format_version") != 1:
        raise ValueError("overwrite destination is not an OmniFold decoration campaign")
    expected = {MANIFEST_NAME}
    for record in manifest.get("files", []):
        name = record.get("output_name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("existing decoration manifest has an invalid output name")
        expected.add(name)
    actual = {path.name for path in destination.iterdir()}
    if actual != expected or any(not path.is_file() for path in destination.iterdir()):
        raise ValueError("overwrite destination contains unrecognized or missing files")


def _input_weight(
    arrays: Mapping[str, np.ndarray], *, branch: str, is_mc: bool
) -> tuple[np.ndarray, str]:
    if branch in arrays:
        weights = np.asarray(arrays[branch], dtype=np.float64)
        expression = branch
    elif not is_mc and branch == "weight_background_removed" and {
        "weight", "background_removal_weight"
    }.issubset(arrays):
        weights = np.asarray(arrays["weight"], dtype=np.float64) * np.asarray(
            arrays["background_removal_weight"], dtype=np.float64
        )
        expression = "weight * background_removal_weight"
    else:
        raise KeyError(f"missing observation/MC weight branch {branch!r}")
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError(f"non-finite or non-scalar event weights in {expression}")
    return weights, expression


def _predict_into(
    output: dict[str, np.ndarray],
    prefix: str,
    model: Any,
    features: np.ndarray,
    mask: np.ndarray,
    iteration_column: int,
) -> None:
    if not np.any(mask):
        return
    prediction = model.predict(features[mask], batch_size=65536)
    shape = np.asarray(prediction["shape_ratio"], dtype=np.float64)
    full = np.asarray(prediction["full_ratio"], dtype=np.float64)
    expected_shape = (int(np.count_nonzero(mask)),)
    if shape.shape != expected_shape or full.shape != expected_shape:
        raise ValueError(f"{prefix} model returned an incorrect prediction shape")
    if not np.all(np.isfinite(shape) & (shape > 0) & np.isfinite(full) & (full > 0)):
        raise ValueError(f"{prefix} model returned non-positive/non-finite ratios")
    if not np.allclose(full, float(model.yield_ratio) * shape, rtol=1e-6, atol=0):
        raise ValueError(f"{prefix} full ratio is inconsistent with yield times shape")
    output[prefix][mask, iteration_column] = full
    output[prefix + "_shape"][mask, iteration_column] = shape


def _decorate_one(
    source: Path,
    destination: Path,
    iterations: Sequence[Any],
    *,
    is_mc: bool,
    weight_branch: str,
    step_size: str | int,
    source_provenance: Mapping[str, Any],
    model_records: list[dict[str, Any]],
    expected_luminosity_fb: float | None,
) -> dict[str, Any]:
    n_iterations = len(iterations)
    entries = 0
    selected_reco = 0
    selected_truth = 0
    weight_expression = weight_branch
    with uproot.open(source) as root_file:
        tree = root_file[TREE_NAME]
        branches = set(tree.keys())
        if any(name.startswith("omnifold_") for name in branches):
            raise ValueError(f"input already contains OmniFold columns: {source}")
        needed = set(FEATURES) | {"reconstructed", "reco_m_ZZ"}
        if is_mc:
            needed |= set(TRUTH_FEATURES) | {"fiducial"}
        if weight_branch not in branches and not (
            not is_mc and weight_branch == "weight_background_removed"
            and {"weight", "background_removal_weight"}.issubset(branches)
        ):
            needed.add(weight_branch)
        missing = needed - branches
        if missing:
            raise KeyError(f"missing required branches in {source}: {sorted(missing)}")
        has_truth = (set(TRUTH_FEATURES) | {"fiducial"}).issubset(branches)
        metadata = _root_metadata(root_file)
        schema = _schema(tree)
        vector_dtype = np.dtype((np.float64, (n_iterations,)))
        for stage in STAGES:
            prefix = f"omnifold_{stage}"
            for suffix in ("", "_shape", "_yield_ratio"):
                schema[prefix + suffix] = vector_dtype
            schema[prefix + "_valid"] = np.dtype(np.bool_)
        schema["omnifold_weight"] = vector_dtype
        schema["omnifold_weight_valid"] = np.dtype(np.bool_)
        with uproot.recreate(destination) as output:
            output.mktree(TREE_NAME, schema, title="Frozen OmniFold iteration vectors")
            for chunk in tree.iterate(step_size=step_size, library="np", how=dict):
                arrays = _as_dict(chunk)
                weights, weight_expression = _input_weight(
                    arrays, branch=weight_branch, is_mc=is_mc
                )
                size = len(weights)
                if expected_luminosity_fb is not None and "luminosity_fb" in arrays:
                    luminosity = np.asarray(arrays["luminosity_fb"], dtype=np.float64)
                    if luminosity.shape != (size,) or not np.all(
                        np.isfinite(luminosity)
                        & np.isclose(luminosity, expected_luminosity_fb, rtol=0, atol=1e-9)
                    ):
                        raise ValueError(
                            f"input luminosity_fb does not match the frozen campaign "
                            f"({expected_luminosity_fb:g} fb^-1): {source}"
                        )
                reco = np.column_stack([arrays[name] for name in FEATURES])
                low, high = ANALYSIS_MASS_WINDOW
                mass = arrays["reco_m_ZZ"]
                reco_domain = (
                    np.asarray(arrays["reconstructed"], dtype=bool)
                    & (mass > low) & (mass < high)
                )
                finite_reco = np.all(np.isfinite(reco), axis=1)
                if np.any(reco_domain & ~finite_reco):
                    raise ValueError(
                        f"non-finite reconstructed features inside the analysis region: {source}"
                    )
                reco_mask = reco_domain & finite_reco
                truth = None
                truth_mask = np.zeros(size, dtype=bool)
                if has_truth:
                    truth = np.column_stack([arrays[name] for name in TRUTH_FEATURES])
                    fiducial = np.asarray(arrays["fiducial"], dtype=bool)
                    finite_truth = np.all(np.isfinite(truth), axis=1)
                    if is_mc and np.any(fiducial & ~finite_truth):
                        raise ValueError(f"non-finite truth features on fiducial MC events: {source}")
                    truth_mask = fiducial & finite_truth
                if is_mc:
                    reco_mask &= truth_mask
                for stage in STAGES:
                    prefix = f"omnifold_{stage}"
                    for suffix in ("", "_shape"):
                        arrays[prefix + suffix] = np.full(
                            (size, n_iterations), np.nan, dtype=np.float64
                        )
                    arrays[prefix + "_yield_ratio"] = np.broadcast_to(
                        np.array([float(getattr(item, stage).yield_ratio) for item in iterations]),
                        (size, n_iterations),
                    ).copy()
                    arrays[prefix + "_valid"] = (
                        truth_mask.copy() if stage == "truth_push" else reco_mask.copy()
                    )
                for column, iteration in enumerate(iterations):
                    for stage in ("reco_pull", "reco_likelihood"):
                        _predict_into(
                            arrays, f"omnifold_{stage}", getattr(iteration, stage),
                            reco, reco_mask, column,
                        )
                    if truth is not None:
                        _predict_into(
                            arrays, "omnifold_truth_push", iteration.truth_push,
                            truth, truth_mask, column,
                        )
                weight_mask = truth_mask if is_mc else reco_mask
                arrays["omnifold_weight_valid"] = weight_mask.copy()
                arrays["omnifold_weight"] = np.full(
                    (size, n_iterations), np.nan, dtype=np.float64
                )
                if is_mc:
                    arrays["omnifold_weight"][weight_mask] = (
                        weights[weight_mask, None]
                        * arrays["omnifold_truth_push"][weight_mask]
                    )
                else:
                    arrays["omnifold_weight"][weight_mask] = weights[weight_mask, None]
                if not np.all(np.isfinite(arrays["omnifold_weight"][weight_mask])):
                    raise ValueError("unfolded event weights overflowed")
                output[TREE_NAME].extend(arrays)
                entries += size
                selected_reco += int(np.count_nonzero(reco_mask))
                selected_truth += int(np.count_nonzero(truth_mask))
            record = {
                "output_name": destination.name,
                "source": str(source),
                "source_provenance": dict(source_provenance),
                "sample_kind": "gg_h_mc" if is_mc else "data",
                "entries": entries,
                "reco_valid_entries": selected_reco,
                "truth_valid_entries": selected_truth,
                "weight_expression": weight_expression,
                "expected_luminosity_fb": expected_luminosity_fb,
                "luminosity_branch_present": "luminosity_fb" in branches,
                "data_truth_is_diagnostic_only": not is_mc and has_truth,
                "iteration_models": model_records,
            }
            for name, content in metadata.items():
                output[name] = content
            output["omnifold_metadata"] = json.dumps(record, sort_keys=True, allow_nan=False)
    assert_file_unchanged(source, source_provenance)
    return record


def decorate_files(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    iterations: Sequence[Any],
    *,
    mc_path: str | Path | None = None,
    data_weight_branch: str = "weight_background_removed",
    mc_weight_branch: str = "weight",
    step_size: str | int = "64 MB",
    overwrite: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write all frozen iteration ratios to independent copies of the inputs.

    ``inputs`` contains data/toy files; ``mc_path`` is appended once if given.
    No fitting is performed.  The reco likelihood reference is the original
    MC encoded by each supplied model, and the truth-push ratio is cumulative.

    ``omnifold_weight`` is base-MC-weight times cumulative truth push for MC.
    For data it is the unchanged, signed background-subtracted observation
    weight repeated across iterations.  Model ratios must *not* be multiplied
    into data to evaluate a likelihood.  Ratios/weights outside their valid
    domain are NaN; the yield ratio vectors are global factors on every row.
    Truth weights on pseudo-data are diagnostic only and are unavailable for
    data without truth columns.  Original branches and TObjString metadata
    are preserved.  Other ROOT object types fail explicitly.

    If ``metadata['luminosity_fb']`` is supplied, any input luminosity column
    must agree with this frozen campaign value.  Real data without a
    luminosity column are permitted.  Non-finite features on reconstructed
    analysis-region events, or on fiducial MC events, cause an explicit error.
    """
    iterations = list(iterations)
    if not iterations:
        raise ValueError("at least one completed iteration is required")
    indices = [item.index for item in iterations]
    if indices != list(range(1, len(iterations) + 1)):
        raise ValueError("iterations must be the complete ordered sequence starting at 1")
    model_records = [
        {"iteration": item.index, **{stage: _model_record(getattr(item, stage)) for stage in STAGES}}
        for item in iterations
    ]
    destination = Path(output_dir).expanduser().resolve()
    sources = [Path(path).expanduser().resolve() for path in inputs]
    mc = Path(mc_path).expanduser().resolve() if mc_path is not None else None
    if mc is not None and mc not in sources:
        sources.append(mc)
    if not sources:
        raise ValueError("no input files were supplied")
    if len(set(sources)) != len(sources):
        raise ValueError("input files contain duplicates")
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("input basenames collide in the output directory")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source == destination or destination in source.parents:
            raise ValueError("output directory cannot contain an input file")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"output directory exists: {destination}")
        _validate_existing(destination)
    manifest: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": indices,
        "iteration_models": model_records,
        "metadata": dict(metadata or {}),
        "semantics": {
            "reco_pull": "incremental data/current-MC ratio at reconstructed level",
            "truth_push": "cumulative updated/original-MC ratio at dressed fiducial level",
            "reco_likelihood": "cumulative folded updated/original-MC ratio at reconstructed level",
            "yield_ratio": "full_ratio = shape_ratio * yield_ratio; global factors repeated on all rows",
            "invalid": "NaN values with scalar false validity mask; no truth inference from reco data",
            "mc_weight": "original signed MC weight * cumulative truth_push; use one iteration column",
            "data_weight": "unchanged signed background-subtracted observation weight in each column",
        },
        "files": [],
    }
    expected_luminosity_fb = manifest["metadata"].get("luminosity_fb")
    if expected_luminosity_fb is not None:
        expected_luminosity_fb = float(expected_luminosity_fb)
        if not np.isfinite(expected_luminosity_fb) or expected_luminosity_fb <= 0:
            raise ValueError("metadata luminosity_fb must be finite and positive")
    json.dumps(manifest, allow_nan=False)
    provenance = {path: capture_file_provenance(path) for path in sources}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    backup: Path | None = None
    try:
        for source in sources:
            record = _decorate_one(
                source, temporary / source.name, iterations,
                is_mc=source == mc,
                weight_branch=mc_weight_branch if source == mc else data_weight_branch,
                step_size=step_size,
                source_provenance=provenance[source], model_records=model_records,
                expected_luminosity_fb=expected_luminosity_fb,
            )
            manifest["files"].append(record)
        for source in sources:
            assert_file_unchanged(source, provenance[source])
        with (temporary / MANIFEST_NAME).open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            _validate_existing(destination)
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup.", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup is not None and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as error:
                warnings.warn(
                    f"decoration campaign was published, but its backup remains at {backup}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return [destination / source.name for source in sources]
