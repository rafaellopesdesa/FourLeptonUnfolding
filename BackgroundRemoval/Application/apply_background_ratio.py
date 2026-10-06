#!/usr/bin/env python3
"""Decorate a reconstructed data ROOT tree with signal-purity weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import uproot

from BackgroundRemoval.common import (
    DIAGNOSTIC_OUTPUT_BRANCHES,
    FEATURES,
    OUTPUT_BRANCHES,
    PRIMARY_OUTPUT_BRANCH,
    TREE_NAME,
    ModelBundle,
    sha256_file,
)


def _tree_schema(tree: Any, branches: list[str]) -> dict[str, np.dtype]:
    if tree.num_entries == 0:
        raise ValueError("the input Analysis tree is empty")
    schema: dict[str, np.dtype] = {}
    for name in branches:
        sample = np.asarray(
            tree[name].array(entry_start=0, entry_stop=1, library="np")
        )
        if sample.dtype == np.dtype(object):
            raise TypeError(
                f"branch {name!r} is variable-length and cannot be preserved by this "
                "compact-tree decorator"
            )
        schema[name] = (
            np.dtype((sample.dtype, sample.shape[1:]))
            if sample.ndim > 1
            else sample.dtype
        )
    return schema


def _metadata_objects(root_file: Any) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for name, class_name in root_file.classnames(cycle=False).items():
        if name == TREE_NAME or name == "background_removal_metadata":
            continue
        if class_name == "TObjString":
            metadata[name] = str(root_file[name])
        elif class_name in {"TTree", "ROOT::RNTuple"}:
            raise ValueError(
                f"input contains an additional tree {name!r}; refusing to silently drop it"
            )
        else:
            raise ValueError(
                f"input contains unsupported object {name!r} ({class_name}); "
                "refusing to silently drop it"
            )
    return metadata


def decorate(
    input_path: Path,
    output_path: Path,
    model_directory: Path,
    *,
    step_size: str,
    inference_batch_size: int,
    device: str,
    overwrite: bool,
    replace_existing_branches: bool,
    write_diagnostic_branches: bool,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if output_path.exists() and output_path != input_path and not overwrite:
        raise FileExistsError(
            f"output file exists: {output_path}; pass --overwrite to replace it"
        )
    if output_path == input_path and not overwrite:
        raise FileExistsError("in-place decoration requires --overwrite")
    bundle = ModelBundle.load(model_directory, device_name=device)
    manifest_path = bundle.root / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.background-removal-",
        suffix=".root",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    required = {*FEATURES, "reconstructed", "weight", "luminosity_fb"}
    written_branches = (
        OUTPUT_BRANCHES
        if write_diagnostic_branches
        else (PRIMARY_OUTPUT_BRANCH,)
    )
    entry_count = 0
    removal_sum = 0.0
    removal_min = math.inf
    removal_max = -math.inf

    try:
        with uproot.open(input_path) as input_file:
            if TREE_NAME not in input_file:
                raise KeyError(f"{input_path} does not contain the {TREE_NAME} tree")
            input_tree = input_file[TREE_NAME]
            branch_order = list(input_tree.keys(recursive=True, full_paths=False))
            all_branches = set(branch_order)
            missing = sorted(required.difference(all_branches))
            if missing:
                raise KeyError(
                    f"{input_path} is missing required branches: {', '.join(missing)}"
                )
            conflicts = sorted(set(OUTPUT_BRANCHES).intersection(all_branches))
            if conflicts and not replace_existing_branches:
                raise ValueError(
                    "input is already decorated with output branches "
                    f"{', '.join(conflicts)}; pass --replace-existing-branches to recompute them"
                )
            original_branches = [
                name for name in branch_order if name not in OUTPUT_BRANCHES
            ]
            schema = _tree_schema(input_tree, original_branches)
            schema[PRIMARY_OUTPUT_BRANCH] = np.dtype(np.float32)
            if write_diagnostic_branches:
                schema.update(
                    {
                        name: (
                            np.dtype(np.float64)
                            if name == "weight_background_removed"
                            else np.dtype(np.float32)
                        )
                        for name in DIAGNOSTIC_OUTPUT_BRANCHES
                    }
                )
            metadata_objects = _metadata_objects(input_file)
            with uproot.recreate(temporary) as output_file:
                output_file.mktree(
                    TREE_NAME,
                    schema,
                    title=(
                        getattr(input_tree, "title", None)
                        or "Background-decorated four-lepton data"
                    ),
                )
                output_tree = output_file[TREE_NAME]
                for arrays in input_tree.iterate(
                    expressions=original_branches,
                    step_size=step_size,
                    library="np",
                    how=dict,
                ):
                    luminosities = np.asarray(arrays["luminosity_fb"], dtype=np.float64)
                    expected_luminosity = float(bundle.manifest["luminosity_fb"])
                    if not np.all(
                        np.isfinite(luminosities)
                        & np.isclose(
                            luminosities,
                            expected_luminosity,
                            rtol=0.0,
                            atol=1.0e-9,
                        )
                    ):
                        raise ValueError(
                            "input is not normalized to the model luminosity of "
                            f"{expected_luminosity:g} fb^-1"
                        )
                    reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
                    if not np.all(reconstructed):
                        failed = int(np.count_nonzero(~reconstructed))
                        raise ValueError(
                            f"input contains {failed} non-reconstructed rows in a processing "
                            "chunk; the background model is defined only after reconstruction"
                        )
                    raw_features = np.column_stack(
                        [np.asarray(arrays[name], dtype=np.float32) for name in FEATURES]
                    )
                    if not np.all(np.isfinite(raw_features)):
                        failed = int(
                            np.count_nonzero(~np.all(np.isfinite(raw_features), axis=1))
                        )
                        raise ValueError(
                            f"input contains {failed} reconstructed rows with non-finite "
                            "background-model features"
                        )
                    observation_weights = np.asarray(
                        arrays["weight"], dtype=np.float64
                    )
                    if not np.all(np.isfinite(observation_weights)):
                        raise ValueError("input contains non-finite observation weights")
                    predictions = bundle.predict(
                        raw_features, batch_size=inference_batch_size
                    )
                    for name in written_branches:
                        if name == "weight_background_removed":
                            arrays[name] = np.asarray(
                                observation_weights, dtype=np.float64
                            ) * np.asarray(
                                predictions["background_removal_weight"],
                                dtype=np.float64,
                            )
                        else:
                            arrays[name] = np.asarray(predictions[name], dtype=np.float32)
                    removal = arrays["background_removal_weight"]
                    entry_count += removal.size
                    removal_sum += float(np.sum(removal, dtype=np.float64))
                    if removal.size:
                        removal_min = min(removal_min, float(np.min(removal)))
                        removal_max = max(removal_max, float(np.max(removal)))
                    output_tree.extend(arrays)

                application_metadata = {
                    "format_version": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source": str(input_path),
                    "model_directory": str(bundle.root),
                    "model_manifest_sha256": sha256_file(manifest_path),
                    "model_created_utc": bundle.manifest.get("created_utc"),
                    "features": list(FEATURES),
                    "written_branches": list(written_branches),
                    "entries": entry_count,
                    "background_removal_weight_mean": (
                        removal_sum / entry_count if entry_count else None
                    ),
                    "background_removal_weight_min": (
                        removal_min if entry_count else None
                    ),
                    "background_removal_weight_max": (
                        removal_max if entry_count else None
                    ),
                    "formula": (
                        "background_removal_weight = ((S/B)*r_shape) / "
                        "(1 + (S/B)*r_shape)"
                    ),
                    "effective_weight": (
                        "multiply the preserved weight by background_removal_weight"
                    ),
                    "diagnostic_branches_written": write_diagnostic_branches,
                    "original_weight_preserved": True,
                    "model_yields": bundle.manifest["yields"],
                }
                for name, value in metadata_objects.items():
                    output_file[name] = value
                output_file["background_removal_metadata"] = json.dumps(
                    application_metadata, sort_keys=True
                )
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return application_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a trained background-removal ensemble and add a bounded "
            "signal-purity weight to data.root"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-size", default="200 MB")
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-existing-branches", action="store_true")
    parser.add_argument(
        "--write-diagnostic-branches",
        action="store_true",
        help="also store scores, ratios, ensemble spread, and the multiplied weight",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = decorate(
        args.input,
        args.output,
        args.model_dir,
        step_size=args.step_size,
        inference_batch_size=args.inference_batch_size,
        device=args.device,
        overwrite=args.overwrite,
        replace_existing_branches=args.replace_existing_branches,
        write_diagnostic_branches=args.write_diagnostic_branches,
    )
    print(
        f"Decorated {metadata['entries']} events -> {args.output}; "
        f"mean background_removal_weight="
        f"{metadata['background_removal_weight_mean']:.7g}"
    )


if __name__ == "__main__":
    main()
