#!/usr/bin/env python3
"""Apply the frozen background-removal and data/MC-correction artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import uproot

from BackgroundRemoval.common import (
    ANALYSIS_MASS_WINDOW,
    ANALYSIS_REGION_BRANCH,
    CORRECTION_MASS_WINDOW,
    FEATURES,
    MASS_BRANCH,
    MODEL_KIND_BACKGROUND_REMOVAL,
    MODEL_KIND_DATA_MC_CORRECTION,
    PRIMARY_OUTPUT_BRANCH,
    TREE_NAME,
    ModelBundle,
    assert_file_unchanged,
    capture_file_provenance,
    known_weight_measures_compatible,
    sha256_file,
    weight_measure_contract,
)


SAMPLE_KINDS = ("data", "gg-h", "zz")
DIAGNOSTIC_OUTPUT_BRANCHES = (
    "signal_score_balanced",
    "signal_score_ensemble_std",
    "background_shape_ratio",
    "signal_to_background_ratio",
    "weight_background_removed",
)
OUTPUT_BRANCHES = (
    PRIMARY_OUTPUT_BRANCH,
    ANALYSIS_REGION_BRANCH,
    *DIAGNOSTIC_OUTPUT_BRANCHES,
)


@dataclass(frozen=True)
class ApplicationModels:
    """Two verified artifacts held in memory for one application campaign."""

    background: ModelBundle
    correction: ModelBundle
    background_manifest_sha256: str
    correction_manifest_sha256: str


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


def _manifest_correction_checksum(manifest: Mapping[str, Any]) -> str:
    correction = manifest.get("correction_model")
    if not isinstance(correction, Mapping):
        raise ValueError(
            "background-removal artifact does not record its correction model"
        )
    checksum = correction.get("manifest_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(
            "background-removal artifact has no valid correction manifest checksum"
        )
    try:
        int(checksum, 16)
    except ValueError as error:
        raise ValueError(
            "background-removal artifact has a non-hexadecimal correction checksum"
        ) from error
    return checksum.lower()


def _validate_model_windows(
    background_manifest: Mapping[str, Any],
    correction_manifest: Mapping[str, Any],
) -> None:
    """Refuse artifacts trained in mass domains different from this recipe."""

    background_selections = background_manifest.get("selections", {})
    background_window = (
        background_selections.get("mass_window_gev", {})
        if isinstance(background_selections, Mapping)
        else {}
    )
    if not isinstance(background_window, Mapping) or not (
        background_window.get("branch") == MASS_BRANCH
        and background_window.get("low_exclusive") == ANALYSIS_MASS_WINDOW[0]
        and background_window.get("high_exclusive") == ANALYSIS_MASS_WINDOW[1]
    ):
        raise ValueError(
            "background-removal artifact does not use the strict analysis mass window"
        )

    correction_selections = correction_manifest.get("selections", {})
    if not isinstance(correction_selections, Mapping):
        raise ValueError("correction artifact has no valid selections")
    correction_window = correction_selections.get("mass_window_gev")
    if isinstance(correction_window, Mapping):
        correction_valid = (
            correction_window.get("branch") == MASS_BRANCH
            and correction_window.get("low_exclusive") == CORRECTION_MASS_WINDOW[0]
            and correction_window.get("high_exclusive") == CORRECTION_MASS_WINDOW[1]
        )
    else:
        correction_valid = (
            correction_selections.get("correction_mass_branch") == MASS_BRANCH
            and correction_selections.get("correction_mass_window_GeV")
            == list(CORRECTION_MASS_WINDOW)
            and correction_selections.get("bounds") == "strict"
        )
    if not correction_valid:
        raise ValueError(
            "correction artifact does not use the strict correction-sideband mass window"
        )


def _manifest_weight_contract(manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
    branch = manifest.get("weight_branch")
    if not isinstance(branch, str) or not branch:
        raise ValueError(f"{label} artifact has no valid weight-branch contract")
    contract = weight_measure_contract(
        branch, luminosity_fb=float(manifest["luminosity_fb"])
    )
    recorded = manifest.get("weight_measure")
    if not isinstance(recorded, Mapping) or recorded != contract:
        raise ValueError(f"{label} artifact has an inconsistent weight-measure contract")
    return contract


def load_application_models(
    background_model_directory: Path,
    correction_model_directory: Path,
    *,
    device: str,
) -> ApplicationModels:
    """Load both artifact kinds and enforce their provenance relationship."""

    background = ModelBundle.load(
        background_model_directory,
        device_name=device,
        expected_model_kind=MODEL_KIND_BACKGROUND_REMOVAL,
    )
    correction = ModelBundle.load(
        correction_model_directory,
        device_name=device,
        expected_model_kind=MODEL_KIND_DATA_MC_CORRECTION,
    )
    background_manifest = background.root / "manifest.json"
    correction_manifest = correction.root / "manifest.json"
    background_checksum = sha256_file(background_manifest)
    correction_checksum = sha256_file(correction_manifest)
    recorded_checksum = _manifest_correction_checksum(background.manifest)
    if recorded_checksum != correction_checksum.lower():
        raise ValueError(
            "the supplied correction artifact is not the one used to train the "
            "background-removal model: "
            f"{correction_checksum} != {recorded_checksum}"
        )
    _validate_model_windows(background.manifest, correction.manifest)

    background_luminosity = float(background.manifest["luminosity_fb"])
    correction_luminosity = float(correction.manifest["luminosity_fb"])
    if not (
        np.isfinite(background_luminosity)
        and np.isfinite(correction_luminosity)
        and math.isclose(
            background_luminosity,
            correction_luminosity,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(
            "background and correction artifacts use different luminosities"
        )
    background_weight_contract = _manifest_weight_contract(
        background.manifest, "background-removal"
    )
    correction_weight_contract = _manifest_weight_contract(
        correction.manifest, "Correction"
    )
    if not known_weight_measures_compatible(
        background_weight_contract, correction_weight_contract
    ):
        correction_link = background.manifest.get("correction_model")
        override_recorded = (
            isinstance(correction_link, Mapping)
            and correction_link.get("weight_measure_override") is True
            and correction_link.get("weight_measures_known_compatible") is False
            and correction_link.get("weight_branch")
            == correction_weight_contract["branch"]
            and correction_link.get("weight_measure")
            == correction_weight_contract
        )
        if not override_recorded:
            raise ValueError(
                "background and correction artifacts use a custom or unknown "
                "weight-measure combination without a recorded Training override"
            )
    return ApplicationModels(
        background=background,
        correction=correction,
        background_manifest_sha256=background_checksum,
        correction_manifest_sha256=correction_checksum,
    )


def _prediction(
    predictions: Mapping[str, np.ndarray], modern: str, legacy: str
) -> np.ndarray:
    if modern in predictions:
        return np.asarray(predictions[modern])
    if legacy in predictions:
        return np.asarray(predictions[legacy])
    raise KeyError(f"model prediction is missing {modern!r}")


def _prediction_arrays(size: int) -> dict[str, np.ndarray]:
    """Unity/undefined defaults for rows on which no network is evaluated."""

    return {
        "signal_score_balanced": np.full(size, np.nan, dtype=np.float32),
        "signal_score_ensemble_std": np.full(size, np.nan, dtype=np.float32),
        "background_shape_ratio": np.ones(size, dtype=np.float32),
        "signal_to_background_ratio": np.ones(size, dtype=np.float32),
    }


def _validate_luminosity(
    luminosities: np.ndarray, expected_luminosity: float
) -> None:
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


def decorate(
    input_path: Path,
    output_path: Path,
    background_model_directory: Path,
    correction_model_directory: Path,
    *,
    sample_kind: str,
    step_size: str,
    inference_batch_size: int,
    device: str,
    overwrite: bool,
    replace_existing_branches: bool,
    write_diagnostic_branches: bool,
    models: ApplicationModels | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decorate one merged sample according to its role in the analysis.

    ``data`` receives the bounded target-purity prediction from the background
    model only in the 115--130 GeV analysis region.  ``gg-h`` receives unity.
    Reconstructed ``zz`` events receive the full physical data/MC ratio from
    the correction model at every mass; non-reconstructed rows receive unity.
    The analysis-region flag is one only for reconstructed events strictly
    inside the mass window.
    """

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if sample_kind not in SAMPLE_KINDS:
        raise ValueError(
            f"sample kind must be one of {', '.join(SAMPLE_KINDS)}; got {sample_kind!r}"
        )
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
    input_provenance = (
        dict(source_provenance)
        if source_provenance is not None
        else capture_file_provenance(input_path)
    )
    recorded_source = input_provenance.get("path")
    if not isinstance(recorded_source, str) or Path(recorded_source).resolve() != input_path:
        raise ValueError("source provenance does not describe the requested input file")
    assert_file_unchanged(
        input_path, input_provenance, verify_sha256=False
    )
    if models is None:
        models = load_application_models(
            background_model_directory,
            correction_model_directory,
            device=device,
        )
    for label, model_root in (
        ("background", models.background.root),
        ("correction", models.correction.root),
    ):
        model_root = model_root.expanduser().resolve()
        if (
            output_path == model_root
            or model_root in output_path.parents
            or output_path in model_root.parents
        ):
            raise ValueError(
                f"output path must not overlap the loaded {label} model artifact"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.background-removal-",
        suffix=".root",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    required = {"reconstructed", MASS_BRANCH, "luminosity_fb", "weight"}
    if sample_kind in {"data", "zz"}:
        required.update(FEATURES)
    written_branches = (
        (PRIMARY_OUTPUT_BRANCH, ANALYSIS_REGION_BRANCH, *DIAGNOSTIC_OUTPUT_BRANCHES)
        if write_diagnostic_branches
        else (PRIMARY_OUTPUT_BRANCH, ANALYSIS_REGION_BRANCH)
    )
    entry_count = 0
    analysis_region_count = 0
    predicted_count = 0
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
            schema[ANALYSIS_REGION_BRANCH] = np.dtype(np.uint8)
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
                        or "Background-decorated four-lepton sample"
                    ),
                )
                output_tree = output_file[TREE_NAME]
                for arrays in input_tree.iterate(
                    expressions=original_branches,
                    step_size=step_size,
                    library="np",
                    how=dict,
                ):
                    size = len(arrays["reconstructed"])
                    luminosities = np.asarray(arrays["luminosity_fb"], dtype=np.float64)
                    expected_luminosity = float(
                        models.background.manifest["luminosity_fb"]
                    )
                    _validate_luminosity(luminosities, expected_luminosity)
                    observation_weights = np.asarray(arrays["weight"], dtype=np.float64)
                    if not np.all(np.isfinite(observation_weights)):
                        raise ValueError("input contains non-finite observation weights")
                    reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
                    masses = np.asarray(arrays[MASS_BRANCH], dtype=np.float64)
                    if np.any(reconstructed & ~np.isfinite(masses)):
                        raise ValueError(
                            "input contains reconstructed rows with non-finite reco_m_ZZ"
                        )
                    lower, upper = ANALYSIS_MASS_WINDOW
                    in_region = reconstructed & (masses > lower) & (masses < upper)
                    analysis_region = in_region.astype(np.uint8)
                    removal = np.ones(size, dtype=np.float32)
                    if sample_kind == "data":
                        active = in_region
                    elif sample_kind == "zz":
                        active = reconstructed
                    else:
                        active = np.zeros(size, dtype=np.bool_)
                    diagnostics = (
                        _prediction_arrays(size) if write_diagnostic_branches else {}
                    )

                    if np.any(active):
                        raw_features = np.column_stack(
                            [
                                np.asarray(arrays[name], dtype=np.float32)[active]
                                for name in FEATURES
                            ]
                        )
                        if not np.all(np.isfinite(raw_features)):
                            failed = int(
                                np.count_nonzero(
                                    ~np.all(np.isfinite(raw_features), axis=1)
                                )
                            )
                            raise ValueError(
                                f"input contains {failed} rows requiring a model prediction "
                                "with non-finite features"
                            )
                        bundle = (
                            models.background
                            if sample_kind == "data"
                            else models.correction
                        )
                        predictions = bundle.predict(
                            raw_features, batch_size=inference_batch_size
                        )
                        if sample_kind == "data":
                            applied = _prediction(
                                predictions,
                                "target_purity",
                                "background_removal_weight",
                            )
                        else:
                            applied = _prediction(
                                predictions,
                                "physical_ratio",
                                "signal_to_background_ratio",
                            )
                        if applied.shape != (int(np.count_nonzero(active)),):
                            raise ValueError("model returned an incompatible prediction shape")
                        lower_bound_ok = (
                            applied > 0.0
                            if sample_kind == "zz"
                            else applied >= 0.0
                        )
                        if not np.all(np.isfinite(applied) & lower_bound_ok):
                            raise ValueError("model returned invalid application weights")
                        if sample_kind == "data" and not np.all(applied <= 1.0):
                            raise ValueError(
                                "background-removal model returned a purity above one"
                            )
                        applied_float32 = np.asarray(applied, dtype=np.float32)
                        if not np.all(np.isfinite(applied_float32)):
                            raise ValueError(
                                "model application weights exceed the float32 output range"
                            )
                        if sample_kind == "zz" and not np.all(
                            applied_float32 > 0.0
                        ):
                            raise ValueError(
                                "correction weights underflow the positive float32 range"
                            )
                        removal[active] = applied_float32

                        if write_diagnostic_branches:
                            diagnostic_sources = {
                                "signal_score_balanced": (
                                    "calibrated_score",
                                    "signal_score_balanced",
                                ),
                                "signal_score_ensemble_std": (
                                    "score_ensemble_std",
                                    "signal_score_ensemble_std",
                                ),
                                "background_shape_ratio": (
                                    "shape_ratio",
                                    "background_shape_ratio",
                                ),
                                "signal_to_background_ratio": (
                                    "physical_ratio",
                                    "signal_to_background_ratio",
                                ),
                            }
                            for output_name, (
                                modern,
                                legacy,
                            ) in diagnostic_sources.items():
                                values = np.asarray(
                                    _prediction(predictions, modern, legacy),
                                    dtype=np.float32,
                                )
                                if values.shape != applied.shape or not np.all(
                                    np.isfinite(values)
                                ):
                                    raise ValueError(
                                        f"model returned invalid {modern} diagnostics"
                                    )
                                diagnostics[output_name][active] = values

                    arrays[PRIMARY_OUTPUT_BRANCH] = removal
                    arrays[ANALYSIS_REGION_BRANCH] = analysis_region
                    if write_diagnostic_branches:
                        for name, values in diagnostics.items():
                            arrays[name] = values
                        arrays["weight_background_removed"] = observation_weights * removal

                    entry_count += size
                    analysis_region_count += int(np.count_nonzero(in_region))
                    predicted_count += int(np.count_nonzero(active))
                    removal_sum += float(np.sum(removal, dtype=np.float64))
                    if size:
                        removal_min = min(removal_min, float(np.min(removal)))
                        removal_max = max(removal_max, float(np.max(removal)))
                    output_tree.extend(arrays)

                application_metadata = {
                    "format_version": 2,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source": str(input_path),
                    "source_provenance": input_provenance,
                    "sample_kind": sample_kind,
                    "analysis_mass_window_GeV": list(ANALYSIS_MASS_WINDOW),
                    "analysis_mass_window_strict": True,
                    "analysis_region_definition": (
                        "reconstructed and lower < reco_m_ZZ < upper"
                    ),
                    "background_model_directory": str(models.background.root),
                    "background_model_manifest_sha256": (
                        models.background_manifest_sha256
                    ),
                    "correction_model_directory": str(models.correction.root),
                    "correction_model_manifest_sha256": (
                        models.correction_manifest_sha256
                    ),
                    "features": list(FEATURES),
                    "written_branches": list(written_branches),
                    "entries": entry_count,
                    "analysis_region_entries": analysis_region_count,
                    "predicted_entries": predicted_count,
                    "background_removal_weight_mean": (
                        removal_sum / entry_count if entry_count else None
                    ),
                    "background_removal_weight_min": (
                        removal_min if entry_count else None
                    ),
                    "background_removal_weight_max": (
                        removal_max if entry_count else None
                    ),
                    "application_rule": {
                        "data": "target_purity in analysis region; unity elsewhere",
                        "gg-h": "unity everywhere",
                        "zz": (
                            "data/MC physical ratio for every reconstructed row; "
                            "unity for non-reconstructed rows"
                        ),
                    }[sample_kind],
                    "effective_weight": (
                        "multiply the preserved weight by background_removal_weight"
                    ),
                    "diagnostic_branches_written": write_diagnostic_branches,
                    "original_weight_preserved": True,
                }
                for name, value in metadata_objects.items():
                    output_file[name] = value
                output_file["background_removal_metadata"] = json.dumps(
                    application_metadata, sort_keys=True
                )
        assert_file_unchanged(input_path, input_provenance)
        if overwrite or output_path == input_path:
            os.replace(temporary, output_path)
        else:
            try:
                os.link(temporary, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output file appeared while decorating: {output_path}; "
                    "rerun with --overwrite to replace it"
                ) from error
            temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return application_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decorate one merged data, ggH, or ZZ file with the common "
            "background-removal recipe"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-kind", choices=SAMPLE_KINDS, required=True)
    parser.add_argument("--background-model-dir", type=Path, required=True)
    parser.add_argument("--correction-model-dir", type=Path, required=True)
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
        args.background_model_dir,
        args.correction_model_dir,
        sample_kind=args.sample_kind,
        step_size=args.step_size,
        inference_batch_size=args.inference_batch_size,
        device=args.device,
        overwrite=args.overwrite,
        replace_existing_branches=args.replace_existing_branches,
        write_diagnostic_branches=args.write_diagnostic_branches,
    )
    print(
        f"Decorated {metadata['entries']} {args.sample_kind} events -> {args.output}; "
        f"analysis-region entries={metadata['analysis_region_entries']}; "
        f"mean background_removal_weight="
        f"{metadata['background_removal_weight_mean']:.7g}"
    )


if __name__ == "__main__":
    main()
