#!/usr/bin/env python3
"""Merge compact analysis samples and build luminosity-scaled Herwig pseudo-data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import uproot


TREE_NAME = "Analysis"
SAMPLE_PATTERNS = {
    "ZZ_pythia": "ZZ_pythia_*.root",
    "ZZ_herwig": "ZZ_herwig_*.root",
    "gg_H_pythia": "gg_H_pythia_*.root",
    "gg_H_herwig": "gg_H_herwig_*.root",
}
ESSENTIAL_BRANCHES = {"event_id", "weight", "reconstructed"}


@dataclass(frozen=True)
class SampleStats:
    entries: int
    sum_weights: float
    reconstructed_entries: int
    reconstructed_sum_weights: float
    cross_section_pb: float
    has_negative_weights: bool


def _branch_names(tree: object) -> set[str]:
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def _input_files(directory: Path, pattern: str) -> list[Path]:
    files = sorted(path.resolve() for path in directory.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"no inputs match {directory / pattern}")
    return files


def _validate_tree(path: Path, expected_branches: set[str] | None = None) -> set[str]:
    with uproot.open(path) as root_file:
        if TREE_NAME not in root_file:
            raise KeyError(f"{path} does not contain the {TREE_NAME} tree")
        branches = _branch_names(root_file[TREE_NAME])
    missing = sorted(ESSENTIAL_BRANCHES.difference(branches))
    if missing:
        raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
    if expected_branches is not None and branches != expected_branches:
        missing_here = sorted(expected_branches.difference(branches))
        extra_here = sorted(branches.difference(expected_branches))
        raise KeyError(
            f"{path} has a different schema; missing={missing_here}, extra={extra_here}"
        )
    return branches


def _cross_section_from_values(values: list[np.ndarray]) -> float:
    finite_positive = np.concatenate(
        [array[np.isfinite(array) & (array > 0.0)] for array in values]
    )
    if finite_positive.size == 0:
        return float("nan")
    # Shower converters may store a running cross-section estimate in every
    # event. The final positive value is the converged estimate for that job.
    return float(finite_positive[-1])


def scan_files(
    files: Iterable[Path],
    *,
    step_size: str,
    cross_section_override_pb: float | None = None,
) -> tuple[SampleStats, set[str]]:
    entries = 0
    sum_weights = 0.0
    reconstructed_entries = 0
    reconstructed_sum_weights = 0.0
    has_negative_weights = False
    cross_sections: list[tuple[float, int]] = []
    expected_branches: set[str] | None = None

    for path in files:
        branches = _validate_tree(path, expected_branches)
        if expected_branches is None:
            expected_branches = branches
        expressions = ["weight", "reconstructed"]
        has_cross_section = "cross_section_pb" in branches
        if has_cross_section:
            expressions.append("cross_section_pb")
        file_entries = 0
        file_cross_sections: list[np.ndarray] = []
        with uproot.open(path) as root_file:
            tree = root_file[TREE_NAME]
            for arrays in tree.iterate(
                expressions=expressions,
                step_size=step_size,
                library="np",
                how=dict,
            ):
                weights = np.asarray(arrays["weight"], dtype=np.float64)
                reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
                if not np.all(np.isfinite(weights)):
                    raise ValueError(f"{path} contains non-finite event weights")
                file_entries += weights.size
                entries += weights.size
                sum_weights += float(np.sum(weights, dtype=np.float64))
                reconstructed_entries += int(np.count_nonzero(reconstructed))
                reconstructed_sum_weights += float(
                    np.sum(weights[reconstructed], dtype=np.float64)
                )
                has_negative_weights |= bool(np.any(weights < 0.0))
                if has_cross_section:
                    file_cross_sections.append(
                        np.asarray(arrays["cross_section_pb"], dtype=np.float64)
                    )
        if file_entries == 0:
            raise ValueError(f"{path} contains no events")
        if has_cross_section:
            cross_sections.append(
                (_cross_section_from_values(file_cross_sections), file_entries)
            )

    if expected_branches is None or entries == 0:
        raise ValueError("sample contains no events")
    if not np.isfinite(sum_weights) or sum_weights <= 0.0:
        raise ValueError("sample has a non-positive total event weight")

    if cross_section_override_pb is not None:
        cross_section_pb = cross_section_override_pb
    else:
        valid = [(value, count) for value, count in cross_sections if np.isfinite(value)]
        cross_section_pb = (
            float(np.average([value for value, _ in valid], weights=[count for _, count in valid]))
            if valid
            else float("nan")
        )

    return (
        SampleStats(
            entries=entries,
            sum_weights=sum_weights,
            reconstructed_entries=reconstructed_entries,
            reconstructed_sum_weights=reconstructed_sum_weights,
            cross_section_pb=cross_section_pb,
            has_negative_weights=has_negative_weights,
        ),
        expected_branches,
    )


def _tree_schema(path: Path, branches: set[str]) -> dict[str, np.dtype]:
    schema: dict[str, np.dtype] = {}
    with uproot.open(path) as root_file:
        tree = root_file[TREE_NAME]
        for name in sorted(branches):
            # Reading one scalar works for both TTrees produced by the reducer
            # and RNTuples that users may create with uproot directly.
            schema[name] = np.asarray(
                tree[name].array(entry_start=0, entry_stop=1, library="np")
            ).dtype
    schema["cross_section_pb"] = np.dtype(np.float64)
    return schema


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary


def merge_sample(
    files: list[Path],
    output_path: Path,
    *,
    step_size: str,
    cross_section_override_pb: float | None,
    overwrite: bool,
) -> SampleStats:
    stats, branches = scan_files(
        files,
        step_size=step_size,
        cross_section_override_pb=cross_section_override_pb,
    )
    scale = stats.entries / stats.sum_weights
    schema = _tree_schema(files[0], branches)
    temporary = _prepare_output(output_path, overwrite)
    next_event_id = 0
    try:
        with uproot.recreate(temporary) as output_file:
            output_file.mktree(TREE_NAME, schema, title="Merged compact four-lepton sample")
            output_tree = output_file[TREE_NAME]
            for path in files:
                with uproot.open(path) as input_file:
                    for arrays in input_file[TREE_NAME].iterate(
                        step_size=step_size,
                        library="np",
                        how=dict,
                    ):
                        size = len(arrays["weight"])
                        arrays["event_id"] = np.arange(
                            next_event_id, next_event_id + size, dtype=np.uint64
                        )
                        arrays["weight"] = np.asarray(arrays["weight"], dtype=np.float64) * scale
                        arrays["cross_section_pb"] = np.full(
                            size, stats.cross_section_pb, dtype=np.float64
                        )
                        output_tree.extend(arrays)
                        next_event_id += size
            output_file["merge_metadata"] = json.dumps(
                {
                    "inputs": [str(path) for path in files],
                    "entries": stats.entries,
                    "input_sum_weights": stats.sum_weights,
                    "output_sum_weights": float(stats.entries),
                    "weight_scale": scale,
                    "cross_section_pb": stats.cross_section_pb,
                },
                sort_keys=True,
            )
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return SampleStats(
        entries=stats.entries,
        sum_weights=float(stats.entries),
        reconstructed_entries=stats.reconstructed_entries,
        reconstructed_sum_weights=stats.reconstructed_sum_weights * scale,
        cross_section_pb=stats.cross_section_pb,
        has_negative_weights=stats.has_negative_weights,
    )


def _pseudo_data_count(
    stats: SampleStats, luminosity_fb: float, rng: np.random.Generator
) -> tuple[float, int]:
    if not np.isfinite(stats.cross_section_pb) or stats.cross_section_pb <= 0.0:
        raise ValueError(
            "a positive cross section is required for pseudo-data; rebuild the Analysis "
            "inputs with the current reducer or pass a process cross-section override"
        )
    if stats.has_negative_weights:
        raise ValueError(
            "cannot turn signed NLO weights into unit-weight pseudo-data; use a positive-weight "
            "POWHEG sample or an explicit positive-probability resampling prescription"
        )
    efficiency = stats.reconstructed_sum_weights / stats.sum_weights
    expected = stats.cross_section_pb * luminosity_fb * 1000.0 * efficiency
    return expected, int(rng.poisson(expected))


def _empty_arrays(schema: dict[str, np.dtype]) -> dict[str, np.ndarray]:
    return {name: np.empty(0, dtype=dtype) for name, dtype in schema.items()}


def _sample_reconstructed(
    path: Path,
    count: int,
    *,
    reconstructed_entries: int,
    rng: np.random.Generator,
    step_size: str,
) -> dict[str, np.ndarray]:
    with uproot.open(path) as root_file:
        tree = root_file[TREE_NAME]
        branches = _branch_names(tree)
        schema = _tree_schema(path, branches)
        if count == 0:
            return _empty_arrays(schema)
        if count > reconstructed_entries:
            raise ValueError(
                f"{path} has only {reconstructed_entries} reconstructed events but the "
                f"pseudo-data draw requests {count}; generate more Herwig events"
            )

        # Exponential-race keys provide probability-proportional-to-weight
        # sampling without replacement. Keep only the best `count` global
        # entry indices, so memory scales with the pseudo-data size rather
        # than with the full Monte Carlo sample.
        selected_keys = np.empty(0, dtype=np.float64)
        selected_indices = np.empty(0, dtype=np.int64)
        entry_offset = 0
        for arrays in tree.iterate(
            expressions=["weight", "reconstructed"],
            step_size=step_size,
            library="np",
            how=dict,
        ):
            weights = np.asarray(arrays["weight"], dtype=np.float64)
            reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
            eligible = reconstructed & (weights > 0.0)
            eligible_weights = weights[eligible]
            if eligible_weights.size:
                uniforms = np.maximum(
                    rng.random(eligible_weights.size), np.finfo(np.float64).tiny
                )
                keys = np.log(uniforms) / eligible_weights
                indices = np.flatnonzero(eligible).astype(np.int64) + entry_offset
                selected_keys = np.concatenate((selected_keys, keys))
                selected_indices = np.concatenate((selected_indices, indices))
                if selected_keys.size > count:
                    keep = np.argpartition(selected_keys, -count)[-count:]
                    selected_keys = selected_keys[keep]
                    selected_indices = selected_indices[keep]
            entry_offset += weights.size

        if selected_indices.size != count:
            raise RuntimeError(f"failed to select all requested events from {path}")
        selected_indices.sort()

        pieces: dict[str, list[np.ndarray]] = {name: [] for name in schema}
        entry_offset = 0
        selected_start = 0
        for arrays in tree.iterate(step_size=step_size, library="np", how=dict):
            size = len(arrays["weight"])
            upper = entry_offset + size
            selected_stop = int(np.searchsorted(selected_indices, upper, side="left"))
            if selected_stop > selected_start:
                local = selected_indices[selected_start:selected_stop] - entry_offset
                for name in schema:
                    pieces[name].append(np.asarray(arrays[name])[local])
            selected_start = selected_stop
            entry_offset = upper
        if selected_start != count:
            raise RuntimeError(f"failed to sample all requested events from {path}")
    return {
        name: np.concatenate(values) if values else np.empty(0, dtype=schema[name])
        for name, values in pieces.items()
    }


def build_pseudo_data(
    zz_path: Path,
    higgs_path: Path,
    output_path: Path,
    *,
    luminosity_fb: float,
    seed: int,
    step_size: str,
    overwrite: bool,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    components: list[tuple[str, Path, SampleStats, float, int]] = []
    for name, path in (("ZZ", zz_path), ("gg_H", higgs_path)):
        stats, _ = scan_files([path], step_size=step_size)
        expected, observed = _pseudo_data_count(stats, luminosity_fb, rng)
        components.append((name, path, stats, expected, observed))

    sampled = [
        _sample_reconstructed(
            path,
            observed,
            reconstructed_entries=stats.reconstructed_entries,
            rng=rng,
            step_size=step_size,
        )
        for _, path, stats, _, observed in components
    ]
    branch_names = set(sampled[0])
    if any(set(component) != branch_names for component in sampled[1:]):
        raise KeyError("the ZZ and gg_H merged trees have different schemas")
    combined = {
        name: np.concatenate([component[name] for component in sampled])
        for name in sorted(branch_names)
    }
    total = len(combined["weight"])
    order = rng.permutation(total)
    for name in combined:
        combined[name] = combined[name][order]
    combined["event_id"] = np.arange(total, dtype=np.uint64)
    combined["weight"] = np.ones(total, dtype=np.float64)
    combined["reconstructed"] = np.ones(total, dtype=np.bool_)
    combined["cross_section_pb"] = np.full(total, np.nan, dtype=np.float64)

    schema = {name: values.dtype for name, values in combined.items()}
    temporary = _prepare_output(output_path, overwrite)
    metadata: dict[str, float | int] = {
        "luminosity_fb": luminosity_fb,
        "seed": seed,
        "ZZ_expected": components[0][3],
        "ZZ_observed": components[0][4],
        "gg_H_expected": components[1][3],
        "gg_H_observed": components[1][4],
        "total_observed": total,
    }
    try:
        with uproot.recreate(temporary) as output_file:
            output_file.mktree(TREE_NAME, schema, title="Unit-weight Herwig pseudo-data")
            output_file[TREE_NAME].extend(combined)
            output_file["merge_metadata"] = json.dumps(metadata, sort_keys=True)
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return metadata


def merge_directory(
    input_directory: Path,
    output_directory: Path,
    *,
    luminosity_fb: float,
    seed: int,
    step_size: str,
    zz_cross_section_pb: float | None,
    higgs_cross_section_pb: float | None,
    overwrite: bool,
) -> dict[str, SampleStats]:
    input_directory = input_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if not input_directory.is_dir():
        raise NotADirectoryError(input_directory)
    outputs: dict[str, SampleStats] = {}
    for sample, pattern in SAMPLE_PATTERNS.items():
        override = higgs_cross_section_pb if sample.startswith("gg_H") else zz_cross_section_pb
        files = _input_files(input_directory, pattern)
        output_path = output_directory / f"{sample}.root"
        outputs[sample] = merge_sample(
            files,
            output_path,
            step_size=step_size,
            cross_section_override_pb=override,
            overwrite=overwrite,
        )
        print(
            f"{sample}: {len(files)} files -> {output_path}; "
            f"entries={outputs[sample].entries}, sum(weight)={outputs[sample].sum_weights:.8g}, "
            f"cross_section_pb={outputs[sample].cross_section_pb:.8g}"
        )

    metadata = build_pseudo_data(
        output_directory / "ZZ_herwig.root",
        output_directory / "gg_H_herwig.root",
        output_directory / "data.root",
        luminosity_fb=luminosity_fb,
        seed=seed,
        step_size=step_size,
        overwrite=overwrite,
    )
    print(
        f"data: {output_directory / 'data.root'}; luminosity={luminosity_fb:g} fb^-1, "
        f"ZZ={metadata['ZZ_observed']} (expected {metadata['ZZ_expected']:.3f}), "
        f"gg_H={metadata['gg_H_observed']} (expected {metadata['gg_H_expected']:.3f})"
    )
    return outputs


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        help="output directory (default: input directory)",
    )
    parser.add_argument("--luminosity-fb", type=_positive_float, default=300.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--step-size", default="100 MB", help="uproot chunk size")
    parser.add_argument("--zz-cross-section-pb", type=_positive_float)
    parser.add_argument("--gg-h-cross-section-pb", type=_positive_float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_directory = args.output_directory or args.input_directory
    merge_directory(
        args.input_directory,
        output_directory,
        luminosity_fb=args.luminosity_fb,
        seed=args.seed,
        step_size=args.step_size,
        zz_cross_section_pb=args.zz_cross_section_pb,
        higgs_cross_section_pb=args.gg_h_cross_section_pb,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
