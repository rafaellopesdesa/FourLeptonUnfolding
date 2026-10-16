#!/usr/bin/env python3
"""Merge compact samples and build luminosity-scaled signed Herwig pseudo-data ensembles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Iterable

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
DEFAULT_LUMINOSITY_FB = 312.0
PUBLICATION_MARKER = ".publication-in-progress"


@dataclass(frozen=True)
class SampleStats:
    entries: int
    sum_weights: float
    sum_squared_weights: float
    reconstructed_entries: int
    reconstructed_sum_weights: float
    reconstructed_sum_squared_weights: float
    reconstructed_positive_entries: int
    reconstructed_negative_entries: int
    reconstructed_positive_sum_weights: float
    reconstructed_negative_sum_abs_weights: float
    reconstructed_positive_sum_squared_weights: float
    reconstructed_negative_sum_squared_weights: float
    cross_section_pb: float
    has_negative_weights: bool


@dataclass(frozen=True)
class SignedExpectation:
    positive: float
    negative: float

    @property
    def net(self) -> float:
        return self.positive - self.negative


@dataclass(frozen=True)
class SourceSnapshot:
    """Stable identity and content digest for a source used in multiple passes."""

    path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PublishedPath:
    """Identity of a file installed by one campaign publication attempt."""

    path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    mode: int
    sha256: str


class CampaignPublicationError(RuntimeError):
    """Publication failure, optionally retaining staging for manual recovery."""

    def __init__(
        self,
        message: str,
        *,
        retained_staging_directory: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.retained_staging_directory = retained_staging_directory


@dataclass(frozen=True)
class PseudoDataComponent:
    name: str
    path: Path
    stats: SampleStats
    expectation: SignedExpectation
    source_snapshot: SourceSnapshot | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _snapshot_source(path: Path) -> SourceSnapshot:
    resolved = path.expanduser().resolve(strict=True)
    before = _stat_identity(resolved)
    sha256 = _sha256_file(resolved)
    after = _stat_identity(resolved)
    if after != before:
        raise RuntimeError(
            f"source file changed while its fingerprint was being recorded: {resolved}"
        )
    return SourceSnapshot(resolved, *before, sha256)


def _assert_source_unchanged(
    snapshot: SourceSnapshot,
    *,
    context: str,
    verify_sha256: bool = True,
) -> None:
    try:
        identity = _stat_identity(snapshot.path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"source file disappeared {context}: {snapshot.path}"
        ) from error
    expected_identity = (
        snapshot.device,
        snapshot.inode,
        snapshot.size_bytes,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if identity != expected_identity:
        raise RuntimeError(
            f"source file changed {context}: {snapshot.path}; refusing to combine "
            "statistics and events from different file versions"
        )
    if verify_sha256:
        sha256 = _sha256_file(snapshot.path)
        if _stat_identity(snapshot.path) != expected_identity:
            raise RuntimeError(
                f"source file changed {context}: {snapshot.path}; refusing to combine "
                "statistics and events from different file versions"
            )
        if sha256 != snapshot.sha256:
            raise RuntimeError(
                f"source file content changed {context}: {snapshot.path}; refusing to "
                "combine statistics and events from different file versions"
            )


def _assert_sources_unchanged(
    snapshots: Iterable[SourceSnapshot],
    *,
    context: str,
    verify_sha256: bool = True,
) -> None:
    for snapshot in snapshots:
        _assert_source_unchanged(
            snapshot,
            context=context,
            verify_sha256=verify_sha256,
        )


def _draw_signed_counts(
    expectation: SignedExpectation,
    positive_rng: np.random.Generator,
    negative_rng: np.random.Generator,
) -> tuple[int, int]:
    """Draw the independent positive and negative Poisson components."""

    return (
        int(positive_rng.poisson(expectation.positive)),
        int(negative_rng.poisson(expectation.negative)),
    )


def _effective_entries(sum_weights: float, sum_squared_weights: float) -> float:
    if sum_weights <= 0.0 or sum_squared_weights <= 0.0:
        return 0.0
    return sum_weights * sum_weights / sum_squared_weights


def _reconstructed_effective_entries(stats: SampleStats, sign: int) -> float:
    if sign > 0:
        return _effective_entries(
            stats.reconstructed_positive_sum_weights,
            stats.reconstructed_positive_sum_squared_weights,
        )
    return _effective_entries(
        stats.reconstructed_negative_sum_abs_weights,
        stats.reconstructed_negative_sum_squared_weights,
    )


def _reconstructed_sign_sum_weights(stats: SampleStats, sign: int) -> float:
    if sign > 0:
        return stats.reconstructed_positive_sum_weights
    return stats.reconstructed_negative_sum_abs_weights


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
    source_snapshots: Iterable[SourceSnapshot] | None = None,
) -> tuple[SampleStats, set[str]]:
    file_list = list(files)
    snapshots = (
        list(source_snapshots)
        if source_snapshots is not None
        else [_snapshot_source(path) for path in file_list]
    )
    if len(snapshots) != len(file_list):
        raise ValueError("the source snapshot count does not match the input file count")
    entries = 0
    sum_weights = 0.0
    sum_squared_weights = 0.0
    reconstructed_entries = 0
    reconstructed_sum_weights = 0.0
    reconstructed_sum_squared_weights = 0.0
    reconstructed_positive_entries = 0
    reconstructed_negative_entries = 0
    reconstructed_positive_sum_weights = 0.0
    reconstructed_negative_sum_abs_weights = 0.0
    reconstructed_positive_sum_squared_weights = 0.0
    reconstructed_negative_sum_squared_weights = 0.0
    has_negative_weights = False
    cross_sections: list[tuple[float, int]] = []
    expected_branches: set[str] | None = None

    for path, snapshot in zip(file_list, snapshots, strict=True):
        _assert_source_unchanged(
            snapshot,
            context="before the statistics scan",
            verify_sha256=False,
        )
        read_path = snapshot.path
        branches = _validate_tree(read_path, expected_branches)
        if expected_branches is None:
            expected_branches = branches
        expressions = ["weight", "reconstructed"]
        has_cross_section = "cross_section_pb" in branches
        if has_cross_section:
            expressions.append("cross_section_pb")
        file_entries = 0
        file_cross_sections: list[np.ndarray] = []
        with uproot.open(read_path) as root_file:
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
                    raise ValueError(f"{read_path} contains non-finite event weights")
                file_entries += weights.size
                entries += weights.size
                sum_weights += float(np.sum(weights, dtype=np.float64))
                sum_squared_weights += float(
                    np.sum(weights * weights, dtype=np.float64)
                )
                reconstructed_entries += int(np.count_nonzero(reconstructed))
                reconstructed_weights = weights[reconstructed]
                reconstructed_sum_weights += float(
                    np.sum(reconstructed_weights, dtype=np.float64)
                )
                reconstructed_sum_squared_weights += float(
                    np.sum(reconstructed_weights * reconstructed_weights, dtype=np.float64)
                )
                positive_weights = reconstructed_weights[reconstructed_weights > 0.0]
                negative_abs_weights = -reconstructed_weights[reconstructed_weights < 0.0]
                reconstructed_positive_entries += int(positive_weights.size)
                reconstructed_negative_entries += int(negative_abs_weights.size)
                reconstructed_positive_sum_weights += float(
                    np.sum(positive_weights, dtype=np.float64)
                )
                reconstructed_negative_sum_abs_weights += float(
                    np.sum(negative_abs_weights, dtype=np.float64)
                )
                reconstructed_positive_sum_squared_weights += float(
                    np.sum(positive_weights * positive_weights, dtype=np.float64)
                )
                reconstructed_negative_sum_squared_weights += float(
                    np.sum(negative_abs_weights * negative_abs_weights, dtype=np.float64)
                )
                has_negative_weights |= bool(np.any(weights < 0.0))
                if has_cross_section:
                    file_cross_sections.append(
                        np.asarray(arrays["cross_section_pb"], dtype=np.float64)
                    )
        if file_entries == 0:
            raise ValueError(f"{read_path} contains no events")
        if has_cross_section:
            cross_sections.append(
                (_cross_section_from_values(file_cross_sections), file_entries)
            )
        _assert_source_unchanged(
            snapshot,
            context="during the statistics scan",
        )

    if expected_branches is None or entries == 0:
        raise ValueError("sample contains no events")
    if not np.isfinite(sum_weights) or sum_weights <= 0.0:
        raise ValueError("sample has a non-positive total event weight")
    aggregate_statistics = (
        sum_squared_weights,
        reconstructed_sum_weights,
        reconstructed_sum_squared_weights,
        reconstructed_positive_sum_weights,
        reconstructed_negative_sum_abs_weights,
        reconstructed_positive_sum_squared_weights,
        reconstructed_negative_sum_squared_weights,
    )
    if not all(np.isfinite(value) for value in aggregate_statistics):
        raise ValueError("sample has non-finite aggregate weight statistics")

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
            sum_squared_weights=sum_squared_weights,
            reconstructed_entries=reconstructed_entries,
            reconstructed_sum_weights=reconstructed_sum_weights,
            reconstructed_sum_squared_weights=reconstructed_sum_squared_weights,
            reconstructed_positive_entries=reconstructed_positive_entries,
            reconstructed_negative_entries=reconstructed_negative_entries,
            reconstructed_positive_sum_weights=reconstructed_positive_sum_weights,
            reconstructed_negative_sum_abs_weights=reconstructed_negative_sum_abs_weights,
            reconstructed_positive_sum_squared_weights=(
                reconstructed_positive_sum_squared_weights
            ),
            reconstructed_negative_sum_squared_weights=(
                reconstructed_negative_sum_squared_weights
            ),
            cross_section_pb=cross_section_pb,
            has_negative_weights=has_negative_weights,
        ),
        expected_branches,
    )


def _tree_schema(
    path: Path, branches: set[str], *, include_merged_weights: bool = False
) -> dict[str, np.dtype]:
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
    if include_merged_weights:
        schema["weight_shape"] = np.dtype(np.float64)
        schema["weight_nominal_pb"] = np.dtype(np.float64)
        schema["lumi"] = np.dtype(np.float64)
        schema["luminosity_fb"] = np.dtype(np.float64)
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


def _assert_close(label: str, observed: float, expected: float) -> None:
    if not np.isclose(observed, expected, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError(
            f"{label} changed between merge passes: observed {observed:.17g}, "
            f"expected {expected:.17g}"
        )


def _validate_merged_output(
    path: Path,
    *,
    entries: int,
    sum_weight_shape: float,
    sum_weight_nominal_pb: float,
    sum_weights: float,
    sum_squared_weights: float,
    step_size: str,
) -> None:
    totals = {
        "weight_shape": 0.0,
        "weight_nominal_pb": 0.0,
        "weight": 0.0,
        "weight_squared": 0.0,
    }
    counted_entries = 0
    with uproot.open(path) as root_file:
        tree = root_file[TREE_NAME]
        if int(tree.num_entries) != entries:
            raise RuntimeError(
                f"merged output has {tree.num_entries} entries, expected {entries}"
            )
        for arrays in tree.iterate(
            expressions=["weight_shape", "weight_nominal_pb", "weight"],
            step_size=step_size,
            library="np",
            how=dict,
        ):
            weights = np.asarray(arrays["weight"], dtype=np.float64)
            counted_entries += weights.size
            totals["weight_shape"] += float(
                np.sum(arrays["weight_shape"], dtype=np.float64)
            )
            totals["weight_nominal_pb"] += float(
                np.sum(arrays["weight_nominal_pb"], dtype=np.float64)
            )
            totals["weight"] += float(np.sum(weights, dtype=np.float64))
            totals["weight_squared"] += float(
                np.sum(weights * weights, dtype=np.float64)
            )
    if counted_entries != entries:
        raise RuntimeError(
            f"merged output iteration found {counted_entries} entries, expected {entries}"
        )
    _assert_close("merged shape-weight sum", totals["weight_shape"], sum_weight_shape)
    _assert_close(
        "merged nominal-pb weight sum",
        totals["weight_nominal_pb"],
        sum_weight_nominal_pb,
    )
    _assert_close("merged event-weight sum", totals["weight"], sum_weights)
    _assert_close(
        "merged event-weight squared sum",
        totals["weight_squared"],
        sum_squared_weights,
    )


def merge_sample(
    files: list[Path],
    output_path: Path,
    *,
    step_size: str,
    cross_section_override_pb: float | None,
    luminosity_fb: float = DEFAULT_LUMINOSITY_FB,
    overwrite: bool,
) -> SampleStats:
    source_snapshots = [_snapshot_source(path) for path in files]
    stats, branches = scan_files(
        files,
        step_size=step_size,
        cross_section_override_pb=cross_section_override_pb,
        source_snapshots=source_snapshots,
    )
    if not np.isfinite(luminosity_fb) or luminosity_fb <= 0.0:
        raise ValueError("merged-sample luminosity must be finite and positive")
    if not np.isfinite(stats.cross_section_pb) or stats.cross_section_pb <= 0.0:
        raise ValueError(
            "a positive cross section is required to construct luminosity-scaled "
            "merged weights; rebuild the Analysis inputs or pass a process "
            "cross-section override"
        )
    luminosity_pb_inverse = luminosity_fb * 1000.0
    shape_scale = stats.entries / stats.sum_weights
    nominal_pb_scale = stats.cross_section_pb / stats.sum_weights
    yield_scale = nominal_pb_scale * luminosity_pb_inverse
    first_snapshot = source_snapshots[0]
    schema = _tree_schema(
        first_snapshot.path,
        branches,
        include_merged_weights=True,
    )
    _assert_sources_unchanged(
        source_snapshots,
        context="between the statistics and schema passes",
        verify_sha256=False,
    )
    temporary = _prepare_output(output_path, overwrite)
    next_event_id = 0
    written_input_sum_weights = 0.0
    written_input_sum_squared_weights = 0.0
    try:
        with uproot.recreate(temporary) as output_file:
            output_file.mktree(TREE_NAME, schema, title="Merged compact four-lepton sample")
            output_tree = output_file[TREE_NAME]
            for snapshot in source_snapshots:
                _assert_source_unchanged(
                    snapshot,
                    context="before the merge write pass",
                    verify_sha256=False,
                )
                with uproot.open(snapshot.path) as input_file:
                    for arrays in input_file[TREE_NAME].iterate(
                        step_size=step_size,
                        library="np",
                        how=dict,
                    ):
                        size = len(arrays["weight"])
                        arrays["event_id"] = np.arange(
                            next_event_id, next_event_id + size, dtype=np.uint64
                        )
                        input_weight = np.asarray(
                            arrays["weight"], dtype=np.float64
                        )
                        written_input_sum_weights += float(
                            np.sum(input_weight, dtype=np.float64)
                        )
                        written_input_sum_squared_weights += float(
                            np.sum(input_weight * input_weight, dtype=np.float64)
                        )
                        arrays["weight_shape"] = input_weight * shape_scale
                        arrays["weight_nominal_pb"] = (
                            input_weight * nominal_pb_scale
                        )
                        arrays["lumi"] = np.full(
                            size, luminosity_pb_inverse, dtype=np.float64
                        )
                        arrays["luminosity_fb"] = np.full(
                            size, luminosity_fb, dtype=np.float64
                        )
                        arrays["weight"] = (
                            arrays["weight_nominal_pb"] * arrays["lumi"]
                        )
                        arrays["cross_section_pb"] = np.full(
                            size, stats.cross_section_pb, dtype=np.float64
                        )
                        output_tree.extend(arrays)
                        next_event_id += size
                _assert_source_unchanged(
                    snapshot,
                    context="during the merge write pass",
                    verify_sha256=False,
                )
            if next_event_id != stats.entries:
                raise RuntimeError(
                    f"merge write pass found {next_event_id} entries, but the statistics "
                    f"pass found {stats.entries}"
                )
            _assert_close(
                "input event-weight sum",
                written_input_sum_weights,
                stats.sum_weights,
            )
            _assert_close(
                "input event-weight squared sum",
                written_input_sum_squared_weights,
                stats.sum_squared_weights,
            )
            output_file["merge_metadata"] = json.dumps(
                {
                    "format_version": 2,
                    "weight_semantics": "expected_events_at_luminosity",
                    "weight_units": "events",
                    "inputs": [str(path) for path in files],
                    "input_sources": [
                        snapshot.as_metadata() for snapshot in source_snapshots
                    ],
                    "entries": stats.entries,
                    "input_sum_weights": stats.sum_weights,
                    "input_sum_squared_weights": stats.sum_squared_weights,
                    "output_sum_weight_shape": float(stats.entries),
                    "output_sum_squared_weight_shape": (
                        stats.sum_squared_weights * shape_scale * shape_scale
                    ),
                    "output_sum_weight_nominal_pb": stats.cross_section_pb,
                    "output_sum_squared_weight_nominal_pb": (
                        stats.sum_squared_weights
                        * nominal_pb_scale
                        * nominal_pb_scale
                    ),
                    "output_sum_weights": (
                        stats.cross_section_pb * luminosity_pb_inverse
                    ),
                    "output_sum_squared_weights": (
                        stats.sum_squared_weights * yield_scale * yield_scale
                    ),
                    "shape_weight_scale": shape_scale,
                    "weight_nominal_pb_scale": nominal_pb_scale,
                    "yield_weight_scale": yield_scale,
                    "weight_formula": "weight = weight_nominal_pb * lumi",
                    "weight_nominal_pb_formula": (
                        "input_weight * cross_section_pb / input_sum_weights"
                    ),
                    "luminosity_fb": luminosity_fb,
                    "lumi_pb_inverse": luminosity_pb_inverse,
                    "cross_section_pb": stats.cross_section_pb,
                    "has_negative_weights": stats.has_negative_weights,
                    "reconstructed_positive_entries": (
                        stats.reconstructed_positive_entries
                    ),
                    "reconstructed_negative_entries": (
                        stats.reconstructed_negative_entries
                    ),
                    "reconstructed_positive_sum_weights": (
                        stats.reconstructed_positive_sum_weights * yield_scale
                    ),
                    "reconstructed_negative_sum_abs_weights": (
                        stats.reconstructed_negative_sum_abs_weights * yield_scale
                    ),
                    "reconstructed_positive_effective_entries": (
                        _reconstructed_effective_entries(stats, +1)
                    ),
                    "reconstructed_negative_effective_entries": (
                        _reconstructed_effective_entries(stats, -1)
                    ),
                },
                sort_keys=True,
            )
        _assert_sources_unchanged(
            source_snapshots,
            context="before finalizing the merged output",
        )
        _validate_merged_output(
            temporary,
            entries=stats.entries,
            sum_weight_shape=float(stats.entries),
            sum_weight_nominal_pb=stats.cross_section_pb,
            sum_weights=stats.cross_section_pb * luminosity_pb_inverse,
            sum_squared_weights=(
                stats.sum_squared_weights * yield_scale * yield_scale
            ),
            step_size=step_size,
        )
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return SampleStats(
        entries=stats.entries,
        sum_weights=stats.cross_section_pb * luminosity_pb_inverse,
        sum_squared_weights=stats.sum_squared_weights * yield_scale * yield_scale,
        reconstructed_entries=stats.reconstructed_entries,
        reconstructed_sum_weights=stats.reconstructed_sum_weights * yield_scale,
        reconstructed_sum_squared_weights=(
            stats.reconstructed_sum_squared_weights * yield_scale * yield_scale
        ),
        reconstructed_positive_entries=stats.reconstructed_positive_entries,
        reconstructed_negative_entries=stats.reconstructed_negative_entries,
        reconstructed_positive_sum_weights=(
            stats.reconstructed_positive_sum_weights * yield_scale
        ),
        reconstructed_negative_sum_abs_weights=(
            stats.reconstructed_negative_sum_abs_weights * yield_scale
        ),
        reconstructed_positive_sum_squared_weights=(
            stats.reconstructed_positive_sum_squared_weights
            * yield_scale
            * yield_scale
        ),
        reconstructed_negative_sum_squared_weights=(
            stats.reconstructed_negative_sum_squared_weights
            * yield_scale
            * yield_scale
        ),
        cross_section_pb=stats.cross_section_pb,
        has_negative_weights=stats.has_negative_weights,
    )


def _signed_expectation(
    stats: SampleStats, luminosity_fb: float
) -> SignedExpectation:
    if not np.isfinite(luminosity_fb) or luminosity_fb <= 0.0:
        raise ValueError("pseudo-data luminosity must be finite and positive")
    if not np.isfinite(stats.cross_section_pb) or stats.cross_section_pb <= 0.0:
        raise ValueError(
            "a positive cross section is required for pseudo-data; rebuild the Analysis "
            "inputs with the current reducer or pass a process cross-section override"
        )
    normalization = stats.cross_section_pb * luminosity_fb * 1000.0 / stats.sum_weights
    expectation = SignedExpectation(
        positive=normalization * stats.reconstructed_positive_sum_weights,
        negative=normalization * stats.reconstructed_negative_sum_abs_weights,
    )
    if stats.reconstructed_sum_weights <= 0.0:
        raise ValueError(
            "the reconstructed signed cross section is non-positive; the available Monte Carlo "
            "sample cannot define pseudo-data"
        )
    return expectation


def _effective_luminosity_fb(stats: SampleStats, sign: int) -> float | None:
    sign_sum_weights = _reconstructed_sign_sum_weights(stats, sign)
    if sign_sum_weights <= 0.0:
        return None
    rate_per_fb = (
        stats.cross_section_pb * 1000.0 * sign_sum_weights / stats.sum_weights
    )
    if rate_per_fb <= 0.0:
        return None
    return _reconstructed_effective_entries(stats, sign) / rate_per_fb


def _net_effective_luminosity_fb(stats: SampleStats) -> float | None:
    if stats.reconstructed_sum_weights <= 0.0:
        return None
    effective_entries = _effective_entries(
        stats.reconstructed_sum_weights,
        stats.reconstructed_sum_squared_weights,
    )
    rate_per_fb = (
        stats.cross_section_pb
        * 1000.0
        * stats.reconstructed_sum_weights
        / stats.sum_weights
    )
    if rate_per_fb <= 0.0:
        return None
    return effective_entries / rate_per_fb


def _component_diagnostics(component: PseudoDataComponent) -> dict[str, Any]:
    stats = component.stats
    diagnostics = {
        "source": str(component.path),
        "cross_section_pb": stats.cross_section_pb,
        "inclusive_sum_weights": stats.sum_weights,
        "inclusive_sum_squared_weights": stats.sum_squared_weights,
        "reconstructed_sum_weights": stats.reconstructed_sum_weights,
        "reconstructed_sum_squared_weights": (
            stats.reconstructed_sum_squared_weights
        ),
        "reconstructed_positive_entries": stats.reconstructed_positive_entries,
        "reconstructed_negative_entries": stats.reconstructed_negative_entries,
        "reconstructed_positive_sum_weights": (
            stats.reconstructed_positive_sum_weights
        ),
        "reconstructed_negative_sum_abs_weights": (
            stats.reconstructed_negative_sum_abs_weights
        ),
        "reconstructed_positive_sum_squared_weights": (
            stats.reconstructed_positive_sum_squared_weights
        ),
        "reconstructed_negative_sum_squared_weights": (
            stats.reconstructed_negative_sum_squared_weights
        ),
        "reconstructed_positive_effective_entries": (
            _reconstructed_effective_entries(stats, +1)
        ),
        "reconstructed_negative_effective_entries": (
            _reconstructed_effective_entries(stats, -1)
        ),
        "reconstructed_net_effective_entries": _effective_entries(
            stats.reconstructed_sum_weights,
            stats.reconstructed_sum_squared_weights,
        ),
        "reconstructed_positive_effective_luminosity_fb": (
            _effective_luminosity_fb(stats, +1)
        ),
        "reconstructed_negative_effective_luminosity_fb": (
            _effective_luminosity_fb(stats, -1)
        ),
        "reconstructed_net_effective_luminosity_fb": (
            _net_effective_luminosity_fb(stats)
        ),
        "expected_positive": component.expectation.positive,
        "expected_negative": component.expectation.negative,
        "expected_net": component.expectation.net,
    }
    if component.source_snapshot is not None:
        diagnostics["source_snapshot"] = component.source_snapshot.as_metadata()
    return diagnostics


def _empty_arrays(schema: dict[str, np.dtype]) -> dict[str, np.ndarray]:
    return {name: np.empty(0, dtype=dtype) for name, dtype in schema.items()}


def _sample_reconstructed(
    path: Path,
    count: int,
    *,
    sign: int,
    total_abs_weight: float,
    rng: np.random.Generator,
    step_size: str,
    source_snapshot: SourceSnapshot | None = None,
) -> dict[str, np.ndarray]:
    if sign not in (-1, +1):
        raise ValueError("the sampling sign must be +1 or -1")
    snapshot = source_snapshot or _snapshot_source(path)
    _assert_source_unchanged(
        snapshot,
        context="before pseudo-data sampling",
        verify_sha256=False,
    )
    with uproot.open(snapshot.path) as root_file:
        tree = root_file[TREE_NAME]
        branches = _branch_names(tree)
        schema = _tree_schema(snapshot.path, branches)
        _assert_source_unchanged(
            snapshot,
            context="during pseudo-data schema discovery",
            verify_sha256=False,
        )
        if count == 0:
            _assert_source_unchanged(
                snapshot,
                context="during zero-count pseudo-data sampling",
                verify_sha256=source_snapshot is None,
            )
            return _empty_arrays(schema)
        if not np.isfinite(total_abs_weight) or total_abs_weight <= 0.0:
            raise ValueError(
                f"{path} has no reconstructed events with sign {sign:+d}, but the "
                f"pseudo-data draw requests {count}"
            )

        # Inverse-CDF sampling provides an exact probability-proportional-to-|w|
        # bootstrap with replacement. Sorted targets let us map the draws to
        # global entry indices in one streaming pass without holding every
        # source weight in memory.
        targets = np.sort(rng.random(count) * total_abs_weight)
        selected_indices = np.empty(count, dtype=np.int64)
        target_start = 0
        cumulative_offset = 0.0
        entry_offset = 0
        for arrays in tree.iterate(
            expressions=["weight", "reconstructed"],
            step_size=step_size,
            library="np",
            how=dict,
        ):
            weights = np.asarray(arrays["weight"], dtype=np.float64)
            reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
            eligible = reconstructed & ((weights > 0.0) if sign > 0 else (weights < 0.0))
            eligible_weights = np.abs(weights[eligible])
            chunk_weight = float(np.sum(eligible_weights, dtype=np.float64))
            if chunk_weight > 0.0:
                cumulative_stop = cumulative_offset + chunk_weight
                target_stop = int(
                    np.searchsorted(targets, cumulative_stop, side="left")
                )
                if target_stop > target_start:
                    local_targets = (
                        targets[target_start:target_stop] - cumulative_offset
                    )
                    local_cumulative = np.cumsum(eligible_weights, dtype=np.float64)
                    positions = np.searchsorted(
                        local_cumulative, local_targets, side="right"
                    )
                    eligible_indices = np.flatnonzero(eligible).astype(np.int64)
                    selected_indices[target_start:target_stop] = (
                        eligible_indices[positions] + entry_offset
                    )
                    target_start = target_stop
                cumulative_offset = cumulative_stop
            entry_offset += weights.size

        if target_start != count:
            raise RuntimeError(
                f"failed to map all {count} sign {sign:+d} bootstrap draws from {path}; "
                "the stored signed-weight statistics are inconsistent with the tree"
            )
        _assert_source_unchanged(
            snapshot,
            context="between pseudo-data selection and materialization passes",
            verify_sha256=False,
        )

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
    _assert_source_unchanged(
        snapshot,
        context="during pseudo-data sampling",
        verify_sha256=source_snapshot is None,
    )
    return {
        name: np.concatenate(values) if values else np.empty(0, dtype=schema[name])
        for name, values in pieces.items()
    }


def _pseudo_data_components(
    zz_path: Path,
    higgs_path: Path,
    *,
    luminosity_fb: float,
    step_size: str,
) -> list[PseudoDataComponent]:
    components: list[PseudoDataComponent] = []
    for name, path in (("ZZ", zz_path), ("gg_H", higgs_path)):
        snapshot = _snapshot_source(path)
        stats, _ = scan_files(
            [path],
            step_size=step_size,
            source_snapshots=[snapshot],
        )
        components.append(
            PseudoDataComponent(
                name=name,
                path=path,
                stats=stats,
                expectation=_signed_expectation(stats, luminosity_fb),
                source_snapshot=snapshot,
            )
        )
    return components


def _validate_pseudo_data_output(
    path: Path,
    *,
    entries: int,
    observed_positive: int,
    observed_negative: int,
    step_size: str,
) -> None:
    counted_entries = 0
    counted_positive = 0
    counted_negative = 0
    sum_weights = 0.0
    sum_squared_weights = 0.0
    with uproot.open(path) as root_file:
        tree = root_file[TREE_NAME]
        if int(tree.num_entries) != entries:
            raise RuntimeError(
                f"pseudo-data output has {tree.num_entries} entries, expected {entries}"
            )
        for arrays in tree.iterate(
            expressions=["weight"],
            step_size=step_size,
            library="np",
            how=dict,
        ):
            weights = np.asarray(arrays["weight"], dtype=np.float64)
            if np.any((weights != 1.0) & (weights != -1.0)):
                raise RuntimeError("pseudo-data output contains a non-unit signed weight")
            counted_entries += weights.size
            counted_positive += int(np.count_nonzero(weights > 0.0))
            counted_negative += int(np.count_nonzero(weights < 0.0))
            sum_weights += float(np.sum(weights, dtype=np.float64))
            sum_squared_weights += float(
                np.sum(weights * weights, dtype=np.float64)
            )
    if counted_entries != entries:
        raise RuntimeError(
            f"pseudo-data output iteration found {counted_entries} entries, "
            f"expected {entries}"
        )
    if counted_positive != observed_positive or counted_negative != observed_negative:
        raise RuntimeError(
            "pseudo-data output sign counts do not match the Poisson draws: "
            f"observed (+{counted_positive}, -{counted_negative}), expected "
            f"(+{observed_positive}, -{observed_negative})"
        )
    _assert_close(
        "pseudo-data signed-weight sum",
        sum_weights,
        float(observed_positive - observed_negative),
    )
    _assert_close(
        "pseudo-data squared-weight sum",
        sum_squared_weights,
        float(observed_positive + observed_negative),
    )


def _build_pseudo_data_file(
    components: list[PseudoDataComponent],
    output_path: Path,
    *,
    luminosity_fb: float,
    base_seed: int,
    ensemble_index: int,
    ensemble_count: int,
    seed_sequence: np.random.SeedSequence,
    step_size: str,
    overwrite: bool,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    luminosity_pb_inverse = luminosity_fb * 1000.0
    source_snapshots = [
        component.source_snapshot or _snapshot_source(component.path)
        for component in components
    ]
    _assert_sources_unchanged(
        source_snapshots,
        context="before building a pseudo-data file",
        verify_sha256=False,
    )
    child_sequences = seed_sequence.spawn(4 * len(components) + 1)
    sampled: list[dict[str, np.ndarray]] = []
    process_metadata: dict[str, dict[str, Any]] = {}

    for component_index, (component, source_snapshot) in enumerate(
        zip(components, source_snapshots, strict=True)
    ):
        stats = component.stats
        expectation = component.expectation
        offset = 4 * component_index
        positive_count_rng = np.random.default_rng(child_sequences[offset])
        negative_count_rng = np.random.default_rng(child_sequences[offset + 1])
        positive_sample_rng = np.random.default_rng(child_sequences[offset + 2])
        negative_sample_rng = np.random.default_rng(child_sequences[offset + 3])
        observed_positive, observed_negative = _draw_signed_counts(
            expectation,
            positive_count_rng,
            negative_count_rng,
        )

        for sign, observed, sample_rng in (
            (+1, observed_positive, positive_sample_rng),
            (-1, observed_negative, negative_sample_rng),
        ):
            arrays = _sample_reconstructed(
                component.path,
                observed,
                sign=sign,
                total_abs_weight=_reconstructed_sign_sum_weights(stats, sign),
                rng=sample_rng,
                step_size=step_size,
                source_snapshot=source_snapshot,
            )
            arrays["weight"] = np.full(observed, sign, dtype=np.float64)
            arrays["weight_shape"] = np.full(
                observed, sign, dtype=np.float64
            )
            arrays["weight_nominal_pb"] = np.full(
                observed, sign / luminosity_pb_inverse, dtype=np.float64
            )
            arrays["lumi"] = np.full(
                observed, luminosity_pb_inverse, dtype=np.float64
            )
            arrays["luminosity_fb"] = np.full(
                observed, luminosity_fb, dtype=np.float64
            )
            sampled.append(arrays)

        details = _component_diagnostics(component)
        details["source_snapshot"] = source_snapshot.as_metadata()
        details.update(
            {
                "observed_positive": observed_positive,
                "observed_negative": observed_negative,
                "observed_net": observed_positive - observed_negative,
                "observed_entries": observed_positive + observed_negative,
                "observed_sum_squared_weights": (
                    observed_positive + observed_negative
                ),
            }
        )
        process_metadata[component.name] = details

    _assert_sources_unchanged(
        source_snapshots,
        context="after pseudo-data sampling",
        verify_sha256=verify_source_hashes,
    )

    branch_names = set(sampled[0])
    if any(set(component) != branch_names for component in sampled[1:]):
        raise KeyError("the ZZ and gg_H merged trees have different schemas")
    combined = {
        name: np.concatenate([component[name] for component in sampled])
        for name in sorted(branch_names)
    }
    total = len(combined["weight"])
    shuffle_rng = np.random.default_rng(child_sequences[-1])
    order = shuffle_rng.permutation(total)
    for name in combined:
        combined[name] = combined[name][order]
    combined["event_id"] = np.arange(total, dtype=np.uint64)
    combined["reconstructed"] = np.ones(total, dtype=np.bool_)
    combined["cross_section_pb"] = np.full(total, np.nan, dtype=np.float64)

    schema = {name: values.dtype for name, values in combined.items()}
    temporary = _prepare_output(output_path, overwrite)
    total_expected_positive = float(
        sum(component.expectation.positive for component in components)
    )
    total_expected_negative = float(
        sum(component.expectation.negative for component in components)
    )
    total_observed_positive = int(
        sum(details["observed_positive"] for details in process_metadata.values())
    )
    total_observed_negative = int(
        sum(details["observed_negative"] for details in process_metadata.values())
    )
    total_expected_net = total_expected_positive - total_expected_negative
    total_observed_signed = total_observed_positive - total_observed_negative
    metadata: dict[str, Any] = {
        "format_version": 2,
        "statistical_model": "signed_poisson_bootstrap_with_replacement",
        "sampling_with_replacement": True,
        "signed_weights": any(
            component.expectation.negative > 0.0 for component in components
        ),
        "luminosity_fb": luminosity_fb,
        "lumi_pb_inverse": luminosity_pb_inverse,
        "weight_formula": "weight = weight_nominal_pb * lumi",
        "seed": base_seed,
        "ensemble_index": ensemble_index,
        "ensemble_count": ensemble_count,
        "seed_spawn_key": list(seed_sequence.spawn_key),
        "components": process_metadata,
        "ZZ_expected": process_metadata["ZZ"]["expected_net"],
        "ZZ_observed": process_metadata["ZZ"]["observed_net"],
        "gg_H_expected": process_metadata["gg_H"]["expected_net"],
        "gg_H_observed": process_metadata["gg_H"]["observed_net"],
        "total_expected_positive": total_expected_positive,
        "total_expected_negative": total_expected_negative,
        "total_expected": total_expected_net,
        "total_observed_positive": total_observed_positive,
        "total_observed_negative": total_observed_negative,
        "total_observed": total_observed_signed,
        "total_entries": total,
        "total_sum_abs_weights": int(
            sum(details["observed_entries"] for details in process_metadata.values())
        ),
        "total_sum_squared_weights": int(
            sum(
                details["observed_sum_squared_weights"]
                for details in process_metadata.values()
            )
        ),
    }
    try:
        with uproot.recreate(temporary) as output_file:
            output_file.mktree(
                TREE_NAME,
                schema,
                title="Unit-magnitude signed Herwig pseudo-data",
            )
            output_file[TREE_NAME].extend(combined)
            output_file["merge_metadata"] = json.dumps(metadata, sort_keys=True)
        _assert_sources_unchanged(
            source_snapshots,
            context="before finalizing the pseudo-data output",
            verify_sha256=False,
        )
        _validate_pseudo_data_output(
            temporary,
            entries=total,
            observed_positive=total_observed_positive,
            observed_negative=total_observed_negative,
            step_size=step_size,
        )
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return metadata


def build_pseudo_data(
    zz_path: Path,
    higgs_path: Path,
    output_path: Path,
    *,
    luminosity_fb: float,
    seed: int,
    step_size: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Build one pseudo-data file while preserving the original public API."""

    components = _pseudo_data_components(
        zz_path,
        higgs_path,
        luminosity_fb=luminosity_fb,
        step_size=step_size,
    )
    return _build_pseudo_data_file(
        components,
        output_path,
        luminosity_fb=luminosity_fb,
        base_seed=seed,
        ensemble_index=0,
        ensemble_count=1,
        seed_sequence=np.random.SeedSequence(seed),
        step_size=step_size,
        overwrite=overwrite,
    )


def _recommended_ensemble_count(
    components: list[PseudoDataComponent], luminosity_fb: float
) -> tuple[int, float]:
    sign_effective_luminosities = [
        effective_luminosity
        for component in components
        for sign, expected in (
            (+1, component.expectation.positive),
            (-1, component.expectation.negative),
        )
        if expected > 0.0
        for effective_luminosity in (
            _effective_luminosity_fb(component.stats, sign),
        )
        if effective_luminosity is not None
    ]
    net_effective_luminosities = [
        effective_luminosity
        for component in components
        for effective_luminosity in (
            _net_effective_luminosity_fb(component.stats),
        )
        if effective_luminosity is not None
    ]
    effective_luminosities = [
        *sign_effective_luminosities,
        *net_effective_luminosities,
    ]
    if not effective_luminosities:
        return 0, 0.0
    limiting_luminosity = float(min(effective_luminosities))
    ratio = limiting_luminosity / luminosity_fb
    recommended = int(np.floor(ratio + 1.0e-12))
    return recommended, limiting_luminosity


def _pseudo_data_output_path(
    output_directory: Path, ensemble_index: int
) -> Path:
    if ensemble_index == 0:
        return output_directory / "data.root"
    return output_directory / f"data_{ensemble_index:04d}.root"


def _write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    temporary = _prepare_output(path, overwrite)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _path_exists_without_following_symlinks(path: Path) -> bool:
    return os.path.lexists(path)


def _published_path(
    staged_path: Path,
    final_path: Path,
    *,
    expected_sha256: str,
) -> PublishedPath:
    """Record the staged inode that is about to be linked into place."""

    stat = staged_path.lstat()
    return PublishedPath(
        path=final_path,
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        mode=int(stat.st_mode),
        sha256=expected_sha256,
    )


def _snapshot_publication_target(path: Path) -> PublishedPath:
    """Fingerprint an existing regular-file destination before publication."""

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        if stat.S_ISDIR(before.st_mode):
            raise IsADirectoryError(path)
        raise OSError(f"output is not a regular file: {path}")
    sha256 = _sha256_file(path)
    after = path.lstat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_mode),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_mode),
    )
    if after_identity != before_identity:
        raise RuntimeError(
            f"output changed while its publication snapshot was recorded: {path}"
        )
    return PublishedPath(
        path=path,
        device=before_identity[0],
        inode=before_identity[1],
        size_bytes=before_identity[2],
        mtime_ns=before_identity[3],
        mode=before_identity[4],
        sha256=sha256,
    )


def _is_unchanged_published_path(
    publication: PublishedPath,
    path: Path | None = None,
) -> bool:
    """Return whether ``path`` is still the exact file we published."""

    candidate = publication.path if path is None else path
    try:
        before = candidate.lstat()
    except FileNotFoundError:
        return False
    observed = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_mode),
    )
    expected = (
        publication.device,
        publication.inode,
        publication.size_bytes,
        publication.mtime_ns,
        publication.mode,
    )
    if observed != expected:
        return False
    try:
        sha256 = _sha256_file(candidate)
        after = candidate.lstat()
    except OSError:
        return False
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_mode),
    )
    return after_identity == expected and sha256 == publication.sha256


def _preserve_rollback_path(
    source_path: Path,
    final_path: Path,
    *,
    label: str,
) -> Path:
    """Move a blocked rollback file outside the ephemeral staging tree."""

    recovery_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{final_path.name}.{label}-",
            suffix=".recovery",
            dir=final_path.parent,
        )
    )
    recovery_path = recovery_directory / final_path.name
    try:
        os.replace(source_path, recovery_path)
    except Exception:
        recovery_directory.rmdir()
        raise
    return recovery_path


@contextmanager
def _campaign_staging_directory(output_directory: Path) -> Iterable[Path]:
    """Clean staging unless publication reports an unrecovered object in it."""

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=".pseudo_data_campaign.",
            dir=output_directory,
        )
    )
    retain = False
    try:
        yield staging_directory
    except CampaignPublicationError as error:
        retain = error.retained_staging_directory is not None
        raise
    except BaseException as error:
        if (staging_directory / PUBLICATION_MARKER).exists():
            retain = True
            raise CampaignPublicationError(
                "campaign rollback was interrupted; automatic cleanup disabled "
                f"and staging retained at {staging_directory}",
                retained_staging_directory=staging_directory,
            ) from error
        raise
    finally:
        if not retain and staging_directory.exists():
            shutil.rmtree(staging_directory)


def _publish_staged_campaign(
    staged_output_paths: list[tuple[Path, Path]],
    staged_manifest_path: Path,
    manifest_path: Path,
    *,
    staging_directory: Path,
    overwrite: bool,
    expected_sha256: dict[Path, str],
) -> None:
    """Publish a staged campaign and safely restore prior files on an error."""

    backup_directory = staging_directory / "backups"
    backups: dict[Path, Path] = {}
    published: list[PublishedPath] = []
    destinations = [
        *(final_path for _, final_path in staged_output_paths),
        manifest_path,
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("campaign publication destinations must be unique")
    initial_destinations: dict[Path, PublishedPath] = {}
    for destination in destinations:
        if not _path_exists_without_following_symlinks(destination):
            continue
        if not overwrite:
            raise FileExistsError(
                f"output appeared while the campaign was being built: {destination}"
            )
        initial_destinations[destination] = _snapshot_publication_target(
            destination
        )
    publication_marker = staging_directory / PUBLICATION_MARKER
    publication_marker.touch(exist_ok=False)

    def publish_one(staged_path: Path, final_path: Path, index: int) -> None:
        initial_destination = initial_destinations.get(final_path)
        if initial_destination is not None:
            backup_directory.mkdir(parents=True, exist_ok=True)
            backup_path = backup_directory / f"{index:04d}-{final_path.name}"
            # Register the claim before the atomic move so an asynchronous
            # interruption immediately after the syscall cannot orphan it.
            backups[final_path] = backup_path
            try:
                # Claim the destination first, then validate the claimed inode.
                # If the pathname was swapped after the initial snapshot, that
                # replacement becomes a tracked backup and is never discarded.
                os.replace(final_path, backup_path)
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"output disappeared before it could be published: {final_path}"
                ) from error
            if not _is_unchanged_published_path(
                initial_destination,
                backup_path,
            ):
                raise RuntimeError(
                    f"output changed while it was being claimed for publication: "
                    f"{final_path}"
                )
        elif _path_exists_without_following_symlinks(final_path):
            raise FileExistsError(
                f"output appeared while the campaign was being built: {final_path}"
            )
        expected_digest = expected_sha256.get(final_path)
        if expected_digest is None:
            expected_digest = _sha256_file(staged_path)
        publication = _published_path(
            staged_path,
            final_path,
            expected_sha256=expected_digest,
        )
        # Staging lives in the output directory. A hard link is therefore an
        # atomic no-clobber install in both modes, including the interval after
        # an overwrite target has been moved to its private backup.
        # Pre-register for the same interruption-safety reason as the backup.
        published.append(publication)
        os.link(staged_path, final_path, follow_symlinks=False)
        staged_path.unlink()

    try:
        for index, (staged_path, final_path) in enumerate(staged_output_paths):
            publish_one(staged_path, final_path, index)

        final_snapshots: list[SourceSnapshot] = []
        for _, final_path in staged_output_paths:
            snapshot = _snapshot_source(final_path)
            if snapshot.sha256 != expected_sha256[final_path]:
                raise RuntimeError(
                    f"published pseudo-data digest does not match its manifest: "
                    f"{final_path}"
                )
            final_snapshots.append(snapshot)
        _assert_sources_unchanged(
            final_snapshots,
            context="before publishing the pseudo-data campaign manifest",
            verify_sha256=False,
        )
        publish_one(
            staged_manifest_path,
            manifest_path,
            len(staged_output_paths),
        )
        _assert_sources_unchanged(
            final_snapshots,
            context="after publishing the pseudo-data campaign manifest",
        )
    except BaseException as publication_error:
        conflicts: list[str] = []
        unpreserved_claims: list[Path] = []
        backup_directory.mkdir(parents=True, exist_ok=True)
        for rollback_index, publication in enumerate(reversed(published)):
            final_path = publication.path
            if not _path_exists_without_following_symlinks(final_path):
                conflicts.append(f"published path disappeared: {final_path}")
                continue
            rollback_path = (
                backup_directory
                / f"rollback-{rollback_index:04d}-{final_path.name}"
            )
            try:
                # Claim the current directory entry before inspecting it. This
                # avoids the check-then-unlink race that could otherwise remove
                # a replacement installed immediately after an identity check.
                os.replace(final_path, rollback_path)
            except FileNotFoundError:
                conflicts.append(f"published path disappeared: {final_path}")
                continue
            if _is_unchanged_published_path(publication, rollback_path):
                rollback_path.unlink()
                if _path_exists_without_following_symlinks(final_path):
                    conflicts.append(
                        f"concurrent replacement preserved at {final_path}"
                    )
                continue

            try:
                recovery_path = _preserve_rollback_path(
                    rollback_path,
                    final_path,
                    label="concurrent-replacement",
                )
            except BaseException as error:
                unpreserved_claims.append(rollback_path)
                conflicts.append(
                    f"concurrent replacement remains at {rollback_path} after a "
                    f"failed recovery move: {error}"
                )
            else:
                restored_to_final = False
                if not _path_exists_without_following_symlinks(final_path):
                    try:
                        os.link(
                            recovery_path,
                            final_path,
                            follow_symlinks=False,
                        )
                        restored_to_final = True
                    except FileExistsError:
                        pass
                    except OSError as error:
                        conflicts.append(
                            f"could not restore concurrent replacement to "
                            f"{final_path}: {error}"
                        )
                if restored_to_final:
                    conflicts.append(
                        f"concurrent replacement restored at {final_path} with "
                        f"a safety copy preserved at {recovery_path}"
                    )
                else:
                    conflicts.append(
                        f"concurrent replacement preserved at {recovery_path} "
                        f"because {final_path} was occupied"
                    )

        for final_path, backup_path in backups.items():
            if not _path_exists_without_following_symlinks(backup_path):
                continue
            try:
                recovery_path = _preserve_rollback_path(
                    backup_path,
                    final_path,
                    label="pre-campaign",
                )
            except BaseException as error:
                unpreserved_claims.append(backup_path)
                conflicts.append(
                    f"pre-campaign output remains at {backup_path} after a failed "
                    f"recovery move: {error}"
                )
            else:
                restored_to_final = False
                if not _path_exists_without_following_symlinks(final_path):
                    try:
                        # The recovery path remains as a safety copy even when
                        # this no-clobber install succeeds. Thus a replacement
                        # immediately after link cannot destroy the last copy.
                        os.link(
                            recovery_path,
                            final_path,
                            follow_symlinks=False,
                        )
                        restored_to_final = True
                    except FileExistsError:
                        pass
                    except OSError as error:
                        conflicts.append(
                            f"could not restore pre-campaign output {final_path}: "
                            f"{error}"
                        )
                if restored_to_final:
                    conflicts.append(
                        f"pre-campaign output restored at {final_path} with a "
                        f"safety copy preserved at {recovery_path}"
                    )
                else:
                    conflicts.append(
                        f"pre-campaign output preserved at {recovery_path} because "
                        f"{final_path} was occupied"
                    )

        if conflicts:
            retained_staging_directory = (
                staging_directory if unpreserved_claims else None
            )
            details = "; ".join(conflicts)
            if retained_staging_directory is not None:
                details += (
                    "; automatic cleanup disabled because unrecovered objects remain "
                    f"in {retained_staging_directory}"
                )
            raise CampaignPublicationError(
                "campaign publication failed and rollback detected concurrent "
                f"changes or retained safety copies; {details}",
                retained_staging_directory=retained_staging_directory,
            ) from publication_error
        raise CampaignPublicationError(
            "campaign publication failed; rollback completed without retained "
            "claims",
        ) from publication_error


def build_pseudo_data_ensembles(
    zz_path: Path,
    higgs_path: Path,
    output_directory: Path,
    *,
    luminosity_fb: float,
    seed: int,
    ensemble_count: int | None,
    allow_ensemble_oversubscription: bool,
    step_size: str,
    overwrite: bool,
) -> dict[str, Any]:
    components = _pseudo_data_components(
        zz_path,
        higgs_path,
        luminosity_fb=luminosity_fb,
        step_size=step_size,
    )
    recommended_count, limiting_luminosity_fb = _recommended_ensemble_count(
        components, luminosity_fb
    )
    auto = ensemble_count is None
    generated_count = recommended_count if auto else ensemble_count
    if generated_count is None or generated_count <= 0:
        raise ValueError(
            "the available Herwig effective luminosity does not support one "
            f"{luminosity_fb:g} fb^-1 pseudo-experiment; explicitly request an "
            "ensemble count and pass --allow-ensemble-oversubscription to bootstrap anyway"
        )
    if (
        generated_count > recommended_count
        and not allow_ensemble_oversubscription
    ):
        raise ValueError(
            f"requested {generated_count} pseudo-data ensembles, but the signed-weight "
            f"effective-luminosity recommendation is {recommended_count}; pass "
            "--allow-ensemble-oversubscription to override"
        )

    output_directory = output_directory.resolve()
    output_paths = [
        _pseudo_data_output_path(output_directory, index)
        for index in range(generated_count)
    ]
    manifest_path = output_directory / "pseudo_data_manifest.json"
    if not overwrite:
        existing = [path for path in [*output_paths, manifest_path] if path.exists()]
        if existing:
            raise FileExistsError(
                f"output already exists: {existing[0]}; pass --overwrite to replace it"
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    seed_sequences = np.random.SeedSequence(seed).spawn(generated_count)
    with _campaign_staging_directory(output_directory) as staging_directory:
        staged_output_paths = [
            staging_directory / path.name for path in output_paths
        ]
        staged_manifest_path = staging_directory / manifest_path.name
        files: list[dict[str, Any]] = []
        output_snapshots: list[SourceSnapshot] = []
        for index, (staged_path, final_path, seed_sequence) in enumerate(
            zip(
                staged_output_paths,
                output_paths,
                seed_sequences,
                strict=True,
            )
        ):
            metadata = _build_pseudo_data_file(
                components,
                staged_path,
                luminosity_fb=luminosity_fb,
                base_seed=seed,
                ensemble_index=index,
                ensemble_count=generated_count,
                seed_sequence=seed_sequence,
                step_size=step_size,
                overwrite=False,
                verify_source_hashes=False,
            )
            output_snapshot = _snapshot_source(staged_path)
            output_snapshots.append(output_snapshot)
            files.append(
                {
                    "ensemble_index": index,
                    "path": final_path.name,
                    "sha256": output_snapshot.sha256,
                    "total_entries": metadata["total_entries"],
                    "total_observed_positive": metadata[
                        "total_observed_positive"
                    ],
                    "total_observed_negative": metadata[
                        "total_observed_negative"
                    ],
                    "total_observed": metadata["total_observed"],
                    "total_sum_squared_weights": metadata[
                        "total_sum_squared_weights"
                    ],
                    "seed_spawn_key": metadata["seed_spawn_key"],
                }
            )

        _assert_sources_unchanged(
            (
                component.source_snapshot
                for component in components
                if component.source_snapshot is not None
            ),
            context="before finalizing the pseudo-data campaign manifest",
        )
        _assert_sources_unchanged(
            output_snapshots,
            context="before finalizing the pseudo-data campaign manifest",
            verify_sha256=False,
        )

        manifest: dict[str, Any] = {
            "format_version": 2,
            "statistical_model": "signed_poisson_bootstrap_with_replacement",
            "sampling_with_replacement": True,
            "signed_weights": any(
                component.expectation.negative > 0.0 for component in components
            ),
            "luminosity_fb": luminosity_fb,
            "seed": seed,
            "ensemble_request": "auto" if auto else generated_count,
            "recommended_ensemble_count": recommended_count,
            "generated_ensemble_count": generated_count,
            "allow_ensemble_oversubscription": allow_ensemble_oversubscription,
            "limiting_effective_luminosity_fb": limiting_luminosity_fb,
            "components": {
                component.name: _component_diagnostics(component)
                for component in components
            },
            "files": files,
        }
        _write_json(staged_manifest_path, manifest, overwrite=False)
        _publish_staged_campaign(
            list(zip(staged_output_paths, output_paths, strict=True)),
            staged_manifest_path,
            manifest_path,
            staging_directory=staging_directory,
            overwrite=overwrite,
            expected_sha256={
                path: entry["sha256"]
                for path, entry in zip(output_paths, files, strict=True)
            },
        )
    return manifest


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
    pseudo_data_ensembles: int | None = None,
    allow_ensemble_oversubscription: bool = False,
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
            luminosity_fb=luminosity_fb,
            overwrite=overwrite,
        )
        print(
            f"{sample}: {len(files)} files -> {output_path}; "
            f"entries={outputs[sample].entries}, sum(weight)={outputs[sample].sum_weights:.8g}, "
            f"cross_section_pb={outputs[sample].cross_section_pb:.8g}"
        )

    manifest = build_pseudo_data_ensembles(
        output_directory / "ZZ_herwig.root",
        output_directory / "gg_H_herwig.root",
        output_directory,
        luminosity_fb=luminosity_fb,
        seed=seed,
        ensemble_count=pseudo_data_ensembles,
        allow_ensemble_oversubscription=allow_ensemble_oversubscription,
        step_size=step_size,
        overwrite=overwrite,
    )
    first_file = manifest["files"][0]
    print(
        f"pseudo-data: {manifest['generated_ensemble_count']} files beginning with "
        f"{output_directory / first_file['path']}; luminosity={luminosity_fb:g} fb^-1, "
        f"recommended={manifest['recommended_ensemble_count']}, "
        f"limiting effective luminosity="
        f"{manifest['limiting_effective_luminosity_fb']:.8g} fb^-1, "
        f"weights={'signed +/-1' if manifest['signed_weights'] else '+1'}"
    )
    if manifest["signed_weights"]:
        print(
            "warning: negative Herwig weights make these signed pseudo-observations; "
            "downstream losses must support signed sample weights"
        )
    return outputs


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _ensemble_count(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be 'auto' or a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be 'auto' or a positive integer"
        )
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
    parser.add_argument(
        "--luminosity-fb", type=_positive_float, default=DEFAULT_LUMINOSITY_FB
    )
    parser.add_argument("--seed", type=_nonnegative_int, default=12345)
    parser.add_argument("--step-size", default="100 MB", help="uproot chunk size")
    parser.add_argument("--zz-cross-section-pb", type=_positive_float)
    parser.add_argument("--gg-h-cross-section-pb", type=_positive_float)
    parser.add_argument(
        "--pseudo-data-ensembles",
        type=_ensemble_count,
        default=None,
        metavar="auto|N",
        help=(
            "number of Herwig pseudo-data files; default auto uses the "
            "signed-weight effective-luminosity recommendation"
        ),
    )
    parser.add_argument(
        "--allow-ensemble-oversubscription",
        action="store_true",
        help="allow an explicit ensemble count above the effective-luminosity recommendation",
    )
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
        pseudo_data_ensembles=args.pseudo_data_ensembles,
        allow_ensemble_oversubscription=args.allow_ensemble_oversubscription,
    )


if __name__ == "__main__":
    main()
