#!/usr/bin/env python3
"""Validate a compact Analysis ROOT file before publishing a batch result."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import uproot


TREE_NAME = "Analysis"
KINEMATIC_FIELDS = (
    "cos_theta_star",
    "cos_theta1",
    "cos_theta2",
    "Phi",
    "Phi1",
    "Psi",
    "m_Z1",
    "m_Z2",
    "m_ZZ",
    "y_ZZ",
    "pT_ZZ",
)
REQUIRED_BRANCHES = {
    "event_id",
    "event_number",
    "weight",
    "cross_section_pb",
    "fiducial",
    "reconstructed",
    "truth_type",
    "reco_type",
    "type",
    *(f"{level}_{field}" for level in ("truth", "reco") for field in KINEMATIC_FIELDS),
}


@dataclass(frozen=True)
class ValidationSummary:
    entries: int
    sum_weights: float
    sum_squared_weights: float
    cross_section_pb: float
    lhe_matched_entries: int | None = None
    lhe_four_lepton_entries: int | None = None
    lhe_higgs_entries: int | None = None


def _branch_names(tree: object) -> set[str]:
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def _lhe_inventory(root_file, path: Path) -> dict[int, int] | None:
    """Check optional full-digest provenance and return each source's size."""
    if "lhe_metadata" not in root_file:
        return None
    metadata = json.loads(str(root_file["lhe_metadata"]))
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1
            or metadata.get("event_number") != "lhe_event_index_1based"
            or not isinstance(metadata.get("sources"), list)):
        raise ValueError(f"{path}: unsupported LHE metadata contract")
    counts, digests = {}, {}
    for source in metadata["sources"]:
        if not isinstance(source, dict):
            raise ValueError(f"{path}: invalid LHE source metadata")
        digest, identifier = source.get("lhe_sha256"), source.get("lhe_source_id")
        count = source.get("lhe_event_count")
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{16}", identifier) is None
                or identifier != digest[:16] or type(count) is not int or count < 1):
            raise ValueError(f"{path}: invalid LHE source ID, SHA-256 or event count")
        source_id = int(identifier, 16)
        if source_id in digests and (digests[source_id] != digest or counts[source_id] != count):
            raise ValueError(f"{path}: conflicting LHE source metadata for one source ID")
        digests[source_id], counts[source_id] = digest, count
    return counts


def _validate_lhe_chunk(arrays, *, path: Path, branch_types, require_lhe: bool,
                        source_inventory: dict[int, int] | None = None) -> tuple[int, int, int]:
    """Validate association and observable flags without assuming an H->4l LHE.

    An undecayed POWHEG Higgs can have an available LHE association and valid
    Higgs kinematics while every four-lepton observable remains undefined.
    Degenerate four-lepton configurations may retain valid masses while their
    angular observables are undefined.
    """
    size = len(arrays["weight"])
    columns = {}
    for name, expected_dtype in branch_types.items():
        column = np.asarray(arrays[name])
        expected_dtype = np.dtype(expected_dtype)
        if (column.shape != (size,) or column.dtype.kind != expected_dtype.kind
                or column.dtype.itemsize != expected_dtype.itemsize):
            raise ValueError(f"{path}: {name} must be a scalar {expected_dtype} branch")
        columns[name] = column
    available = columns["lhe_available"]
    four_leptons = columns["lhe_has_four_leptons"]
    valid = columns["lhe_kinematics_valid"]
    higgs = columns["lhe_has_higgs"]
    higgs_valid = columns["lhe_higgs_kinematics_valid"]
    taus = columns["lhe_has_taus"]
    if require_lhe and not np.all(available):
        raise ValueError(f"{path}: --require-lhe requires a matched LHE event for every row")
    if np.any((four_leptons | valid | higgs | higgs_valid | taus) & ~available):
        raise ValueError(f"{path}: LHE observable flags require lhe_available")
    if np.any(valid & ~four_leptons):
        raise ValueError(f"{path}: lhe_kinematics_valid requires lhe_has_four_leptons")
    if np.any(higgs_valid & ~higgs):
        raise ValueError(f"{path}: lhe_higgs_kinematics_valid requires lhe_has_higgs")

    indices, source_ids = columns["lhe_event_index"], columns["lhe_source_id"]
    # The uint64 source ID is an unmodified digest prefix; zero is therefore
    # rare but valid.  Availability and the one-based index disambiguate it
    # from the absent-row sentinel.
    if np.any(available & (indices < 1)):
        raise ValueError(f"{path}: matched LHE rows require a one-based source event index")
    if np.any(~available & ((indices != -1) | (source_ids != 0))):
        raise ValueError(f"{path}: unavailable LHE rows require event index -1 and source ID 0")
    if source_inventory is not None:
        for identifier in np.unique(source_ids[available]):
            source_id = int(identifier)
            if source_id not in source_inventory:
                raise ValueError(f"{path}: matched LHE source ID is absent from lhe_metadata")
            selected = available & (source_ids == identifier)
            if np.any(indices[selected] > source_inventory[source_id]):
                raise ValueError(f"{path}: LHE source event index exceeds its metadata event count")
    counts = columns["lhe_n_final_leptons"]
    if np.any(counts < 0) or np.any(four_leptons & (counts < 4)):
        raise ValueError(f"{path}: inconsistent LHE final-lepton count")
    if np.any(~available & ((counts != 0) | (columns["lhe_process_id"] != 0))):
        raise ValueError(f"{path}: unavailable LHE rows must have zero lepton count and process ID")
    types = columns["lhe_type"]
    if np.any(four_leptons & ~np.isin(types, (0, 1, 2, 3))) or np.any(~four_leptons & (types != -1)):
        raise ValueError(f"{path}: LHE channel type disagrees with the four-lepton flag")

    floating = {name: values for name, values in columns.items() if values.dtype.kind == "f"}
    for name, values in floating.items():
        if np.any(np.isinf(values)):
            raise ValueError(f"{path}: {name} contains infinite LHE values")
        if np.any(~available & ~np.isnan(values)):
            raise ValueError(f"{path}: unavailable LHE rows require NaN in {name}")
    for name in ("lhe_weight", "lhe_scale"):
        # XWGTUP is the signed generator weight, not a probability.  Do not
        # impose positivity, unit magnitude, or agreement with shower weight.
        if not np.all(np.isfinite(columns[name][available])):
            raise ValueError(f"{path}: matched LHE rows require finite {name}")

    kinematics = np.column_stack([columns[f"lhe_{field}"] for field in KINEMATIC_FIELDS])
    all_finite = np.all(np.isfinite(kinematics), axis=1)
    if np.any(valid != all_finite):
        raise ValueError(f"{path}: lhe_kinematics_valid must agree with finite four-lepton kinematics")
    if np.any(~four_leptons & ~np.all(np.isnan(kinematics), axis=1)):
        raise ValueError(f"{path}: rows without an LHE four-lepton candidate require NaN four-lepton kinematics")
    higgs_kinematics = np.column_stack([columns[f"lhe_higgs_{field}"] for field in ("m", "y", "pT")])
    if (np.any(higgs_valid & ~np.all(np.isfinite(higgs_kinematics), axis=1))
            or np.any(~higgs_valid & ~np.all(np.isnan(higgs_kinematics), axis=1))):
        raise ValueError(f"{path}: lhe_higgs_kinematics_valid must agree with finite Higgs kinematics")
    return int(np.count_nonzero(available)), int(np.count_nonzero(four_leptons)), int(np.count_nonzero(higgs))


def validate_analysis_output(
    path: Path, *, expected_entries: int | None = None, step_size: str = "100 MB",
    require_lhe: bool = False,
) -> ValidationSummary:
    """Validate schema, signed normalization and any available LHE provenance.

    Legacy files without LHE branches remain valid unless ``require_lhe`` is
    true.  A partially present LHE schema is always an error.
    """

    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"analysis output is missing or empty: {path}")

    with uproot.open(path) as root_file:
        if TREE_NAME not in root_file:
            raise KeyError(f"{path} does not contain the {TREE_NAME} tree")
        tree = root_file[TREE_NAME]
        available_branches = _branch_names(tree)
        missing = sorted(REQUIRED_BRANCHES.difference(available_branches))
        if missing:
            raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
        lhe_branch_types = None
        source_inventory = None
        if require_lhe or any(name.startswith("lhe_") for name in available_branches):
            from Analysis.lhe import LHE_BRANCH_TYPES

            lhe_branch_types = LHE_BRANCH_TYPES
            missing_lhe = sorted(set(lhe_branch_types) - available_branches)
            if missing_lhe:
                raise KeyError(f"{path} is missing required LHE branches: {', '.join(missing_lhe)}")
            source_inventory = _lhe_inventory(root_file, path)
        entries = int(tree.num_entries)
        if entries <= 0:
            raise ValueError(f"{path} contains no analysis entries")
        if expected_entries is not None and entries != expected_entries:
            raise ValueError(
                f"{path} contains {entries} entries; expected {expected_entries}"
            )

        sum_weights = 0.0
        sum_squared_weights = 0.0
        final_cross_section = float("nan")
        lhe_counts = [0, 0, 0] if lhe_branch_types is not None else None
        expressions = ["weight", "cross_section_pb"]
        if lhe_branch_types is not None:
            expressions.extend(lhe_branch_types)
        for arrays in tree.iterate(
            expressions=expressions,
            step_size=step_size,
            library="np",
            how=dict,
        ):
            weights = np.asarray(arrays["weight"], dtype=np.float64)
            cross_sections = np.asarray(
                arrays["cross_section_pb"], dtype=np.float64
            )
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"{path} contains non-finite event weights")
            sum_weights += float(np.sum(weights, dtype=np.float64))
            sum_squared_weights += float(
                np.sum(weights * weights, dtype=np.float64)
            )
            valid_cross_sections = cross_sections[
                np.isfinite(cross_sections) & (cross_sections > 0.0)
            ]
            if valid_cross_sections.size:
                final_cross_section = float(valid_cross_sections[-1])
            if lhe_branch_types is not None:
                local_counts = _validate_lhe_chunk(
                    arrays, path=path, branch_types=lhe_branch_types, require_lhe=require_lhe,
                    source_inventory=source_inventory,
                )
                for index, count in enumerate(local_counts):
                    lhe_counts[index] += count

    # An individual NLO shard may legitimately have a zero or negative net
    # sum even though the complete process sample has a positive cross
    # section.  Reject only numerically invalid aggregates here; the merger
    # performs the positive-net check at the point where it normalizes the
    # complete sample.
    if not np.isfinite(sum_weights):
        raise ValueError(f"{path} has a non-finite total event weight")
    if not np.isfinite(sum_squared_weights):
        raise ValueError(f"{path} has a non-finite sum of squared event weights")
    if not np.isfinite(final_cross_section) or final_cross_section <= 0.0:
        raise ValueError(f"{path} has no finite positive cross section")
    return ValidationSummary(
        entries=entries,
        sum_weights=sum_weights,
        sum_squared_weights=sum_squared_weights,
        cross_section_pb=final_cross_section,
        lhe_matched_entries=lhe_counts[0] if lhe_counts is not None else None,
        lhe_four_lepton_entries=lhe_counts[1] if lhe_counts is not None else None,
        lhe_higgs_entries=lhe_counts[2] if lhe_counts is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-entries", type=int)
    parser.add_argument("--step-size", default="100 MB")
    parser.add_argument("--require-lhe", action="store_true",
                        help="require a matched LHE event for every row (no four-lepton or Higgs requirement)")
    args = parser.parse_args()
    if args.expected_entries is not None and args.expected_entries <= 0:
        parser.error("--expected-entries must be positive")
    summary = validate_analysis_output(
        args.path,
        expected_entries=args.expected_entries,
        step_size=args.step_size,
        require_lhe=args.require_lhe,
    )
    print(
        f"validated {args.path}: entries={summary.entries}, "
        f"sum_weights={summary.sum_weights:.12g}, "
        f"sum_squared_weights={summary.sum_squared_weights:.12g}, "
        f"cross_section_pb={summary.cross_section_pb:.12g}"
        + (f", lhe_matched={summary.lhe_matched_entries}, "
           f"lhe_four_leptons={summary.lhe_four_lepton_entries}, "
           f"lhe_higgs={summary.lhe_higgs_entries}"
           if summary.lhe_matched_entries is not None else "")
    )


if __name__ == "__main__":
    main()
