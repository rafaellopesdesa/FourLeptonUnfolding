#!/usr/bin/env python3
"""Validate a compact Analysis ROOT file before publishing a batch result."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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
    cross_section_pb: float


def _branch_names(tree: object) -> set[str]:
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def validate_analysis_output(
    path: Path, *, expected_entries: int | None = None, step_size: str = "100 MB"
) -> ValidationSummary:
    """Validate schema, entry count, weights, and physical normalization."""

    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"analysis output is missing or empty: {path}")

    with uproot.open(path) as root_file:
        if TREE_NAME not in root_file:
            raise KeyError(f"{path} does not contain the {TREE_NAME} tree")
        tree = root_file[TREE_NAME]
        missing = sorted(REQUIRED_BRANCHES.difference(_branch_names(tree)))
        if missing:
            raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
        entries = int(tree.num_entries)
        if entries <= 0:
            raise ValueError(f"{path} contains no analysis entries")
        if expected_entries is not None and entries != expected_entries:
            raise ValueError(
                f"{path} contains {entries} entries; expected {expected_entries}"
            )

        sum_weights = 0.0
        final_cross_section = float("nan")
        for arrays in tree.iterate(
            expressions=["weight", "cross_section_pb"],
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
            valid_cross_sections = cross_sections[
                np.isfinite(cross_sections) & (cross_sections > 0.0)
            ]
            if valid_cross_sections.size:
                final_cross_section = float(valid_cross_sections[-1])

    if not np.isfinite(sum_weights) or sum_weights <= 0.0:
        raise ValueError(f"{path} has a non-positive total event weight")
    if not np.isfinite(final_cross_section) or final_cross_section <= 0.0:
        raise ValueError(f"{path} has no finite positive cross section")
    return ValidationSummary(entries, sum_weights, final_cross_section)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-entries", type=int)
    parser.add_argument("--step-size", default="100 MB")
    args = parser.parse_args()
    if args.expected_entries is not None and args.expected_entries <= 0:
        parser.error("--expected-entries must be positive")
    summary = validate_analysis_output(
        args.path,
        expected_entries=args.expected_entries,
        step_size=args.step_size,
    )
    print(
        f"validated {args.path}: entries={summary.entries}, "
        f"sum_weights={summary.sum_weights:.12g}, "
        f"cross_section_pb={summary.cross_section_pb:.12g}"
    )


if __name__ == "__main__":
    main()
