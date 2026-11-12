"""Keep detector-scale variations distinct during reduction and merging."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import uproot


MUON_SCALE_FACTORS = {"nominal": 1.0, "up": 1.01, "down": 0.99}
MUON_SCALE_MODEL = "uniform_relative_momentum_scale_v1"
CONTRACT_KEYS = (
    "muon_scale_variation",
    "muon_scale_factor",
    "muon_scale_one_sigma",
    "muon_scale_model",
)


def _contract(values: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Validate explicit settings; an entirely absent contract is legacy nominal."""
    present = set(CONTRACT_KEYS).intersection(values)
    if not present:
        return {
            "muon_scale_variation": "nominal",
            "muon_scale_factor": 1.0,
            "muon_scale_one_sigma": 0.01,
            "muon_scale_model": MUON_SCALE_MODEL,
        }
    missing = set(CONTRACT_KEYS).difference(values)
    if missing:
        raise ValueError(f"Incomplete muon-scale metadata in {source}: missing {sorted(missing)}")
    variation = values["muon_scale_variation"]
    if not isinstance(variation, str) or variation not in MUON_SCALE_FACTORS:
        raise ValueError(f"Invalid muon-scale variation in {source}: {variation!r}")
    normalized: dict[str, Any] = {"muon_scale_variation": variation}
    for key, expected in (
        ("muon_scale_factor", MUON_SCALE_FACTORS[variation]),
        ("muon_scale_one_sigma", 0.01),
    ):
        try:
            value = float(values[key])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {key} in {source}: {values[key]!r}") from None
        if isinstance(values[key], bool) or not math.isfinite(value) or not math.isclose(
            value, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Invalid {key} in {source}: expected {expected}, got {values[key]!r}")
        normalized[key] = expected
    if values["muon_scale_model"] != MUON_SCALE_MODEL:
        raise ValueError(f"Unsupported muon-scale model in {source}: {values['muon_scale_model']!r}")
    normalized["muon_scale_model"] = MUON_SCALE_MODEL
    return normalized


def _combine(records: Iterable[tuple[Path, dict[str, Any], list[str]]]) -> dict[str, Any]:
    selected = None
    first_source = None
    legacy_inputs: list[str] = []
    for source, contract, legacy in records:
        if selected is None:
            selected, first_source = contract, source
        elif contract != selected:
            raise ValueError(
                "Cannot combine different muon-scale variations: "
                f"{first_source} ({selected['muon_scale_variation']}) and "
                f"{source} ({contract['muon_scale_variation']}); "
                "reduce and merge nominal/up/down samples separately"
            )
        legacy_inputs.extend(legacy)
    if selected is None:
        raise ValueError("No inputs supplied for simulation metadata")
    return {
        "schema_version": 1,
        **selected,
        "legacy_nominal_inputs": list(dict.fromkeys(legacy_inputs)),
    }


def collect_delphes_metadata(files: Iterable[Path]) -> dict[str, Any]:
    """Read simulation sidecars before combining Delphes runs into one ntuple."""
    records = []
    for path in files:
        sidecar = path.parent / "simulation-metadata.txt"
        values: dict[str, str] = {}
        if sidecar.is_file():
            for line in sidecar.read_text().splitlines():
                key, separator, value = line.partition("=")
                key = key.strip()
                if separator and key in CONTRACT_KEYS:
                    if key in values:
                        raise ValueError(f"Duplicate {key} in {sidecar}")
                    values[key] = value.strip()
        contract = _contract(values, sidecar)
        records.append((path, contract, [] if values else [str(path)]))
    return _combine(records)


def merge_simulation_metadata(files: Iterable[Path]) -> dict[str, Any]:
    """Propagate the scale contract through compact-sample and pseudo-data merges."""
    records = []
    for path in files:
        with uproot.open(path) as root_file:
            if "simulation_metadata" not in root_file:
                records.append((path, _contract({}, path), [str(path)]))
                continue
            try:
                metadata = json.loads(str(root_file["simulation_metadata"]))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid simulation_metadata JSON in {path}") from None
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 1
        ):
            raise ValueError(f"Unsupported simulation_metadata schema in {path}")
        # An explicit object must carry the full contract, unlike older files
        # that have no simulation_metadata object at all.
        if not set(CONTRACT_KEYS).issubset(metadata):
            raise ValueError(f"Incomplete muon-scale metadata in {path}")
        legacy = metadata.get("legacy_nominal_inputs", [])
        if not isinstance(legacy, list) or any(not isinstance(item, str) for item in legacy):
            raise ValueError(f"Invalid legacy_nominal_inputs in {path}")
        contract = _contract(metadata, path)
        if legacy and contract["muon_scale_variation"] != "nominal":
            raise ValueError(f"Varied muon-scale metadata cannot contain legacy nominal inputs in {path}")
        records.append((path, contract, legacy))
    return _combine(records)
