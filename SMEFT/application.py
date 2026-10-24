"""Apply HB matrix-element ratios to certified native Higgs decay rows.

The original event measure is preserved. New ratio/weight vectors follow the
requested coefficient order; signed nominal weights are never absolutized.
This module has no neural-network, shower-reader, or generator dependency.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import uproot

from Tools.born_projection import BORN_PROJECTION_METADATA


TREE_NAME = "Analysis"
PREFIX = "smeft_hb_"
METADATA_NAME = "smeft_hb_metadata"
STATUS_NAMES = {
    0: "evaluated", 1: "background_unity", 2: "unavailable_born",
    3: "tau_unsupported", 4: "unsupported_channel", 5: "invalid_born_kinematics",
}
DEFINITIONS = {
    1: ("ZZ", "lhe_status1_charged_leptons"),
    2: ("gg_H", "pythia_event_before_decay_fsr"),
    3: ("gg_H", "herwig_qtilde_original_decay"),
}
BORN_KINEMATICS = (
    "m_ZZ", "m_Z1", "m_Z2", "cos_theta_star", "cos_theta1", "cos_theta2",
    "Phi", "Phi1", "Psi", "y_ZZ", "pT_ZZ",
)
REQUIRED_BRANCHES = {
    "born_source", "born_available", "born_kinematics_valid", "born_type",
    "born_n_taus", "born_shower_event_index", "lhe_available", "lhe_source_id",
    "lhe_event_index", *("born_" + name for name in BORN_KINEMATICS),
}


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _as_dict(arrays) -> dict[str, np.ndarray]:
    if isinstance(arrays, Mapping):
        return {key: np.asarray(value) for key, value in arrays.items()}
    if getattr(arrays.dtype, "names", None):
        return {key: np.asarray(arrays[key]) for key in arrays.dtype.names}
    raise TypeError("ROOT tree did not produce named arrays")


def _schema(tree) -> dict[str, np.dtype]:
    schema = {}
    for name in tree.keys():
        values = np.asarray(tree[name].array(entry_start=0, entry_stop=1, library="np"))
        if values.dtype.hasobject:
            raise TypeError(f"cannot preserve variable-length/object branch {tree.name}/{name}; convert it to a supported fixed-width schema")
        schema[name] = np.dtype((values.dtype, values.shape[1:])) if values.ndim > 1 else values.dtype
    return schema


def _metadata(root_file) -> tuple[dict[str, str], dict[str, dict[str, np.dtype]]]:
    strings, trees = {}, {}
    for name, kind in root_file.classnames(cycle=False).items():
        if name == METADATA_NAME or name.startswith(PREFIX):
            raise ValueError(f"input already contains SMEFT output object {name!r}")
        if kind == "TObjString":
            strings[name] = str(root_file[name])
        elif kind in {"TTree", "ROOT::RNTuple", "ROOT::Experimental::RNTuple"}:
            trees[name] = _schema(root_file[name])
        else:
            raise TypeError(f"cannot preserve ROOT object {name!r} ({kind}); refusing to drop it")
    if TREE_NAME not in trees:
        raise KeyError("input ROOT file has no Analysis tree")
    if any(name.startswith(PREFIX) for name in trees[TREE_NAME]):
        raise ValueError("input already contains SMEFT HB branches")
    return strings, trees


def _json_object(strings: Mapping[str, str], key: str) -> dict:
    if key not in strings:
        raise ValueError(f"missing required {key}; regenerate certified Born analysis inputs")
    try:
        value = json.loads(strings[key])
    except (ValueError, TypeError) as error:
        raise ValueError(f"malformed {key}") from error
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"unsupported {key} schema")
    if not isinstance(value.get("sources"), list):
        raise ValueError(f"{key} lacks a source inventory")
    return value


def _source_record(record: Any) -> tuple[int, str, str, str, str, int]:
    if not isinstance(record, dict):
        raise ValueError("invalid source record in provenance")
    identifier, digest = record.get("lhe_source_id"), record.get("lhe_sha256")
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{16}", identifier):
        raise ValueError("source provenance requires a 16-digit hexadecimal lhe_source_id")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest[:16] != identifier:
        raise ValueError("LHE full SHA-256 does not agree with its source ID")
    process, shower, definition = (record.get(key) for key in ("process", "shower", "hard_decay_definition"))
    if process not in {"gg_H", "ZZ"} or shower not in {"pythia", "herwig"}:
        raise ValueError("source provenance has unsupported process or shower")
    expected = DEFINITIONS[1][1] if process == "ZZ" else DEFINITIONS[2 if shower == "pythia" else 3][1]
    if definition != expected:
        raise ValueError("source provenance has an incompatible hard-decay definition")
    count = record.get("lhe_event_count")
    if type(count) is not int or count < 1:
        raise ValueError("source provenance has no positive LHE event count")
    return int(identifier, 16), digest, process, shower, definition, count


def _source_inventory(strings: Mapping[str, str]) -> dict[tuple[int, int], int]:
    lhe = _json_object(strings, "lhe_metadata")
    born = _json_object(strings, "born_metadata")
    if lhe.get("event_number") != "lhe_event_index_1based":
        raise ValueError("LHE metadata does not certify source-index event numbering")
    if born.get("projection") != BORN_PROJECTION_METADATA:
        raise ValueError("Born projection metadata is not the supported canonical definition")
    origins = {}
    lhe_records = set()
    for record in lhe["sources"]:
        parsed = _source_record(record)
        identifier, digest, process, _, _, count = parsed
        summary = (digest, process, count)
        if identifier in origins and origins[identifier] != summary:
            raise ValueError("LHE source-ID collision or conflicting process/source metadata")
        origins[identifier] = summary
        lhe_records.add(parsed)
    inventory = {}
    for record in born["sources"]:
        parsed = _source_record(record)
        if parsed not in lhe_records:
            raise ValueError("Born source record is not matched by the full LHE provenance inventory")
        identifier, _, process, shower, _, count = parsed
        code = record.get("born_source")
        expected_code = 1 if process == "ZZ" else 2 if shower == "pythia" else 3
        if type(code) is not int or code != expected_code:
            raise ValueError("Born source code conflicts with certified process/shower identity")
        inventory[(identifier, code)] = count
    return inventory


def _scalar(arrays, name: str, size: int, kind: str | None = None) -> np.ndarray:
    values = np.asarray(arrays[name])
    if values.shape != (size,) or (kind is not None and values.dtype.kind not in kind):
        raise ValueError(f"{name} must be a scalar branch of type {kind or 'numeric'}")
    return values


def _classify(arrays, inventory, size: int) -> np.ndarray:
    source = _scalar(arrays, "born_source", size, "iu")
    identifiers = _scalar(arrays, "lhe_source_id", size, "iu")
    lhe_index = _scalar(arrays, "lhe_event_index", size, "iu")
    if not np.all(_scalar(arrays, "lhe_available", size, "b")):
        raise ValueError("every row requires certified original LHE identity")
    native_index = _scalar(arrays, "born_shower_event_index", size, "iu")
    for code in np.unique(source):
        if int(code) not in DEFINITIONS:
            raise ValueError(f"uncertified or unsupported born_source code {code}")
        selected = source == code
        if int(code) == 1:
            if np.any(native_index[selected] != -1):
                raise ValueError("ZZ LHE rows must not have a native shower-decay ordinal")
        elif np.any(native_index[selected] < 1):
            raise ValueError("native Higgs rows require a positive accepted shower-event ordinal")
        for identifier in np.unique(identifiers[selected]):
            key = (int(identifier), int(code))
            if key not in inventory:
                raise ValueError(f"row source {key[0]:016x}/{key[1]} is absent from certified Born provenance")
            mask = selected & (identifiers == identifier)
            if np.any((lhe_index[mask] < 1) | (lhe_index[mask] > inventory[key])):
                raise ValueError("row LHE event index lies outside its certified source")
    available = _scalar(arrays, "born_available", size, "b")
    valid = _scalar(arrays, "born_kinematics_valid", size, "b")
    channel = _scalar(arrays, "born_type", size, "iu")
    taus = _scalar(arrays, "born_n_taus", size, "iu")
    status = np.zeros(size, dtype=np.int8)
    signal = np.isin(source, (2, 3))
    status[source == 1] = 1
    status[signal & ~available] = 2
    status[signal & available & ((taus > 0) | np.isin(channel, (4, 5, 6, 7, 8)))] = 3
    status[signal & available & (status == 0) & ~np.isin(channel, (0, 1, 2, 3))] = 4
    status[signal & available & (status == 0) & ~valid] = 5
    if np.any(signal & available & np.isin(channel, (0, 1, 2, 3)) & (taus != 0)):
        raise ValueError("Born channel and direct-tau count are inconsistent")
    if np.any(signal & valid & ~available):
        raise ValueError("Born kinematics cannot be valid when Born input is unavailable")
    evaluated = status == 0
    if np.any(evaluated):
        for name in BORN_KINEMATICS:
            values = _scalar(arrays, "born_" + name, size, "fiu")
            status[evaluated & ~np.isfinite(values)] = 5
    return status


def _copy_tree(source_tree, destination, name, schema, step_size):
    destination.mktree(name, schema, title=getattr(source_tree, "title", ""))
    for chunk in source_tree.iterate(step_size=step_size, library="np", how=dict):
        destination[name].extend(_as_dict(chunk))


def decorate_file(
    input_path: str | Path,
    output_path: str | Path,
    coefficients: Sequence[float],
    *,
    backend,
    lambda_tev: float = 1.,
    weight_branch: str = "weight",
    unsupported: str = "error",
    step_size: str | int = "64 MB",
    overwrite: bool = False,
    reconstruct: Callable | None = None,
) -> dict[str, Any]:
    """Decorate one file atomically, retaining every original row and field.

    ``backend.evaluate`` consumes arrays (N,4,4) in E,px,py,pz order and
    (N,4) PDG IDs, returning linear/quadratic arrays for x=C_HB/Lambda_TeV^2.
    Unsupported native Higgs rows fail unless explicitly retained at unity.
    Certified ZZ rows are always unity and remain marked as background.
    """
    source, destination = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().resolve()
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError("input and output must be different files")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    if unsupported not in {"error", "unity"}:
        raise ValueError("unsupported policy must be 'error' or 'unity'")
    points = np.asarray(coefficients, dtype=np.float64)
    if points.ndim != 1 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("at least one finite C_HB coefficient is required")
    if not np.isfinite(lambda_tev) or lambda_tev <= 0:
        raise ValueError("Lambda in TeV must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        scale_squared = np.square(np.float64(lambda_tev))
        effective = points / scale_squared
    if not np.isfinite(scale_squared) or scale_squared <= 0:
        raise ValueError("Lambda squared is outside the finite float64 range")
    if not np.all(np.isfinite(effective)):
        raise ValueError("effective C_HB/Lambda_TeV^2 coefficients overflow")
    if reconstruct is None:
        from SMEFT.kinematics import reconstruct_born_momenta
        reconstruct = reconstruct_born_momenta
    snapshot = _identity(source)
    digest = _sha256(source)
    if _identity(source) != snapshot:
        raise RuntimeError("input changed while capturing provenance")
    backend_metadata = getattr(backend, "metadata", {})
    if callable(backend_metadata):
        backend_metadata = backend_metadata()
    backend_metadata = json.loads(json.dumps(backend_metadata, allow_nan=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.smeft-", suffix=".root", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    counts = Counter()
    entries = 0
    input_sum = 0.
    output_sums = np.zeros(len(points), dtype=np.float64)
    try:
        with uproot.open(source) as root_file:
            strings, schemas = _metadata(root_file)
            tree_entries = {name: int(root_file[name].num_entries) for name in schemas}
            inventory = _source_inventory(strings)
            missing = (REQUIRED_BRANCHES | {weight_branch}) - schemas[TREE_NAME].keys()
            if missing:
                raise KeyError(f"missing required Analysis branches: {sorted(missing)}")
            schema = dict(schemas[TREE_NAME])
            for name in ("ratio", "weight"):
                schema[PREFIX + name] = np.dtype((np.float64, (len(points),)))
            for name in ("linear", "quadratic"):
                schema[PREFIX + name] = np.dtype(np.float64)
            schema[PREFIX + "valid"] = np.dtype(np.bool_)
            schema[PREFIX + "status"] = np.dtype(np.int8)
            with uproot.recreate(temporary) as output:
                for name, value in strings.items():
                    output[name] = value
                for name, other_schema in schemas.items():
                    if name != TREE_NAME:
                        _copy_tree(root_file[name], output, name, other_schema, step_size)
                tree = root_file[TREE_NAME]
                output.mktree(TREE_NAME, schema, title=getattr(tree, "title", ""))
                for chunk in tree.iterate(step_size=step_size, library="np", how=dict):
                    arrays = _as_dict(chunk)
                    size = len(arrays[weight_branch])
                    weights = np.asarray(_scalar(arrays, weight_branch, size, "fiu"), dtype=np.float64)
                    if not np.all(np.isfinite(weights)):
                        raise ValueError("input weights must be finite; negative weights are supported")
                    status = _classify(arrays, inventory, size)
                    if unsupported == "error" and np.any(status > 1):
                        reasons = {STATUS_NAMES[int(code)]: int(np.count_nonzero(status == code)) for code in np.unique(status[status > 1])}
                        raise ValueError(f"unsupported Higgs rows: {reasons}; explicitly use unsupported='unity' to retain them unmodified")
                    selected = status == 0
                    linear, quadratic = np.zeros(size), np.zeros(size)
                    if np.any(selected):
                        momenta, pdg_ids = reconstruct({key: value[selected] for key, value in arrays.items()}, prefix="born_")
                        predicted_linear, predicted_quadratic = backend.evaluate(momenta, pdg_ids)
                        predicted_linear, predicted_quadratic = np.asarray(predicted_linear, dtype=np.float64), np.asarray(predicted_quadratic, dtype=np.float64)
                        expected = (int(np.count_nonzero(selected)),)
                        if predicted_linear.shape != expected or predicted_quadratic.shape != expected:
                            raise ValueError("backend returned incorrect coefficient shapes")
                        if not np.all(np.isfinite(predicted_linear)) or not np.all(np.isfinite(predicted_quadratic)):
                            raise ValueError("backend returned nonfinite coefficients")
                        linear[selected], quadratic[selected] = predicted_linear, predicted_quadratic
                    with np.errstate(over="ignore", invalid="ignore"):
                        ratio = 1. + linear[:, None] * effective + quadratic[:, None] * effective**2
                        reweighted = weights[:, None] * ratio
                    if not np.all(np.isfinite(ratio)) or np.any(ratio < 0):
                        raise ValueError("full squared-amplitude ratios must be finite and nonnegative")
                    if not np.all(np.isfinite(reweighted)):
                        raise ValueError("reweighted signed event weights overflow")
                    arrays.update({PREFIX + "ratio": ratio, PREFIX + "weight": reweighted,
                                   PREFIX + "linear": linear, PREFIX + "quadratic": quadratic,
                                   PREFIX + "valid": status <= 1, PREFIX + "status": status})
                    output[TREE_NAME].extend(arrays)
                    counts.update(int(code) for code in status)
                    entries += size
                    input_sum += float(np.sum(weights, dtype=np.float64))
                    output_sums += np.sum(reweighted, axis=0, dtype=np.float64)
                if entries != tree.num_entries:
                    raise RuntimeError("written Analysis entry count differs from input")
                metadata = {
                    "schema_version": 1, "artifact_type": "smeft_hb_reweighting",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "input_file": str(source), "input_sha256": digest,
                    "operator": "cHB", "coefficients": points.tolist(), "lambda_tev": float(lambda_tev),
                    "effective_coefficients": effective.tolist(), "ratio_formula": "1 + linear*x + quadratic*x*x; x=cHB/(Lambda_TeV)^2",
                    "weight_branch": weight_branch, "weight_formula": f"smeft_hb_weight[:,i] = {weight_branch} * smeft_hb_ratio[:,i]",
                    "unsupported_policy": unsupported, "status_codes": {str(key): value for key, value in STATUS_NAMES.items()},
                    "status_counts": {value: counts[key] for key, value in STATUS_NAMES.items()},
                    "entries": entries, "input_sum_weights": input_sum, "output_sum_weights": output_sums.tolist(),
                    "valid_semantics": "evaluated native Higgs or certified unchanged ZZ background; unsupported-unity rows are false",
                    "point_axis": "requested coefficient order; fixed-length vector, not iterative weights",
                    "backend": backend_metadata,
                }
                output[METADATA_NAME] = json.dumps(metadata, sort_keys=True, allow_nan=False)
        if _identity(source) != snapshot or _sha256(source) != digest:
            raise RuntimeError("input changed while applying SMEFT reweighting")
        with uproot.open(temporary) as check:
            if check[TREE_NAME].num_entries != entries or set(check[TREE_NAME].keys()) != set(schema):
                raise RuntimeError("written SMEFT output failed schema/entry verification")
            for name in schemas:
                if name != TREE_NAME and check[name].num_entries != tree_entries[name]:
                    raise RuntimeError(f"copied tree {name!r} failed entry verification")
        if destination.exists() and os.path.samefile(source, destination):
            raise ValueError("output became an alias of the input")
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic no-clobber publication on the supported POSIX filesystem.
            os.link(temporary, destination)
            temporary.unlink()
        return metadata
    finally:
        temporary.unlink(missing_ok=True)
