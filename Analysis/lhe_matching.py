"""Join hard events by source identity, never by accepted-shower row order."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np

from Analysis.lhe import event_observables, iter_lhe_events


EVENT_NUMBER_CONVENTION = "lhe_event_index_1based"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _digest(manifest: dict, name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"LHE provenance has no valid {name}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"LHE provenance {name} must be a positive integer")
    return value


@dataclass
class MatchedLHESource:
    observations: dict[int, dict]
    source_id: int
    metadata: dict

    def decorate(self, output: dict[str, np.ndarray]) -> None:
        """Fill reduced rows using their preserved HepMC/Delphes event number."""
        for row, number in enumerate(output["event_number"]):
            try:
                observation = self.observations[int(number)]
            except KeyError as error:
                raise ValueError(f"No LHE match for Delphes Event.Number={number}") from error
            for name, value in observation.items():
                output[name][row] = value
            output["lhe_source_id"][row] = self.source_id


def load_lhe_source(
    input_path: Path,
    tree: object,
    *,
    lhe_path: Path | None = None,
    manifest_path: Path | None = None,
    require_lhe: bool = False,
) -> MatchedLHESource | None:
    """Load only needed LHE observables after verifying both input digests.

    Simulation binds its ROOT output to the generation provenance. Merely
    supplying an LHE file does not establish a trustworthy event mapping.
    """
    explicit_manifest = manifest_path is not None
    manifest_path = (
        input_path.parent / "lhe-provenance.json"
        if manifest_path is None else manifest_path.expanduser().resolve()
    )
    if not manifest_path.is_file():
        if require_lhe or explicit_manifest or lhe_path is not None:
            raise FileNotFoundError(
                f"Missing simulation-bound LHE provenance: {manifest_path}; "
                "rerun the current generation/simulation chain. Row-order matching is unsafe."
            )
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1):
        raise ValueError("Unsupported LHE provenance schema")
    if manifest.get("event_number") != EVENT_NUMBER_CONVENTION:
        raise ValueError("LHE provenance does not certify source-index event numbering")
    lhe_digest = _digest(manifest, "lhe_sha256")
    root_digest = _digest(manifest, "delphes_sha256")
    count = _positive_integer(manifest.get("lhe_event_count"), "lhe_event_count")
    root_count = _positive_integer(manifest.get("delphes_event_count"), "delphes_event_count")
    if root_count != tree.num_entries:
        raise ValueError("Delphes entry count differs from its LHE provenance")
    if file_sha256(input_path) != root_digest:
        raise ValueError("Delphes SHA-256 differs from its LHE provenance")
    if lhe_path is None:
        stored_path = manifest.get("lhe_file")
        if not isinstance(stored_path, str) or not stored_path:
            raise ValueError("LHE provenance has no source file path")
        lhe_path = Path(stored_path)
        if not lhe_path.is_absolute():
            lhe_path = manifest_path.parent / lhe_path
    lhe_path = lhe_path.expanduser().resolve()
    original_identity = file_identity(lhe_path)
    if file_sha256(lhe_path) != lhe_digest:
        raise ValueError("LHE SHA-256 differs from the source used for showering")

    requested: set[int] = set()
    for arrays in tree.iterate(expressions=["Event.Number"], library="ak", how=dict):
        for row in ak.to_list(arrays["Event.Number"]):
            values = row if isinstance(row, list) else [row]
            if len(values) != 1:
                raise ValueError("Each Delphes row must have exactly one Event.Number")
            number = values[0]
            if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= count:
                raise ValueError(f"Invalid source LHE index in Delphes Event.Number: {number}")
            requested.add(number)
    observations: dict[int, dict] = {}
    parsed_count = 0
    for event in iter_lhe_events(lhe_path):
        parsed_count = event.ordinal
        if event.ordinal in requested:
            observations[event.ordinal] = event_observables(event)
    if parsed_count != count:
        raise ValueError(f"LHE contains {parsed_count} events; provenance declares {count}")
    if observations.keys() != requested:
        raise ValueError("Some Delphes event numbers have no source LHE event")
    if file_identity(lhe_path) != original_identity:
        raise ValueError("LHE source changed while reading its observables")
    source_id = int(lhe_digest[:16], 16)
    metadata = dict(manifest)
    metadata.update(
        lhe_file=str(lhe_path), delphes_file=str(input_path),
        provenance_file=str(manifest_path), lhe_source_id=f"{source_id:016x}",
        matched_entries=root_count, matched_source_events=len(requested),
    )
    return MatchedLHESource(observations, source_id, metadata)


def merge_lhe_metadata(paths: list[Path]) -> dict | None:
    """Preserve full source hashes through compact merging and pseudo-data."""
    import uproot

    sources: dict[str, dict] = {}
    source_ids: dict[str, str] = {}
    for path in paths:
        with uproot.open(path) as root_file:
            if "lhe_metadata" not in root_file:
                continue
            metadata = json.loads(str(root_file["lhe_metadata"]))
            if metadata.get("schema_version") != 1:
                raise ValueError(f"Unsupported LHE metadata in {path}")
            for record in metadata.get("sources", []):
                source_id = record["lhe_source_id"]
                digest = record["lhe_sha256"]
                if source_ids.setdefault(source_id, digest) != digest:
                    raise ValueError("LHE source ID collision between distinct SHA-256 digests")
                key = json.dumps(record, sort_keys=True)
                sources[key] = record
    if not sources:
        return None
    return {"schema_version": 1, "event_number": EVENT_NUMBER_CONVENTION,
            "sources": list(sources.values())}
