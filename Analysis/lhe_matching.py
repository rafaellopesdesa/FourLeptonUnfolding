"""Match original LHE identities and separately bind accepted native decays."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np

from Analysis.lhe import event_observables, iter_lhe_events
from Analysis.born import born_observables, lhe_born_observables
from Tools.born_projection import BORN_PROJECTION_METADATA


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
    born_rows: list[dict] | None = None
    born_metadata: dict | None = None

    def decorate(self, output: dict[str, np.ndarray], *, first_input_entry: int = 0) -> None:
        """Fill reduced rows using their preserved HepMC/Delphes event number."""
        for row, number in enumerate(output["event_number"]):
            try:
                observation = self.observations[int(number)]
            except KeyError as error:
                raise ValueError(f"No LHE match for Delphes Event.Number={number}") from error
            for name, value in observation.items():
                output[name][row] = value
            output["lhe_source_id"][row] = self.source_id
            if self.born_rows is not None:
                # A repeated source LHE event can be showered/decayed again.
                # Its native decay belongs to this accepted output occurrence.
                for name, value in self.born_rows[first_input_entry + row].items():
                    output[name][row] = value


def load_lhe_source(
    input_path: Path,
    tree: object,
    *,
    lhe_path: Path | None = None,
    manifest_path: Path | None = None,
    require_lhe: bool = False,
    hepmc_path: Path | None = None,
    require_born: bool = False,
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
        if require_lhe or require_born or explicit_manifest or lhe_path is not None or hepmc_path is not None:
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
    process = manifest.get("process")
    definition = manifest.get("hard_decay_definition")
    expected_definition = {
        ("ZZ", "pythia"): "lhe_status1_charged_leptons",
        ("ZZ", "herwig"): "lhe_status1_charged_leptons",
        ("gg_H", "pythia"): "pythia_event_before_decay_fsr",
        ("gg_H", "herwig"): "herwig_qtilde_original_decay",
    }.get((process, manifest.get("shower")))
    born_enabled = expected_definition is not None and definition == expected_definition
    if (require_born or hepmc_path is not None) and not born_enabled:
        raise ValueError("No certified Born input definition; rerun the updated generation chain")
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
    requested_sequence: list[int] = []
    for arrays in tree.iterate(expressions=["Event.Number"], library="ak", how=dict):
        for row in ak.to_list(arrays["Event.Number"]):
            values = row if isinstance(row, list) else [row]
            if len(values) != 1:
                raise ValueError("Each Delphes row must have exactly one Event.Number")
            number = values[0]
            if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= count:
                raise ValueError(f"Invalid source LHE index in Delphes Event.Number: {number}")
            requested.add(number)
            requested_sequence.append(number)
    observations: dict[int, dict] = {}
    parsed_count = 0
    for event in iter_lhe_events(lhe_path):
        parsed_count = event.ordinal
        if event.ordinal in requested:
            observations[event.ordinal] = event_observables(event)
            if born_enabled and process == "ZZ":
                observations[event.ordinal].update(lhe_born_observables(event))
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
    result = MatchedLHESource(observations, source_id, metadata)
    if born_enabled:
        if process == "gg_H":
            result.born_rows, decay_counts, used_hepmc = _load_shower_born(
                manifest, manifest_path, requested_sequence, hepmc_path=hepmc_path,
            )
            rows = result.born_rows
            born_source = 2 if manifest["shower"] == "pythia" else 3
        else:
            rows = [observations[number] for number in requested_sequence]
            decay_counts = {"lhe_quartet": sum(bool(row["born_available"]) for row in rows),
                            "unavailable_lhe_quartet": sum(not row["born_available"] for row in rows)}
            used_hepmc = None
            born_source = 1
        result.born_metadata = dict(metadata)
        result.born_metadata.update(
            born_source=born_source, hard_decay_definition=definition,
            born_available_entries=sum(bool(row["born_available"]) for row in rows),
            born_kinematics_valid_entries=sum(bool(row["born_kinematics_valid"]) for row in rows),
            decay_status_counts=dict(decay_counts),
        )
        if used_hepmc is not None:
            result.born_metadata["hepmc_file"] = str(used_hepmc)
    return result


def _load_shower_born(manifest, manifest_path, event_numbers, *, hepmc_path=None):
    """Join accepted HepMC occurrences to unfiltered, ordered Delphes rows.

    Delphes' two readers emit one row per accepted HepMC event, in input order;
    run_simulation verifies event retention and binds the ROOT/HepMC hashes.
    Check every source number as well as that order, including repeated IDs.
    This is not an ordinal join between LHE and accepted shower events.
    """
    from Analysis.hard_decay import iterate_hard_decays

    if hepmc_path is None:
        stored_path = manifest.get("hepmc_file")
        if not isinstance(stored_path, str) or not stored_path:
            raise ValueError("Born input requires the original accepted HepMC file")
        hepmc_path = Path(stored_path)
        if not hepmc_path.is_absolute():
            hepmc_path = manifest_path.parent / hepmc_path
    hepmc_path = hepmc_path.expanduser().resolve()
    before = file_identity(hepmc_path)
    if file_sha256(hepmc_path) != _digest(manifest, "hepmc_sha256"):
        raise ValueError("HepMC SHA-256 differs from the source used for Delphes")
    count = _positive_integer(manifest.get("hepmc_event_count"), "hepmc_event_count")
    if count < len(event_numbers):
        raise ValueError("Delphes contains more rows than the accepted HepMC source")
    source = 2 if manifest["shower"] == "pythia" else 3
    records = iterate_hard_decays(hepmc_path, manifest["shower"], manifest["hard_decay_definition"])
    rows, statuses = [], Counter()
    try:
        for index, expected_number in enumerate(event_numbers, start=1):
            try:
                number, record = next(records)
            except StopIteration as error:
                raise ValueError("HepMC ended before all Delphes rows were matched") from error
            if number != expected_number:
                raise ValueError(
                    f"HepMC/Delphes source-number sequence differs at accepted event {index}: "
                    f"{number} != {expected_number}"
                )
            if record.status in {
                "missing_native_attribute", "invalid_native_attribute",
                "unsupported_attribute_schema", "unexpected_attribute_source",
                "unexpected_attribute_units", "invalid_attribute_status",
            }:
                raise ValueError(
                    f"Invalid native decay transport at accepted event {index}: "
                    f"{record.status}; rerun generation with the updated Pythia bridge"
                )
            rows.append(born_observables(
                record.particles if record.valid else (), source=source,
                shower_event_index=index,
            ))
            statuses[record.status] += 1
        # A deliberate --max-events reduction uses a prefix. For a full run,
        # also verify that the reader did not hide trailing accepted events.
        if len(event_numbers) == count and next(records, None) is not None:
            raise ValueError("HepMC contains more accepted events than its provenance declares")
    finally:
        records.close()
    if file_identity(hepmc_path) != before:
        raise ValueError("HepMC changed while extracting its native decay record")
    return rows, statuses, hepmc_path


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


def merge_born_metadata(paths: list[Path]) -> dict | None:
    """Keep the projection and input-stage definitions with merged samples."""
    import uproot

    sources: dict[str, dict] = {}
    saw_metadata = False
    for path in paths:
        with uproot.open(path) as root_file:
            if "born_metadata" not in root_file:
                continue
            metadata = json.loads(str(root_file["born_metadata"]))
            if metadata.get("schema_version") != 1 or metadata.get("projection") != BORN_PROJECTION_METADATA:
                raise ValueError(f"Incompatible Born projection metadata in {path}")
            saw_metadata = True
            for record in metadata.get("sources", []):
                sources[json.dumps(record, sort_keys=True)] = record
    if not saw_metadata:
        return None
    return {"schema_version": 1, "projection": BORN_PROJECTION_METADATA,
            "sources": list(sources.values())}
