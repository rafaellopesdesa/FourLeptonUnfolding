#!/usr/bin/env python3
"""Preserve exact source-LHE identities through shower skips and retries.

The pinned ThePEG 2.3 reader understands a legacy ``<event num>`` line
inside each event and propagates it with ``EventNumbering LHE``.  Such lines
are written only to a private HERWIG input copy: the original LHE remains
unchanged and is the source recorded for later analysis.  PYTHIA numbers
events using its source-reading counter and does not consume these tags.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, TextIO


EVENT_NUMBER_CONVENTION = "lhe_event_index_1based"
MAX_HEPMC_EVENT_NUMBER = 2**31 - 1
_EVENT_OPEN = re.compile(r"^\s*<event(?:\s+[^>]*)?>\s*$")
_EVENT_CLOSE = re.compile(r"^\s*</event>\s*$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _without_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    result = ""
    while line:
        if in_comment:
            end = line.find("-->")
            if end < 0:
                return result, True
            line = line[end + 3:]
            in_comment = False
        else:
            start = line.find("<!--")
            if start < 0:
                return result + line, False
            result += line[:start]
            line = line[start + 4:]
            in_comment = True
    return result, in_comment


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_lhe(
    lhe_file: str | Path,
    output: str | Path,
    *,
    tagged_output: str | Path | None = None,
    plain_output: str | Path | None = None,
) -> dict[str, Any]:
    """Count/hash original events and optionally write a private shower copy."""
    source = Path(lhe_file).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if tagged_output is not None and plain_output is not None:
        raise ValueError("choose tagged_output or plain_output, not both")
    copy_path = tagged_output if tagged_output is not None else plain_output
    copy_path = Path(copy_path).expanduser().resolve() if copy_path is not None else None
    if destination == source or copy_path in {source, destination}:
        raise ValueError("source, source-info output, and shower copy must be distinct")
    initial_stat = _stat_identity(source)
    source_hash = sha256_file(source)
    if copy_path is not None:
        copy_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    inside_event = False
    in_comment = False
    saw_root = False
    saw_end = False
    created_copy = False
    try:
        copy_context = copy_path.open("x", encoding="utf-8", newline="") if copy_path else nullcontext(None)
        created_copy = copy_path is not None
        with copy_context as target, _open_text(source) as stream:
            for line_number, line in enumerate(stream, 1):
                code, in_comment = _without_comments(line, in_comment)
                if code.lstrip().startswith("#"):
                    code = ""
                if re.match(r"^\s*<LesHouchesEvents(?:\s|>)", code):
                    saw_root = True
                if re.match(r"^\s*</LesHouchesEvents>\s*$", code):
                    saw_end = True
                if re.search(r"<event(?: num|_num)>", code) and inside_event:
                    # The original remains authoritative: do not silently
                    # reinterpret a prior custom identity convention.
                    raise ValueError(f"source already contains a legacy event-number tag at line {line_number}")
                if _EVENT_OPEN.match(code):
                    if inside_event or saw_end:
                        raise ValueError(f"nested or misplaced event at line {line_number}")
                    inside_event = True
                    count += 1
                    if count > MAX_HEPMC_EVENT_NUMBER:
                        raise ValueError("source index exceeds HepMC's signed 32-bit event-number range")
                elif _EVENT_CLOSE.match(code):
                    if not inside_event:
                        raise ValueError(f"unmatched event closing tag at line {line_number}")
                    if target is not None and tagged_output is not None:
                        target.write(f"<event num>{count}</event num>\n")
                    inside_event = False
                if target is not None:
                    target.write(line)
        if inside_event or in_comment or not saw_root or not saw_end or count == 0:
            raise ValueError("incomplete LHE document or no complete source events")
        if _stat_identity(source) != initial_stat:
            raise RuntimeError("original LHE changed while preparing its shower provenance")
        info = {
            "schema_version": 1,
            "kind": "lhe_source_info",
            "lhe_file": str(source),
            "lhe_sha256": source_hash,
            "lhe_event_count": count,
            "event_number": EVENT_NUMBER_CONVENTION,
            "source_stat": list(initial_stat),
        }
        if copy_path is not None:
            info["private_shower_copy"] = str(copy_path)
            info["private_shower_copy_kind"] = "herwig_native_event_tags" if tagged_output else "decompressed_lhe"
        _write_json(destination, info)
        return info
    except BaseException:
        if created_copy and copy_path is not None:
            copy_path.unlink(missing_ok=True)
        raise


def inspect_hepmc(hepmc_file: str | Path, source_event_count: int) -> dict[str, Any]:
    """Check every explicit source ID, permitting skips and repeated showers."""
    if not 0 < source_event_count <= MAX_HEPMC_EVENT_NUMBER:
        raise ValueError("invalid source LHE event count")
    path = Path(hepmc_file).expanduser().resolve()
    seen = bytearray((source_event_count + 7) // 8)
    total = 0
    unique = 0
    format_name = None
    ended = False
    starts = {
        "HepMC::Asciiv3-START_EVENT_LISTING": "HepMC3",
        "HepMC::IO_GenEvent-START_EVENT_LISTING": "HepMC2",
    }
    ends = {
        "HepMC::Asciiv3-END_EVENT_LISTING": "HepMC3",
        "HepMC::IO_GenEvent-END_EVENT_LISTING": "HepMC2",
    }
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if stripped in starts:
                if format_name is not None:
                    raise ValueError("multiple HepMC event listings are not supported")
                format_name = starts[stripped]
            elif stripped in ends:
                if format_name != ends[stripped] or ended:
                    raise ValueError("inconsistent HepMC listing footer")
                ended = True
            elif re.match(r"^E(?:\s|$)", stripped):
                fields = stripped.split()
                if format_name is None or ended or len(fields) < 2 or not re.fullmatch(r"[0-9]+", fields[1]):
                    raise ValueError(f"invalid HepMC event record at line {line_number}")
                number = int(fields[1])
                if not 1 <= number <= source_event_count:
                    raise ValueError(
                        f"HepMC source event number {number} is outside [1, {source_event_count}]"
                    )
                index = number - 1
                byte, bit = divmod(index, 8)
                if not (seen[byte] & (1 << bit)):
                    seen[byte] |= 1 << bit
                    unique += 1
                total += 1
    if format_name is None or not ended or total == 0:
        raise ValueError("incomplete or empty HepMC event listing")
    return {
        "hepmc_format": format_name,
        "hepmc_event_count": total,
        "hepmc_unique_source_events": unique,
        "hepmc_repeated_source_events": total - unique,
    }


def finalize_provenance(
    source_info: str | Path,
    hepmc_file: str | Path,
    shower: str,
    output: str | Path,
) -> dict[str, Any]:
    """Publish provenance only after the shower and all source checks succeed."""
    if shower not in {"pythia", "herwig"}:
        raise ValueError("shower must be pythia or herwig")
    with Path(source_info).open(encoding="utf-8") as stream:
        info = json.load(stream)
    if info.get("schema_version") != 1 or info.get("kind") != "lhe_source_info" or info.get("event_number") != EVENT_NUMBER_CONVENTION:
        raise ValueError("unrecognized prepared LHE source information")
    source = Path(info["lhe_file"]).resolve()
    if sha256_file(source) != info["lhe_sha256"]:
        raise RuntimeError("original LHE changed between preparation and shower completion")
    hepmc = Path(hepmc_file).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination in {source, hepmc, Path(source_info).resolve()}:
        raise ValueError("provenance output must differ from its input files")
    initial_stat = _stat_identity(hepmc)
    counts = inspect_hepmc(hepmc, int(info["lhe_event_count"]))
    hepmc_hash = sha256_file(hepmc)
    if _stat_identity(hepmc) != initial_stat:
        raise RuntimeError("HepMC changed while validating its source identities")
    manifest = {
        "schema_version": 1,
        "lhe_file": str(source),
        "lhe_sha256": info["lhe_sha256"],
        "lhe_event_count": int(info["lhe_event_count"]),
        "hepmc_file": str(hepmc),
        "hepmc_sha256": hepmc_hash,
        "event_number": EVENT_NUMBER_CONVENTION,
        "shower": shower,
        "association": "explicit source-event identity; not accepted-output ordinal",
        **counts,
    }
    _write_json(destination, manifest)
    return manifest


def verify_provenance(manifest_file: str | Path, hepmc_file: str | Path) -> dict[str, Any]:
    with Path(manifest_file).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("event_number") != EVENT_NUMBER_CONVENTION:
        raise ValueError("unrecognized LHE provenance manifest")
    if sha256_file(hepmc_file) != manifest["hepmc_sha256"]:
        raise ValueError("HepMC checksum does not match LHE provenance")
    if sha256_file(manifest["lhe_file"]) != manifest["lhe_sha256"]:
        raise ValueError("original LHE checksum does not match provenance")
    return manifest


def beam_energies(lhe_file: str | Path) -> tuple[float, float]:
    in_init = False
    in_comment = False
    with _open_text(Path(lhe_file)) as stream:
        for line in stream:
            code, in_comment = _without_comments(line, in_comment)
            if re.match(r"^\s*<init(?:\s|>)", code):
                in_init = True
                continue
            if in_init and code.strip() and not code.lstrip().startswith("#"):
                fields = code.split()
                if len(fields) < 4:
                    break
                return tuple(float(value.replace("D", "e").replace("d", "e")) for value in fields[2:4])
    raise ValueError("cannot read beam energies from the LHE init record")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--lhe", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    copies = prepare.add_mutually_exclusive_group()
    copies.add_argument("--tagged-output", type=Path)
    copies.add_argument("--plain-output", type=Path)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--source-info", required=True, type=Path)
    finalize.add_argument("--hepmc", required=True, type=Path)
    finalize.add_argument("--shower", required=True, choices=("pythia", "herwig"))
    finalize.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--hepmc", required=True, type=Path)
    beams = commands.add_parser("beams")
    beams.add_argument("--lhe", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        info = prepare_lhe(args.lhe, args.output, tagged_output=args.tagged_output, plain_output=args.plain_output)
        print(info["lhe_event_count"])
    elif args.command == "finalize":
        finalize_provenance(args.source_info, args.hepmc, args.shower, args.output)
    elif args.command == "verify":
        verify_provenance(args.manifest, args.hepmc)
    else:
        print(" ".join(f"{value:.17g}" for value in beam_energies(args.lhe)))


if __name__ == "__main__":
    main()
