#!/usr/bin/env python3
"""Bind a generation LHE provenance manifest to its Delphes output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, *, expected_sha256: str | None = None) -> dict:
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{path}: generation manifest changed during simulation")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: provenance manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported LHE provenance schema")
    if manifest.get("event_number") != "lhe_event_index_1based":
        raise ValueError(f"{path}: no exact LHE event-number mapping")
    for key in ("lhe_sha256", "hepmc_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError(f"{path}: invalid {key}")
    count = manifest.get("lhe_event_count")
    if type(count) is not int or count <= 0:
        raise ValueError(f"{path}: invalid LHE event count")
    if not isinstance(manifest.get("lhe_file"), str) or not manifest["lhe_file"]:
        raise ValueError(f"{path}: missing original LHE source path")
    return manifest


def verify_hepmc(manifest_path: Path, hepmc_path: Path) -> dict:
    manifest = read_manifest(manifest_path)
    if sha256_file(hepmc_path) != manifest["hepmc_sha256"]:
        raise ValueError(f"{hepmc_path}: HepMC hash does not match LHE provenance")
    return manifest


def bind_output(
    manifest_path: Path, delphes_path: Path, output_path: Path, *, event_count: int,
    expected_manifest_sha256: str | None = None,
) -> None:
    """Create a portable manifest for an already validated Delphes output."""

    if event_count <= 0:
        raise ValueError("Delphes event count must be positive")
    if output_path.resolve() in {manifest_path.resolve(), delphes_path.resolve()}:
        raise ValueError("bound manifest must not overwrite a source file")
    manifest = read_manifest(manifest_path, expected_sha256=expected_manifest_sha256)
    manifest.update(
        delphes_file=str(delphes_path.resolve()),
        delphes_sha256=sha256_file(delphes_path),
        delphes_event_count=event_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_output(
    source_path: Path, bound_path: Path, delphes_path: Path, *, event_count: int
) -> None:
    source = read_manifest(source_path)
    bound = read_manifest(bound_path)
    # Compare identities, not paths: source files may have been relocated.
    for key in ("lhe_sha256", "hepmc_sha256", "lhe_event_count", "event_number"):
        if source[key] != bound[key]:
            raise ValueError(f"{bound_path}: source provenance changed ({key})")
    if bound.get("delphes_event_count") != event_count:
        raise ValueError(f"{bound_path}: Delphes event count does not match this run")
    if bound.get("delphes_sha256") != sha256_file(delphes_path):
        raise ValueError(f"{delphes_path}: Delphes hash does not match LHE provenance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-input")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("hepmc", type=Path)
    for name in ("bind-output", "verify-output"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("delphes", type=Path)
        command.add_argument("bound_manifest", type=Path)
        command.add_argument("--events", type=int, required=True)
        if name == "bind-output":
            command.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "verify-input":
        verify_hepmc(args.manifest, args.hepmc)
    elif args.command == "bind-output":
        bind_output(
            args.manifest, args.delphes, args.bound_manifest, event_count=args.events,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    else:
        verify_output(args.manifest, args.bound_manifest, args.delphes, event_count=args.events)


if __name__ == "__main__":
    main()
