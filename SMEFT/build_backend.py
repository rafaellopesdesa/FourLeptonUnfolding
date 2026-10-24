#!/usr/bin/env python3
"""Compile the vendored matrix element with a local C++17 compiler, offline."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
FLAGS = ("-O3", "-std=c++17", "-fPIC", "-shared", "-pthread", "-fno-fast-math")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> str:
    files = [ROOT / "native/hb_wrapper.cc", ROOT / "cards/param_card.dat"]
    files.extend(path for path in sorted((ROOT / "generated").rglob("*"))
                 if path.is_file() and path.suffix in {".h", ".cc", ".json"})
    if not (ROOT / "generated/provenance.json").is_file():
        raise FileNotFoundError("missing vendored SMEFT matrix-element provenance")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def ensure_backend(build_dir: str | Path | None = None, *, force: bool = False) -> Path:
    """Reuse only a matching compiled binary; serialize concurrent builders."""
    destination = ROOT / ".build" if build_dir is None else Path(build_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise ValueError("CXX must name a C++17 compiler")
    signature = hashlib.sha256(json.dumps(
        [fingerprint, compiler, FLAGS], sort_keys=True).encode()).hexdigest()[:24]
    # A new source/configuration gets a new path: dlopen can otherwise reuse
    # an old library already loaded in a long-lived notebook process.
    library = destination / f"libhb-{signature}.so"
    manifest_path = library.with_suffix(".json")
    with (destination / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and library.is_file() and manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text())
                if (previous.get("source_sha256") == fingerprint
                        and previous.get("compiler_command") == compiler
                        and previous.get("flags") == list(FLAGS)
                        and previous.get("library_sha256") == sha256(library)):
                    return library
            except (OSError, ValueError):
                pass
        try:
            version = subprocess.run([*compiler, "--version"], text=True, capture_output=True, check=True).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError("SMEFT backend requires a C++17 compiler; load the Unity GCC module or set CXX") from error
        sources = [ROOT / "native/hb_wrapper.cc", *sorted((ROOT / "generated/src").glob("*.cc"))]
        if len(sources) < 2:
            raise FileNotFoundError("missing vendored SMEFT matrix-element sources")
        descriptor, name = tempfile.mkstemp(prefix=".libhb-", suffix=".so", dir=destination)
        os.close(descriptor)
        temporary = Path(name)
        try:
            command = [*compiler, *FLAGS, "-I", str(ROOT / "generated/src"),
                       *map(str, sources), "-o", str(temporary)]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"SMEFT backend compilation failed:\n{result.stderr}")
            if source_fingerprint() != fingerprint:
                raise RuntimeError("SMEFT sources changed during compilation; retry after the update finishes")
            manifest = {"source_sha256": fingerprint, "library_sha256": sha256(temporary),
                        "compiler_command": compiler, "compiler_version": version.strip(),
                        "flags": list(FLAGS)}
            os.replace(temporary, library)
            descriptor, name = tempfile.mkstemp(prefix=".build-", suffix=".json", dir=destination)
            temporary = Path(name)
            with os.fdopen(descriptor, "w") as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(ensure_backend(args.build_dir, force=args.force))


if __name__ == "__main__":
    main()
