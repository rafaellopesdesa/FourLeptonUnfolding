#!/usr/bin/env python3
"""Decorate certified ROOT ntuples with a scan of HB decay-ME deformations."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SMEFT.application import decorate_file


def expand_inputs(specifications: list[str]) -> list[Path]:
    result, seen = [], set()
    for specification in specifications:
        candidate = Path(specification).expanduser()
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.root"))
        elif candidate.is_file():
            matches = [candidate]
        else:
            matches = [Path(name) for name in sorted(glob.glob(str(candidate)))]
        matches = [path.resolve() for path in matches if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"no input ROOT files match {specification!r}")
        for path in matches:
            if path not in seen:
                result.append(path)
                seen.add(path)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="ROOT files, quoted globs, or directories containing ROOT files")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("-o", "--output", type=Path, help="output file for exactly one input")
    destination.add_argument("--output-dir", type=Path, help="output directory; each input retains its basename")
    parser.add_argument("--cHB", nargs="+", type=float, required=True, dest="coefficients", help="dimensionless Wilson coefficients in requested vector order")
    parser.add_argument("--lambda-tev", type=float, default=1., help="EFT scale in TeV (default: 1)")
    parser.add_argument("--weight-branch", default="weight", help="scalar signed input weight (default: weight)")
    parser.add_argument("--unsupported", choices=("error", "unity"), default="error", help="policy for unavailable/tau/unsupported Higgs rows (default: error)")
    parser.add_argument("--step-size", default="64 MB", help="uproot chunk size (default: 64 MB)")
    parser.add_argument("--build-dir", type=Path, help="writable directory for the local matrix-element backend build")
    parser.add_argument("--overwrite", action="store_true", help="replace output files; inputs may never be overwritten")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        inputs = expand_inputs(args.inputs)
        if args.output is not None:
            if len(inputs) != 1:
                raise ValueError("--output requires exactly one input; use --output-dir for multiple files")
            destinations = [args.output.expanduser().resolve()]
        else:
            destinations = [args.output_dir.expanduser().resolve() / path.name for path in inputs]
            if len({path.name for path in inputs}) != len(inputs):
                raise ValueError("input basenames collide in --output-dir")
        input_paths = set(inputs)
        for destination in destinations:
            if destination in input_paths or any(destination.exists() and destination.samefile(path) for path in inputs):
                raise ValueError("an output path aliases an input file")
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"output already exists: {destination}")
        from SMEFT.backend import NativeHBBackend
        backend = NativeHBBackend(build_dir=args.build_dir)
        for source, destination in zip(inputs, destinations):
            metadata = decorate_file(
                source, destination, args.coefficients, backend=backend,
                lambda_tev=args.lambda_tev, weight_branch=args.weight_branch,
                unsupported=args.unsupported, step_size=args.step_size,
                overwrite=args.overwrite,
            )
            print(json.dumps({"input": str(source), "output": str(destination),
                              "entries": metadata["entries"], "status_counts": metadata["status_counts"]}, sort_keys=True))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(1, f"SMEFT reweighting failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
