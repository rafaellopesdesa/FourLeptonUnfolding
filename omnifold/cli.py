"""Command-line access to the same checkpointed workflow used by the notebook."""

from __future__ import annotations

import argparse
from dataclasses import fields
import glob
import json
import os
from pathlib import Path
import sys

from .ratios import RatioConfig
from .reporting import save_reports
from .workflow import OmniFoldSession


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _runtime_options(parser: argparse.ArgumentParser, *, training: bool = False) -> None:
    parser.add_argument("--run-dir", type=Path, required=True, help="Campaign checkpoint directory")
    parser.add_argument("--device", default="auto" if training else None,
                        help="PyTorch device, e.g. auto, cpu, cuda, cuda:0; resume defaults to the saved device")
    parser.add_argument("--cpu-threads", type=_positive_int,
                        help="PyTorch CPU threads (default: SLURM_CPUS_PER_TASK, otherwise 4)")


def _iteration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=_positive_int, default=1,
                        help="Number of additional iterations to fit (default: 1)")
    parser.add_argument("--no-reports", action="store_true",
                        help="Skip automatic PDF reports; generate them later with report")
    parser.add_argument("--shape-only", action="store_true",
                        help="Normalize comparison plots to unit signed yield")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, inspect, and apply the notebook's checkpointed OmniFold workflow.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Create a new campaign and fit iterations")
    _runtime_options(train, training=True)
    _iteration_options(train)
    train.add_argument("--data-root", type=Path, required=True,
                       help="Nominal background-removal-decorated data.root")
    train.add_argument("--gg-h-root", type=Path, required=True,
                       help="Original paired gg_H_pythia.root")
    train.add_argument("--truth-root", type=Path,
                       help="Optional full signal-only gg_H_herwig.root, for closure plots only")
    train.add_argument("--data-weight-branch", default="weight_background_removed")
    train.add_argument("--mc-weight-branch", default="weight")
    train.add_argument("--truth-weight-branch", default="weight")
    train.add_argument("--split-seed", type=_nonnegative_int, default=12345,
                       help="Fixed event-partition seed (default: 12345)")
    network = train.add_argument_group("Network settings (same defaults as RatioConfig)")
    defaults = RatioConfig()
    descriptions = {
        "patience": "Validation checks without sufficient improvement before stopping",
        "maximum_epochs": "Optimizer-update budget in nominal epochs (maximum_epochs times steps_per_epoch)",
        "learning_rate_decay": "Learning-rate multiplier per learning_rate_decay_steps optimizer updates",
        "weight_decay": "L2 regularization coefficient",
        "validation_interval_steps": "Maximum optimizer updates between validation checks (automatically reduced for the smaller fit bank)",
        "learning_rate_decay_steps": "Optimizer updates over which learning_rate_decay is applied",
        "validation_min_delta": "Minimum validation BCE improvement that resets patience",
        "baseline_tolerance": "Maximum accepted validation BCE excess above ln(2)",
    }
    for field in fields(RatioConfig):
        if field.name in {"device", "overwrite"}:
            continue
        default = getattr(defaults, field.name)
        kind = (float if isinstance(default, float) else
                _nonnegative_int if field.name in {"max_retries", "seed"} else _positive_int)
        network.add_argument("--" + field.name.replace("_", "-"), type=kind, default=default,
                             help=f"{descriptions.get(field.name, field.name.replace('_', ' '))} (default: {default})")

    resume = commands.add_parser("resume", help="Append iterations to a notebook or CLI campaign")
    _runtime_options(resume)
    _iteration_options(resume)

    report = commands.add_parser("report", help="Save diagnostics and closure PDFs without fitting")
    _runtime_options(report)
    selection = report.add_mutually_exclusive_group()
    selection.add_argument("--iteration", type=_positive_int,
                           help="Completed iteration to plot (default: latest)")
    selection.add_argument("--all-iterations", action="store_true")
    report.add_argument("--shape-only", action="store_true")

    apply = commands.add_parser("apply", help="Decorate ROOT files with the frozen iteration sequence")
    _runtime_options(apply)
    apply.add_argument("--inputs", nargs="+", required=True,
                       help="ROOT files, quoted globs, or directories (directories select data*.root)")
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--no-mc", action="store_true", help="Do not append the campaign's nominal MC")
    apply.add_argument("--overwrite", action="store_true",
                       help="Replace an existing recognized OmniFold export")

    history = commands.add_parser("history", help="Print completed yields and objectives as JSON")
    _runtime_options(history)
    history.add_argument("--output", type=Path, help="Also save JSON to a new file")

    likelihood = commands.add_parser("likelihood", help="Evaluate frozen likelihood ratios without fitting")
    _runtime_options(likelihood)
    likelihood.add_argument("--data-root", type=Path,
                            help="Other decorated data/toy file (default: campaign's nominal data)")
    likelihood.add_argument("--output", type=Path, help="Also save JSON to a new file")
    return parser


def _configure_runtime(args: argparse.Namespace) -> None:
    # Batch jobs must work without a graphical display or an IPython kernel.
    import matplotlib
    matplotlib.use("Agg")
    import torch

    threads = args.cpu_threads
    if threads is None:
        try:
            threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
        except ValueError as error:
            raise ValueError("SLURM_CPUS_PER_TASK must be an integer; set --cpu-threads explicitly") from error
        if threads < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be positive; set --cpu-threads explicitly")
    torch.set_num_threads(threads)
    if args.device not in (None, "auto"):
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable in this allocation; request a GPU or use --device cpu")


def _expand_inputs(specifications: list[str]) -> list[Path]:
    paths: list[Path] = []
    for specification in specifications:
        expanded = os.path.expanduser(specification)
        direct = Path(expanded)
        matches = [direct] if direct.exists() else [Path(item) for item in sorted(glob.glob(expanded))]
        if not matches:
            raise FileNotFoundError(f"No inputs match {specification!r}")
        for match in matches:
            candidates = sorted(match.glob("data*.root")) if match.is_dir() else [match]
            if not candidates:
                raise FileNotFoundError(f"No data*.root files in {match}")
            for candidate in candidates:
                if not candidate.is_file() or candidate.suffix != ".root":
                    raise ValueError(f"Expected a ROOT file: {candidate}")
                resolved = candidate.resolve()
                if resolved not in paths:
                    paths.append(resolved)
    return paths


def _print_json(rows: list[dict], output: Path | None) -> None:
    payload = json.dumps(rows, indent=2, allow_nan=False) + "\n"
    if output is not None:
        # Never replace a session manifest, input, or existing result by accident.
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


def _check_output_path(output: Path | None, run_dir: Path) -> None:
    """Keep user-selected exports out of the campaign's checkpoint namespace."""
    if output is None:
        return
    destination = output.expanduser().resolve()
    campaign = run_dir.expanduser().resolve()
    for name in ("iterations", ".training.lock", "session.json"):
        protected = (campaign / name).resolve()
        if destination == protected or protected in destination.parents:
            raise ValueError(f"Output must not use the campaign's reserved {name} path: {destination}")


def _reports(session: OmniFoldSession, iteration: int, *, shape_only: bool) -> None:
    try:
        paths = save_reports(session, iteration=iteration, shape_only=shape_only)
    except Exception:
        print(f"Iteration {iteration} is already checkpointed. Fix the report error and run "
              f"'python -m omnifold report --run-dir {str(session.run_dir)!r} --iteration {iteration}' "
              "to regenerate its reports without fitting again.", file=sys.stderr)
        raise
    for path in paths:
        print(f"Saved {path}", flush=True)


def _run(args: argparse.Namespace) -> None:
    if args.command in {"history", "likelihood"}:
        _check_output_path(args.output, args.run_dir)
    elif args.command == "apply":
        _check_output_path(args.output_dir, args.run_dir)
    _configure_runtime(args)
    if args.command == "train":
        config = RatioConfig(**{field.name: getattr(args, field.name)
                                for field in fields(RatioConfig) if field.name != "overwrite"})
        config.validate()
        session = OmniFoldSession.create(
            data_path=args.data_root, mc_path=args.gg_h_root, truth_path=args.truth_root,
            run_dir=args.run_dir, config=config, seed=args.split_seed,
            data_weight_branch=args.data_weight_branch, mc_weight_branch=args.mc_weight_branch,
            truth_weight_branch=args.truth_weight_branch)
    else:
        session = OmniFoldSession.load(args.run_dir, device=args.device)

    if args.command in {"train", "resume"}:
        for _ in range(args.iterations):
            result = session.step()
            if not args.no_reports:
                _reports(session, result.index, shape_only=args.shape_only)
        print(f"Campaign {session.run_dir}: {len(session.iterations)} completed iterations.", flush=True)
    elif args.command == "report":
        if not session.iterations:
            raise ValueError("No completed iterations; fit an iteration before requesting reports")
        indices = ([item.index for item in session.iterations] if args.all_iterations else
                   [args.iteration if args.iteration is not None else session.iterations[-1].index])
        if any(index > len(session.iterations) for index in indices):
            raise ValueError("Requested iteration has not been completed")
        for index in indices:
            _reports(session, index, shape_only=args.shape_only)
    elif args.command == "apply":
        inputs = _expand_inputs(args.inputs)
        paths = session.decorate(inputs=inputs, output_dir=args.output_dir,
                                 include_mc=not args.no_mc, overwrite=args.overwrite)
        for path in paths:
            print(f"Saved {path}", flush=True)
    elif args.command == "history":
        _print_json(session.history(), args.output)
    elif args.command == "likelihood":
        _print_json(session.evaluate_likelihood(data_path=args.data_root), args.output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except KeyboardInterrupt:
        print("Interrupted; completed iterations remain available for resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        print(f"omnifold: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
