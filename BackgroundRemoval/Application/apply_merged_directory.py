#!/usr/bin/env python3
"""Apply one frozen model pair consistently to a merged-output directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import uproot

from BackgroundRemoval.Application.apply_background_ratio import (
    ApplicationModels,
    decorate,
    load_application_models,
)
from BackgroundRemoval.common import (
    assert_file_unchanged,
    capture_file_provenance,
    sha256_file,
    TREE_NAME,
)


SUMMARY_NAME = "background_removal_application_manifest.json"
UPSTREAM_PSEUDO_DATA_MANIFEST = "pseudo_data_manifest.json"
PSEUDO_DATA_PATTERN = re.compile(
    r"data_((?:[0-9]{4}|[1-9][0-9]{4,}))\.root\Z"
)


def _is_managed_output(name: str) -> bool:
    return name in {
        "data.root",
        "ZZ_pythia.root",
        "gg_H_pythia.root",
        SUMMARY_NAME,
        UPSTREAM_PSEUDO_DATA_MANIFEST,
    } or PSEUDO_DATA_PATTERN.fullmatch(name) is not None


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _validate_pseudo_data_file(
    path: Path,
    entry: dict[str, Any],
    campaign: dict[str, Any],
    *,
    ensemble_index: int,
    ensemble_count: int,
) -> None:
    """Cross-check one pseudo-data ROOT file against the campaign manifest."""

    required_entry_fields = (
        "total_entries",
        "total_observed_positive",
        "total_observed_negative",
        "total_observed",
        "seed_spawn_key",
    )
    if any(field not in entry for field in required_entry_fields):
        raise ValueError(f"pseudo-data manifest entry is incomplete for {path.name}")
    try:
        with uproot.open(path) as root_file:
            if TREE_NAME not in root_file or "merge_metadata" not in root_file:
                raise ValueError(
                    f"pseudo-data file lacks Analysis or merge_metadata: {path}"
                )
            entries = int(root_file[TREE_NAME].num_entries)
            metadata = json.loads(str(root_file["merge_metadata"]))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid pseudo-data ROOT metadata: {path}") from error
    expected_pairs = {
        "format_version": campaign["format_version"],
        "ensemble_index": ensemble_index,
        "ensemble_count": ensemble_count,
        "seed": campaign.get("seed"),
        "luminosity_fb": campaign.get("luminosity_fb"),
        "seed_spawn_key": entry["seed_spawn_key"],
        "total_entries": entry["total_entries"],
        "total_observed_positive": entry["total_observed_positive"],
        "total_observed_negative": entry["total_observed_negative"],
        "total_observed": entry["total_observed"],
    }
    if not isinstance(metadata, dict) or any(
        metadata.get(field) != expected
        for field, expected in expected_pairs.items()
    ):
        raise ValueError(
            f"pseudo-data ROOT metadata does not match its manifest entry: {path}"
        )
    if entries != entry["total_entries"]:
        raise ValueError(
            f"pseudo-data ROOT entry count does not match its manifest: {path}"
        )


def discover_inputs(merged_directory: Path) -> list[tuple[Path, str]]:
    """Return only the files covered by the application recipe."""

    merged_directory = merged_directory.expanduser().resolve()
    if not merged_directory.is_dir():
        raise NotADirectoryError(
            f"merged input directory does not exist: {merged_directory}"
        )
    required = [
        (merged_directory / "data.root", "data"),
        (merged_directory / "ZZ_pythia.root", "zz"),
        (merged_directory / "gg_H_pythia.root", "gg-h"),
    ]
    missing = [str(path) for path, _ in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "merged directory is missing required inputs: " + ", ".join(missing)
        )
    pseudo_manifest_path = merged_directory / UPSTREAM_PSEUDO_DATA_MANIFEST
    if pseudo_manifest_path.is_file():
        try:
            manifest = json.loads(pseudo_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid pseudo-data manifest: {pseudo_manifest_path}"
            ) from error
        files = manifest.get("files") if isinstance(manifest, dict) else None
        generated_count = (
            manifest.get("generated_ensemble_count")
            if isinstance(manifest, dict)
            else None
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("format_version") != 2
            or not isinstance(files, list)
            or isinstance(generated_count, bool)
            or not isinstance(generated_count, int)
            or generated_count < 1
            or len(files) != generated_count
            or isinstance(manifest.get("seed"), bool)
            or not isinstance(manifest.get("seed"), int)
            or isinstance(manifest.get("luminosity_fb"), bool)
            or not isinstance(manifest.get("luminosity_fb"), (int, float))
            or not (0.0 < manifest.get("luminosity_fb") < float("inf"))
        ):
            raise ValueError("pseudo-data manifest has an invalid campaign contract")
        pseudo_entries: list[tuple[int, Path]] = []
        indices: set[int] = set()
        seen: set[str] = set()
        for entry in files:
            raw_name = entry.get("path") if isinstance(entry, dict) else None
            ensemble_index = (
                entry.get("ensemble_index") if isinstance(entry, dict) else None
            )
            if (
                not isinstance(raw_name, str)
                or Path(raw_name).name != raw_name
                or isinstance(ensemble_index, bool)
                or not isinstance(ensemble_index, int)
                or ensemble_index < 0
            ):
                raise ValueError("pseudo-data manifest contains an unsafe file path")
            expected_name = (
                "data.root"
                if ensemble_index == 0
                else f"data_{ensemble_index:04d}.root"
            )
            if raw_name != expected_name:
                raise ValueError(
                    "pseudo-data manifest path does not match its ensemble index: "
                    f"{raw_name!r} != {expected_name!r}"
                )
            if raw_name in seen:
                raise ValueError(
                    f"pseudo-data manifest contains duplicate path {raw_name!r}"
                )
            if ensemble_index in indices:
                raise ValueError(
                    f"pseudo-data manifest contains duplicate index {ensemble_index}"
                )
            seen.add(raw_name)
            indices.add(ensemble_index)
            path = merged_directory / raw_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"pseudo-data manifest entry does not exist: {path}"
                )
            _validate_pseudo_data_file(
                path,
                entry,
                manifest,
                ensemble_index=ensemble_index,
                ensemble_count=generated_count,
            )
            if raw_name == "data.root":
                continue
            pseudo_entries.append((ensemble_index, path))
        if indices != set(range(generated_count)):
            raise ValueError(
                "pseudo-data manifest ensemble indices are not unique and contiguous"
            )
        pseudo_paths = [path for _, path in sorted(pseudo_entries)]
    else:
        pseudo_paths = [
            path
            for path in merged_directory.iterdir()
            if path.is_file()
            and (match := PSEUDO_DATA_PATTERN.fullmatch(path.name)) is not None
            and int(match.group(1)) > 0
        ]
    pseudo_paths.sort(
        key=lambda path: int(PSEUDO_DATA_PATTERN.fullmatch(path.name).group(1))
    )
    pseudoexperiments = [(path, "data") for path in pseudo_paths]
    return [required[0], *pseudoexperiments, *required[1:]]


def _expected_input_checksum(
    models: ApplicationModels, artifact: str, input_name: str
) -> str | None:
    bundle = models.background if artifact == "background" else models.correction
    inputs = bundle.manifest.get("inputs")
    entry = inputs.get(input_name) if isinstance(inputs, dict) else None
    checksum = entry.get("sha256") if isinstance(entry, dict) else None
    if checksum is not None and (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in checksum)
    ):
        raise ValueError(
            f"{artifact} artifact has an invalid {input_name} input checksum"
        )
    return checksum.lower() if isinstance(checksum, str) else None


def _validate_campaign_inputs(
    models: ApplicationModels,
    provenance: dict[str, dict[str, Any]],
    *,
    allow_input_mismatch: bool,
) -> list[dict[str, Any]]:
    checks = []
    for artifact, input_name in (
        ("correction", "data"),
        ("correction", "gg_H_pythia"),
        ("correction", "ZZ_pythia"),
        ("background", "gg_H_pythia"),
        ("background", "ZZ_pythia"),
    ):
        expected = _expected_input_checksum(models, artifact, input_name)
        observed = provenance[input_name]["sha256"]
        matched = None if expected is None else expected == observed
        checks.append(
            {
                "artifact": artifact,
                "input": input_name,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": matched,
            }
        )
        if matched is False and not allow_input_mismatch:
            raise ValueError(
                f"merged {input_name} does not match the {artifact} artifact input; "
                "pass --allow-input-mismatch only for an intentional cross-campaign use"
            )
    return checks


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _commit_staged_directory(staging: Path, destination: Path) -> None:
    """Publish a complete campaign, restoring the old directory on failure."""

    if not destination.exists():
        os.replace(staging, destination)
        return
    backup_container = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.background-removal-backup-",
            dir=destination.parent,
        )
    )
    backup = backup_container / "previous"
    moved_old = False
    committed = False
    try:
        os.replace(destination, backup)
        moved_old = True
        try:
            os.replace(staging, destination)
            committed = True
        except Exception:
            os.replace(backup, destination)
            moved_old = False
            raise
    finally:
        if committed and moved_old and backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                # Publication already succeeded. A recoverable backup is
                # preferable to reporting a false campaign failure here.
                pass
        try:
            backup_container.rmdir()
        except OSError:
            # If even rollback failed, preserve the old campaign in this
            # explicitly named backup rather than deleting recoverable data.
            pass


def _commit_new_directory_no_clobber(staging: Path, destination: Path) -> None:
    """Publish to a previously absent path without replacing a late arrival."""

    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"output directory appeared while decorating: {destination}"
        ) from error
    reservation = destination.stat(follow_symlinks=False)
    try:
        # On the supported POSIX platform, replacing our own empty reservation
        # publishes the fully staged directory in one rename. A third party
        # that writes into the reservation makes this fail rather than clobber.
        os.replace(staging, destination)
    except Exception:
        try:
            current = destination.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (
                reservation.st_dev,
                reservation.st_ino,
            ):
                destination.rmdir()
        except OSError:
            pass
        raise


def apply_directory(
    merged_directory: Path,
    output_directory: Path,
    background_model_directory: Path,
    correction_model_directory: Path,
    *,
    step_size: str,
    inference_batch_size: int,
    device: str,
    overwrite: bool,
    replace_existing_branches: bool,
    write_diagnostic_branches: bool,
    allow_input_mismatch: bool = False,
) -> dict[str, Any]:
    """Decorate nominal data, all pseudo-data, and the two Pythia samples."""

    merged_directory = merged_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    background_model_directory = background_model_directory.expanduser().resolve()
    correction_model_directory = correction_model_directory.expanduser().resolve()
    protected_directories = {
        "merged input": merged_directory,
        "background model": background_model_directory,
        "correction model": correction_model_directory,
    }
    overlaps = [
        label
        for label, protected in protected_directories.items()
        if _paths_overlap(output_directory, protected)
    ]
    if overlaps:
        raise ValueError(
            "output directory must not equal, contain, or be contained by the "
            + ", ".join(overlaps)
            + " directory"
        )
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_directory}")
    output_existed_initially = output_directory.exists()
    inputs = discover_inputs(merged_directory)
    upstream_manifest_path = merged_directory / UPSTREAM_PSEUDO_DATA_MANIFEST
    upstream_manifest_present = upstream_manifest_path.is_file()
    provenance_by_path = {
        source: capture_file_provenance(source) for source, _ in inputs
    }
    upstream_manifest_provenance = (
        capture_file_provenance(upstream_manifest_path)
        if upstream_manifest_present
        else None
    )
    destinations = [output_directory / source.name for source, _ in inputs]
    source_targets = {source.resolve() for source, _ in inputs}
    aliased = [
        destination
        for destination in destinations
        if destination.exists() and destination.resolve() in source_targets
    ]
    if aliased:
        raise ValueError(
            "output files must not alias merged inputs: "
            + ", ".join(str(path) for path in aliased)
        )
    existing_managed = (
        sorted(
            path
            for path in output_directory.iterdir()
            if _is_managed_output(path.name)
        )
        if output_directory.is_dir()
        else []
    )
    if existing_managed and not overwrite:
        raise FileExistsError(
            "application outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_managed)
        )

    # Load exactly once: every pseudo-experiment is evaluated with the same
    # in-memory ensemble and the same verified artifact pair.
    models = load_application_models(
        background_model_directory,
        correction_model_directory,
        device=device,
    )
    verified_inputs = discover_inputs(merged_directory)
    if verified_inputs != inputs:
        raise RuntimeError("merged input file set changed while models were loading")
    if upstream_manifest_path.is_file() != upstream_manifest_present:
        raise RuntimeError(
            "pseudo-data manifest presence changed while models were loading"
        )
    for source, _ in inputs:
        assert_file_unchanged(
            source, provenance_by_path[source], verify_sha256=False
        )
    if upstream_manifest_provenance is not None:
        assert_file_unchanged(
            upstream_manifest_path, upstream_manifest_provenance
        )
    sources_by_name = {source.name: source for source, _ in inputs}
    campaign_provenance = {
        "data": provenance_by_path[sources_by_name["data.root"]],
        "gg_H_pythia": provenance_by_path[sources_by_name["gg_H_pythia.root"]],
        "ZZ_pythia": provenance_by_path[sources_by_name["ZZ_pythia.root"]],
    }
    input_provenance_checks = _validate_campaign_inputs(
        models,
        campaign_provenance,
        allow_input_mismatch=allow_input_mismatch,
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.background-removal-staging-",
            dir=output_directory.parent,
        )
    )
    try:
        if output_directory.exists():
            # Hard links make the snapshot cheap. Decorated files and the
            # summary are published with os.replace, so the old files behind
            # these links remain untouched until the campaign commits.
            shutil.copytree(
                output_directory,
                staging,
                dirs_exist_ok=True,
                copy_function=os.link,
                symlinks=True,
            )
            # Managed files are one coherent campaign. Drop every prior
            # managed file in the staging snapshot, including pseudo-data
            # files that are stale because the new input has fewer ensembles.
            for path in staging.iterdir():
                if not _is_managed_output(path.name):
                    continue
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()

        copied_manifest_sha256 = None
        if upstream_manifest_provenance is not None:
            copied_manifest_path = staging / UPSTREAM_PSEUDO_DATA_MANIFEST
            shutil.copy2(
                upstream_manifest_path,
                copied_manifest_path,
            )
            assert_file_unchanged(
                upstream_manifest_path, upstream_manifest_provenance
            )
            copied_manifest_sha256 = sha256_file(copied_manifest_path)
            if copied_manifest_sha256 != upstream_manifest_provenance["sha256"]:
                raise RuntimeError("copied pseudo-data manifest checksum mismatch")

        outputs: list[dict[str, Any]] = []
        for (source, sample_kind), destination in zip(
            inputs, destinations, strict=True
        ):
            staged_destination = staging / destination.name
            source_provenance = provenance_by_path[source]
            metadata = decorate(
                source,
                staged_destination,
                background_model_directory,
                correction_model_directory,
                sample_kind=sample_kind,
                step_size=step_size,
                inference_batch_size=inference_batch_size,
                device=device,
                overwrite=overwrite,
                replace_existing_branches=replace_existing_branches,
                write_diagnostic_branches=write_diagnostic_branches,
                models=models,
                source_provenance=source_provenance,
            )
            outputs.append(
                {
                    "source": str(source),
                    "source_sha256": source_provenance["sha256"],
                    "output": str(destination),
                    "output_sha256": sha256_file(staged_destination),
                    "sample_kind": sample_kind,
                    "entries": metadata["entries"],
                    "analysis_region_entries": metadata[
                        "analysis_region_entries"
                    ],
                    "predicted_entries": metadata["predicted_entries"],
                }
            )

        summary = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "merged_directory": str(merged_directory),
            "output_directory": str(output_directory),
            "background_model_directory": str(models.background.root),
            "background_model_manifest_sha256": models.background_manifest_sha256,
            "correction_model_directory": str(models.correction.root),
            "correction_model_manifest_sha256": models.correction_manifest_sha256,
            "input_provenance_checks": input_provenance_checks,
            "allow_input_mismatch": allow_input_mismatch,
            "upstream_pseudo_data_manifest": (
                {
                    "source": str(upstream_manifest_path),
                    "sha256": upstream_manifest_provenance["sha256"],
                    "copied_sha256": copied_manifest_sha256,
                    "copied_to": str(
                        output_directory / UPSTREAM_PSEUDO_DATA_MANIFEST
                    ),
                }
                if upstream_manifest_provenance is not None
                else None
            ),
            "pseudo_data_discovery": (
                "authoritative_manifest"
                if upstream_manifest_provenance is not None
                else "filename_fallback"
            ),
            "options": {
                "step_size": step_size,
                "inference_batch_size": inference_batch_size,
                "device": device,
                "overwrite": overwrite,
                "allow_input_mismatch": allow_input_mismatch,
                "replace_existing_branches": replace_existing_branches,
                "write_diagnostic_branches": write_diagnostic_branches,
            },
            "pseudoexperiment_policy": (
                "all numbered data_XXXX.root files use the same artifacts as "
                "nominal data.root"
            ),
            "files": outputs,
        }
        _write_summary(staging / SUMMARY_NAME, summary)
        if output_directory.exists():
            shutil.copymode(output_directory, staging, follow_symlinks=False)
        else:
            staging.chmod(0o755)
        if output_existed_initially:
            _commit_staged_directory(staging, output_directory)
        else:
            _commit_new_directory_no_clobber(staging, output_directory)
        return summary
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decorate merged nominal data, pseudo-data, ZZ Pythia, and ggH Pythia "
            "with one frozen background-removal model pair"
        )
    )
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-model-dir", type=Path, required=True)
    parser.add_argument("--correction-model-dir", type=Path, required=True)
    parser.add_argument("--step-size", default="200 MB")
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-existing-branches", action="store_true")
    parser.add_argument("--write-diagnostic-branches", action="store_true")
    parser.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help=(
            "allow ROOT inputs whose checksums differ from the training artifacts; "
            "the mismatch remains recorded in the output manifest"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = apply_directory(
        args.merged_dir,
        args.output_dir,
        args.background_model_dir,
        args.correction_model_dir,
        step_size=args.step_size,
        inference_batch_size=args.inference_batch_size,
        device=args.device,
        overwrite=args.overwrite,
        replace_existing_branches=args.replace_existing_branches,
        write_diagnostic_branches=args.write_diagnostic_branches,
        allow_input_mismatch=args.allow_input_mismatch,
    )
    print(
        f"Decorated {len(summary['files'])} files -> {args.output_dir}; "
        f"summary: {args.output_dir / SUMMARY_NAME}"
    )


if __name__ == "__main__":
    main()
