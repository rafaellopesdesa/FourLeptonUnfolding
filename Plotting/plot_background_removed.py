#!/usr/bin/env python3
"""Compare background-removed data with fiducial ggH at reconstruction level."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402

if __package__:
    from .plot_data_mc import (
        DEFAULT_LUMINOSITY_FB,
        HistogramValues,
        OBSERVABLES,
        TREE_NAME,
        _add_histograms,
        _branch_names,
        _draw_panel,
        _weighted_histogram,
    )
else:  # Support direct execution as ``python Plotting/plot_background_removed.py``.
    from plot_data_mc import (  # type: ignore[no-redef]
        DEFAULT_LUMINOSITY_FB,
        HistogramValues,
        OBSERVABLES,
        TREE_NAME,
        _add_histograms,
        _branch_names,
        _draw_panel,
        _weighted_histogram,
    )


ANALYSIS_MASS_WINDOW_GEV = (115.0, 130.0)
BACKGROUND_REMOVAL_METADATA = "background_removal_metadata"
APPLICATION_MANIFEST = "background_removal_application_manifest.json"
SampleKind = Literal["data", "gg-h"]


@dataclass(frozen=True)
class CampaignSnapshot:
    manifest_path: Path
    manifest_sha256: str
    file_sha256: dict[Path, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _validate_application_metadata(
    root_file: uproot.ReadOnlyDirectory,
    path: Path,
    *,
    expected_sample_kind: SampleKind,
) -> dict[str, object]:
    if BACKGROUND_REMOVAL_METADATA not in root_file:
        raise KeyError(
            f"{path} has no {BACKGROUND_REMOVAL_METADATA} object; "
            "run BackgroundRemoval Application before plotting"
        )
    try:
        metadata = json.loads(str(root_file[BACKGROUND_REMOVAL_METADATA]))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"{path} contains invalid {BACKGROUND_REMOVAL_METADATA} JSON"
        ) from error
    if not isinstance(metadata, dict) or metadata.get("format_version") != 2:
        raise ValueError(f"{path} has unsupported background-removal metadata")
    if metadata.get("sample_kind") != expected_sample_kind:
        raise ValueError(
            f"{path} was decorated as {metadata.get('sample_kind')!r}, "
            f"not {expected_sample_kind!r}"
        )
    recorded_window = metadata.get("analysis_mass_window_GeV")
    try:
        window_matches = bool(
            np.array_equal(
                np.asarray(recorded_window, dtype=np.float64),
                np.asarray(ANALYSIS_MASS_WINDOW_GEV, dtype=np.float64),
            )
        )
    except (TypeError, ValueError):
        window_matches = False
    if (
        not window_matches
        or metadata.get("analysis_mass_window_strict") is not True
    ):
        raise ValueError(
            f"{path} was not decorated with the strict "
            "115 < reco_m_ZZ < 130 GeV analysis region"
        )
    return metadata


def _validate_campaign(
    decorated_directory: Path,
    paths: dict[str, Path],
) -> CampaignSnapshot:
    """Verify that selected files belong to one directory-level application."""

    manifest_path = decorated_directory / APPLICATION_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {APPLICATION_MANIFEST}: run the directory-level "
            "BackgroundRemoval Application before plotting"
        )
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid application manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError(f"unsupported application manifest: {manifest_path}")

    artifact_hashes = {}
    for field in (
        "background_model_manifest_sha256",
        "correction_model_manifest_sha256",
    ):
        value = manifest.get(field)
        if not _valid_sha256(value):
            raise ValueError(f"application manifest has an invalid {field}")
        artifact_hashes[field] = str(value).lower()

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("application manifest has no valid files list")
    entries_by_name: dict[str, dict[str, object]] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("output"), str):
            raise ValueError("application manifest contains an invalid file entry")
        name = Path(str(item["output"])).name
        if name in entries_by_name:
            raise ValueError(f"application manifest repeats output {name!r}")
        entries_by_name[name] = item

    expected_kinds: dict[str, SampleKind] = {
        "data": "data",
        "gg_H_pythia": "gg-h",
    }
    file_hashes: dict[Path, str] = {}
    for role, path in paths.items():
        entry = entries_by_name.get(path.name)
        if entry is None:
            raise ValueError(
                f"{path.name} is not listed in the application manifest"
            )
        expected_kind = expected_kinds[role]
        if entry.get("sample_kind") != expected_kind:
            raise ValueError(
                f"application manifest gives {path.name} sample kind "
                f"{entry.get('sample_kind')!r}, not {expected_kind!r}"
            )
        recorded_hash = entry.get("output_sha256")
        if not _valid_sha256(recorded_hash):
            raise ValueError(
                f"application manifest has no valid checksum for {path.name}"
            )
        observed_hash = _sha256_file(path)
        if observed_hash != str(recorded_hash).lower():
            raise ValueError(
                f"{path} does not match its application-manifest checksum"
            )
        file_hashes[path] = observed_hash

        with uproot.open(path) as root_file:
            metadata = _validate_application_metadata(
                root_file, path, expected_sample_kind=expected_kind
            )
        for field, expected_hash in artifact_hashes.items():
            if metadata.get(field) != expected_hash:
                raise ValueError(
                    f"{path} metadata does not match the campaign {field}"
                )

    return CampaignSnapshot(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        file_sha256=file_hashes,
    )


def _assert_campaign_unchanged(snapshot: CampaignSnapshot) -> None:
    if _sha256_file(snapshot.manifest_path) != snapshot.manifest_sha256:
        raise RuntimeError("application manifest changed while plotting")
    for path, expected_hash in snapshot.file_sha256.items():
        if _sha256_file(path) != expected_hash:
            raise RuntimeError(f"input file changed while plotting: {path}")


def _read_signal_region_histogram(
    path: Path,
    *,
    value_branch: str,
    edges: np.ndarray,
    expected_lumi_pb_inverse: float,
    step_size: str,
    sample_kind: SampleKind,
) -> HistogramValues:
    """Read one reco histogram under the background-removal target definition."""

    pieces: list[HistogramValues] = []
    with uproot.open(path) as root_file:
        _validate_application_metadata(
            root_file, path, expected_sample_kind=sample_kind
        )
        if TREE_NAME not in root_file:
            raise KeyError(f"{path} does not contain the {TREE_NAME} tree")
        tree = root_file[TREE_NAME]
        branches = _branch_names(tree)
        required = {
            value_branch,
            "analysis_region",
            "background_removal_weight",
            "lumi",
            "reco_m_ZZ",
            "reconstructed",
            "weight",
        }
        if sample_kind == "gg-h":
            required.add("fiducial")
        missing = sorted(required.difference(branches))
        if missing:
            raise KeyError(
                f"{path} is missing required decorated branches: {', '.join(missing)}"
            )

        for arrays in tree.iterate(
            expressions=sorted(required),
            step_size=step_size,
            library="np",
            how=dict,
        ):
            luminosities = np.asarray(arrays["lumi"], dtype=np.float64)
            if not np.all(
                np.isfinite(luminosities)
                & np.isclose(
                    luminosities,
                    expected_lumi_pb_inverse,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise ValueError(
                    f"{path} is not normalized to "
                    f"{expected_lumi_pb_inverse:g} pb^-1"
                )

            region_raw = np.asarray(arrays["analysis_region"])
            if not np.all((region_raw == 0) | (region_raw == 1)):
                raise ValueError(f"{path} contains a non-binary analysis_region")
            region = region_raw.astype(np.bool_)
            reconstructed = np.asarray(arrays["reconstructed"], dtype=np.bool_)
            masses = np.asarray(arrays["reco_m_ZZ"], dtype=np.float64)
            lower, upper = ANALYSIS_MASS_WINDOW_GEV
            expected_region = (
                reconstructed
                & np.isfinite(masses)
                & (masses > lower)
                & (masses < upper)
            )
            if not np.array_equal(region, expected_region):
                raise ValueError(
                    f"{path} has analysis_region values inconsistent with "
                    "reconstructed && 115 < reco_m_ZZ < 130"
                )

            base_weights = np.asarray(arrays["weight"], dtype=np.float64)
            removal_weights = np.asarray(
                arrays["background_removal_weight"], dtype=np.float64
            )
            if not np.all(np.isfinite(base_weights)):
                raise ValueError(f"{path} contains non-finite event weights")
            if not np.all(np.isfinite(removal_weights)):
                raise ValueError(
                    f"{path} contains non-finite background-removal weights"
                )
            if sample_kind == "data":
                if not np.all(
                    (removal_weights >= 0.0) & (removal_weights <= 1.0)
                ):
                    raise ValueError(
                        f"{path} contains data background-removal weights outside [0, 1]"
                    )
                if not np.all(removal_weights[~region] == 1.0):
                    raise ValueError(
                        f"{path} contains non-unity data removal weights outside "
                        "the analysis region"
                    )
                selected = region
            else:
                if not np.all(removal_weights == 1.0):
                    raise ValueError(
                        f"{path} contains non-unity ggH background-removal weights"
                    )
                fiducial = np.asarray(arrays["fiducial"], dtype=np.bool_)
                selected = region & fiducial

            values = np.asarray(arrays[value_branch], dtype=np.float64)
            effective_weights = base_weights * removal_weights
            selected &= np.isfinite(values)
            pieces.append(
                _weighted_histogram(
                    values[selected], effective_weights[selected], edges
                )
            )

    if not pieces:
        empty = np.zeros(len(edges) - 1, dtype=np.float64)
        return HistogramValues(empty.copy(), empty.copy())
    return _add_histograms(pieces)


def create_background_removed_comparison_pdf(
    decorated_directory: Path,
    output_path: Path,
    *,
    data_file: str = "data.root",
    luminosity_fb: float = DEFAULT_LUMINOSITY_FB,
    step_size: str = "100 MB",
    overwrite: bool = False,
) -> int:
    """Create the reco-only starting-point report and return its page count."""

    decorated_directory = decorated_directory.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not decorated_directory.is_dir():
        raise NotADirectoryError(decorated_directory)
    if Path(data_file).name != data_file:
        raise ValueError(
            "--data-file must be a file name within the decorated directory"
        )
    if not np.isfinite(luminosity_fb) or luminosity_fb <= 0.0:
        raise ValueError("luminosity_fb must be finite and positive")
    paths = {
        "data": (decorated_directory / data_file).resolve(),
        "gg_H_pythia": (decorated_directory / "gg_H_pythia.root").resolve(),
    }
    if output_path in paths.values():
        raise ValueError(
            f"plot output must not overwrite an input ROOT file: {output_path}"
        )
    if output_path.suffix.lower() != ".pdf":
        raise ValueError(f"plot output must use a .pdf extension: {output_path}")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing plotting inputs: " + ", ".join(missing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    snapshot = _validate_campaign(decorated_directory, paths)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".tmp.pdf", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    expected_lumi_pb_inverse = luminosity_fb * 1000.0
    hep.style.use("ATLAS")
    pages = 0
    try:
        with PdfPages(temporary) as pdf:
            for observable in OBSERVABLES:
                reco_branch = (
                    "reco_type"
                    if observable.field == "type"
                    else f"reco_{observable.field}"
                )
                data_histogram = _read_signal_region_histogram(
                    paths["data"],
                    value_branch=reco_branch,
                    edges=observable.edges,
                    expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                    step_size=step_size,
                    sample_kind="data",
                )
                gg_h_histogram = _read_signal_region_histogram(
                    paths["gg_H_pythia"],
                    value_branch=reco_branch,
                    edges=observable.edges,
                    expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                    step_size=step_size,
                    sample_kind="gg-h",
                )

                figure = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
                grid = figure.add_gridspec(2, 1, height_ratios=(3.2, 1.0))
                axis = figure.add_subplot(grid[0, 0])
                ratio_axis = figure.add_subplot(grid[1, 0], sharex=axis)
                plt.setp(axis.get_xticklabels(), visible=False)
                _draw_panel(
                    axis,
                    ratio_axis,
                    observable,
                    data_histogram,
                    "Purity-weighted Herwig pseudo-data",
                    [
                        (
                            r"$gg\to H\to ZZ$ (Pythia, reco. & fid.)",
                            "#F58518",
                            gg_h_histogram,
                        )
                    ],
                    level_label=(
                        r"Reconstruction level, "
                        r"$115 < m_{4\ell}^{\mathrm{reco}} < 130$ GeV"
                    ),
                    luminosity_fb=luminosity_fb,
                )
                ratio_axis.set_ylabel("Data/ggH")
                figure.suptitle(observable.label, fontsize=15)
                pdf.savefig(figure)
                plt.close(figure)
                pages += 1

        _assert_campaign_unchanged(snapshot)
        if overwrite:
            os.replace(temporary, output_path)
        else:
            try:
                os.link(temporary, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output appeared while plotting: {output_path}; "
                    "pass --overwrite to replace it"
                ) from error
            temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return pages


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "decorated_directory",
        type=Path,
        help="output directory from BackgroundRemoval apply-directory",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--data-file",
        default="data.root",
        help="decorated nominal or numbered pseudo-data file within the directory",
    )
    parser.add_argument(
        "--luminosity-fb", type=_positive_float, default=DEFAULT_LUMINOSITY_FB
    )
    parser.add_argument("--step-size", default="100 MB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pages = create_background_removed_comparison_pdf(
        args.decorated_directory,
        args.output,
        data_file=args.data_file,
        luminosity_fb=args.luminosity_fb,
        step_size=args.step_size,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {pages} background-removed comparison pages to "
        f"{args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
