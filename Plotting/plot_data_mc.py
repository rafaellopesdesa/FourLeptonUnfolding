#!/usr/bin/env python3
"""Make a multipage Herwig-data versus stacked-Pythia comparison PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable

import hist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402


TREE_NAME = "Analysis"
DEFAULT_LUMINOSITY_FB = 312.0
KINEMATIC_FIELDS = (
    "cos_theta_star",
    "cos_theta1",
    "cos_theta2",
    "Phi",
    "Phi1",
    "Psi",
    "m_Z1",
    "m_Z2",
    "m_ZZ",
    "y_ZZ",
    "pT_ZZ",
)
CHANNEL_LABELS = (r"$4\mu$", r"$2\mu2e$", r"$2e2\mu$", r"$4e$")


@dataclass(frozen=True)
class Observable:
    field: str
    label: str
    edges: np.ndarray
    categorical: bool = False


@dataclass(frozen=True)
class HistogramValues:
    values: np.ndarray
    variances: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variances, 0.0))


def _regular_edges(start: float, stop: float, bins: int) -> np.ndarray:
    return np.linspace(start, stop, bins + 1, dtype=np.float64)


OBSERVABLES = (
    Observable("cos_theta_star", r"$\cos\theta^{*}$", _regular_edges(-1.0, 1.0, 24)),
    Observable("cos_theta1", r"$\cos\theta_{1}$", _regular_edges(-1.0, 1.0, 24)),
    Observable("cos_theta2", r"$\cos\theta_{2}$", _regular_edges(-1.0, 1.0, 24)),
    Observable("Phi", r"$\Phi$", _regular_edges(-math.pi, math.pi, 24)),
    Observable("Phi1", r"$\Phi_{1}$", _regular_edges(-math.pi, math.pi, 24)),
    Observable("Psi", r"$\Psi$", _regular_edges(-math.pi, math.pi, 24)),
    Observable("m_Z1", r"$m_{Z_1}$ [GeV]", _regular_edges(50.0, 106.0, 28)),
    Observable("m_Z2", r"$m_{Z_2}$ [GeV]", _regular_edges(12.0, 115.0, 26)),
    Observable("m_ZZ", r"$m_{4\ell}$ [GeV]", _regular_edges(105.0, 160.0, 28)),
    Observable("y_ZZ", r"$y_{4\ell}$", _regular_edges(-3.0, 3.0, 24)),
    Observable("pT_ZZ", r"$p_{\mathrm{T}}^{4\ell}$ [GeV]", _regular_edges(0.0, 250.0, 25)),
    Observable("type", "Four-lepton channel", np.arange(-0.5, 4.5, 1.0), True),
)


def _branch_names(tree: object) -> set[str]:
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def _weighted_histogram(
    values: np.ndarray, weights: np.ndarray, edges: np.ndarray
) -> HistogramValues:
    """Fill a weighted histogram and fold under/overflow into edge bins."""

    histogram = hist.Hist(
        hist.axis.Variable(edges, underflow=True, overflow=True),
        storage=hist.storage.Weight(),
    )
    histogram.fill(values, weight=weights)
    view = histogram.view(flow=True)
    sums = np.asarray(view.value, dtype=np.float64)
    sum_squares = np.asarray(view.variance, dtype=np.float64)
    regular = sums[1:-1].copy()
    regular_variance = sum_squares[1:-1].copy()
    regular[0] += sums[0]
    regular[-1] += sums[-1]
    regular_variance[0] += sum_squares[0]
    regular_variance[-1] += sum_squares[-1]
    return HistogramValues(regular, regular_variance)


def _add_histograms(items: Iterable[HistogramValues]) -> HistogramValues:
    items = tuple(items)
    if not items:
        raise ValueError("at least one histogram is required")
    return HistogramValues(
        np.sum([item.values for item in items], axis=0),
        np.sum([item.variances for item in items], axis=0),
    )


def _read_histogram(
    path: Path,
    *,
    value_branch: str,
    selection_branch: str,
    edges: np.ndarray,
    expected_lumi_pb_inverse: float,
    step_size: str,
) -> HistogramValues:
    pieces: list[HistogramValues] = []
    with uproot.open(path) as root_file:
        if TREE_NAME not in root_file:
            raise KeyError(f"{path} does not contain the {TREE_NAME} tree")
        tree = root_file[TREE_NAME]
        branches = _branch_names(tree)
        required = {value_branch, selection_branch, "weight", "lumi"}
        missing = sorted(required.difference(branches))
        if missing:
            raise KeyError(f"{path} is missing required branches: {', '.join(missing)}")
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
            values = np.asarray(arrays[value_branch], dtype=np.float64)
            weights = np.asarray(arrays["weight"], dtype=np.float64)
            selected = np.asarray(arrays[selection_branch], dtype=np.bool_)
            mask = selected & np.isfinite(values) & np.isfinite(weights)
            pieces.append(_weighted_histogram(values[mask], weights[mask], edges))
    if not pieces:
        empty = np.zeros(len(edges) - 1, dtype=np.float64)
        return HistogramValues(empty.copy(), empty.copy())
    return _add_histograms(pieces)


def _step_values(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[-1]]


def _draw_stack(
    axis: plt.Axes,
    edges: np.ndarray,
    components: list[tuple[str, str, HistogramValues]],
) -> HistogramValues:
    positive_baseline = np.zeros(len(edges) - 1, dtype=np.float64)
    negative_baseline = np.zeros_like(positive_baseline)
    total_variance = np.zeros_like(positive_baseline)
    for label, color, component in components:
        baseline = np.where(component.values >= 0.0, positive_baseline, negative_baseline)
        top = baseline + component.values
        axis.fill_between(
            edges,
            _step_values(baseline),
            _step_values(top),
            step="post",
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            label=label,
        )
        positive_baseline = np.where(
            component.values >= 0.0, top, positive_baseline
        )
        negative_baseline = np.where(
            component.values < 0.0, top, negative_baseline
        )
        total_variance += component.variances
    return HistogramValues(positive_baseline + negative_baseline, total_variance)


def _draw_uncertainty_band(
    axis: plt.Axes,
    edges: np.ndarray,
    center: np.ndarray,
    error: np.ndarray,
    *,
    ratio: bool = False,
) -> None:
    lower = center - error
    upper = center + error
    axis.fill_between(
        edges,
        _step_values(lower),
        _step_values(upper),
        step="post",
        facecolor="none",
        edgecolor="0.35",
        hatch="////",
        linewidth=0.0,
        label="Pythia MC stat." if not ratio else None,
        zorder=3,
    )


def _draw_panel(
    axis: plt.Axes,
    ratio_axis: plt.Axes,
    observable: Observable,
    reference: HistogramValues,
    reference_label: str,
    components: list[tuple[str, str, HistogramValues]],
    *,
    level_label: str,
    luminosity_fb: float,
) -> None:
    edges = observable.edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = _draw_stack(axis, edges, components)
    _draw_uncertainty_band(axis, edges, total.values, total.errors)
    axis.errorbar(
        centers,
        reference.values,
        yerr=reference.errors,
        fmt="o",
        color="black",
        markersize=3.5,
        linewidth=1.0,
        capsize=0.0,
        label=reference_label,
        zorder=5,
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_ylabel("Expected events")
    axis.set_title(level_label, loc="left", fontsize=13)
    axis.text(
        0.98,
        0.96,
        f"√s = 13.6 TeV, {luminosity_fb:g} fb$^{{-1}}$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    handles, labels = axis.get_legend_handles_labels()
    order = [len(handles) - 1, *range(len(handles) - 1)]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        fontsize=9,
        frameon=False,
    )

    valid = np.isfinite(total.values) & (total.values > 0.0)
    ratio = np.full_like(reference.values, np.nan)
    ratio_error = np.full_like(reference.values, np.nan)
    ratio[valid] = reference.values[valid] / total.values[valid]
    ratio_error[valid] = reference.errors[valid] / total.values[valid]
    relative_mc_error = np.full_like(total.values, np.nan)
    relative_mc_error[valid] = total.errors[valid] / total.values[valid]
    _draw_uncertainty_band(
        ratio_axis,
        edges,
        np.ones_like(total.values),
        relative_mc_error,
        ratio=True,
    )
    ratio_axis.errorbar(
        centers,
        ratio,
        yerr=ratio_error,
        fmt="o",
        color="black",
        markersize=3.5,
        linewidth=1.0,
    )
    ratio_axis.axhline(1.0, color="black", linewidth=0.8)
    ratio_axis.set_ylabel("Ref./MC")
    ratio_axis.set_xlabel(observable.label)
    finite_ratio = np.isfinite(ratio) & np.isfinite(ratio_error)
    finite_mc_band = np.isfinite(relative_mc_error)
    lower_bounds = []
    upper_bounds = []
    if np.any(finite_ratio):
        lower_bounds.append(ratio[finite_ratio] - ratio_error[finite_ratio])
        upper_bounds.append(ratio[finite_ratio] + ratio_error[finite_ratio])
    if np.any(finite_mc_band):
        lower_bounds.append(1.0 - relative_mc_error[finite_mc_band])
        upper_bounds.append(1.0 + relative_mc_error[finite_mc_band])
    if lower_bounds:
        observed_low = float(np.min(np.concatenate(lower_bounds)))
        observed_high = float(np.max(np.concatenate(upper_bounds)))
        span = max(observed_high - observed_low, 0.2)
        ratio_axis.set_ylim(
            min(0.45, observed_low - 0.05 * span),
            max(1.55, observed_high + 0.05 * span),
        )
    else:
        ratio_axis.set_ylim(0.45, 1.55)
    if observable.categorical:
        ratio_axis.set_xticks(np.arange(4), CHANNEL_LABELS)
        axis.set_xticks(np.arange(4), [])


def create_comparison_pdf(
    merged_directory: Path,
    output_path: Path,
    *,
    data_file: str = "data.root",
    luminosity_fb: float = DEFAULT_LUMINOSITY_FB,
    step_size: str = "100 MB",
    overwrite: bool = False,
) -> int:
    """Create the full reco/fiducial validation report and return its page count."""

    merged_directory = merged_directory.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not merged_directory.is_dir():
        raise NotADirectoryError(merged_directory)
    paths = {
        "data": (merged_directory / data_file).resolve(),
        "ZZ_pythia": (merged_directory / "ZZ_pythia.root").resolve(),
        "gg_H_pythia": (merged_directory / "gg_H_pythia.root").resolve(),
        "ZZ_herwig": (merged_directory / "ZZ_herwig.root").resolve(),
        "gg_H_herwig": (merged_directory / "gg_H_herwig.root").resolve(),
    }
    if output_path in paths.values():
        raise ValueError(f"plot output must not overwrite an input ROOT file: {output_path}")
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
    temporary = output_path.with_name(f".{output_path.name}.tmp.pdf")
    if temporary.exists():
        temporary.unlink()

    expected_lumi_pb_inverse = luminosity_fb * 1000.0
    hep.style.use("ATLAS")
    pages = 0
    try:
        with PdfPages(temporary) as pdf:
            for observable in OBSERVABLES:
                reco_branch = (
                    "reco_type" if observable.field == "type" else f"reco_{observable.field}"
                )
                truth_branch = (
                    "truth_type"
                    if observable.field == "type"
                    else f"truth_{observable.field}"
                )
                reco_data = _read_histogram(
                    paths["data"],
                    value_branch=reco_branch,
                    selection_branch="reconstructed",
                    edges=observable.edges,
                    expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                    step_size=step_size,
                )
                reco_mc = [
                    (
                        r"$q\bar{q}\to ZZ$ (Pythia)",
                        "#4C78A8",
                        _read_histogram(
                            paths["ZZ_pythia"],
                            value_branch=reco_branch,
                            selection_branch="reconstructed",
                            edges=observable.edges,
                            expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                            step_size=step_size,
                        ),
                    ),
                    (
                        r"$gg\to H\to ZZ$ (Pythia)",
                        "#F58518",
                        _read_histogram(
                            paths["gg_H_pythia"],
                            value_branch=reco_branch,
                            selection_branch="reconstructed",
                            edges=observable.edges,
                            expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                            step_size=step_size,
                        ),
                    ),
                ]

                # data.root is deliberately reco-selected and therefore cannot
                # supply an unbiased fiducial distribution. Use the two merged,
                # luminosity-scaled Herwig truth samples as the data-like
                # pseudo-truth reference instead.
                fiducial_reference = _add_histograms(
                    [
                        _read_histogram(
                            paths[name],
                            value_branch=truth_branch,
                            selection_branch="fiducial",
                            edges=observable.edges,
                            expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                            step_size=step_size,
                        )
                        for name in ("ZZ_herwig", "gg_H_herwig")
                    ]
                )
                fiducial_mc = [
                    (
                        r"$q\bar{q}\to ZZ$ (Pythia)",
                        "#4C78A8",
                        _read_histogram(
                            paths["ZZ_pythia"],
                            value_branch=truth_branch,
                            selection_branch="fiducial",
                            edges=observable.edges,
                            expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                            step_size=step_size,
                        ),
                    ),
                    (
                        r"$gg\to H\to ZZ$ (Pythia)",
                        "#F58518",
                        _read_histogram(
                            paths["gg_H_pythia"],
                            value_branch=truth_branch,
                            selection_branch="fiducial",
                            edges=observable.edges,
                            expected_lumi_pb_inverse=expected_lumi_pb_inverse,
                            step_size=step_size,
                        ),
                    ),
                ]

                figure = plt.figure(figsize=(13.0, 7.0), constrained_layout=True)
                grid = figure.add_gridspec(2, 2, height_ratios=(3.2, 1.0))
                reco_axis = figure.add_subplot(grid[0, 0])
                reco_ratio = figure.add_subplot(grid[1, 0], sharex=reco_axis)
                truth_axis = figure.add_subplot(grid[0, 1])
                truth_ratio = figure.add_subplot(grid[1, 1], sharex=truth_axis)
                plt.setp(reco_axis.get_xticklabels(), visible=False)
                plt.setp(truth_axis.get_xticklabels(), visible=False)
                _draw_panel(
                    reco_axis,
                    reco_ratio,
                    observable,
                    reco_data,
                    "Herwig pseudo-data",
                    reco_mc,
                    level_label="Reconstruction level",
                    luminosity_fb=luminosity_fb,
                )
                _draw_panel(
                    truth_axis,
                    truth_ratio,
                    observable,
                    fiducial_reference,
                    "Herwig pseudo-truth",
                    fiducial_mc,
                    level_label="Fiducial level",
                    luminosity_fb=luminosity_fb,
                )
                figure.suptitle(observable.label, fontsize=15)
                pdf.savefig(figure)
                plt.close(figure)
                pages += 1
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return pages


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_directory", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--data-file",
        default="data.root",
        help="reco-level pseudo-data file within the merged directory",
    )
    parser.add_argument(
        "--luminosity-fb", type=_positive_float, default=DEFAULT_LUMINOSITY_FB
    )
    parser.add_argument("--step-size", default="100 MB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pages = create_comparison_pdf(
        args.merged_directory,
        args.output,
        data_file=args.data_file,
        luminosity_fb=args.luminosity_fb,
        step_size=args.step_size,
        overwrite=args.overwrite,
    )
    print(f"Wrote {pages} comparison pages to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
