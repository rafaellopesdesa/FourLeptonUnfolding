"""Independent validation plots for the background-removal model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import hist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

from BackgroundRemoval.common import FEATURES, ratio_outputs


hep.style.use("ATLAS")


FEATURE_LABELS = {
    "reco_Phi": r"Reco $\Phi$",
    "reco_Phi1": r"Reco $\Phi_1$",
    "reco_Psi": r"Reco $\Psi$",
    "reco_cos_theta1": r"Reco $\cos\theta_1$",
    "reco_cos_theta2": r"Reco $\cos\theta_2$",
    "reco_cos_theta_star": r"Reco $\cos\theta^{*}$",
    "reco_m_Z1": r"Reco $m_{Z_1}$ [GeV]",
    "reco_m_Z2": r"Reco $m_{Z_2}$ [GeV]",
    "reco_m_ZZ": r"Reco $m_{4\ell}$ [GeV]",
}


def _edges(name: str) -> np.ndarray:
    if name in {"reco_Phi", "reco_Phi1", "reco_Psi"}:
        return np.linspace(-math.pi, math.pi, 25)
    if name.startswith("reco_cos_theta"):
        return np.linspace(-1.0, 1.0, 25)
    if name == "reco_m_Z1":
        return np.linspace(50.0, 106.0, 25)
    if name == "reco_m_Z2":
        return np.linspace(12.0, 115.0, 25)
    if name == "reco_m_ZZ":
        return np.linspace(105.0, 160.0, 25)
    raise KeyError(name)


def _histogram(
    values: np.ndarray, weights: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    histogram = hist.Hist(
        hist.axis.Variable(edges, underflow=True, overflow=True),
        storage=hist.storage.Weight(),
    )
    histogram.fill(values, weight=weights)
    view = histogram.view(flow=True)
    sums = np.asarray(view.value, dtype=np.float64)
    variances = np.asarray(view.variance, dtype=np.float64)
    regular = sums[1:-1].copy()
    regular_variance = variances[1:-1].copy()
    regular[0] += sums[0]
    regular[-1] += sums[-1]
    regular_variance[0] += variances[0]
    regular_variance[-1] += variances[-1]
    return regular, regular_variance


def _model_outputs(predictions: Any, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    calibration = manifest["calibration"]
    return ratio_outputs(
        predictions.member_scores,
        calibration_scale=float(calibration["scale"]),
        calibration_bias=float(calibration["bias_after_normalization"]),
        yield_ratio=float(manifest["yields"]["signal_to_background"]),
        logit_clip=float(calibration["logit_clip"]),
    )


def _reliability(
    signal_scores: np.ndarray,
    background_scores: np.ndarray,
    signal_weights: np.ndarray,
    background_weights: np.ndarray,
    *,
    bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    scores = np.concatenate([signal_scores, background_scores])
    labels = np.concatenate(
        [np.ones(signal_scores.size), np.zeros(background_scores.size)]
    )
    weights = np.concatenate(
        [
            0.5 * signal_weights / np.sum(signal_weights, dtype=np.float64),
            0.5 * background_weights / np.sum(background_weights, dtype=np.float64),
        ]
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = np.full(bins, np.nan)
    fractions = np.full(bins, np.nan)
    errors = np.full(bins, np.nan)
    ece = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            (scores < edges[index + 1]) if index + 1 < bins else (scores <= 1.0)
        )
        if not np.any(selected):
            continue
        selected_weights = weights[selected]
        weight_sum = float(np.sum(selected_weights, dtype=np.float64))
        centers[index] = float(
            np.sum(selected_weights * scores[selected], dtype=np.float64) / weight_sum
        )
        fractions[index] = float(
            np.sum(selected_weights * labels[selected], dtype=np.float64) / weight_sum
        )
        effective = weight_sum * weight_sum / float(
            np.sum(selected_weights * selected_weights, dtype=np.float64)
        )
        errors[index] = math.sqrt(
            max(fractions[index] * (1.0 - fractions[index]) / effective, 0.0)
        )
        ece += weight_sum * abs(fractions[index] - centers[index])
    clipped = np.clip(scores, 1.0e-12, 1.0 - 1.0e-12)
    nll = float(
        np.sum(
            weights * (-labels * np.log(clipped) - (1.0 - labels) * np.log1p(-clipped)),
            dtype=np.float64,
        )
    )
    brier = float(np.sum(weights * (scores - labels) ** 2, dtype=np.float64))
    return centers, fractions, errors, nll, brier, ece


def _draw_comparison(
    pdf: PdfPages,
    *,
    values_target: np.ndarray,
    weights_target: np.ndarray,
    values_estimate: np.ndarray,
    weights_estimate: np.ndarray,
    edges: np.ndarray,
    xlabel: str,
    title: str,
    target_label: str,
    estimate_label: str,
    normalize: bool,
    shared_values: np.ndarray | None = None,
    shared_covariance_weights: np.ndarray | None = None,
) -> dict[str, float]:
    target, target_variance = _histogram(values_target, weights_target, edges)
    estimate, estimate_variance = _histogram(
        values_estimate, weights_estimate, edges
    )
    covariance = np.zeros_like(target)
    if (shared_values is None) != (shared_covariance_weights is None):
        raise ValueError("shared values and covariance weights must be supplied together")
    if shared_values is not None and shared_covariance_weights is not None:
        covariance, _ = _histogram(
            shared_values, shared_covariance_weights, edges
        )
    if normalize:
        target_scale = float(np.sum(target, dtype=np.float64))
        estimate_scale = float(np.sum(estimate, dtype=np.float64))
        if target_scale > 0.0:
            target /= target_scale
            target_variance /= target_scale * target_scale
        if estimate_scale > 0.0:
            estimate /= estimate_scale
            estimate_variance /= estimate_scale * estimate_scale
        if target_scale > 0.0 and estimate_scale > 0.0:
            covariance /= target_scale * estimate_scale

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    figure, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )
    axis.bar(
        edges[:-1],
        estimate,
        width=widths,
        align="edge",
        color="#4C78A8",
        alpha=0.55,
        edgecolor="#2F4B6C",
        linewidth=0.8,
        label=estimate_label,
    )
    axis.errorbar(
        centers,
        target,
        yerr=np.sqrt(np.maximum(target_variance, 0.0)),
        fmt="o",
        color="black",
        markersize=4.0,
        capsize=1.5,
        label=target_label,
    )
    axis.set_ylabel("Normalized entries" if normalize else "Weighted yield")
    axis.set_title(title)
    axis.legend(fontsize=10)
    axis.tick_params(labelbottom=False)

    valid = target != 0.0
    ratio = np.full_like(target, np.nan)
    ratio_error = np.full_like(target, np.nan)
    ratio[valid] = estimate[valid] / target[valid]
    ratio_variance = (
        estimate_variance[valid] / (target[valid] ** 2)
        + estimate[valid] ** 2 * target_variance[valid] / (target[valid] ** 4)
        - 2.0 * estimate[valid] * covariance[valid] / (target[valid] ** 3)
    )
    ratio_error[valid] = np.sqrt(np.maximum(ratio_variance, 0.0))
    ratio_axis.errorbar(
        centers[valid],
        ratio[valid],
        yerr=ratio_error[valid],
        fmt="o",
        color="#2F4B6C",
        markersize=3.5,
    )
    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    finite_ratio = valid & np.isfinite(ratio) & np.isfinite(ratio_error)
    if np.any(finite_ratio):
        lower = min(0.8, float(np.min(ratio[finite_ratio] - ratio_error[finite_ratio])))
        upper = max(1.2, float(np.max(ratio[finite_ratio] + ratio_error[finite_ratio])))
        padding = 0.08 * max(upper - lower, 0.1)
        ratio_axis.set_ylim(max(0.0, lower - padding), upper + padding)
    ratio_axis.set_ylabel("Estimate / target")
    ratio_axis.set_xlabel(xlabel)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.11, top=0.92)
    pdf.savefig(figure)
    plt.close(figure)

    combined_variance = np.maximum(
        target_variance + estimate_variance - 2.0 * covariance, 0.0
    )
    # Estimate-only support bins must contribute to the discrepancy even
    # though a target ratio cannot be drawn there.
    chi2_mask = combined_variance > 0.0
    chi2 = float(
        np.sum(
            (estimate[chi2_mask] - target[chi2_mask]) ** 2
            / combined_variance[chi2_mask],
            dtype=np.float64,
        )
    )
    return {
        "chi2": chi2,
        "bins": int(np.count_nonzero(chi2_mask)),
        "target_integral": float(np.sum(target, dtype=np.float64)),
        "estimate_integral": float(np.sum(estimate, dtype=np.float64)),
    }


def make_diagnostics_pdf(
    output_path: Path,
    *,
    manifest: dict[str, Any],
    cache: dict[str, Any],
    validation: dict[str, Any],
    closure: dict[str, Any],
) -> dict[str, Any]:
    """Create the calibration and all-variable closure evidence report."""

    validation_outputs = {
        name: _model_outputs(item, manifest) for name, item in validation.items()
    }
    closure_outputs = {
        name: _model_outputs(item, manifest) for name, item in closure.items()
    }
    validation_weights = {
        name: np.asarray(cache[name].weights[item.indices], dtype=np.float64)
        for name, item in validation.items()
    }
    closure_weights = {
        name: np.asarray(cache[name].weights[item.indices], dtype=np.float64)
        for name, item in closure.items()
    }
    report_metrics: dict[str, Any] = {"shape_closure": {}, "purity_closure": {}}

    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(8.3, 11.7))
        figure.suptitle(
            "Background-removal density-ratio validation", fontsize=18, y=0.97
        )
        lines = [
            "Model domain: reconstructed decay observables only",
            "Label 1: ggH reconstructed and fiducial",
            "Label 0: reconstructed ZZ + reconstructed, nonfiducial ggH",
            "",
            f"Signal yield: {manifest['yields']['signal']:.8g} {manifest['yields']['units']}",
            (
                f"Background yield: {manifest['yields']['background']:.8g} "
                f"{manifest['yields']['units']}"
            ),
            f"Yield ratio S/B: {manifest['yields']['signal_to_background']:.8g}",
            f"Luminosity check: {manifest['luminosity_fb']:g} fb$^{{-1}}$",
            "",
            "Balanced BCE: score -> r_shape = score/(1-score)",
            "Physical odds: r_phys = (S/B) r_shape",
            "Applied data factor: w_remove = r_phys/(1+r_phys)",
            "",
            "The validation split sets early stopping and calibration.",
            "Every plotted closure below uses the untouched 25% closure split.",
            "Selected negative training weights are rejected, never abs-weighted.",
            "",
            f"Toolkit pin: {manifest['toolkit']['pinned_commit']}",
            f"Ensemble: {manifest['ensemble']['size']} arithmetic-mean members",
            (
                f"Network: {manifest['architecture']['hidden_layers']} x "
                f"{manifest['architecture']['neurons']} SiLU; no dropout/weight decay"
            ),
        ]
        figure.text(
            0.08,
            0.90,
            "\n".join(lines),
            va="top",
            fontfamily="DejaVu Sans Mono",
            fontsize=10.5,
        )
        pdf.savefig(figure)
        plt.close(figure)

        members = manifest["members"]
        columns = 2
        rows = math.ceil(len(members) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(11.0, 4.1 * rows), squeeze=False)
        for axis, member in zip(axes.flat, members, strict=False):
            training_loss = member["history"]["training_loss"]
            validation_loss = member["history"]["validation_loss"]
            axis.plot(training_loss, label="fit batches")
            axis.plot(validation_loss, label="validation")
            axis.set_title(
                f"Member {member['slot']} (attempt {member['attempt']}, seed {member['seed']})"
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Balanced BCE")
            axis.legend(fontsize=9)
        for axis in axes.flat[len(members):]:
            axis.set_visible(False)
        figure.suptitle("Learning curves and member selection", y=1.0)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
        bins = np.linspace(0.0, 1.0, 51)
        for split_name, predictions, outputs, linestyle in (
            ("validation", validation, validation_outputs, "--"),
            ("closure", closure, closure_outputs, "-"),
        ):
            for class_name, color in (("signal", "#D62728"), ("background", "#4C78A8")):
                values = outputs[class_name]["signal_score_balanced"]
                weights = (
                    validation_weights[class_name]
                    if split_name == "validation"
                    else closure_weights[class_name]
                )
                axes[0].hist(
                    values,
                    bins=bins,
                    weights=weights / np.sum(weights),
                    histtype="step",
                    linewidth=1.5,
                    linestyle=linestyle,
                    color=color,
                    label=f"{class_name}, {split_name}",
                )
        axes[0].set_xlabel("Calibrated balanced-prior score")
        axes[0].set_ylabel("Normalized entries")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Score stability")

        calibration_summary = {}
        raw_centers, raw_fractions, raw_errors, raw_nll, raw_brier, raw_ece = (
            _reliability(
                closure["signal"].ensemble_score,
                closure["background"].ensemble_score,
                closure_weights["signal"],
                closure_weights["background"],
            )
        )
        raw_valid = np.isfinite(raw_centers) & np.isfinite(raw_fractions)
        axes[1].errorbar(
            raw_centers[raw_valid],
            raw_fractions[raw_valid],
            yerr=raw_errors[raw_valid],
            fmt="^",
            markersize=4,
            capsize=1.5,
            color="#7F7F7F",
            label=f"closure raw: NLL={raw_nll:.3f}, ECE={raw_ece:.3f}",
        )
        calibration_summary["closure_raw"] = {
            "nll": raw_nll,
            "brier": raw_brier,
            "ece": raw_ece,
        }
        for label, predictions, outputs, marker in (
            ("validation", validation, validation_outputs, "s"),
            ("closure", closure, closure_outputs, "o"),
        ):
            weights = validation_weights if label == "validation" else closure_weights
            centers, fractions, errors, nll, brier, ece = _reliability(
                outputs["signal"]["signal_score_balanced"],
                outputs["background"]["signal_score_balanced"],
                weights["signal"],
                weights["background"],
            )
            valid = np.isfinite(centers) & np.isfinite(fractions)
            axes[1].errorbar(
                centers[valid],
                fractions[valid],
                yerr=errors[valid],
                fmt=marker,
                markersize=4,
                capsize=1.5,
                label=f"{label}: NLL={nll:.3f}, ECE={ece:.3f}",
            )
            calibration_summary[label] = {"nll": nll, "brier": brier, "ece": ece}
        axes[1].plot([0.0, 1.0], [0.0, 1.0], "--", color="black", linewidth=1.0)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("Mean predicted score")
        axes[1].set_ylabel("Weighted signal fraction")
        axes[1].set_title("Balanced-prior reliability")
        axes[1].legend(fontsize=9)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
        report_metrics["calibration"] = calibration_summary

        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
        split_labels = []
        forward_normalizations = []
        inverse_normalizations = []
        for label, outputs, weights in (
            ("validation", validation_outputs, validation_weights),
            ("closure", closure_outputs, closure_weights),
        ):
            forward = float(
                np.sum(
                    weights["background"] * outputs["background"]["background_shape_ratio"],
                    dtype=np.float64,
                )
                / np.sum(weights["background"], dtype=np.float64)
            )
            inverse = float(
                np.sum(
                    weights["signal"]
                    / outputs["signal"]["background_shape_ratio"],
                    dtype=np.float64,
                )
                / np.sum(weights["signal"], dtype=np.float64)
            )
            split_labels.append(label)
            forward_normalizations.append(forward)
            inverse_normalizations.append(inverse)
        positions = np.arange(len(split_labels), dtype=np.float64)
        axes[0].bar(
            positions - 0.18,
            forward_normalizations,
            width=0.36,
            color="#4C78A8",
            label=r"$E_B[r_{\mathrm{shape}}]$",
        )
        axes[0].bar(
            positions + 0.18,
            inverse_normalizations,
            width=0.36,
            color="#F58518",
            label=r"$E_S[1/r_{\mathrm{shape}}]$",
        )
        axes[0].set_xticks(positions, split_labels)
        axes[0].axhline(1.0, color="black", linestyle="--")
        axes[0].set_ylabel("Weighted expectation")
        axes[0].set_title("Forward and inverse ratio normalization")
        axes[0].legend(fontsize=9)
        closure_spread = np.concatenate(
            [
                closure_outputs["signal"]["signal_score_ensemble_std"],
                closure_outputs["background"]["signal_score_ensemble_std"],
            ]
        )
        axes[1].hist(closure_spread, bins=50, histtype="stepfilled", alpha=0.6)
        axes[1].set_xlabel("Per-event ensemble score standard deviation")
        axes[1].set_ylabel("Events")
        axes[1].set_yscale("log")
        axes[1].set_title("Untouched closure ensemble spread")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
        report_metrics["ratio_normalization"] = {
            label: {"forward": forward, "inverse": inverse}
            for label, forward, inverse in zip(
                split_labels,
                forward_normalizations,
                inverse_normalizations,
                strict=True,
            )
        }

        signal_indices = closure["signal"].indices
        background_indices = closure["background"].indices
        signal_features = cache["signal"].features
        background_features = cache["background"].features
        for feature_index, feature_name in enumerate(FEATURES):
            edges = _edges(feature_name)
            shape_metrics = _draw_comparison(
                pdf,
                values_target=np.asarray(signal_features[signal_indices, feature_index]),
                weights_target=closure_weights["signal"],
                values_estimate=np.asarray(
                    background_features[background_indices, feature_index]
                ),
                weights_estimate=(
                    closure_weights["background"]
                    * closure_outputs["background"]["background_shape_ratio"]
                ),
                edges=edges,
                xlabel=FEATURE_LABELS[feature_name],
                title=f"Shape closure: background reweighted to signal — {feature_name}",
                target_label="Signal target",
                estimate_label=r"Background $\times\ r_{\mathrm{shape}}$",
                normalize=True,
            )
            report_metrics["shape_closure"][feature_name] = shape_metrics

            mixture_values = np.concatenate(
                [
                    np.asarray(signal_features[signal_indices, feature_index]),
                    np.asarray(background_features[background_indices, feature_index]),
                ]
            )
            mixture_weights = np.concatenate(
                [
                    closure_weights["signal"]
                    * closure_outputs["signal"]["background_removal_weight"],
                    closure_weights["background"]
                    * closure_outputs["background"]["background_removal_weight"],
                ]
            )
            purity_metrics = _draw_comparison(
                pdf,
                values_target=np.asarray(signal_features[signal_indices, feature_index]),
                weights_target=closure_weights["signal"],
                values_estimate=mixture_values,
                weights_estimate=mixture_weights,
                edges=edges,
                xlabel=FEATURE_LABELS[feature_name],
                title=f"Yield closure: (signal + background) weighted to signal — {feature_name}",
                target_label="True signal",
                estimate_label=r"$(S+B)\times w_{\mathrm{remove}}$",
                normalize=False,
                shared_values=np.asarray(
                    signal_features[signal_indices, feature_index]
                ),
                shared_covariance_weights=(
                    closure_weights["signal"] ** 2
                    * closure_outputs["signal"]["background_removal_weight"]
                ),
            )
            report_metrics["purity_closure"][feature_name] = purity_metrics

        figure = plt.figure(figsize=(8.3, 11.7))
        figure.suptitle("Closure summary", fontsize=18, y=0.97)
        summary_lines = [
            "Closure discrepancy metrics (chi2 / populated bins; diagnostic only)",
            "",
            f"{'Variable':32s} {'B*r -> S shape':>18s} {'(S+B)*w -> S yield':>22s}",
        ]
        for name in FEATURES:
            shape = report_metrics["shape_closure"][name]
            purity = report_metrics["purity_closure"][name]
            summary_lines.append(
                f"{name:32s} {shape['chi2']:7.2f}/{shape['bins']:<7d} "
                f"{purity['chi2']:9.2f}/{purity['bins']:<7d}"
            )
        signal_total = float(np.sum(closure_weights["signal"], dtype=np.float64))
        estimate_total = float(
            np.sum(
                closure_weights["signal"]
                * closure_outputs["signal"]["background_removal_weight"],
                dtype=np.float64,
            )
            + np.sum(
                closure_weights["background"]
                * closure_outputs["background"]["background_removal_weight"],
                dtype=np.float64,
            )
        )
        summary_lines.extend(
            [
                "",
                f"Closure signal yield: {signal_total:.9g}",
                f"Purity-weighted mixture yield: {estimate_total:.9g}",
                f"Relative yield difference: {(estimate_total / signal_total - 1.0):+.3%}",
                "",
                "Interpretation: this is reconstructed, fiducial ggH expressed in reco variables.",
                "Detector-resolution/migration unfolding has not yet been applied.",
                "The chi2 values include shared-signal covariance but omit model uncertainty.",
                "They are not p-values.",
            ]
        )
        figure.text(
            0.06,
            0.91,
            "\n".join(summary_lines),
            va="top",
            fontfamily="DejaVu Sans Mono",
            fontsize=9.3,
        )
        pdf.savefig(figure)
        plt.close(figure)
        report_metrics["closure_signal_yield"] = signal_total
        report_metrics["closure_weighted_mixture_yield"] = estimate_total

    return report_metrics
