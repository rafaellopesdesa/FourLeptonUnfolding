"""Notebook diagnostics for signed-weight density ratios and unfolding.

All distributions retain signed bin contents and use sum(w**2) uncertainties.
Absolute expected yields are the default.  Histograms used for visual checks
do not enter ratio training or calibration.  Ratio-model closure uses only
events with split == 2; these plots must not be used to choose a fitted model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np


FEATURES = (
    "Phi", "Phi1", "Psi", "cos_theta1", "cos_theta2", "cos_theta_star",
    "m_Z1", "m_Z2",
)
LABELS = {
    "Phi": r"$\Phi$", "Phi1": r"$\Phi_1$", "Psi": r"$\Psi$",
    "cos_theta1": r"$\cos\theta_1$", "cos_theta2": r"$\cos\theta_2$",
    "cos_theta_star": r"$\cos\theta^*$",
    "m_Z1": r"$m_{Z_1}$ [GeV]", "m_Z2": r"$m_{Z_2}$ [GeV]",
    "m_ZZ": r"$m_{4\ell}$ [GeV]", "y_ZZ": r"$y_{4\ell}$",
    "pT_ZZ": r"$p_{\mathrm{T}}^{4\ell}$ [GeV]",
}


def _base_name(name: str) -> str:
    for prefix in ("reco_", "truth_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _sample(features: Any, weights: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError(f"{name} features must have shape (events, variables)")
    if weights.ndim != 1 or weights.size != features.shape[0]:
        raise ValueError(f"{name} weights must have one entry per event")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} weights must be finite")
    return features, weights


def _names(feature_names: Sequence[str] | None, dimension: int) -> tuple[str, ...]:
    if feature_names is None:
        feature_names = FEATURES if dimension == len(FEATURES) else tuple(
            f"variable_{index}" for index in range(dimension)
        )
    names = tuple(feature_names)
    if len(names) != dimension:
        raise ValueError("feature_names must match the number of feature columns")
    return names


def _edges(name: str, *samples: np.ndarray, bins: int = 24) -> np.ndarray:
    name = _base_name(name)
    if name in {"Phi", "Phi1", "Psi"}:
        return np.linspace(-np.pi, np.pi, bins + 1)
    if name.startswith("cos_theta"):
        return np.linspace(-1.0, 1.0, bins + 1)
    values = np.concatenate([sample[np.isfinite(sample)] for sample in samples])
    if not values.size:
        return np.linspace(0.0, 1.0, bins + 1)
    # Folding the outer 0.5% into the edge bins keeps sparse tails from hiding
    # the populated region; no finite event weight is dropped.
    low, high = np.quantile(values, [0.005, 0.995])
    if high <= low:
        delta = max(1.0, abs(float(low)) * 0.05)
        low, high = low - delta, high + delta
    else:
        margin = 0.02 * (high - low)
        low, high = low - margin, high + margin
    if name.startswith(("m_", "pT_")) and np.min(values) >= 0.0:
        low = max(0.0, low)
    return np.linspace(low, high, bins + 1)


def _histogram(values: np.ndarray, weights: np.ndarray,
               edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed sumw and sumw2, folding flow and excluding nonfinite values."""
    finite = np.isfinite(values)
    indices = np.clip(np.searchsorted(edges, values[finite], side="right") - 1,
                      0, len(edges) - 2)
    contents = np.bincount(indices, weights=weights[finite], minlength=len(edges) - 1)
    variance = np.bincount(indices, weights=weights[finite] ** 2,
                          minlength=len(edges) - 1)
    return contents, variance


def _normalize(contents: np.ndarray, covariance: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate same-sample shape normalization with its full Jacobian."""
    total = float(np.sum(contents))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("shape normalization requires a positive net signed yield")
    normalized = contents / total
    jacobian = (np.eye(contents.size) - normalized[:, None]) / total
    return normalized, jacobian @ covariance @ jacobian.T, jacobian


def _ratio(numerator: np.ndarray, denominator: np.ndarray,
           numerator_covariance: np.ndarray, denominator_covariance: np.ndarray,
           cross_covariance: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Delta-method errors, retaining negative denominators and contents."""
    cross = np.zeros_like(numerator) if cross_covariance is None else np.diag(cross_covariance)
    ratio = np.full_like(numerator, np.nan)
    error = np.full_like(numerator, np.nan)
    valid = denominator != 0.0
    ratio[valid] = numerator[valid] / denominator[valid]
    variance = (
        np.diag(numerator_covariance)[valid]
        + ratio[valid] ** 2 * np.diag(denominator_covariance)[valid]
        - 2.0 * ratio[valid] * cross[valid]
    ) / denominator[valid] ** 2
    # This variance is nonnegative analytically; remove floating-point residue.
    error[valid] = np.sqrt(np.maximum(variance, 0.0))
    return ratio, error


def histogram_discrepancy(target: np.ndarray, estimate: np.ndarray,
                          target_covariance: np.ndarray, estimate_covariance: np.ndarray,
                          cross_covariance: np.ndarray | None = None) -> dict[str, float | int]:
    """A descriptive binned discrepancy, conditional on fitted event weights.

    This does not propagate neural-network/training uncertainty or provide a
    calibrated goodness-of-fit p-value.  The pseudoinverse handles the null
    normalization mode in shape-only comparisons.
    """
    covariance = target_covariance + estimate_covariance
    if cross_covariance is not None:
        covariance = covariance - cross_covariance - cross_covariance.T
    covariance = 0.5 * (covariance + covariance.T)
    delta = estimate - target
    rank = int(np.linalg.matrix_rank(covariance))
    return {
        "target_yield": float(np.sum(target)),
        "estimate_yield": float(np.sum(estimate)),
        "chi2": float(delta @ np.linalg.pinv(covariance, hermitian=True) @ delta),
        "covariance_rank": rank,
        "max_absolute_bin_difference": float(np.max(np.abs(delta), initial=0.0)),
    }


def _distribution_grid(features: np.ndarray, nominal_weights: np.ndarray,
                       updated_weights: np.ndarray,
                       truth_features: np.ndarray | None, truth_weights: np.ndarray | None,
                       *, feature_names: Sequence[str], title: str, shape_only: bool,
                       labels: tuple[str, str, str], target_shares_events: bool = False) -> Figure:
    names = _names(feature_names, features.shape[1])
    rows = (len(names) + 1) // 2
    figure = plt.figure(figsize=(14, 4.25 * rows), layout="constrained")
    outer = figure.add_gridspec(rows, 2)
    metrics: dict[str, Any] = {}
    denominator_label = labels[2] if truth_features is not None else labels[0]
    for index, name in enumerate(names):
        inner = outer[index // 2, index % 2].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.04)
        axis = figure.add_subplot(inner[0])
        ratio_axis = figure.add_subplot(inner[1], sharex=axis)
        values = features[:, index]
        target_values = truth_features[:, index] if truth_features is not None else values
        edges = _edges(name, values, target_values)
        centers = (edges[:-1] + edges[1:]) / 2.0
        histograms, covariances, transforms = [], [], []
        samples = [(values, nominal_weights), (values, updated_weights)]
        if truth_features is not None:
            samples.append((target_values, truth_weights))
        for sample_values, sample_weights in samples:
            contents, variance = _histogram(sample_values, sample_weights, edges)
            covariance = np.diag(variance)
            transform = np.eye(contents.size)
            if shape_only:
                contents, covariance, transform = _normalize(contents, covariance)
            histograms.append(contents)
            covariances.append(covariance)
            transforms.append(transform)
        cross, _ = _histogram(values, nominal_weights * updated_weights, edges)
        nominal_updated_covariance = transforms[0] @ np.diag(cross) @ transforms[1].T
        colors = ("#6a6a6a", "#0072b2", "#b55220")
        for item, (contents, covariance) in enumerate(zip(histograms, covariances)):
            error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            if item == 2:
                axis.errorbar(centers, contents, yerr=error, fmt="o", markersize=3,
                              color=colors[item], label=labels[item], linewidth=0.9)
            else:
                axis.stairs(contents, edges, color=colors[item], label=labels[item], linewidth=1.5)
                axis.fill_between(edges, np.r_[contents - error, (contents - error)[-1]],
                                  np.r_[contents + error, (contents + error)[-1]],
                                  step="post", color=colors[item], alpha=0.13)
        denominator_index = 2 if truth_features is not None else 0
        ratios_to_plot = (0, 1) if truth_features is not None else (1,)
        feature_metrics = {}
        for item in ratios_to_plot:
            cross_cov = nominal_updated_covariance.T if truth_features is None else None
            if truth_features is not None and target_shares_events:
                cross, _ = _histogram(values, samples[item][1] * truth_weights, edges)
                cross_cov = transforms[item] @ np.diag(cross) @ transforms[2].T
            ratio, error = _ratio(histograms[item], histograms[denominator_index],
                                  covariances[item], covariances[denominator_index], cross_cov)
            ratio_axis.errorbar(centers, ratio, yerr=error, fmt="o", markersize=2.5,
                                color=colors[item], linewidth=0.8)
            feature_metrics[labels[item]] = histogram_discrepancy(
                histograms[denominator_index], histograms[item],
                covariances[denominator_index], covariances[item],
                cross_cov.T if cross_cov is not None else None,
            )
        feature_metrics["nonfinite_entries"] = [int(np.count_nonzero(~np.isfinite(s[0]))) for s in samples]
        metrics[name] = feature_metrics
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Fraction / bin" if shape_only else "Expected events / bin")
        ratio_axis.set_ylabel(f"/ {denominator_label}", fontsize=8)
        ratio_axis.set_xlabel(LABELS.get(_base_name(name), name))
        axis.tick_params(labelbottom=False)
        axis.set_yscale("linear")
        ratio_axis.set_yscale("linear")
        axis.legend(fontsize=8, loc="best")
        missing = feature_metrics["nonfinite_entries"]
        if any(missing):
            axis.text(0.02, 0.97, f"Nonfinite omitted: {missing}", transform=axis.transAxes,
                      va="top", fontsize=8)
    scope = "Shape normalized" if shape_only else "Absolute yields"
    figure.suptitle(f"{title}\n{scope}; flow folded; bands/error bars: MC sumw² (fixed weights)", fontsize=13)
    # Notebook users can inspect these numbers without adding another figure.
    figure.diagnostic_metrics = metrics
    return figure


def plot_distributions(features: Any, nominal_weights: Any, updated_weights: Any,
                       truth_features: Any = None, truth_weights: Any = None, *,
                       feature_names: Sequence[str], title: str = "",
                       shape_only: bool = False) -> Figure:
    """One grid comparing nominal, updated, and optional independent target.

    Pass reco and truth coordinates in separate calls; ``truth_features`` is
    the known target in the *same coordinate space* as ``features``.  Without
    a target, the lower panels show updated/nominal with their shared-event
    covariance.  With a target they show both simulation/target ratios and
    assume the target is statistically independent.  Ratio uncertainties are
    first-order approximations and become unreliable near a zero net bin.
    Nonfinite observables are counted on each panel; finite flow is folded.
    """
    features, nominal_weights = _sample(features, nominal_weights, "nominal")
    _, updated_weights = _sample(features, updated_weights, "updated")
    if (truth_features is None) != (truth_weights is None):
        raise ValueError("truth_features and truth_weights must be supplied together")
    if truth_features is not None:
        truth_features, truth_weights = _sample(truth_features, truth_weights, "target")
        if truth_features.shape[1] != features.shape[1]:
            raise ValueError("target and simulation must have the same feature columns")
    return _distribution_grid(features, nominal_weights, updated_weights, truth_features,
                              truth_weights, feature_names=feature_names, title=title,
                              shape_only=shape_only,
                              labels=("Nominal", "Updated", "Known target"))


def _reliability(target_scores: np.ndarray, reference_scores: np.ndarray,
                 target_weights: np.ndarray, reference_weights: np.ndarray,
                 *, bins: int = 15, shared_events: bool = False) -> dict[str, np.ndarray]:
    """Balanced-prior signed reliability; retain values outside [0, 1].

    Bin errors include each class's same-sample normalization variance.  A
    negative local signed mass is retained; only zero/cancelled denominators
    are undefined.  Such finite-sample ratios are diagnostics, not probabilities.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    sums, variances, score_sums, transforms = [], [], [], []
    for scores, weights in ((target_scores, target_weights), (reference_scores, reference_weights)):
        if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("classifier scores must be finite and within [0, 1]")
        contents, variance = _histogram(scores, weights, edges)
        normalized, covariance, transform = _normalize(contents, np.diag(variance))
        weighted_scores, _ = _histogram(scores, weights * scores, edges)
        sums.append(0.5 * normalized)
        variances.append(0.25 * np.diag(covariance))
        score_sums.append(0.5 * weighted_scores / np.sum(weights))
        transforms.append(transform)
    target, reference = sums
    total = target + reference
    threshold = np.finfo(np.float64).eps * (np.abs(target) + np.abs(reference)) * 8.0
    valid = np.abs(total) > threshold
    fractions = np.full(bins, np.nan)
    centers = np.full(bins, np.nan)
    errors = np.full(bins, np.nan)
    fractions[valid] = target[valid] / total[valid]
    centers[valid] = (score_sums[0][valid] + score_sums[1][valid]) / total[valid]
    cross_variance = np.zeros(bins)
    if shared_events:
        if not np.array_equal(target_scores, reference_scores):
            raise ValueError("shared-event reliability requires matching event scores")
        cross, _ = _histogram(target_scores, target_weights * reference_weights, edges)
        cross_variance = 0.25 * np.diag(transforms[0] @ np.diag(cross) @ transforms[1].T)
    errors[valid] = np.sqrt(np.maximum(
        ((1.0 - fractions[valid]) ** 2 * variances[0][valid]
         + fractions[valid] ** 2 * variances[1][valid]
         - 2.0 * fractions[valid] * (1.0 - fractions[valid]) * cross_variance[valid])
        / total[valid] ** 2,
        0.0,
    ))
    return {"centers": centers, "fractions": fractions, "errors": errors,
            "net_bin_mass": total, "valid": valid}


def _predictions(model: Any, features: np.ndarray) -> dict[str, np.ndarray]:
    predictions = model.predict(features)
    output = {}
    for key in ("shape_ratio", "full_ratio", "score_raw", "score_calibrated", "ensemble_std"):
        if key not in predictions:
            raise ValueError(f"model.predict is missing {key!r}")
        value = np.asarray(predictions[key], dtype=np.float64)
        if value.shape != (features.shape[0],) or np.any(~np.isfinite(value)):
            raise ValueError(f"model prediction {key!r} must be a finite vector")
        output[key] = value
    if np.any(output["shape_ratio"] <= 0.0) or np.any(output["full_ratio"] <= 0.0):
        raise ValueError("predicted density ratios must be positive")
    if np.any(output["ensemble_std"] < 0.0):
        raise ValueError("ensemble standard deviations must be nonnegative")
    return output


def _heldout(features: np.ndarray, weights: np.ndarray, splits: Any,
             name: str) -> tuple[np.ndarray, np.ndarray]:
    splits = np.asarray(splits)
    if splits.shape != weights.shape or np.any(~np.isin(splits, [0, 1, 2])):
        raise ValueError(f"{name} splits must contain one 0/1/2 entry per event")
    selected = splits == 2
    if not np.any(selected):
        raise ValueError(f"{name} has no held-out (split == 2) events")
    features, weights = features[selected], weights[selected]
    if np.any(~np.isfinite(features)):
        raise ValueError(f"{name} held-out features must be finite for model prediction")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError(f"{name} held-out sample needs a positive net signed yield")
    return features, weights


def _draw_weighted_histogram(axis: Any, values: np.ndarray, weights: np.ndarray,
                             edges: np.ndarray, label: str, color: str) -> None:
    contents, variance = _histogram(values, weights, edges)
    axis.stairs(contents, edges, label=label, color=color)
    axis.errorbar((edges[:-1] + edges[1:]) / 2.0, contents,
                  yerr=np.sqrt(variance), fmt="none", color=color, alpha=0.5, linewidth=0.8)
    axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    axis.set_ylabel("Signed fraction / bin")
    axis.legend(fontsize=9)


def plot_ratio_diagnostics(model: Any, target_features: Any, target_weights: Any,
                           reference_features: Any, reference_weights: Any, *,
                           target_splits: Any, reference_splits: Any, title: str = "",
                           feature_names: Sequence[str] | None = None,
                           shared_events: bool = False) -> dict[str, Figure]:
    """Return closure, signed reliability, score, ensemble and learning figures.

    ``model`` supplies ``predict(features)`` and optionally ``history`` (member
    records with training_loss/validation_loss/learning_rate histories).  The
    fitted full ratio, including its learned yield convention, is used as-is.
    Held-out target and reference yields are never forced to agree here.
    Use ``shared_events=True`` when both classes reweight the same ordered
    events (truth push and reco likelihood); this includes their covariance.
    """
    target_features, target_weights = _sample(target_features, target_weights, "target")
    reference_features, reference_weights = _sample(reference_features, reference_weights, "reference")
    if target_features.shape[1] != reference_features.shape[1]:
        raise ValueError("target and reference must have the same feature columns")
    names = _names(feature_names, target_features.shape[1])
    target_features, target_weights = _heldout(target_features, target_weights, target_splits, "target")
    reference_features, reference_weights = _heldout(reference_features, reference_weights, reference_splits, "reference")
    if shared_events and not np.array_equal(target_features, reference_features):
        raise ValueError("shared-event diagnostics require matching ordered held-out features")
    predictions = [_predictions(model, target_features), _predictions(model, reference_features)]
    reference_prediction = predictions[1]
    figures = {"closure": _distribution_grid(
        reference_features, reference_weights, reference_weights * reference_prediction["full_ratio"],
        target_features, target_weights, feature_names=names,
        title=f"{title} — held-out reweighting closure", shape_only=False,
        labels=("Reference", "Reweighted reference", "Target"),
        target_shares_events=shared_events,
    )}
    figures["closure"].ratio_normalization = {
        "reference_shape_mean": float(np.sum(reference_weights * reference_prediction["shape_ratio"]) / np.sum(reference_weights)),
        "target_yield": float(np.sum(target_weights)),
        "reference_yield": float(np.sum(reference_weights)),
        "reweighted_reference_yield": float(np.sum(reference_weights * reference_prediction["full_ratio"])),
    }
    reliability_figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, key, label in zip(axes, ("score_raw", "score_calibrated"), ("Raw", "Calibrated")):
        result = _reliability(predictions[0][key], predictions[1][key], target_weights,
                              reference_weights, shared_events=shared_events)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.errorbar(result["centers"], result["fractions"], yerr=result["errors"], fmt="o", markersize=4)
        axis.set_xlabel("Signed mean classifier score")
        axis.set_ylabel("Signed target fraction (balanced classes)")
        axis.set_title(f"{label}; undefined bins: {np.count_nonzero(~result['valid'])}")
        axis.grid(alpha=0.15)
    reliability_figure.suptitle(f"{title} — held-out reliability; signed ratios may leave [0, 1]")
    figures["reliability"] = reliability_figure
    normalized_weights = [target_weights / np.sum(target_weights), reference_weights / np.sum(reference_weights)]
    score_figure, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for axis, key, label in zip(axes, ("score_raw", "score_calibrated", "shape_ratio"),
                                ("Raw score", "Calibrated score", "log(shape ratio)")):
        values = [np.log(p[key]) if key == "shape_ratio" else p[key] for p in predictions]
        edges = _edges("log_ratio", *values, bins=30) if key == "shape_ratio" else np.linspace(0, 1, 31)
        for value, weight, sample_label, color in zip(values, normalized_weights,
                                                     ("Target", "Reference"), ("#b55220", "#0072b2")):
            _draw_weighted_histogram(axis, value, weight, edges, sample_label, color)
        axis.set_xlabel(label)
    score_figure.suptitle(f"{title} — held-out classifier and ratio distributions; flow folded")
    figures["scores"] = score_figure
    ensemble_figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    values = [p["ensemble_std"] for p in predictions]
    edges = _edges("ensemble_std", *values, bins=30)
    for value, weight, sample_label, color in zip(values, normalized_weights,
                                                 ("Target", "Reference"), ("#b55220", "#0072b2")):
        _draw_weighted_histogram(axis, value, weight, edges, sample_label, color)
    axis.set_xlabel("Standard deviation of member scores")
    axis.set_title(f"{title} — held-out ensemble spread (not a calibrated uncertainty)")
    figures["ensemble"] = ensemble_figure
    records = getattr(model, "history", None)
    if records is None:
        records = getattr(model, "manifest", {}).get("members", [])
    if records and any(record.get("history") for record in records):
        learning_figure, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
        for index, record in enumerate(records):
            history = record.get("history", {})
            for axis, key, label in zip(axes, ("training_loss", "validation_loss", "learning_rate"),
                                       ("Training signed BCE", "Validation signed BCE", "Learning rate")):
                values = np.asarray(history.get(key, []), dtype=float)
                if values.size:
                    axis.plot(np.arange(1, values.size + 1), values,
                              label=f"Member {record.get('slot', index)}")
                axis.set_xlabel("Epoch")
                axis.set_ylabel(label)
                if key == "learning_rate" and values.size and np.all(values > 0):
                    axis.set_yscale("log")
        if axes[0].lines:
            axes[0].legend(fontsize=8)
        learning_figure.suptitle(f"{title} — optimization histories")
        figures["learning"] = learning_figure
    return figures


def save_figures(figures: Mapping[str, Figure] | Figure, path: str | Path) -> Path:
    """Save notebook figures to one PDF without closing their live displays."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("save_figures expects a .pdf output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = [figures] if isinstance(figures, Figure) else list(figures.values())
    if not pages:
        raise ValueError("there are no figures to save")
    with PdfPages(path) as output:
        for figure in pages:
            output.savefig(figure, bbox_inches="tight")
    return path
