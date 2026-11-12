"""Compatibility checks for weights supplied to raw-ratio OmniFold campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from BackgroundRemoval.common import FEATURES, MODEL_FEATURES, RAW_RATIO_ESTIMATOR


def validate_background_weight_metadata(
    root_file: Any, *, weight_branch: str, path: Path
) -> None:
    """Reject known legacy BG weights, while permitting generic weighted inputs.

    The standard observation branch also covers its ``weight *
    background_removal_weight`` fallback. Metadata-free files and explicitly
    selected custom branches have no inferred BackgroundRemoval provenance.
    """
    if weight_branch not in {"weight_background_removed", "background_removal_weight"}:
        return
    if "background_removal_metadata" not in root_file:
        return
    message = (
        f"{path}: background_removal_metadata does not identify raw, uncalibrated "
        "background-removal weights using seven inputs with Phi and Psi in radians. "
        "Retrain the BackgroundRemoval models and "
        "regenerate/redecorate this input before using a new OmniFold campaign."
    )
    try:
        metadata = json.loads(str(root_file["background_removal_metadata"]))
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if (not isinstance(metadata, dict)
            or metadata.get("format_version") != 3
            or metadata.get("ratio_estimator") != RAW_RATIO_ESTIMATOR
            or metadata.get("features") != list(FEATURES)
            or metadata.get("model_features") != list(MODEL_FEATURES)):
        raise ValueError(message)
