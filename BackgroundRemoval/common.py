"""Shared model, preprocessing, and ratio utilities.

The neural network is supplied by the pinned IRIS-HEP NSBI toolkit.  This
module owns the analysis-specific feature transform and the distinction
between the balanced-class shape ratio and the physical signal-purity weight.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TREE_NAME = "Analysis"
ARTIFACT_FORMAT_VERSION = 1
TOOLKIT_COMMIT = "fc09848fc6540fd32310faebbe9db6eea7ecd17b"

# Keep this order identical to the order requested for the decay-only model.
FEATURES = (
    "reco_Phi",
    "reco_Phi1",
    "reco_Psi",
    "reco_cos_theta1",
    "reco_cos_theta2",
    "reco_cos_theta_star",
    "reco_m_Z1",
    "reco_m_Z2",
    "reco_m_ZZ",
)
MODEL_FEATURES = (
    "sin(reco_Phi)",
    "cos(reco_Phi)",
    "sin(reco_Phi1)",
    "cos(reco_Phi1)",
    "sin(reco_Psi)",
    "cos(reco_Psi)",
    "reco_cos_theta1",
    "reco_cos_theta2",
    "reco_cos_theta_star",
    "reco_m_Z1",
    "reco_m_Z2",
    "reco_m_ZZ",
)

PRIMARY_OUTPUT_BRANCH = "background_removal_weight"
DIAGNOSTIC_OUTPUT_BRANCHES = (
    "signal_score_balanced",
    "signal_score_ensemble_std",
    "background_shape_ratio",
    "signal_to_background_ratio",
    "weight_background_removed",
)
OUTPUT_BRANCHES = (PRIMARY_OUTPUT_BRANCH, *DIAGNOSTIC_OUTPUT_BRANCHES)

FIT_SPLIT = np.uint8(0)
VALIDATION_SPLIT = np.uint8(1)
CLOSURE_SPLIT = np.uint8(2)
SPLIT_NAMES = {0: "fit", 1: "validation", 2: "closure"}
SOURCE_SALTS = {
    "gg_H": np.uint64(0x243F6A8885A308D3),
    "ZZ": np.uint64(0x13198A2E03707344),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toolkit_runtime_provenance() -> dict[str, Any]:
    """Report and, when possible, verify the installed toolkit revision."""

    result: dict[str, Any] = {
        "pinned_commit": TOOLKIT_COMMIT,
        "runtime_version": None,
        "runtime_commit": None,
        "runtime_commit_verified": False,
    }
    try:
        distribution = importlib.metadata.distribution("nsbi-common-utils")
    except importlib.metadata.PackageNotFoundError:
        result["note"] = "package metadata unavailable; runtime commit not verifiable"
        return result
    result["runtime_version"] = distribution.version
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
            runtime_commit = direct_url.get("vcs_info", {}).get("commit_id")
        except (json.JSONDecodeError, AttributeError):
            runtime_commit = None
        result["runtime_commit"] = runtime_commit
        if runtime_commit:
            if runtime_commit != TOOLKIT_COMMIT:
                raise RuntimeError(
                    "installed nsbi-common-utils revision does not match the model "
                    f"environment pin: {runtime_commit} != {TOOLKIT_COMMIT}"
                )
            result["runtime_commit_verified"] = True
    if not result["runtime_commit_verified"]:
        result["note"] = "runtime package has no verifiable VCS commit metadata"
    return result


def transformed_features(raw: np.ndarray) -> np.ndarray:
    """Apply periodic encoding while using only the nine requested inputs."""

    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURES):
        raise ValueError(
            f"expected a two-dimensional array with {len(FEATURES)} columns; "
            f"received shape {raw.shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("model inputs contain non-finite values")
    output = np.empty((raw.shape[0], len(MODEL_FEATURES)), dtype=np.float32)
    output[:, 0] = np.sin(raw[:, 0])
    output[:, 1] = np.cos(raw[:, 0])
    output[:, 2] = np.sin(raw[:, 1])
    output[:, 3] = np.cos(raw[:, 1])
    output[:, 4] = np.sin(raw[:, 2])
    output[:, 5] = np.cos(raw[:, 2])
    output[:, 6:] = raw[:, 3:]
    return output


def deterministic_split(
    event_ids: np.ndarray,
    *,
    source: str,
    seed: int,
    fit_fraction: float = 0.60,
    validation_fraction: float = 0.15,
) -> np.ndarray:
    """Hash event identities into a reproducible 60/15/25-style split."""

    if source not in SOURCE_SALTS:
        raise ValueError(f"unknown event source {source!r}")
    if seed < 0:
        raise ValueError("split seed must be non-negative")
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError("fit fraction must lie strictly between zero and one")
    if not (0.0 < validation_fraction < 1.0 - fit_fraction):
        raise ValueError("validation fraction leaves no closure sample")

    values = np.asarray(event_ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = values ^ SOURCE_SALTS[source] ^ np.uint64(seed)
        mixed = mixed + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        mixed = mixed ^ (mixed >> np.uint64(31))
    uniform = (mixed >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    boundary = fit_fraction + validation_fraction
    return np.where(
        uniform < fit_fraction,
        FIT_SPLIT,
        np.where(uniform < boundary, VALIDATION_SPLIT, CLOSURE_SPLIT),
    ).astype(np.uint8)


def stable_sigmoid(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def stable_logit(probabilities: np.ndarray, *, logit_clip: float) -> np.ndarray:
    if not np.isfinite(logit_clip) or not (0.0 < logit_clip <= 100.0):
        raise ValueError("logit clip must be finite and lie in (0, 100]")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    epsilon = max(
        float(stable_sigmoid(-abs(logit_clip))), np.finfo(np.float64).eps
    )
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def ratio_outputs(
    member_scores: np.ndarray,
    *,
    calibration_scale: float,
    calibration_bias: float,
    yield_ratio: float,
    logit_clip: float,
) -> dict[str, np.ndarray]:
    """Turn balanced member scores into shape, physical-odds, and purity terms."""

    scores = np.asarray(member_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("member scores must have shape (events, ensemble members)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("ensemble predictions contain non-finite values")
    if not np.isfinite(yield_ratio) or yield_ratio <= 0.0:
        raise ValueError("yield ratio must be finite and positive")
    if not np.isfinite(calibration_scale) or calibration_scale <= 0.0:
        raise ValueError("calibration scale must be finite and positive")
    if not np.isfinite(calibration_bias):
        raise ValueError("calibration bias must be finite")
    if not np.isfinite(logit_clip) or not (0.0 < logit_clip <= 100.0):
        raise ValueError("logit clip must be finite and lie in (0, 100]")

    raw_score = np.mean(scores, axis=1)
    raw_logit = stable_logit(raw_score, logit_clip=logit_clip)
    log_shape_ratio = np.clip(
        calibration_scale * raw_logit + calibration_bias,
        -abs(logit_clip),
        abs(logit_clip),
    )
    shape_ratio = np.exp(log_shape_ratio)
    log_physical_ratio = np.clip(
        log_shape_ratio + math.log(yield_ratio),
        -2.0 * abs(logit_clip),
        2.0 * abs(logit_clip),
    )
    physical_ratio = np.exp(log_physical_ratio)
    return {
        "raw_score": raw_score,
        "signal_score_balanced": stable_sigmoid(log_shape_ratio),
        "signal_score_ensemble_std": np.std(scores, axis=1),
        "background_shape_ratio": shape_ratio,
        "signal_to_background_ratio": physical_ratio,
        "background_removal_weight": stable_sigmoid(log_physical_ratio),
    }


def build_toolkit_model(architecture: dict[str, Any]):
    """Construct the pinned NSBI toolkit density-ratio network."""

    try:
        from nsbi_common_utils.lightning_tools.density_ratio_model import (  # type: ignore
            DensityRatioLightning,
        )
    except ImportError as error:  # pragma: no cover - environment error path
        raise RuntimeError(
            "nsbi-common-utils is required; install the pinned BackgroundRemoval "
            "environment with `pixi install --manifest-path BackgroundRemoval/pixi.toml`"
        ) from error

    return DensityRatioLightning(
        n_hidden=int(architecture["hidden_layers"]),
        n_neurons=int(architecture["neurons"]),
        input_dim=len(MODEL_FEATURES),
        learning_rate=float(architecture["learning_rate"]),
        use_log_loss=True,
        activation="swish",
        callback_factor=float(architecture.get("learning_rate_decay", 1.0)),
        callback_patience=1,
    )


@dataclass
class ModelBundle:
    root: Path
    manifest: dict[str, Any]
    models: list[Any]
    device: Any

    @classmethod
    def load(cls, model_directory: Path, *, device_name: str = "auto") -> "ModelBundle":
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment error path
            raise RuntimeError("PyTorch is required to apply the background model") from error

        root = model_directory.expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"model manifest does not exist: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported model format {manifest.get('format_version')!r}; "
                f"expected {ARTIFACT_FORMAT_VERSION}"
            )
        if tuple(manifest.get("features", ())) != FEATURES:
            raise ValueError("model feature order is incompatible with this application")
        if tuple(manifest.get("model_features", ())) != MODEL_FEATURES:
            raise ValueError("model preprocessing schema is incompatible with this application")
        if manifest.get("toolkit", {}).get("pinned_commit") != TOOLKIT_COMMIT:
            raise ValueError("model toolkit revision is incompatible with this application")
        toolkit_runtime_provenance()

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        models: list[Any] = []
        for member in manifest.get("members", []):
            relative = Path(member["file"])
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"model path escapes artifact directory: {relative}") from error
            if not path.is_file():
                raise FileNotFoundError(f"ensemble member is missing: {path}")
            expected_sha = member.get("sha256")
            if expected_sha and sha256_file(path) != expected_sha:
                raise ValueError(f"checksum mismatch for ensemble member {path.name}")
            model = build_toolkit_model(manifest["architecture"])
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)
        if len(models) < 4:
            raise ValueError("a background-removal artifact must contain at least four members")
        return cls(root=root, manifest=manifest, models=models, device=device)

    def predict_member_scores(
        self, raw_features: np.ndarray, *, batch_size: int = 65536
    ) -> np.ndarray:
        import torch

        transformed = transformed_features(raw_features)
        mean = np.asarray(self.manifest["scaler"]["mean"], dtype=np.float32)
        scale = np.asarray(self.manifest["scaler"]["scale"], dtype=np.float32)
        if mean.shape != (len(MODEL_FEATURES),) or scale.shape != mean.shape:
            raise ValueError("invalid scaler dimensions in model manifest")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale) & (scale > 0.0)):
            raise ValueError("model scaler contains non-finite or non-positive values")
        if batch_size < 1:
            raise ValueError("inference batch size must be positive")
        standardized = (transformed - mean) / scale
        predictions = np.empty(
            (standardized.shape[0], len(self.models)), dtype=np.float64
        )
        with torch.inference_mode():
            for start in range(0, standardized.shape[0], batch_size):
                stop = min(start + batch_size, standardized.shape[0])
                tensor = torch.from_numpy(standardized[start:stop]).to(self.device)
                for member_index, model in enumerate(self.models):
                    logits = model(tensor).reshape(-1)
                    predictions[start:stop, member_index] = (
                        torch.sigmoid(logits).detach().cpu().numpy()
                    )
        return predictions

    def predict(
        self, raw_features: np.ndarray, *, batch_size: int = 65536
    ) -> dict[str, np.ndarray]:
        scores = self.predict_member_scores(raw_features, batch_size=batch_size)
        calibration = self.manifest["calibration"]
        return ratio_outputs(
            scores,
            calibration_scale=float(calibration["scale"]),
            calibration_bias=float(calibration["bias_after_normalization"]),
            yield_ratio=float(self.manifest["yields"]["signal_to_background"]),
            logit_clip=float(calibration["logit_clip"]),
        )
