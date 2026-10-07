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
from typing import Any, Mapping

import numpy as np


TREE_NAME = "Analysis"
ARTIFACT_FORMAT_VERSION = 2
TOOLKIT_COMMIT = "fc09848fc6540fd32310faebbe9db6eea7ecd17b"

MASS_BRANCH = "reco_m_ZZ"
ANALYSIS_MASS_WINDOW = (115.0, 130.0)
CORRECTION_MASS_WINDOW = (130.0, 160.0)
MODEL_KIND_BACKGROUND_REMOVAL = "background_removal"
MODEL_KIND_DATA_MC_CORRECTION = "data_mc_correction"

# Keep this order identical for the correction and background-removal models.
# reco_m_ZZ is deliberately only a region-selection variable and never enters
# either neural network.
FEATURES = (
    "reco_Phi",
    "reco_Phi1",
    "reco_Psi",
    "reco_cos_theta1",
    "reco_cos_theta2",
    "reco_cos_theta_star",
    "reco_m_Z1",
    "reco_m_Z2",
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
)

PRIMARY_OUTPUT_BRANCH = "background_removal_weight"
ANALYSIS_REGION_BRANCH = "analysis_region"
DIAGNOSTIC_OUTPUT_BRANCHES = (
    "signal_score_balanced",
    "signal_score_ensemble_std",
    "background_shape_ratio",
    "signal_to_background_ratio",
    "weight_background_removed",
)
OUTPUT_BRANCHES = (
    PRIMARY_OUTPUT_BRANCH,
    ANALYSIS_REGION_BRANCH,
    *DIAGNOSTIC_OUTPUT_BRANCHES,
)

FIT_SPLIT = np.uint8(0)
VALIDATION_SPLIT = np.uint8(1)
CLOSURE_SPLIT = np.uint8(2)
SPLIT_NAMES = {0: "fit", 1: "validation", 2: "closure"}
SOURCE_SALTS = {
    "data": np.uint64(0xA4093822299F31D0),
    "gg_H": np.uint64(0x243F6A8885A308D3),
    "ZZ": np.uint64(0x13198A2E03707344),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_file_provenance(
    path: Path, *, include_sha256: bool = True
) -> dict[str, Any]:
    """Snapshot an input before it is read by a multi-pass workflow."""

    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path) if include_sha256 else None,
    }


def assert_file_unchanged(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    verify_sha256: bool = True,
) -> None:
    """Fail if an input changed after its provenance snapshot was captured."""

    path = path.expanduser().resolve()
    stat = path.stat()
    if (
        stat.st_size != provenance.get("size_bytes")
        or stat.st_mtime_ns != provenance.get("mtime_ns")
    ):
        raise RuntimeError(f"input changed while it was being processed: {path}")
    expected_sha256 = provenance.get("sha256")
    if (
        verify_sha256
        and expected_sha256 is not None
        and sha256_file(path) != expected_sha256
    ):
        raise RuntimeError(f"input changed while it was being processed: {path}")


def weight_measure_contract(
    branch: str, *, luminosity_fb: float
) -> dict[str, Any]:
    """Describe how a training branch represents the nominal event measure.

    ``weight`` and ``weight_nominal_pb`` differ only by the common integrated
    luminosity factor written by the merge step.  That constant cancels from
    each calibrated density ratio, so those two branches are a known-compatible
    pair.  Any other branch is deliberately classified as custom rather than
    guessing its units or normalization semantics.
    """

    if not isinstance(branch, str) or not branch:
        raise ValueError("weight branch must be a non-empty string")
    if not np.isfinite(luminosity_fb) or luminosity_fb <= 0.0:
        raise ValueError("luminosity must be finite and positive")
    if branch == "weight":
        return {
            "branch": branch,
            "family": "nominal_cross_section",
            "units": "expected_events_at_luminosity",
            "scale_to_pb": 1.0 / (1000.0 * float(luminosity_fb)),
        }
    if branch == "weight_nominal_pb":
        return {
            "branch": branch,
            "family": "nominal_cross_section",
            "units": "pb",
            "scale_to_pb": 1.0,
        }
    return {
        "branch": branch,
        "family": "custom",
        "units": "arbitrary",
        "scale_to_pb": None,
    }


def known_weight_measures_compatible(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    """Return true only for the explicitly understood nominal-weight family."""

    return (
        first.get("family") == "nominal_cross_section"
        and second.get("family") == "nominal_cross_section"
        and first.get("branch") in {"weight", "weight_nominal_pb"}
        and second.get("branch") in {"weight", "weight_nominal_pb"}
    )


def validate_manifest_mass_window(
    manifest: Mapping[str, Any],
    expected_window: tuple[float, float],
    *,
    artifact_label: str,
) -> None:
    """Validate the standard strict-open mass-window artifact contract."""

    selections = manifest.get("selections")
    window = selections.get("mass_window_gev") if isinstance(selections, Mapping) else None
    if not isinstance(window, Mapping) or not (
        window.get("branch") == MASS_BRANCH
        and window.get("low_exclusive") == expected_window[0]
        and window.get("high_exclusive") == expected_window[1]
    ):
        raise ValueError(
            f"{artifact_label} does not use the required strict "
            f"{expected_window[0]:g} < {MASS_BRANCH} < {expected_window[1]:g} GeV window"
        )


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
    """Apply periodic encoding while using only the eight requested inputs."""

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


def row_fingerprint_ids(values: np.ndarray) -> np.ndarray:
    """Create deterministic uint64 identities for exact duplicate rows.

    Pseudo-data are sampled with replacement and receive fresh sequential
    ``event_id`` values.  Hashing the selected reconstructed values keeps all
    copies of one underlying event in the same fit/validation/closure split.
    This is a grouping identity, not a cryptographic content checksum.
    """

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("row fingerprints require a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("row fingerprints cannot contain non-finite values")
    fingerprints = np.full(
        values.shape[0], np.uint64(0x6A09E667F3BCC909), dtype=np.uint64
    )
    with np.errstate(over="ignore"):
        for column in range(values.shape[1]):
            bits = np.ascontiguousarray(values[:, column]).view(np.uint32).astype(
                np.uint64
            )
            mixed = bits ^ np.uint64(
                (0x9E3779B97F4A7C15 * (column + 1)) & ((1 << 64) - 1)
            )
            mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
                0xBF58476D1CE4E5B9
            )
            mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
                0x94D049BB133111EB
            )
            mixed ^= mixed >> np.uint64(31)
            fingerprints ^= mixed
            fingerprints = (fingerprints ^ (fingerprints >> np.uint64(29))) * np.uint64(
                0x319642B2D24D8EC3
            )
    return fingerprints


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
    result = {
        "raw_score": raw_score,
        "signal_score_balanced": stable_sigmoid(log_shape_ratio),
        "signal_score_ensemble_std": np.std(scores, axis=1),
        "background_shape_ratio": shape_ratio,
        "signal_to_background_ratio": physical_ratio,
        "background_removal_weight": stable_sigmoid(log_physical_ratio),
    }
    # Generic names make the same calibrated estimator usable both for
    # target/reference data-MC correction and for signal/background purity.
    result["shape_ratio"] = result["background_shape_ratio"]
    result["physical_ratio"] = result["signal_to_background_ratio"]
    result["target_purity"] = result["background_removal_weight"]
    return result


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
    def load(
        cls,
        model_directory: Path,
        *,
        device_name: str = "auto",
        expected_model_kind: str | None = None,
    ) -> "ModelBundle":
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
        model_kind = manifest.get("model_kind")
        if model_kind not in {
            MODEL_KIND_BACKGROUND_REMOVAL,
            MODEL_KIND_DATA_MC_CORRECTION,
        }:
            raise ValueError(f"unsupported model kind {model_kind!r}")
        if expected_model_kind is not None and model_kind != expected_model_kind:
            raise ValueError(
                f"model kind {model_kind!r} is incompatible with the requested "
                f"{expected_model_kind!r} application"
            )
        if tuple(manifest.get("features", ())) != FEATURES:
            raise ValueError("model feature order is incompatible with this application")
        if tuple(manifest.get("model_features", ())) != MODEL_FEATURES:
            raise ValueError("model preprocessing schema is incompatible with this application")
        if manifest.get("toolkit", {}).get("pinned_commit") != TOOLKIT_COMMIT:
            raise ValueError("model toolkit revision is incompatible with this application")
        ratio_convention = manifest.get("ratio_convention")
        if not isinstance(ratio_convention, Mapping) or (
            ratio_convention.get("orientation") != "target_to_reference"
        ):
            raise ValueError(
                "model ratio orientation is incompatible with target/reference application"
            )
        toolkit_runtime_provenance()

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        members = manifest.get("members")
        if not isinstance(members, list):
            raise ValueError("model manifest has no valid ensemble member list")
        ensemble = manifest.get("ensemble")
        if not isinstance(ensemble, dict):
            raise ValueError("model manifest has no valid ensemble metadata")
        declared_size = ensemble.get("size")
        if declared_size != len(members):
            raise ValueError(
                "model manifest ensemble size does not match its member list"
            )
        slots = [member.get("slot") for member in members if isinstance(member, dict)]
        if len(slots) != len(members) or set(slots) != set(range(len(members))):
            raise ValueError("ensemble member slots must be unique and contiguous")
        seeds = [member.get("seed") for member in members if isinstance(member, dict)]
        if (
            len(seeds) != len(members)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("ensemble member seeds must be present and unique")
        member_paths: list[Path] = []
        for member in members:
            relative = Path(member["file"])
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"model path escapes artifact directory: {relative}"
                ) from error
            member_paths.append(path)
        if len(set(member_paths)) != len(member_paths):
            raise ValueError("duplicate ensemble member path")
        checksums = [
            member.get("sha256") for member in members if isinstance(member, dict)
        ]
        if len(checksums) != len(members) or len(set(checksums)) != len(checksums):
            raise ValueError("ensemble member checksums must be unique")
        models: list[Any] = []
        for member, path in zip(members, member_paths, strict=True):
            if not path.is_file():
                raise FileNotFoundError(f"ensemble member is missing: {path}")
            expected_sha = member.get("sha256")
            if not expected_sha:
                raise ValueError(f"ensemble member checksum is missing for {path.name}")
            if sha256_file(path) != expected_sha:
                raise ValueError(f"checksum mismatch for ensemble member {path.name}")
            model = build_toolkit_model(manifest["architecture"])
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)
        if len(models) < 4:
            raise ValueError("a density-ratio artifact must contain at least four members")
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
        yields = self.manifest["yields"]
        yield_ratio = yields.get(
            "target_to_reference", yields.get("signal_to_background")
        )
        if yield_ratio is None:
            raise ValueError("model manifest does not define a target/reference yield ratio")
        return ratio_outputs(
            scores,
            calibration_scale=float(calibration["scale"]),
            calibration_bias=float(calibration["bias_after_normalization"]),
            yield_ratio=float(yield_ratio),
            logit_clip=float(calibration["logit_clip"]),
        )
