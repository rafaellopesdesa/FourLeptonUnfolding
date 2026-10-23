"""Three calibrated ratios per iteration, with a fixed original MC reference.

The truth push is fitted on reconstructed fiducial MC, then applied to every
fiducial MC event. Its target/reference selection efficiencies cancel at fixed
truth; regions with zero efficiency cannot be identified by this construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import uproot

from BackgroundRemoval.common import (
    ANALYSIS_MASS_WINDOW, FEATURES, TREE_NAME, capture_file_provenance,
    assert_file_unchanged, deterministic_split, row_fingerprint_ids, sha256_file,
)
from .ratios import RatioConfig, RatioEstimator

TRUTH_FEATURES = tuple(name.replace("reco_", "truth_", 1) for name in FEATURES)
FORMAT_VERSION = 1


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _np_dict(arrays: Any) -> dict[str, np.ndarray]:
    if isinstance(arrays, dict):
        return arrays
    return {name: arrays[name] for name in arrays.dtype.names}


@dataclass
class Sample:
    """Only selected reco data or all fiducial signal MC are held in memory."""

    path: Path
    weights: np.ndarray
    reco: np.ndarray | None
    truth: np.ndarray | None
    matched: np.ndarray
    splits: np.ndarray
    luminosity_fb: float | None
    total_entries: int
    provenance: dict[str, Any]


def _read_sample(path: Path, *, kind: str, weight_branch: str, seed: int,
                 step_size: str = "64 MB", provenance: dict[str, Any] | None = None) -> Sample:
    """Read data without any truth access; truth-only closure never enters fits."""
    path = Path(path).expanduser().resolve()
    provenance = provenance or capture_file_provenance(path)
    assert_file_unchanged(path, provenance)
    weights_parts, reco_parts, truth_parts, matched_parts, split_parts = [], [], [], [], []
    luminosity = None
    entries = 0
    with uproot.open(path) as root_file:
        if TREE_NAME not in root_file:
            raise KeyError(f"{path} has no {TREE_NAME} tree")
        if kind in {"mc", "truth"} and "merge_metadata" in root_file:
            source_metadata = json.loads(str(root_file["merge_metadata"]))
            if "ensemble_index" in source_metadata or "components" in source_metadata:
                raise ValueError(f"{path}: mixed/reco-selected pseudo-data cannot supply full fiducial signal MC or truth closure")
        tree = root_file[TREE_NAME]
        available = set(tree.keys())
        fallback = (kind == "data" and weight_branch == "weight_background_removed"
                    and weight_branch not in available)
        required = {"weight", "background_removal_weight"} if fallback else {weight_branch}
        if kind in {"data", "mc"}:
            required.update((*FEATURES, "reco_m_ZZ", "reconstructed"))
        if kind in {"mc", "truth"}:
            required.update((*TRUTH_FEATURES, "fiducial", "event_id"))
        missing = required - available
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        if "luminosity_fb" in available:
            required.add("luminosity_fb")
        for raw_arrays in tree.iterate(sorted(required), step_size=step_size, library="np", how=dict):
            arrays = _np_dict(raw_arrays)
            size = len(next(iter(arrays.values())))
            entries += size
            if "luminosity_fb" in arrays:
                values = np.asarray(arrays["luminosity_fb"], dtype=float)
                if size:
                    if luminosity is None:
                        luminosity = float(values[0])
                    if (not np.isfinite(luminosity) or luminosity <= 0 or
                            not np.all(np.isclose(values, luminosity, rtol=0, atol=1e-9))):
                        raise ValueError(f"{path}: inconsistent/invalid luminosity_fb")
            if kind in {"data", "mc"}:
                masses = np.asarray(arrays["reco_m_ZZ"], dtype=float)
                reconstructed = np.asarray(arrays["reconstructed"], dtype=bool)
                if np.any(reconstructed & ~np.isfinite(masses)):
                    raise ValueError(f"{path}: reconstructed row has undefined reco mass")
                matched = reconstructed & (masses > ANALYSIS_MASS_WINDOW[0]) & (masses < ANALYSIS_MASS_WINDOW[1])
            else:
                matched = np.zeros(size, dtype=bool)
            keep = matched if kind == "data" else np.asarray(arrays["fiducial"], dtype=bool)
            if not np.any(keep):
                continue
            weights = (np.asarray(arrays["weight"], dtype=float) *
                       np.asarray(arrays["background_removal_weight"], dtype=float)) if fallback else np.asarray(arrays[weight_branch], dtype=float)
            weights = weights[keep]
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"{path}: selected weights are nonfinite")
            weights_parts.append(weights)
            local_matched = matched[keep]
            matched_parts.append(local_matched)
            if kind in {"data", "mc"}:
                reco = np.column_stack([arrays[name][keep] for name in FEATURES]).astype(np.float32)
                if not np.all(np.isfinite(reco[local_matched])):
                    raise ValueError(f"{path}: selected reconstructed inputs are nonfinite")
                reco_parts.append(reco)
            if kind in {"mc", "truth"}:
                truth = np.column_stack([arrays[name][keep] for name in TRUTH_FEATURES]).astype(np.float32)
                if not np.all(np.isfinite(truth)):
                    raise ValueError(f"{path}: fiducial inputs are undefined; rebuild or explicitly resolve these events")
                truth_parts.append(truth)
                split_ids = np.asarray(arrays["event_id"][keep], dtype=np.uint64)
            else:
                # Group bootstrap replicas even though their event_id was reassigned.
                split_ids = row_fingerprint_ids(np.column_stack([reco, masses[keep].astype(np.float32)]))
            split_parts.append(deterministic_split(split_ids, source="data" if kind == "data" else "gg_H", seed=seed))
    assert_file_unchanged(path, provenance)
    if not weights_parts:
        raise ValueError(f"{path}: no selected {kind} events")
    weights = np.concatenate(weights_parts)
    if float(weights.sum(dtype=np.float64)) <= 0:
        raise ValueError(f"{path}: selected net signed yield must be positive")
    return Sample(path, weights, np.concatenate(reco_parts) if reco_parts else None,
                  np.concatenate(truth_parts) if truth_parts else None,
                  np.concatenate(matched_parts), np.concatenate(split_parts), luminosity,
                  entries, provenance)


@dataclass
class IterationResult:
    index: int
    reco_pull: RatioEstimator
    truth_push: RatioEstimator
    reco_likelihood: RatioEstimator
    root: Path
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        self.record_sha256 = sha256_file(self.root / "iteration.json")

    @property
    def truth_weights(self) -> np.ndarray:
        """Cumulative multiplier in the retained fiducial MC event order."""
        return np.load(self.root / "truth_weights.npy", mmap_mode="r", allow_pickle=False)


class OmniFoldSession:
    """One nominal-data unfolding fit, reusable frozen models for every toy.

    Re-execute ``step()`` to advance once. Each iteration is committed as an
    atomic directory; an interrupted training leaves prior iterations usable.
    """

    def __init__(self, run_dir: Path, manifest: dict[str, Any], config: RatioConfig):
        self.run_dir = run_dir
        self.manifest = manifest
        self.config = config
        self.iterations: list[IterationResult] = []
        self._truth: Sample | None = None
        self._manifest_sha = sha256_file(run_dir / "session.json")
        settings = manifest["settings"]
        self._data = _read_sample(Path(manifest["inputs"]["data"]["path"]), kind="data",
                                  weight_branch=settings["data_weight_branch"], seed=settings["seed"],
                                  provenance=manifest["inputs"]["data"])
        self._mc = _read_sample(Path(manifest["inputs"]["mc"]["path"]), kind="mc",
                                weight_branch=settings["mc_weight_branch"], seed=settings["seed"],
                                provenance=manifest["inputs"]["mc"])
        if not np.any(self._mc.matched):
            raise ValueError("No fiducial MC events pass the reconstructed signal region")
        self._check_luminosity(self._data, self._mc)

    @staticmethod
    def _check_luminosity(first: Sample, second: Sample) -> None:
        if first.luminosity_fb is not None and second.luminosity_fb is not None:
            if not np.isclose(first.luminosity_fb, second.luminosity_fb, rtol=0, atol=1e-9):
                raise ValueError("Samples have different luminosities; use consistently normalized inputs")

    @classmethod
    def create(cls, *, data_path: Path | str, mc_path: Path | str, run_dir: Path | str,
               truth_path: Path | str | None = None, config: RatioConfig | None = None,
               data_weight_branch: str = "weight_background_removed", mc_weight_branch: str = "weight",
               truth_weight_branch: str = "weight", seed: int = 12345) -> "OmniFoldSession":
        run_dir = Path(run_dir).expanduser().resolve()
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} exists; use OmniFoldSession.load or a fresh directory")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        config = config or RatioConfig()
        config.validate()
        if mc_weight_branch in {"weight_nominal_pb", "weight_shape"} or truth_weight_branch in {"weight_nominal_pb", "weight_shape"}:
            raise ValueError("MC and truth weights must be expected event yields at the data luminosity; use 'weight'")
        if data_weight_branch == "background_removal_weight":
            raise ValueError("background_removal_weight is a factor; use 'weight_background_removed' to include the signed observation weight")
        inputs = {"data": capture_file_provenance(Path(data_path).expanduser().resolve()),
                  "mc": capture_file_provenance(Path(mc_path).expanduser().resolve())}
        if inputs["data"]["path"] == inputs["mc"]["path"]:
            raise ValueError("Data and signal MC must be different input files")
        if truth_path is not None:
            inputs["truth"] = capture_file_provenance(Path(truth_path).expanduser().resolve())
            if inputs["truth"]["path"] in {inputs["data"]["path"], inputs["mc"]["path"]}:
                raise ValueError("Closure requires a separate full signal-only Herwig sample")
        manifest = {"format_version": FORMAT_VERSION, "kind": "omnifold_session",
                    "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
                    "config": asdict(config), "features_reco": list(FEATURES),
                    "features_truth": list(TRUTH_FEATURES),
                    "settings": {"seed": seed, "data_weight_branch": data_weight_branch,
                                 "mc_weight_branch": mc_weight_branch, "truth_weight_branch": truth_weight_branch},
                    "semantics": {"truth_push": "cumulative relative to original fiducial signal MC",
                                  "reco_pull": "incremental data / previously truth-weighted MC",
                                  "reco_likelihood": "updated folded signal / original folded signal",
                                  "data_truth_used_for_training": False,
                                  "likelihood": "weighted pseudo-log-likelihood after background removal",
                                  "reco_mass_window_gev": list(ANALYSIS_MASS_WINDOW)}}
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        try:
            _json_write(run_dir / "session.json", manifest)
            (run_dir / "iterations").mkdir()
            return cls(run_dir, manifest, config)
        except BaseException:
            shutil.rmtree(run_dir)
            raise

    @classmethod
    def load(cls, run_dir: Path | str, *, device: str | None = None) -> "OmniFoldSession":
        run_dir = Path(run_dir).expanduser().resolve()
        with (run_dir / "session.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("kind") != "omnifold_session" or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported OmniFold session")
        config = RatioConfig(**manifest["config"])
        if device is not None:
            config = replace(config, device=device)
        session = cls(run_dir, manifest, config)
        for expected, path in enumerate(sorted((run_dir / "iterations").glob("iteration_*")), start=1):
            if path.name != f"iteration_{expected:04d}":
                raise ValueError("Iteration sequence is not contiguous")
            with (path / "iteration.json").open(encoding="utf-8") as stream:
                record = json.load(stream)
            if record.get("index") != expected or record.get("session_sha256") != session._manifest_sha:
                raise ValueError("Iteration belongs to another session or sequence")
            previous_sha = (sha256_file(session.iterations[-1].root / "iteration.json")
                            if session.iterations else None)
            if record.get("previous_iteration_sha256") != previous_sha:
                raise ValueError("Broken iteration checkpoint chain")
            if sha256_file(path / "truth_weights.npy") != record["truth_weights_sha256"]:
                raise ValueError("Truth-weight checkpoint checksum mismatch")
            stages = ("reco_pull", "truth_push", "reco_likelihood")
            if set(record.get("model_manifest_sha256", {})) != set(stages):
                raise ValueError("Missing ratio-manifest checksums")
            for stage, digest in record["model_manifest_sha256"].items():
                if sha256_file(path / stage / "manifest.json") != digest:
                    raise ValueError("Ratio-manifest checksum mismatch")
            weights = np.load(path / "truth_weights.npy", mmap_mode="r", allow_pickle=False)
            if weights.shape != session._mc.weights.shape or not np.all(np.isfinite(weights) & (weights > 0)):
                raise ValueError("Invalid cumulative truth weights")
            models = [RatioEstimator.load(path / name, device=config.device)
                      for name in ("reco_pull", "truth_push", "reco_likelihood")]
            for stage, model in zip(stages, models):
                metadata = model.manifest.get("metadata", {})
                if metadata.get("stage") != stage or metadata.get("iteration") != expected:
                    raise ValueError("Ratio model belongs to another stage or iteration")
            session.iterations.append(IterationResult(expected, *models, path, record))
        return session

    def _result(self, iteration: int | None = None) -> IterationResult:
        if not self.iterations:
            raise RuntimeError("Run session.step() first")
        index = len(self.iterations) if iteration is None else int(iteration)
        if index < 1 or index > len(self.iterations):
            raise IndexError("iteration is a 1-based completed iteration number")
        return self.iterations[index - 1]

    def _previous_weights(self, index: int) -> np.ndarray:
        return np.ones_like(self._mc.weights) if index == 1 else self.iterations[index - 2].truth_weights

    def step(self) -> IterationResult:
        """Train and commit exactly one iteration; previous truth weights are replaced."""
        # A kernel restart/concurrent notebook must never overwrite another run.
        index = len(self.iterations) + 1
        destination = self.run_dir / "iterations" / f"iteration_{index:04d}"
        lock = self.run_dir / ".training.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError(f"Training lock exists: {lock}. If a kernel died, confirm no training is running before removing it.") from error
        os.close(descriptor)
        temporary = None
        try:
            if destination.exists():
                raise RuntimeError("Another session has committed this iteration; reload the session")
            if sha256_file(self.run_dir / "session.json") != self._manifest_sha:
                raise ValueError("Session settings changed on disk; start a new session")
            if self.iterations:
                previous_result = self.iterations[-1]
                if (sha256_file(previous_result.root / "iteration.json") != previous_result.record_sha256 or
                        sha256_file(previous_result.root / "truth_weights.npy") != previous_result.manifest["truth_weights_sha256"]):
                    raise ValueError("Previous iteration checkpoint changed on disk")
            for name in ("data", "mc"):
                record = self.manifest["inputs"][name]
                assert_file_unchanged(Path(record["path"]), record)
            previous = self._previous_weights(index)
            mc, data = self._mc, self._data
            matched = mc.matched
            base = mc.weights[matched]
            old = np.asarray(previous[matched])
            xs, zs, splits = mc.reco[matched], mc.truth[matched], mc.splits[matched]
            temporary = Path(tempfile.mkdtemp(prefix=f".iteration_{index:04d}_", dir=destination.parent))
            config = replace(self.config, seed=self.config.seed + 10000 * index, overwrite=False)
            print(f"Iteration {index}: 1/3 reconstructed data / current signal MC", flush=True)
            pull = RatioEstimator.fit(data.reco, data.weights, xs, base * old,
                                      target_splits=data.splits, reference_splits=splits,
                                      output_dir=temporary / "reco_pull", config=config,
                                      metadata={"stage": "reco_pull", "iteration": index, "reference": "previous truth-weighted MC"})
            pulled = base * old * pull.predict(xs)["full_ratio"]
            print(f"Iteration {index}: 2/3 cumulative particle-level ratio", flush=True)
            push = RatioEstimator.fit(zs, pulled, zs, base,
                                      target_splits=splits, reference_splits=splits,
                                      output_dir=temporary / "truth_push", config=replace(config, seed=config.seed + 1),
                                      metadata={"stage": "truth_push", "iteration": index, "reference": "original matched fiducial MC", "cumulative": True})
            # This is already cumulative; do not multiply by previous again.
            cumulative = np.asarray(push.predict(mc.truth)["full_ratio"], dtype=np.float64)
            if not np.all(np.isfinite(cumulative) & (cumulative > 0)):
                raise RuntimeError("Truth push returned invalid cumulative weights")
            print(f"Iteration {index}: 3/3 updated folded MC / original folded MC", flush=True)
            folded = RatioEstimator.fit(xs, base * cumulative[matched], xs, base,
                                        target_splits=splits, reference_splits=splits,
                                        output_dir=temporary / "reco_likelihood", config=replace(config, seed=config.seed + 2),
                                        metadata={"stage": "reco_likelihood", "iteration": index, "reference": "original unweighted-by-OmniFold MC", "cumulative": True})
            np.save(temporary / "truth_weights.npy", cumulative, allow_pickle=False)
            record = {"index": index, "session_sha256": self._manifest_sha,
                      "previous_iteration_sha256": sha256_file(self.iterations[-1].root / "iteration.json") if self.iterations else None,
                      "truth_weights_sha256": sha256_file(temporary / "truth_weights.npy"),
                      "model_manifest_sha256": {name: sha256_file(temporary / name / "manifest.json")
                                                for name in ("reco_pull", "truth_push", "reco_likelihood")},
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "data_yield": float(data.weights.sum(dtype=np.float64)),
                      "reference_reco_yield": float(base.sum(dtype=np.float64)),
                      "updated_reco_yield": float(np.sum(base * cumulative[matched], dtype=np.float64)),
                      "reference_truth_yield": float(mc.weights.sum(dtype=np.float64)),
                      "updated_truth_yield": float(np.sum(mc.weights * cumulative, dtype=np.float64)),
                      "yield_ratios": {"reco_pull": float(pull.yield_ratio), "truth_push": float(push.yield_ratio), "reco_likelihood": float(folded.yield_ratio)},
                      "fiducial_mc_entries": int(mc.weights.size), "matched_mc_entries": int(matched.sum())}
            record.update(self._likelihood_values(folded, data, record))
            _json_write(temporary / "iteration.json", record)
            os.rename(temporary, destination)
            temporary = None
            # Models are loaded from their committed location so paths remain valid.
            models = [RatioEstimator.load(destination / name, device=self.config.device)
                      for name in ("reco_pull", "truth_push", "reco_likelihood")]
            result = IterationResult(index, *models, destination, record)
            self.iterations.append(result)
            print(f"Committed iteration {index} to {destination}", flush=True)
            return result
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            lock.unlink(missing_ok=True)

    @staticmethod
    def _likelihood_values(model: RatioEstimator, data: Sample, record: dict[str, Any]) -> dict[str, float]:
        predictions = model.predict(data.reco)
        shape, full = np.asarray(predictions["shape_ratio"]), np.asarray(predictions["full_ratio"])
        if not np.all(np.isfinite(shape) & (shape > 0) & np.isfinite(full) & (full > 0)):
            raise ValueError("Likelihood ratio predictions must be finite and positive")
        return {"weighted_shape_log_ratio": float(np.sum(data.weights * np.log(shape), dtype=np.float64)),
                "weighted_extended_log_ratio": float(np.sum(data.weights * np.log(full), dtype=np.float64)
                                                      - (record["updated_reco_yield"] - record["reference_reco_yield"]))}

    def history(self) -> list[dict[str, Any]]:
        return [{key: result.manifest[key] for key in
                 ("index", "data_yield", "reference_reco_yield", "updated_reco_yield",
                  "reference_truth_yield", "updated_truth_yield", "yield_ratios",
                  "weighted_shape_log_ratio", "weighted_extended_log_ratio")}
                for result in self.iterations]

    def evaluate_likelihood(self, data_path: Path | str | None = None) -> list[dict[str, Any]]:
        """Evaluate frozen nominal-fit models; this does not refit an unfolding toy."""
        data = self._data if data_path is None else _read_sample(Path(data_path), kind="data",
                    weight_branch=self.manifest["settings"]["data_weight_branch"], seed=self.manifest["settings"]["seed"])
        self._check_luminosity(data, self._mc)
        return [{"iteration": result.index, **self._likelihood_values(result.reco_likelihood, data, result.manifest)}
                for result in self.iterations]

    def diagnostics(self, iteration: int | None = None) -> dict[str, Any]:
        from .diagnostics import plot_ratio_diagnostics
        result = self._result(iteration)
        mc, data = self._mc, self._data
        matched = mc.matched
        base = mc.weights[matched]
        previous = self._previous_weights(result.index)[matched]
        xs, zs, splits = mc.reco[matched], mc.truth[matched], mc.splits[matched]
        pulled = base * previous * result.reco_pull.predict(xs)["full_ratio"]
        stages = (("reco_pull", result.reco_pull, data.reco, data.weights, xs, base * previous, data.splits, splits, FEATURES),
                  ("truth_push", result.truth_push, zs, pulled, zs, base, splits, splits, TRUTH_FEATURES),
                  ("reco_likelihood", result.reco_likelihood, xs, base * result.truth_weights[matched], xs, base, splits, splits, FEATURES))
        figures = {}
        for name, model, tx, tw, rx, rw, ts, rs, names in stages:
            plots = plot_ratio_diagnostics(model, tx, tw, rx, rw, target_splits=ts, reference_splits=rs,
                                           title=f"Iteration {result.index}: {name}", feature_names=names,
                                           shared_events=(name != "reco_pull"))
            figures.update({f"{name}_{key}": value for key, value in plots.items()})
        return figures

    def compare(self, iteration: int | None = None, *, shape_only: bool = False) -> dict[str, Any]:
        from .diagnostics import plot_distributions
        result = self._result(iteration)
        mc, data = self._mc, self._data
        matched = mc.matched
        updated = mc.weights * result.truth_weights
        figures = {"reco": plot_distributions(mc.reco[matched], mc.weights[matched], updated[matched],
                    data.reco, data.weights, feature_names=FEATURES,
                    title=f"Iteration {result.index}: folded signal vs background-removed data", shape_only=shape_only)}
        if "truth" in self.manifest["inputs"]:
            if self._truth is None:
                record = self.manifest["inputs"]["truth"]
                self._truth = _read_sample(Path(record["path"]), kind="truth",
                           weight_branch=self.manifest["settings"]["truth_weight_branch"],
                           seed=self.manifest["settings"]["seed"], provenance=record)
                self._check_luminosity(self._truth, mc)
            figures["truth"] = plot_distributions(mc.truth, mc.weights, updated,
                         self._truth.truth, self._truth.weights, feature_names=TRUTH_FEATURES,
                         title=f"Iteration {result.index}: unfolded fiducial signal vs full Herwig truth", shape_only=shape_only)
        else:
            figures["truth"] = plot_distributions(mc.truth, mc.weights, updated,
                         feature_names=TRUTH_FEATURES, title=f"Iteration {result.index}: fiducial signal (no Herwig target configured)", shape_only=shape_only)
        return figures

    def decorate(self, inputs: list[Path | str], *, output_dir: Path | str,
                 include_mc: bool = True, overwrite: bool = False) -> list[Path]:
        from .application import decorate_files
        self._result()
        for name in ("data", "mc"):
            record = self.manifest["inputs"][name]
            assert_file_unchanged(Path(record["path"]), record)
        paths = [Path(path).expanduser().resolve() for path in inputs]
        if include_mc and self._mc.path not in paths:
            paths.append(self._mc.path)
        return decorate_files(paths, output_dir, self.iterations, mc_path=self._mc.path,
                              data_weight_branch=self.manifest["settings"]["data_weight_branch"],
                              mc_weight_branch=self.manifest["settings"]["mc_weight_branch"], overwrite=overwrite,
                              metadata={"session_path": str(self.run_dir), "session_sha256": self._manifest_sha,
                                        "nominal_data_path": str(self._data.path),
                                        "luminosity_fb": self._mc.luminosity_fb})
