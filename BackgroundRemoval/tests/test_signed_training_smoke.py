from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

import BackgroundRemoval.Training.train_background_ratio as training
from BackgroundRemoval.common import (
    CORRECTION_MASS_WINDOW,
    FEATURES,
    MODEL_FEATURES,
    MODEL_KIND_DATA_MC_CORRECTION,
    TOOLKIT_COMMIT,
    deterministic_split,
    weight_measure_contract,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _weights_with_one_negative_per_selected_split(
    event_ids: np.ndarray,
    *,
    source: str,
    split_seed: int,
    masks: tuple[np.ndarray, ...],
) -> np.ndarray:
    weights = np.ones(event_ids.size, dtype=np.float64)
    splits = deterministic_split(event_ids, source=source, seed=split_seed)
    for mask in masks:
        for split_code in (0, 1, 2):
            candidates = np.flatnonzero(mask & (splits == split_code))
            if candidates.size == 0:  # pragma: no cover - fixed sample sizes guard this
                raise RuntimeError("test sample does not populate every split")
            weights[candidates[0]] = -0.25
    return weights


def _write_signed_sample(
    path: Path,
    *,
    source: str,
    size: int,
    split_seed: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    event_ids = np.arange(size, dtype=np.uint64)
    if source == "gg_H":
        fiducial = event_ids % 2 == 0
        masks = (fiducial, ~fiducial)
    else:
        fiducial = np.ones(size, dtype=np.bool_)
        masks = (np.ones(size, dtype=np.bool_),)
    weights = _weights_with_one_negative_per_selected_split(
        event_ids,
        source=source,
        split_seed=split_seed,
        masks=masks,
    )
    arrays: dict[str, np.ndarray] = {
        "event_id": event_ids,
        "reconstructed": np.ones(size, dtype=np.bool_),
        "fiducial": fiducial,
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "weight_nominal_pb": weights,
        "reco_m_ZZ": np.full(size, 122.0, dtype=np.float32),
    }
    for name in FEATURES:
        if name in {"reco_Phi", "reco_Phi1", "reco_Psi"}:
            arrays[name] = rng.uniform(-np.pi, np.pi, size).astype(np.float32)
        elif name.startswith("reco_cos_theta"):
            arrays[name] = rng.uniform(-1.0, 1.0, size).astype(np.float32)
        elif name == "reco_m_Z1":
            arrays[name] = rng.uniform(50.0, 100.0, size).astype(np.float32)
        else:
            arrays[name] = rng.uniform(12.0, 50.0, size).astype(np.float32)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


class _UnitCorrection:
    def __init__(self, root: Path, manifest: dict[str, object]):
        self.root = root
        self.manifest = manifest

    def predict(self, raw_features: np.ndarray, *, batch_size: int):
        del batch_size
        return {"physical_ratio": np.ones(raw_features.shape[0], dtype=np.float64)}


@unittest.skipUnless(TORCH_AVAILABLE, "real signed-loss smoke test requires PyTorch")
class SignedTrainingSmokeTest(unittest.TestCase):
    def test_signed_training_calibration_and_diagnostics_pipeline(self):
        import torch

        split_seed = 24680

        def tiny_model(_architecture):
            return torch.nn.Sequential(
                torch.nn.Linear(len(MODEL_FEATURES), 8),
                torch.nn.SiLU(),
                torch.nn.Linear(8, 1),
            )

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            correction_directory = directory / "correction"
            correction_directory.mkdir()
            output_directory = directory / "background"
            _write_signed_sample(
                gg_h,
                source="gg_H",
                size=240,
                split_seed=split_seed,
                seed=10,
            )
            _write_signed_sample(
                zz,
                source="ZZ",
                size=120,
                split_seed=split_seed,
                seed=11,
            )

            correction_manifest: dict[str, object] = {
                "format_version": 2,
                "model_kind": MODEL_KIND_DATA_MC_CORRECTION,
                "created_utc": "test",
                "luminosity_fb": 312.0,
                "weight_branch": "weight",
                "weight_measure": weight_measure_contract(
                    "weight", luminosity_fb=312.0
                ),
                "selections": {
                    "mass_window_gev": {
                        "branch": "reco_m_ZZ",
                        "low_exclusive": CORRECTION_MASS_WINDOW[0],
                        "high_exclusive": CORRECTION_MASS_WINDOW[1],
                    }
                },
                "inputs": {},
            }
            (correction_directory / "manifest.json").write_text(
                json.dumps(correction_manifest), encoding="utf-8"
            )
            correction = _UnitCorrection(correction_directory, correction_manifest)

            args = training._parser().parse_args(
                [
                    "--gg-h-root",
                    str(gg_h),
                    "--zz-root",
                    str(zz),
                    "--correction-model-dir",
                    str(correction_directory),
                    "--output-dir",
                    str(output_directory),
                    "--ensemble-size",
                    "4",
                    "--hidden-layers",
                    "1",
                    "--neurons",
                    "8",
                    "--batch-size",
                    "32",
                    "--maximum-epochs",
                    "1",
                    "--patience",
                    "1",
                    "--steps-per-epoch",
                    "1",
                    "--inference-batch-size",
                    "128",
                    "--split-seed",
                    str(split_seed),
                    "--device",
                    "cpu",
                    "--skip-input-checksums",
                ]
            )
            toolkit = {
                "pinned_commit": TOOLKIT_COMMIT,
                "runtime_version": "test",
                "runtime_commit": TOOLKIT_COMMIT,
                "runtime_commit_verified": True,
            }
            with (
                patch.object(training, "build_toolkit_model", side_effect=tiny_model),
                patch.object(
                    training,
                    "toolkit_runtime_provenance",
                    return_value=toolkit,
                ),
                patch(
                    "BackgroundRemoval.common.build_toolkit_model",
                    side_effect=tiny_model,
                ),
                patch(
                    "BackgroundRemoval.common.toolkit_runtime_provenance",
                    return_value=toolkit,
                ),
            ):
                with patch.object(
                    training.ModelBundle, "load", return_value=correction
                ):
                    training.train(args)

                # Exercise the serialized signed-weight artifact through the
                # real loader and prediction API, including inference-time
                # logit clipping and ratio conversion.
                bundle = training.ModelBundle.load(
                    output_directory,
                    device_name="cpu",
                    expected_model_kind=training.MODEL_KIND_BACKGROUND_REMOVAL,
                )
                with uproot.open(gg_h) as root_file:
                    raw_features = np.column_stack(
                        [
                            root_file["Analysis"][name].array(library="np")[:5]
                            for name in FEATURES
                        ]
                    )
                predictions = bundle.predict(raw_features, batch_size=3)
                for values in predictions.values():
                    self.assertTrue(np.all(np.isfinite(values)))

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(
                manifest["classes"]["signal"]["negative_weight_entries"], 0
            )
            self.assertGreater(
                manifest["classes"]["background"]["negative_weight_entries"], 0
            )
            self.assertEqual(
                manifest["architecture"]["class_risk"],
                "sum(w * BCE) / sum(w)",
            )
            self.assertEqual(
                manifest["architecture"]["loss_logit_clip"], args.logit_clip
            )
            self.assertEqual(
                manifest["scaler"]["type"], "balanced_absolute_weight_standard"
            )
            self.assertEqual(
                manifest["calibration"]["weight_semantics"],
                "signed class-balanced weights",
            )
            diagnostics = output_directory / "diagnostics.pdf"
            self.assertTrue(diagnostics.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
