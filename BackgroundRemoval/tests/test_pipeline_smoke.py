from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import uproot

import BackgroundRemoval.Correction.train_data_mc_correction as correction_training
import BackgroundRemoval.Training.train_background_ratio as background_training
from BackgroundRemoval.common import FEATURES, TOOLKIT_COMMIT, sha256_file


def _write_sample(
    path: Path,
    *,
    sideband_entries: int,
    signal_region_entries: int,
    weight: float,
    seed: int,
) -> None:
    size = sideband_entries + signal_region_entries
    rng = np.random.default_rng(seed)
    masses = np.concatenate(
        [
            np.full(sideband_entries, 140.0, dtype=np.float32),
            np.full(signal_region_entries, 122.0, dtype=np.float32),
        ]
    )
    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "reconstructed": np.ones(size, dtype=np.bool_),
        "fiducial": (np.arange(size) % 2 == 0),
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "weight": np.full(size, weight, dtype=np.float64),
        "weight_nominal_pb": np.full(
            size, weight / 312_000.0, dtype=np.float64
        ),
        "reco_m_ZZ": masses,
    }
    for index, name in enumerate(FEATURES):
        if name.startswith("reco_m_Z"):
            arrays[name] = rng.uniform(15.0, 100.0, size).astype(np.float32)
        elif name.startswith("reco_cos"):
            arrays[name] = rng.uniform(-1.0, 1.0, size).astype(np.float32)
        else:
            arrays[name] = rng.uniform(-np.pi, np.pi, size).astype(np.float32)
        arrays[name] += np.float32(index * 1.0e-3)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


def _fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: False)
    module.device = lambda name: name

    def save(state, path):
        Path(path).write_bytes(repr(state).encode())

    module.save = save
    return module


def _fake_diagnostics() -> types.ModuleType:
    module = types.ModuleType("BackgroundRemoval.Training.diagnostics")

    def make_diagnostics_pdf(path, **kwargs):
        del kwargs
        Path(path).write_bytes(b"%PDF-1.4\n% pipeline smoke diagnostic\n")
        return {"pipeline_smoke": True}

    module.make_diagnostics_pdf = make_diagnostics_pdf
    return module


def _runtime_module_patches(fake_torch: types.ModuleType) -> dict[str, object]:
    modules: dict[str, object] = {"torch": fake_torch}
    try:
        __import__("hist")
        __import__("mplhep")
    except ImportError:
        modules["BackgroundRemoval.Training.diagnostics"] = _fake_diagnostics()
    return modules


def _fake_member(slot, attempt, *args, base_seed, **kwargs):
    return background_training.MemberResult(
        slot=slot,
        attempt=attempt,
        seed=(base_seed + slot) % (2**32),
        state_dict={"slot": slot},
        history={
            "training_loss": [0.70 - 0.01 * slot],
            "validation_loss": [0.69 + 0.001 * slot],
            "learning_rate": [3.0e-4],
        },
        validation_loss=0.69 + 0.001 * slot,
        validation_score_mean=0.5,
        validation_score_std=0.1,
        validation_saturated_fraction=0.0,
    )


def _fake_predict_split(cache, results, *, split_code, **kwargs):
    output = {}
    offsets = np.linspace(-0.015, 0.015, len(results), dtype=np.float64)
    for class_name, arrays in cache.items():
        indices = np.flatnonzero(arrays.splits == split_code)
        raw = np.asarray(arrays.features[indices], dtype=np.float64)
        central = 0.5 + 0.12 * np.tanh(raw[:, 0])
        member_scores = np.clip(
            central[:, np.newaxis] + offsets[np.newaxis, :], 0.05, 0.95
        ).astype(np.float32)
        output[class_name] = background_training.ClassPredictions(
            indices=indices,
            member_scores=member_scores,
        )
    return output


class _CorrectionBundle:
    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest

    def predict(self, raw_features, *, batch_size):
        del batch_size
        size = len(raw_features)
        return {"physical_ratio": np.full(size, 1.25, dtype=np.float64)}


class PipelineSmokeTest(unittest.TestCase):
    def test_correction_then_training_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            data_path = directory / "data.root"
            gg_h_path = directory / "gg_H_pythia.root"
            zz_path = directory / "ZZ_pythia.root"
            correction_path = directory / "correction"
            background_path = directory / "background"
            _write_sample(
                data_path,
                sideband_entries=180,
                signal_region_entries=0,
                weight=1.0,
                seed=1,
            )
            _write_sample(
                gg_h_path,
                sideband_entries=180,
                signal_region_entries=240,
                weight=0.8,
                seed=2,
            )
            _write_sample(
                zz_path,
                sideband_entries=180,
                signal_region_entries=240,
                weight=1.2,
                seed=3,
            )

            correction_args = correction_training._parser().parse_args(
                [
                    "--data-root",
                    str(data_path),
                    "--gg-h-root",
                    str(gg_h_path),
                    "--zz-root",
                    str(zz_path),
                    "--output-dir",
                    str(correction_path),
                    "--device",
                    "cpu",
                ]
            )
            fake_torch = _fake_torch()
            toolkit = {
                "pinned_commit": TOOLKIT_COMMIT,
                "runtime_version": "test",
                "runtime_commit": TOOLKIT_COMMIT,
                "runtime_commit_verified": True,
            }
            with (
                patch.dict(
                    sys.modules,
                    _runtime_module_patches(fake_torch),
                ),
                patch.object(
                    correction_training,
                    "toolkit_runtime_provenance",
                    return_value=toolkit,
                ),
                patch.object(
                    correction_training,
                    "_train_member",
                    side_effect=_fake_member,
                ),
                patch.object(
                    correction_training,
                    "_predict_split",
                    side_effect=_fake_predict_split,
                ),
            ):
                correction_training.train(correction_args)

            correction_manifest = json.loads(
                (correction_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(correction_manifest["features"], list(FEATURES))
            self.assertEqual(
                correction_manifest["selections"]["mass_window_gev"],
                {
                    "branch": "reco_m_ZZ",
                    "low_exclusive": 130.0,
                    "high_exclusive": 160.0,
                },
            )
            self.assertEqual(correction_manifest["weight_branch"], "weight")
            self.assertGreater(correction_manifest["architecture"]["steps_per_epoch"], 0)
            self.assertEqual(
                correction_manifest["architecture"]["steps_per_epoch_policy"],
                "full_fit_split_coverage",
            )
            self.assertTrue((correction_path / "diagnostics.pdf").read_bytes().startswith(b"%PDF"))

            background_args = background_training._parser().parse_args(
                [
                    "--gg-h-root",
                    str(gg_h_path),
                    "--zz-root",
                    str(zz_path),
                    "--correction-model-dir",
                    str(correction_path),
                    "--output-dir",
                    str(background_path),
                    "--device",
                    "cpu",
                ]
            )
            correction_bundle = _CorrectionBundle(
                correction_path, correction_manifest
            )
            with (
                patch.dict(
                    sys.modules,
                    _runtime_module_patches(fake_torch),
                ),
                patch.object(
                    background_training,
                    "toolkit_runtime_provenance",
                    return_value=toolkit,
                ),
                patch.object(
                    background_training.ModelBundle,
                    "load",
                    return_value=correction_bundle,
                ),
                patch.object(
                    background_training,
                    "_train_member",
                    side_effect=_fake_member,
                ),
                patch.object(
                    background_training,
                    "_predict_split",
                    side_effect=_fake_predict_split,
                ),
            ):
                background_training.train(background_args)

            background_manifest = json.loads(
                (background_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                background_manifest["selections"]["mass_window_gev"],
                {
                    "branch": "reco_m_ZZ",
                    "low_exclusive": 115.0,
                    "high_exclusive": 130.0,
                },
            )
            self.assertEqual(
                background_manifest["correction_model"]["manifest_sha256"],
                sha256_file(correction_path / "manifest.json"),
            )
            self.assertGreater(background_manifest["architecture"]["steps_per_epoch"], 0)
            self.assertTrue(
                background_manifest["correction_model"][
                    "all_input_checksums_verified"
                ]
            )
            self.assertAlmostEqual(
                background_manifest["components"][
                    "ZZ_reconstructed_correction_weighted"
                ]["sum_weights"],
                240 * (1.2 / 312_000.0) * 1.25,
            )
            self.assertTrue((background_path / "diagnostics.pdf").read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
