from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from BackgroundRemoval.Application.apply_background_ratio import (
    ApplicationModels,
    decorate,
    load_application_models,
)
from BackgroundRemoval.common import FEATURES, sha256_file, weight_measure_contract


class _FakeBundle:
    def __init__(self, root: Path, *, kind: str, value: float):
        self.root = root.resolve()
        weight_branch = (
            "weight_nominal_pb" if kind == "background_removal" else "weight"
        )
        self.manifest = {
            "created_utc": "2026-01-01T00:00:00+00:00",
            "model_kind": kind,
            "luminosity_fb": 312.0,
            "weight_branch": weight_branch,
            "weight_measure": weight_measure_contract(
                weight_branch, luminosity_fb=312.0
            ),
            "selections": (
                {
                    "mass_window_gev": {
                        "branch": "reco_m_ZZ",
                        "low_exclusive": 115.0,
                        "high_exclusive": 130.0,
                    }
                }
                if kind == "background_removal"
                else {
                    "correction_mass_branch": "reco_m_ZZ",
                    "correction_mass_window_GeV": [130.0, 160.0],
                    "bounds": "strict",
                }
            ),
        }
        self.value = value
        self.predicted_rows: list[np.ndarray] = []

    def predict(self, raw_features: np.ndarray, *, batch_size: int):
        self.predicted_rows.append(raw_features.copy())
        size = raw_features.shape[0]
        return {
            "calibrated_score": np.full(size, 0.4),
            "score_ensemble_std": np.full(size, 0.01),
            "shape_ratio": np.full(size, 1.25),
            "physical_ratio": np.full(size, self.value),
            "target_purity": np.full(size, self.value),
        }


def _models(directory: Path) -> ApplicationModels:
    background = _FakeBundle(
        directory / "background", kind="background_removal", value=0.25
    )
    correction = _FakeBundle(
        directory / "correction", kind="data_mc_correction", value=2.5
    )
    return ApplicationModels(
        background=background,  # type: ignore[arg-type]
        correction=correction,  # type: ignore[arg-type]
        background_manifest_sha256="a" * 64,
        correction_manifest_sha256="b" * 64,
    )


def _write_sample(
    path: Path,
    *,
    reconstructed: np.ndarray,
    masses: np.ndarray,
    nan_feature_rows: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    size = reconstructed.size
    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight": np.resize(np.array([1.0, -1.0]), size).astype(np.float64),
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "reconstructed": reconstructed.astype(np.bool_),
        "fiducial": np.ones(size, dtype=np.bool_),
        "reco_m_ZZ": masses.astype(np.float32),
        "fixed_vector": np.arange(size * 3, dtype=np.float32).reshape(size, 3),
    }
    for index, name in enumerate(FEATURES):
        arrays[name] = np.linspace(index, index + 1.0, size, dtype=np.float32)
        if nan_feature_rows:
            arrays[name][list(nan_feature_rows)] = np.nan
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays
        root_file["merge_metadata"] = json.dumps({"kept": True})
    return arrays


def _decorate(
    source: Path,
    output: Path,
    models: ApplicationModels,
    *,
    sample_kind: str,
    diagnostics: bool = False,
) -> dict[str, object]:
    return decorate(
        source,
        output,
        Path("unused-background"),
        Path("unused-correction"),
        sample_kind=sample_kind,
        step_size="1 MB",
        inference_batch_size=10,
        device="cpu",
        overwrite=False,
        replace_existing_branches=False,
        write_diagnostic_branches=diagnostics,
        models=models,
    )


class ApplicationTest(unittest.TestCase):
    def test_sample_specific_recipe_and_strict_analysis_region(self):
        reconstructed = np.array([True, True, True, True, False])
        masses = np.array([120.0, 115.0, 130.0, 140.0, np.nan])
        for sample_kind, expected, predicted_rows in (
            ("data", [0.25, 1.0, 1.0, 1.0, 1.0], (1, 0)),
            ("gg-h", [1.0, 1.0, 1.0, 1.0, 1.0], (0, 0)),
            ("zz", [2.5, 2.5, 2.5, 2.5, 1.0], (0, 4)),
        ):
            with self.subTest(sample_kind=sample_kind):
                with tempfile.TemporaryDirectory() as directory_name:
                    directory = Path(directory_name)
                    source = directory / "source.root"
                    output = directory / "output.root"
                    # Data must tolerate NaNs outside its prediction region;
                    # ggH never evaluates a network. ZZ only tolerates them on
                    # the non-reconstructed row.
                    nan_rows = (
                        (1, 2, 3, 4)
                        if sample_kind in {"data", "gg-h"}
                        else (4,)
                    )
                    _write_sample(
                        source,
                        reconstructed=reconstructed,
                        masses=masses,
                        nan_feature_rows=nan_rows,
                    )
                    models = _models(directory)
                    metadata = _decorate(
                        source, output, models, sample_kind=sample_kind
                    )
                    with uproot.open(output) as root_file:
                        arrays = root_file["Analysis"].arrays(library="np", how=dict)
                    np.testing.assert_array_equal(
                        arrays["analysis_region"], np.array([1, 0, 0, 0, 0], np.uint8)
                    )
                    np.testing.assert_allclose(
                        arrays["background_removal_weight"], expected
                    )
                    self.assertEqual(
                        arrays["background_removal_weight"].dtype, np.dtype(np.float32)
                    )
                    self.assertEqual(
                        arrays["analysis_region"].dtype, np.dtype(np.uint8)
                    )
                    self.assertEqual(metadata["predicted_entries"], sum(predicted_rows))
                    self.assertEqual(
                        sum(x.shape[0] for x in models.background.predicted_rows),
                        predicted_rows[0],
                    )
                    self.assertEqual(
                        sum(x.shape[0] for x in models.correction.predicted_rows),
                        predicted_rows[1],
                    )

    def test_preserves_weights_vectors_metadata_and_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            original = _write_sample(
                source,
                reconstructed=np.array([True, True, False]),
                masses=np.array([120.0, 140.0, np.nan]),
                nan_feature_rows=(1, 2),
            )
            source_sha256 = sha256_file(source)
            metadata = _decorate(
                source, output, _models(directory), sample_kind="data", diagnostics=True
            )
            self.assertEqual(metadata["entries"], 3)
            with uproot.open(output) as root_file:
                arrays = root_file["Analysis"].arrays(library="np", how=dict)
                np.testing.assert_array_equal(arrays["weight"], original["weight"])
                np.testing.assert_array_equal(
                    arrays["fixed_vector"], original["fixed_vector"]
                )
                np.testing.assert_allclose(
                    arrays["weight_background_removed"], [0.25, -1.0, 1.0]
                )
                self.assertTrue(np.isnan(arrays["signal_score_balanced"][1]))
                self.assertEqual(
                    json.loads(str(root_file["merge_metadata"])), {"kept": True}
                )
                stored = json.loads(str(root_file["background_removal_metadata"]))
                self.assertEqual(stored["sample_kind"], "data")
                self.assertEqual(
                    stored["source_provenance"]["sha256"], source_sha256
                )

    def test_in_place_decoration_retains_pre_replacement_provenance(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            _write_sample(
                source,
                reconstructed=np.array([True]),
                masses=np.array([120.0]),
            )
            source_sha256 = sha256_file(source)
            decorate(
                source,
                source,
                Path("unused-background"),
                Path("unused-correction"),
                sample_kind="data",
                step_size="1 MB",
                inference_batch_size=10,
                device="cpu",
                overwrite=True,
                replace_existing_branches=False,
                write_diagnostic_branches=False,
                models=_models(directory),
            )
            with uproot.open(source) as root_file:
                stored = json.loads(str(root_file["background_removal_metadata"]))
            self.assertEqual(stored["source_provenance"]["sha256"], source_sha256)
            self.assertNotEqual(sha256_file(source), source_sha256)

    def test_reconstructed_nonfinite_mass_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            _write_sample(
                source,
                reconstructed=np.array([True, False]),
                masses=np.array([np.nan, np.nan]),
                nan_feature_rows=(0, 1),
            )
            with self.assertRaisesRegex(ValueError, "non-finite reco_m_ZZ"):
                _decorate(source, output, _models(directory), sample_kind="data")
            self.assertFalse(output.exists())

    def test_zero_zz_correction_factor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "zz.root"
            output = directory / "decorated.root"
            _write_sample(
                source,
                reconstructed=np.array([True]),
                masses=np.array([120.0]),
            )
            models = _models(directory)
            models.correction.value = 0.0
            with self.assertRaisesRegex(ValueError, "invalid application weights"):
                _decorate(source, output, models, sample_kind="zz")
            self.assertFalse(output.exists())

    def test_output_cannot_overlap_loaded_model_artifacts(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            _write_sample(
                source,
                reconstructed=np.array([True]),
                masses=np.array([120.0]),
            )
            models = _models(directory)
            dangerous_output = models.background.root / "member_000.pt"
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                _decorate(
                    source,
                    dangerous_output,
                    models,
                    sample_kind="data",
                )
            self.assertFalse(dangerous_output.exists())

    def test_nominal_weight_branch_is_required(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            arrays = _write_sample(
                source,
                reconstructed=np.array([True]),
                masses=np.array([120.0]),
            )
            arrays.pop("weight")
            with uproot.recreate(source) as root_file:
                root_file["Analysis"] = arrays
            with self.assertRaisesRegex(KeyError, "weight"):
                _decorate(source, output, _models(directory), sample_kind="data")
            self.assertFalse(output.exists())

    def test_nonfinite_nominal_weight_is_rejected_without_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            arrays = _write_sample(
                source,
                reconstructed=np.array([True]),
                masses=np.array([120.0]),
            )
            arrays["weight"][0] = np.nan
            with uproot.recreate(source) as root_file:
                root_file["Analysis"] = arrays

            with self.assertRaisesRegex(ValueError, "non-finite observation weights"):
                _decorate(
                    source,
                    output,
                    _models(directory),
                    sample_kind="data",
                    diagnostics=False,
                )
            self.assertFalse(output.exists())

    def test_model_pair_checksum_and_kinds_are_verified(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            background_dir = directory / "background"
            correction_dir = directory / "correction"
            background_dir.mkdir()
            correction_dir.mkdir()
            correction_manifest = correction_dir / "manifest.json"
            correction_manifest.write_text('{"model": "correction"}\n', encoding="utf-8")
            checksum = sha256_file(correction_manifest)
            background_manifest = background_dir / "manifest.json"
            background_manifest.write_text("{}\n", encoding="utf-8")
            background = _FakeBundle(
                background_dir, kind="background_removal", value=0.25
            )
            background.manifest["correction_model"] = {
                "manifest_sha256": checksum
            }
            correction = _FakeBundle(
                correction_dir, kind="data_mc_correction", value=2.5
            )
            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                side_effect=[background, correction],
            ) as loader:
                loaded = load_application_models(
                    background_dir, correction_dir, device="cpu"
                )
            self.assertEqual(loaded.correction_manifest_sha256, checksum)
            self.assertEqual(
                loader.call_args_list[0].kwargs["expected_model_kind"],
                "background_removal",
            )
            self.assertEqual(
                loader.call_args_list[1].kwargs["expected_model_kind"],
                "data_mc_correction",
            )

            background.manifest["correction_model"]["manifest_sha256"] = "0" * 64
            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                side_effect=[background, correction],
            ):
                with self.assertRaisesRegex(ValueError, "not the one used"):
                    load_application_models(
                        background_dir, correction_dir, device="cpu"
                    )

            background.manifest["correction_model"]["manifest_sha256"] = checksum
            background.manifest["selections"]["mass_window_gev"][
                "low_exclusive"
            ] = 105.0
            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                side_effect=[background, correction],
            ):
                with self.assertRaisesRegex(ValueError, "analysis mass window"):
                    load_application_models(
                        background_dir, correction_dir, device="cpu"
                    )

    def test_custom_weight_measure_pair_requires_training_override(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            background_dir = directory / "background"
            correction_dir = directory / "correction"
            background_dir.mkdir()
            correction_dir.mkdir()
            (background_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            correction_manifest = correction_dir / "manifest.json"
            correction_manifest.write_text("{}\n", encoding="utf-8")
            correction_checksum = sha256_file(correction_manifest)

            background = _FakeBundle(
                background_dir, kind="background_removal", value=0.25
            )
            correction = _FakeBundle(
                correction_dir, kind="data_mc_correction", value=2.5
            )
            background.manifest["weight_branch"] = "training_custom_weight"
            background.manifest["weight_measure"] = weight_measure_contract(
                "training_custom_weight", luminosity_fb=312.0
            )
            correction.manifest["weight_branch"] = "correction_custom_weight"
            correction.manifest["weight_measure"] = weight_measure_contract(
                "correction_custom_weight", luminosity_fb=312.0
            )
            background.manifest["correction_model"] = {
                "manifest_sha256": correction_checksum,
                "weight_branch": "correction_custom_weight",
                "weight_measure_override": False,
            }

            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                side_effect=[background, correction],
            ):
                with self.assertRaisesRegex(
                    ValueError, "without a recorded Training override"
                ):
                    load_application_models(
                        background_dir, correction_dir, device="cpu"
                    )

    def test_custom_weight_measure_pair_accepts_linked_training_override(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            background_dir = directory / "background"
            correction_dir = directory / "correction"
            background_dir.mkdir()
            correction_dir.mkdir()
            (background_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            correction_manifest = correction_dir / "manifest.json"
            correction_manifest.write_text("{}\n", encoding="utf-8")
            correction_checksum = sha256_file(correction_manifest)

            background = _FakeBundle(
                background_dir, kind="background_removal", value=0.25
            )
            correction = _FakeBundle(
                correction_dir, kind="data_mc_correction", value=2.5
            )
            background.manifest["weight_branch"] = "training_custom_weight"
            background.manifest["weight_measure"] = weight_measure_contract(
                "training_custom_weight", luminosity_fb=312.0
            )
            correction.manifest["weight_branch"] = "correction_custom_weight"
            correction_contract = weight_measure_contract(
                "correction_custom_weight", luminosity_fb=312.0
            )
            correction.manifest["weight_measure"] = correction_contract
            background.manifest["correction_model"] = {
                "manifest_sha256": correction_checksum,
                "weight_branch": "correction_custom_weight",
                "weight_measure": correction_contract,
                "weight_measures_known_compatible": False,
                "weight_measure_override": True,
            }

            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                side_effect=[background, correction],
            ):
                loaded = load_application_models(
                    background_dir, correction_dir, device="cpu"
                )

            self.assertIs(loaded.background, background)
            self.assertIs(loaded.correction, correction)


if __name__ == "__main__":
    unittest.main()
