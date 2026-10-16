from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from BackgroundRemoval.common import FEATURES, transformed_features
from BackgroundRemoval.Correction.train_data_mc_correction import (
    _COMPONENT_DATA,
    _COMPONENT_GGH,
    _COMPONENT_ZZ,
    _parser,
    _scan_inputs,
    _selected_chunks,
)
from BackgroundRemoval.Training.train_background_ratio import SelectedChunk


def _write_sample(
    path: Path,
    *,
    masses: np.ndarray,
    weights: np.ndarray,
    reconstructed: np.ndarray | None = None,
) -> None:
    size = masses.size
    if reconstructed is None:
        reconstructed = np.ones(size, dtype=np.bool_)
    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight": weights.astype(np.float64),
        "weight_nominal_pb": weights.astype(np.float64),
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "reconstructed": np.asarray(reconstructed, dtype=np.bool_),
        "reco_m_ZZ": masses.astype(np.float32),
    }
    for index, name in enumerate(FEATURES):
        arrays[name] = np.linspace(index, index + 0.5, size, dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays


class CorrectionDataTest(unittest.TestCase):
    def test_scan_uses_absolute_weights_only_for_scaler_moments(self):
        raw_target = np.arange(6 * len(FEATURES), dtype=np.float32).reshape(
            6, len(FEATURES)
        ) / 20.0
        raw_reference = raw_target + np.float32(0.3)
        weights = np.array([2.0, -1.0, 2.0, -1.0, 2.0, -1.0])
        splits = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8)
        chunks = [
            SelectedChunk(
                class_name="signal",
                component_name=_COMPONENT_DATA,
                raw_features=raw_target,
                weights=weights,
                splits=splits,
                zero_count=0,
            ),
            SelectedChunk(
                class_name="background",
                component_name=_COMPONENT_GGH,
                raw_features=raw_reference,
                weights=weights,
                splits=splits,
                zero_count=0,
            ),
        ]
        with patch(
            "BackgroundRemoval.Correction.train_data_mc_correction._selected_chunks",
            return_value=iter(chunks),
        ):
            _, classes, mean, scale = _scan_inputs(
                Path("data.root"),
                Path("gg.root"),
                Path("zz.root"),
                weight_branch="weight",
                split_seed=1,
                expected_luminosity_fb=312.0,
                step_size="1 MB",
            )

        transformed_target = transformed_features(raw_target[:2]).astype(np.float64)
        transformed_reference = transformed_features(raw_reference[:2]).astype(
            np.float64
        )
        absolute_fit_weights = np.array([2.0, 1.0])
        target_mean = np.sum(
            transformed_target * absolute_fit_weights[:, np.newaxis], axis=0
        ) / 3.0
        reference_mean = np.sum(
            transformed_reference * absolute_fit_weights[:, np.newaxis], axis=0
        ) / 3.0
        np.testing.assert_allclose(mean, 0.5 * (target_mean + reference_mean))
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale > 0.0))
        self.assertAlmostEqual(classes["signal"].sum_weights, 3.0)
        self.assertAlmostEqual(classes["signal"].sum_absolute_weights, 9.0)
        self.assertEqual(classes["signal"].negative_weight_entries, 3)

    def test_strict_sideband_and_physical_reference_mixture(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            data = directory / "data.root"
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                data,
                masses=np.array([129.0, 130.0, 131.0, 159.0, 160.0, 140.0]),
                weights=np.array([50.0, 40.0, 2.0, 3.0, 30.0, 20.0]),
                reconstructed=np.array([True, True, True, True, True, False]),
            )
            _write_sample(
                gg_h,
                masses=np.array([131.0, 145.0, 161.0]),
                weights=np.array([5.0, 7.0, 100.0]),
            )
            _write_sample(
                zz,
                masses=np.array([130.0, 150.0, 159.0]),
                weights=np.array([200.0, 11.0, 13.0]),
            )
            chunks = list(
                _selected_chunks(
                    data,
                    gg_h,
                    zz,
                    weight_branch="weight",
                    split_seed=17,
                    expected_luminosity_fb=312.0,
                    step_size="1 MB",
                )
            )
            by_component = {chunk.component_name: chunk for chunk in chunks}
            self.assertEqual(by_component[_COMPONENT_DATA].weights.tolist(), [2.0, 3.0])
            self.assertEqual(by_component[_COMPONENT_GGH].weights.tolist(), [5.0, 7.0])
            self.assertEqual(by_component[_COMPONENT_ZZ].weights.tolist(), [11.0, 13.0])
            reference_yield = sum(
                float(np.sum(chunk.weights, dtype=np.float64))
                for chunk in chunks
                if chunk.class_name == "background"
            )
            self.assertEqual(reference_yield, 36.0)

    def test_selected_negative_reference_weight_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            data = directory / "data.root"
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                data,
                masses=np.array([120.0, 140.0]),
                weights=np.array([-9.0, 1.0]),
            )
            _write_sample(
                gg_h,
                masses=np.array([140.0]),
                weights=np.array([1.0]),
            )
            _write_sample(
                zz,
                masses=np.array([140.0]),
                weights=np.array([-1.0]),
            )
            chunks = list(
                _selected_chunks(
                    data,
                    gg_h,
                    zz,
                    weight_branch="weight",
                    split_seed=17,
                    expected_luminosity_fb=312.0,
                    step_size="1 MB",
                )
            )
            by_component = {chunk.component_name: chunk for chunk in chunks}
            self.assertEqual(by_component[_COMPONENT_DATA].weights.tolist(), [1.0])
            self.assertEqual(by_component[_COMPONENT_GGH].weights.tolist(), [1.0])
            self.assertEqual(by_component[_COMPONENT_ZZ].weights.tolist(), [-1.0])

    def test_selected_negative_data_weight_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            data = directory / "data.root"
            gg_h = directory / "gg_H_pythia.root"
            zz = directory / "ZZ_pythia.root"
            _write_sample(
                data,
                masses=np.array([120.0, 140.0]),
                weights=np.array([-9.0, -1.0]),
            )
            _write_sample(
                gg_h,
                masses=np.array([140.0]),
                weights=np.array([1.0]),
            )
            _write_sample(
                zz,
                masses=np.array([140.0]),
                weights=np.array([1.0]),
            )
            chunks = list(
                _selected_chunks(
                    data,
                    gg_h,
                    zz,
                    weight_branch="weight",
                    split_seed=17,
                    expected_luminosity_fb=312.0,
                    step_size="1 MB",
                )
            )
            by_component = {chunk.component_name: chunk for chunk in chunks}
            self.assertEqual(by_component[_COMPONENT_DATA].weights.tolist(), [-1.0])

    def test_training_defaults_match_ensemble_prescription(self):
        parser = _parser()
        args = parser.parse_args(
            [
                "--data-root",
                "data.root",
                "--gg-h-root",
                "gg_H_pythia.root",
                "--zz-root",
                "ZZ_pythia.root",
                "--output-dir",
                "model",
            ]
        )
        self.assertGreaterEqual(args.ensemble_size, 4)
        self.assertEqual(args.hidden_layers, 4)
        self.assertEqual(args.neurons, 1024)
        self.assertEqual(args.weight_branch, "weight")


if __name__ == "__main__":
    unittest.main()
