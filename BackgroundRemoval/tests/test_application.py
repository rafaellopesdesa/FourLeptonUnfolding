from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from BackgroundRemoval.Application.apply_background_ratio import decorate
from BackgroundRemoval.common import FEATURES


class _FakeBundle:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = {
            "created_utc": "2026-01-01T00:00:00+00:00",
            "luminosity_fb": 312.0,
            "yields": {"signal": 2.0, "background": 1.0, "signal_to_background": 2.0},
        }

    def predict(self, raw_features: np.ndarray, *, batch_size: int):
        size = raw_features.shape[0]
        removal = np.linspace(0.2, 0.8, size)
        return {
            "signal_score_balanced": np.full(size, 0.5),
            "signal_score_ensemble_std": np.full(size, 0.01),
            "background_shape_ratio": np.ones(size),
            "signal_to_background_ratio": np.full(size, 2.0),
            "background_removal_weight": removal,
        }


def _write_data(path: Path, *, reconstructed: np.ndarray) -> dict[str, np.ndarray]:
    size = reconstructed.size
    arrays: dict[str, np.ndarray] = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight": np.resize(np.array([1.0, -1.0]), size).astype(np.float64),
        "luminosity_fb": np.full(size, 312.0, dtype=np.float64),
        "reconstructed": reconstructed.astype(np.bool_),
        "fiducial": np.ones(size, dtype=np.bool_),
        "fixed_vector": np.arange(size * 3, dtype=np.float32).reshape(size, 3),
    }
    for index, name in enumerate(FEATURES):
        arrays[name] = np.linspace(index, index + 1.0, size, dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = arrays
        root_file["merge_metadata"] = json.dumps({"kept": True})
    return arrays


class ApplicationTest(unittest.TestCase):
    def test_default_writes_only_the_requested_compact_branch(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            model = directory / "model"
            model.mkdir()
            (model / "manifest.json").write_text("{}\n", encoding="utf-8")
            _write_data(source, reconstructed=np.ones(3, dtype=np.bool_))
            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                return_value=_FakeBundle(model.resolve()),
            ):
                decorate(
                    source,
                    output,
                    model,
                    step_size="1 MB",
                    inference_batch_size=10,
                    device="cpu",
                    overwrite=False,
                    replace_existing_branches=False,
                    write_diagnostic_branches=False,
                )
            with uproot.open(output) as root_file:
                branches = set(root_file["Analysis"].keys())
                self.assertIn("background_removal_weight", branches)
                self.assertNotIn("weight_background_removed", branches)
                self.assertEqual(
                    root_file["Analysis"]["background_removal_weight"].array(
                        library="np"
                    ).dtype,
                    np.dtype(np.float32),
                )

    def test_decorates_without_overwriting_signed_weight_or_metadata(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            model = directory / "model"
            model.mkdir()
            (model / "manifest.json").write_text("{}\n", encoding="utf-8")
            original = _write_data(source, reconstructed=np.ones(4, dtype=np.bool_))
            fake = _FakeBundle(model.resolve())
            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                return_value=fake,
            ):
                metadata = decorate(
                    source,
                    output,
                    model,
                    step_size="1 MB",
                    inference_batch_size=10,
                    device="cpu",
                    overwrite=False,
                    replace_existing_branches=False,
                    write_diagnostic_branches=True,
                )
            self.assertEqual(metadata["entries"], 4)
            with uproot.open(output) as root_file:
                arrays = root_file["Analysis"].arrays(library="np", how=dict)
                np.testing.assert_array_equal(arrays["weight"], original["weight"])
                np.testing.assert_array_equal(
                    arrays["fixed_vector"], original["fixed_vector"]
                )
                expected_removal = np.linspace(0.2, 0.8, 4)
                np.testing.assert_allclose(
                    arrays["background_removal_weight"], expected_removal
                )
                np.testing.assert_allclose(
                    arrays["weight_background_removed"],
                    original["weight"] * expected_removal,
                )
                self.assertIn("merge_metadata", root_file)
                self.assertIn("background_removal_metadata", root_file)
                self.assertEqual(json.loads(str(root_file["merge_metadata"])), {"kept": True})

    def test_nonreconstructed_input_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "data.root"
            output = directory / "decorated.root"
            model = directory / "model"
            model.mkdir()
            (model / "manifest.json").write_text("{}\n", encoding="utf-8")
            _write_data(source, reconstructed=np.array([True, False]))
            with patch(
                "BackgroundRemoval.Application.apply_background_ratio.ModelBundle.load",
                return_value=_FakeBundle(model.resolve()),
            ):
                with self.assertRaisesRegex(ValueError, "non-reconstructed"):
                    decorate(
                        source,
                        output,
                        model,
                        step_size="1 MB",
                        inference_batch_size=10,
                        device="cpu",
                        overwrite=False,
                        replace_existing_branches=False,
                        write_diagnostic_branches=False,
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
