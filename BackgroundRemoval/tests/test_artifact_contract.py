from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from BackgroundRemoval.common import (
    ARTIFACT_FORMAT_VERSION,
    FEATURES,
    MODEL_FEATURES,
    MODEL_KIND_BACKGROUND_REMOVAL,
    MODEL_KIND_DATA_MC_CORRECTION,
    TOOLKIT_COMMIT,
    ModelBundle,
)
from BackgroundRemoval.Training.train_background_ratio import (
    _validate_existing_artifact,
)


class _FakeModel:
    def load_state_dict(self, state):
        self.state = state

    def to(self, device):
        return self

    def eval(self):
        return self


def _fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: False)
    module.device = lambda name: name
    module.load = lambda *args, **kwargs: {}
    return module


def _write_artifact(path: Path, *, model_kind: str) -> dict:
    path.mkdir()
    members = []
    for slot in range(4):
        member_path = path / f"member_{slot:03d}.pt"
        member_path.write_bytes(f"member-{slot}".encode())
        members.append(
            {
                "slot": slot,
                "seed": 1000 + slot,
                "file": member_path.name,
                "sha256": hashlib.sha256(member_path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_kind": model_kind,
        "features": list(FEATURES),
        "model_features": list(MODEL_FEATURES),
        "toolkit": {"pinned_commit": TOOLKIT_COMMIT},
        "ratio_convention": {"orientation": "target_to_reference"},
        "architecture": {},
        "ensemble": {"size": 4},
        "members": members,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


class ArtifactContractTest(unittest.TestCase):
    def _load(self, path: Path, *, expected_kind: str):
        with (
            patch.dict(sys.modules, {"torch": _fake_torch()}),
            patch(
                "BackgroundRemoval.common.toolkit_runtime_provenance",
                return_value={},
            ),
            patch(
                "BackgroundRemoval.common.build_toolkit_model",
                side_effect=lambda architecture: _FakeModel(),
            ),
        ):
            return ModelBundle.load(
                path, device_name="cpu", expected_model_kind=expected_kind
            )

    def test_loader_enforces_kind_checksums_and_independent_members(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name) / "artifact"
            manifest = _write_artifact(
                root, model_kind=MODEL_KIND_BACKGROUND_REMOVAL
            )
            bundle = self._load(
                root, expected_kind=MODEL_KIND_BACKGROUND_REMOVAL
            )
            self.assertEqual(len(bundle.models), 4)

            with self.assertRaisesRegex(ValueError, "incompatible"):
                self._load(root, expected_kind=MODEL_KIND_DATA_MC_CORRECTION)

            manifest["members"][0].pop("sha256")
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum is missing"):
                self._load(root, expected_kind=MODEL_KIND_BACKGROUND_REMOVAL)

            manifest = _write_replacement_manifest(root)
            manifest["members"][3]["file"] = manifest["members"][0]["file"]
            manifest["members"][3]["sha256"] = manifest["members"][0]["sha256"]
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate ensemble member"):
                self._load(root, expected_kind=MODEL_KIND_BACKGROUND_REMOVAL)

            manifest = _write_replacement_manifest(root)
            manifest["members"][3]["seed"] = manifest["members"][0]["seed"]
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "seeds must be present and unique"):
                self._load(root, expected_kind=MODEL_KIND_BACKGROUND_REMOVAL)

            manifest = _write_replacement_manifest(root)
            last_path = root / manifest["members"][3]["file"]
            last_path.write_bytes((root / manifest["members"][0]["file"]).read_bytes())
            manifest["members"][3]["sha256"] = manifest["members"][0]["sha256"]
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksums must be unique"):
                self._load(root, expected_kind=MODEL_KIND_BACKGROUND_REMOVAL)

    def test_overwrite_validation_rejects_the_other_artifact_kind(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name) / "artifact"
            _write_artifact(root, model_kind=MODEL_KIND_DATA_MC_CORRECTION)
            with self.assertRaisesRegex(ValueError, "different model kind"):
                _validate_existing_artifact(
                    root,
                    expected_model_kind=MODEL_KIND_BACKGROUND_REMOVAL,
                )


def _write_replacement_manifest(root: Path) -> dict:
    """Restore a valid manifest without replacing the already-written members."""

    members = []
    for slot in range(4):
        path = root / f"member_{slot:03d}.pt"
        members.append(
            {
                "slot": slot,
                "seed": 1000 + slot,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_kind": MODEL_KIND_BACKGROUND_REMOVAL,
        "features": list(FEATURES),
        "model_features": list(MODEL_FEATURES),
        "toolkit": {"pinned_commit": TOOLKIT_COMMIT},
        "ratio_convention": {"orientation": "target_to_reference"},
        "architecture": {},
        "ensemble": {"size": 4},
        "members": members,
    }


if __name__ == "__main__":
    unittest.main()
