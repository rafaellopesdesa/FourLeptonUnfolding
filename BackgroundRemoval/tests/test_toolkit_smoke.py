from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from BackgroundRemoval.common import MODEL_FEATURES, build_toolkit_model


TOOLKIT_AVAILABLE = importlib.util.find_spec("nsbi_common_utils") is not None


@unittest.skipUnless(
    TOOLKIT_AVAILABLE,
    "the lightweight Analysis environment does not install the NSBI toolkit",
)
class ToolkitSmokeTest(unittest.TestCase):
    def test_pinned_model_class_constructs_and_produces_logits(self):
        import torch

        architecture = {
            "hidden_layers": 2,
            "neurons": 8,
            "learning_rate": 1.0e-3,
            "learning_rate_decay": 0.98,
        }
        model = build_toolkit_model(architecture)
        modules = tuple(model.modules())
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in modules))
        inputs = torch.from_numpy(
            np.zeros((3, len(MODEL_FEATURES)), dtype=np.float32)
        )
        with torch.inference_mode():
            output = model(inputs)
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
