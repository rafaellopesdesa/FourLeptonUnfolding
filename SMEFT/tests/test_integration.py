from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import uproot

from SMEFT.backend import NativeHBBackend
from SMEFT.build_backend import ensure_backend, sha256
from SMEFT.kinematics import reconstruct_born_momenta
from SMEFT.tests.test_application import fixture


REPO = Path(__file__).resolve().parents[2]


class StandaloneIntegrationTest(unittest.TestCase):
    def test_cli_native_backend_scan_matches_direct_amplitudes_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, output = directory / "data.root", directory / "weighted.root"
            original = fixture(source, sources=(2, 3, 2, 3, 1),
                               column_change=lambda a: a["born_type"].__setitem__(slice(0, 4), [0, 1, 2, 3]))
            digest = sha256(source)
            result = subprocess.run(
                [sys.executable, str(REPO / "SMEFT/reweight.py"), str(source), "-o", str(output),
                 "--cHB", "-.3", "0", ".7", "--lambda-tev", "2", "--step-size", "1 B"],
                cwd=directory, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sha256(source), digest)
            report = json.loads(result.stdout)
            self.assertEqual(report["status_counts"]["evaluated"], 4)
            self.assertEqual(report["status_counts"]["background_unity"], 1)
            p, ids = reconstruct_born_momenta({name: value[:4] for name, value in original.items()})
            backend = NativeHBBackend()
            denominator = backend.matrix_elements(p, ids, 0.)
            expected = np.column_stack([backend.matrix_elements(p, ids, x) / denominator for x in [-.075, 0., .175]])
            with uproot.open(output) as file:
                arrays = file["Analysis"].arrays(library="np")
                np.testing.assert_allclose(arrays["smeft_hb_ratio"][:4], expected, rtol=2.e-12)
                np.testing.assert_array_equal(arrays["smeft_hb_ratio"][:, 1], 1.)
                np.testing.assert_array_equal(arrays["smeft_hb_ratio"][4], 1.)
                np.testing.assert_allclose(arrays["smeft_hb_weight"], original["weight"][:, None] * arrays["smeft_hb_ratio"])
                for name, value in original.items():
                    np.testing.assert_equal(arrays[name], value)
                metadata = json.loads(str(file["smeft_hb_metadata"]))
                self.assertEqual(metadata["input_sha256"], digest)
                self.assertEqual(metadata["backend"]["generated"]["model"]["version"], "3.0.2")

    def test_existing_native_cache_is_reused_without_recompilation(self):
        library = ensure_backend()
        before = (library.stat().st_mtime_ns, sha256(library))
        self.assertEqual(ensure_backend(), library)
        self.assertEqual((library.stat().st_mtime_ns, sha256(library)), before)


if __name__ == "__main__":
    unittest.main()
