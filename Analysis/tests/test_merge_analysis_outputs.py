from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import uproot

from Analysis.merge_analysis_outputs import merge_directory


def write_sample(path: Path, cross_section_pb: float) -> None:
    with uproot.recreate(path) as root_file:
        root_file["Analysis"] = {
            "event_id": np.arange(4, dtype=np.uint64),
            "event_number": np.arange(10, 14, dtype=np.int64),
            "weight": np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64),
            "cross_section_pb": np.full(4, cross_section_pb, dtype=np.float64),
            "fiducial": np.array([True, False, True, False]),
            "reconstructed": np.array([True, False, True, True]),
            "type": np.array([0, 1, 2, 3], dtype=np.int8),
            "reco_m_ZZ": np.array([120.0, 121.0, 122.0, 123.0], dtype=np.float32),
        }


class MergeAnalysisOutputsTest(unittest.TestCase):
    def test_merge_and_unit_weight_pseudo_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for sample in ("ZZ_pythia", "ZZ_herwig", "gg_H_pythia", "gg_H_herwig"):
                cross_section = 1.0e-5
                write_sample(directory / f"{sample}_seed101.root", cross_section)
                write_sample(directory / f"{sample}_seed102.root", cross_section)

            merge_directory(
                directory,
                directory,
                luminosity_fb=300.0,
                seed=7,
                step_size="1 MB",
                zz_cross_section_pb=None,
                higgs_cross_section_pb=None,
                overwrite=False,
            )

            for sample in ("ZZ_pythia", "ZZ_herwig", "gg_H_pythia", "gg_H_herwig"):
                with uproot.open(directory / f"{sample}.root") as root_file:
                    tree = root_file["Analysis"]
                    arrays = tree.arrays(library="np")
                    self.assertEqual(tree.num_entries, 8)
                    self.assertAlmostEqual(float(np.sum(arrays["weight"])), 8.0)
                    self.assertEqual(arrays["event_id"].tolist(), list(range(8)))

            with uproot.open(directory / "data.root") as root_file:
                tree = root_file["Analysis"]
                arrays = tree.arrays(library="np")
                self.assertGreater(tree.num_entries, 0)
                self.assertTrue(np.all(arrays["reconstructed"]))
                self.assertTrue(np.all(arrays["weight"] == 1.0))
                self.assertEqual(arrays["event_id"].tolist(), list(range(tree.num_entries)))
                metadata = json.loads(str(root_file["merge_metadata"]))
                self.assertEqual(metadata["total_observed"], tree.num_entries)


if __name__ == "__main__":
    unittest.main()
