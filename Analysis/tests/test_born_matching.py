from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyhepmc
import uproot
import vector

from Analysis.hard_decay import HERWIG_DEFINITION, PYTHIA_ATTRIBUTE, PYTHIA_DEFINITION
from Analysis.lhe_matching import file_sha256, load_lhe_source
from Analysis.tests.test_hard_decay import graph, payload
from Analysis.tests.test_lhe import _event_text, _leptons
from Analysis.tests.test_lhe_matching import REPO, write_inputs
from Analysis.validate_analysis_output import validate_analysis_output


def _lab(values):
    p = vector.obj(px=values[0], py=values[1], pz=values[2], E=values[3])
    p = p.rotateY(.65).boost_beta3(vector.obj(x=.2, y=-.1, z=.3))
    return [float(p.px), float(p.py), float(p.pz), float(p.E)]


def write_native(directory, shower, *, numbers=(4, 2, 2), hepmc_numbers=None,
                 attribute=True, failed_decay=False):
    lhe, root, manifest_path, manifest = write_inputs(directory, numbers=numbers)
    path = directory / ("events.hepmc3" if shower == "pythia" else "events.hepmc")
    definition = PYTHIA_DEFINITION if shower == "pythia" else HERWIG_DEFINITION
    event_numbers = numbers if hepmc_numbers is None else hepmc_numbers
    with pyhepmc.open(path, "w", format="HepMC3" if shower == "pythia" else "HepMC2") as writer:
        for index, number in enumerate(event_numbers):
            angle = .2 + .4 * index
            event, _ = graph(angle=angle)
            event.event_number = number
            for particle in event.particles:
                p = particle.momentum
                particle.momentum = pyhepmc.FourVector(*_lab([p.px, p.py, p.pz, p.e]))
            if shower == "pythia" and attribute:
                data = payload(angle=angle)
                data["parent"] = _lab(data["parent"])
                data["bosons"] = [_lab(p) for p in data["bosons"]]
                data["particles"] = [[row[0], *_lab(row[1:])] for row in data["particles"]]
                if failed_decay:
                    data.update(status="non_zz_or_radiative_higgs_decay", parent=[], bosons=[], particles=[])
                event.attributes[PYTHIA_ATTRIBUTE] = json.dumps(data)
            writer.write(event)
    manifest.update(process="gg_H", shower=shower, hard_decay_definition=definition,
                    hepmc_file=str(path), hepmc_sha256=file_sha256(path),
                    hepmc_event_count=len(event_numbers))
    manifest_path.write_text(json.dumps(manifest))
    return lhe, root, manifest_path, manifest, path


def build(root, output, *args):
    return subprocess.run(
        [sys.executable, str(REPO / "Analysis/build_analysis_tree.py"), str(root),
         "-o", str(output), "--require-lhe", "--require-born", "--step-size", "1 B", *args],
        text=True, capture_output=True,
    )


class BornMatchingTest(unittest.TestCase):
    def test_qqzz_projection_keeps_raw_lhe_and_signed_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            lhe, root, manifest_path, manifest = write_inputs(base)
            lhe.write_text('<LesHouchesEvents version="3.0"><header/><init/>\n'
                           + _event_text(_leptons()) * 4 + '</LesHouchesEvents>\n')
            manifest.update(lhe_sha256=file_sha256(lhe), process="ZZ", shower="pythia",
                            hard_decay_definition="lhe_status1_charged_leptons")
            manifest_path.write_text(json.dumps(manifest))
            output = base / "analysis.root"
            result = build(root, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = validate_analysis_output(output, require_lhe=True, require_born=True)
            self.assertEqual(summary.born_kinematics_valid_entries, 3)
            with uproot.open(output) as file:
                a = file["Analysis"].arrays(library="np")
                np.testing.assert_array_equal(a["born_source"], 1)
                np.testing.assert_array_equal(a["born_shower_event_index"], -1)
                np.testing.assert_allclose(a["lhe_weight"], -2.5)
                np.testing.assert_allclose(a["weight"], [.01, .02, .03])
                for name in ("m_Z1", "m_Z2", "m_ZZ", "y_ZZ"):
                    np.testing.assert_allclose(a[f"born_{name}"], a[f"lhe_{name}"], atol=1e-12)
                np.testing.assert_allclose(a["born_pT_ZZ"], 0., atol=1e-12)
                np.testing.assert_allclose(a["born_input_pT_ZZ"], a["lhe_pT_ZZ"])
                self.assertTrue(np.all(a["born_input_pT_ZZ"] > 1.))
                source = json.loads(str(file["born_metadata"]))["sources"][0]
                self.assertEqual(source["decay_status_counts"], {"lhe_quartet": 3, "unavailable_lhe_quartet": 0})

    def test_native_repeated_lhe_id_retains_distinct_decays_across_chunks(self):
        for shower, source in (("pythia", 2), ("herwig", 3)):
            with self.subTest(shower=shower), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                _, root, _, _, _ = write_native(base, shower)
                output = base / "analysis.root"
                result = build(root, output)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = validate_analysis_output(output, require_lhe=True, require_born=True)
                self.assertEqual(summary.born_kinematics_valid_entries, 3)
                with uproot.open(output) as file:
                    a = file["Analysis"].arrays(library="np")
                    np.testing.assert_array_equal(a["lhe_event_index"], [4, 2, 2])
                    np.testing.assert_array_equal(a["born_shower_event_index"], [1, 2, 3])
                    np.testing.assert_array_equal(a["born_source"], source)
                    self.assertFalse(np.any(a["lhe_has_four_leptons"]))
                    self.assertTrue(np.all(a["born_kinematics_valid"]))
                    self.assertNotAlmostEqual(a["born_Phi"][1], a["born_Phi"][2])
                    np.testing.assert_allclose(a["born_m_ZZ"], 120., atol=1e-10)
                    np.testing.assert_allclose(a["born_pT_ZZ"], 0., atol=1e-10)
                    np.testing.assert_allclose(a["born_y_ZZ"], a["born_input_y_ZZ"], atol=1e-12)
                    self.assertTrue(np.all(a["born_input_pT_ZZ"] > 1.))
                    np.testing.assert_allclose(a["weight"], [.01, .02, .03])
                    record = json.loads(str(file["born_metadata"]))["sources"][0]
                    self.assertEqual(record["decay_status_counts"], {"ok": 3})

    def test_native_hash_sequence_count_and_attribute_fail_closed(self):
        for mutation in ("hash", "sequence", "short", "trailing", "attribute"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                numbers = {"sequence": (4, 2, 3), "short": (4, 2), "trailing": (4, 2, 2, 1)}.get(mutation)
                _, root, manifest_path, manifest, _ = write_native(
                    base, "pythia", hepmc_numbers=numbers, attribute=mutation != "attribute")
                if mutation == "hash":
                    manifest["hepmc_sha256"] = "0" * 64
                if mutation in {"short", "trailing"}:
                    manifest["hepmc_event_count"] = 3
                manifest_path.write_text(json.dumps(manifest))
                with uproot.open(root) as file, self.assertRaises(ValueError):
                    load_lhe_source(root, file["Delphes"], require_born=True)

    def test_relocated_hepmc_and_delphes_prefix_remain_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, root, _, _, hepmc = write_native(base, "pythia", numbers=(4, 2), hepmc_numbers=(4, 2, 2))
            moved = base / "relocated.hepmc3"
            hepmc.rename(moved)
            output = base / "analysis.root"
            result = build(root, output, "--hepmc", str(moved))
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = validate_analysis_output(output, require_born=True)
            self.assertEqual(summary.born_source_entries, 2)
            with uproot.open(output) as file:
                source = json.loads(str(file["born_metadata"]))["sources"][0]
                self.assertEqual(source["hepmc_file"], str(moved))
                self.assertEqual(source["decay_status_counts"], {"ok": 2})

    def test_certified_unsupported_decay_is_explicit_but_missing_definition_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, root, manifest_path, manifest, _ = write_native(base, "pythia", failed_decay=True)
            output = base / "analysis.root"
            result = build(root, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = validate_analysis_output(output, require_born=True)
            self.assertEqual(summary.born_source_entries, 3)
            self.assertEqual(summary.born_available_entries, 0)
            with uproot.open(output) as file:
                self.assertTrue(np.isnan(file["Analysis"]["born_m_ZZ"].array(library="np")).all())
            del manifest["hard_decay_definition"]
            manifest_path.write_text(json.dumps(manifest))
            before = output.read_bytes()
            result = build(root, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No certified Born input definition", result.stderr)
            self.assertEqual(output.read_bytes(), before)

    def test_analysis_cannot_overwrite_native_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, root, _, _, hepmc = write_native(base, "pythia")
            before = hepmc.read_bytes()
            result = build(root, hepmc)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite", result.stderr)
            self.assertEqual(hepmc.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
