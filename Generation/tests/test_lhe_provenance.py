from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


GENERATION = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generation_lhe_provenance", GENERATION / "lhe_provenance.py")
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def lhe_text(count: int = 3) -> str:
    events = "".join(
        f"<event>\n1 1 {'-1' if index % 2 else '1'} 125 0.007 0.1\n"
        "25 1 0 0 0 0 0 0 0 125 125 0 9\n</event>\n"
        for index in range(count)
    )
    return (
        '<LesHouchesEvents version="3.0">\n'
        "<!--\n<event>\nA commented event is not a source event.\n</event>\n-->\n"
        "<init>\n2212 2212 6.8D3 6.8e3 0 0 0 0 -4 1\n1 0 1 1\n</init>\n"
        + events + "</LesHouchesEvents>\n"
    )


def native_attribute(status="ok") -> str:
    payload = {
        "schema_version": 1, "source": "pythia_event_before_decay_fsr",
        "momentum_unit": "GEV", "status": status,
        "parent": [0., 0., 0., 120.] if status == "ok" else [],
        "bosons": [[0., 0., 36., 60.], [0., 0., -36., 60.]] if status == "ok" else [],
        "particles": [[11, 24., 0., 18., 30.], [-11, -24., 0., 18., 30.],
                      [13, 0., 24., -18., 30.], [-13, 0., -24., -18., 30.]] if status == "ok" else [],
    }
    return "A 0 FourLeptonHardDecay " + json.dumps(payload) + "\n"


def hepmc_text(ids=(1, 3, 3), version=3, native_attributes=False) -> str:
    listing = "Asciiv3" if version == 3 else "IO_GenEvent"
    attribute = native_attribute() if native_attributes else ""
    return (
        f"HepMC::Version {version}.0.0\nHepMC::{listing}-START_EVENT_LISTING\n"
        + "".join(f"E {number} 0 0\n{attribute}" for number in ids)
        + f"HepMC::{listing}-END_EVENT_LISTING\n"
    )


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lhe = self.root / "events.lhe"
        self.lhe.write_text(lhe_text(), encoding="utf-8")
        self.info = self.root / "lhe-source-info.json"
        self.hepmc = self.root / "events.hepmc3"
        self.hepmc.write_text(hepmc_text(), encoding="utf-8")
        self.manifest = self.root / "lhe-provenance.json"

    def test_herwig_temporary_tags_preserve_every_original_byte_and_weight(self):
        original = self.lhe.read_bytes()
        copy = self.root / "tagged.lhe"
        info = provenance.prepare_lhe(self.lhe, self.info, tagged_output=copy)
        self.assertEqual(info["lhe_event_count"], 3)
        self.assertEqual(info["lhe_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(self.lhe.read_bytes(), original)
        tagged = copy.read_text()
        for number in range(1, 4):
            tag = f"<event num>{number}</event num>\n"
            self.assertEqual(tagged.count(tag), 1)
            tagged = tagged.replace(tag, "")
        self.assertEqual(tagged.encode(), original)

    def test_gzip_hashes_raw_source_and_decompresses_for_showering(self):
        compressed = self.root / "events.lhe.gz"
        with gzip.open(compressed, "wb") as stream:
            stream.write(self.lhe.read_bytes())
        copy = self.root / "plain.lhe"
        info = provenance.prepare_lhe(compressed, self.info, plain_output=copy)
        self.assertEqual(info["lhe_sha256"], hashlib.sha256(compressed.read_bytes()).hexdigest())
        self.assertEqual(copy.read_bytes(), self.lhe.read_bytes())
        self.assertEqual(provenance.beam_energies(compressed), (6800.0, 6800.0))

    def test_explicit_ids_allow_skips_and_repeated_source_showers_in_both_formats(self):
        provenance.prepare_lhe(self.lhe, self.info)
        for version in (2, 3):
            with self.subTest(version=version):
                self.hepmc.write_text(hepmc_text(version=version))
                result = provenance.finalize_provenance(self.info, self.hepmc, "herwig", self.manifest)
                self.assertEqual(result["event_number"], "lhe_event_index_1based")
                self.assertEqual(result["lhe_file"], str(self.lhe.resolve()))
                self.assertEqual(result["hepmc_event_count"], 3)
                self.assertEqual(result["hepmc_unique_source_events"], 2)
                self.assertEqual(result["hepmc_repeated_source_events"], 1)
                self.assertEqual(result["hepmc_format"], f"HepMC{version}")
                self.assertEqual(provenance.verify_provenance(self.manifest, self.hepmc), result)

    def test_zero_negative_out_of_range_and_noninteger_ids_fail_closed(self):
        for number in (0, -1, 4, "1.0", "not-an-id"):
            with self.subTest(number=number):
                self.hepmc.write_text(hepmc_text((number,)))
                with self.assertRaisesRegex(ValueError, "event record|outside"):
                    provenance.inspect_hepmc(self.hepmc, 3)
        self.hepmc.write_text(hepmc_text(()))
        with self.assertRaisesRegex(ValueError, "empty"):
            provenance.inspect_hepmc(self.hepmc, 3)
        self.hepmc.write_text(hepmc_text().replace("HepMC::Asciiv3-END_EVENT_LISTING", ""))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            provenance.inspect_hepmc(self.hepmc, 3)

    def test_changed_original_cannot_be_certified_after_showering(self):
        provenance.prepare_lhe(self.lhe, self.info)
        self.lhe.write_text(lhe_text().replace("125 125", "126 126"))
        with self.assertRaisesRegex(RuntimeError, "original LHE changed"):
            provenance.finalize_provenance(self.info, self.hepmc, "pythia", self.manifest)
        self.assertFalse(self.manifest.exists())

    def test_process_definitions_are_explicit_and_legacy_api_does_not_claim_born_support(self):
        provenance.prepare_lhe(self.lhe, self.info)
        legacy = provenance.finalize_provenance(self.info, self.hepmc, "pythia", self.manifest)
        self.assertNotIn("hard_decay_definition", legacy)
        self.hepmc.write_text(hepmc_text(native_attributes=True))
        for process, shower, definition in (
            ("gg_H", "pythia", "pythia_event_before_decay_fsr"),
            ("gg_H", "herwig", "herwig_qtilde_original_decay"),
            ("ZZ", "pythia", "lhe_status1_charged_leptons"),
            ("ZZ", "herwig", "lhe_status1_charged_leptons"),
        ):
            with self.subTest(process=process, shower=shower):
                result = provenance.finalize_provenance(self.info, self.hepmc, shower, self.manifest, process=process)
                self.assertEqual(result["process"], process)
                self.assertEqual(result["hard_decay_definition"], definition)
                if process == "gg_H" and shower == "pythia":
                    self.assertEqual(result["hard_decay_attribute"], "FourLeptonHardDecay")
                    self.assertEqual(result["hard_decay_momentum_unit"], "GEV")

    def test_gg_h_pythia_requires_exactly_one_valid_native_attribute_per_event(self):
        provenance.prepare_lhe(self.lhe, self.info)
        valid = hepmc_text(native_attributes=True)
        attribute = native_attribute()
        cases = {
            "old bridge": hepmc_text(),
            "one missing snapshot": valid.replace(attribute, "", 1),
            "particle attribute only": valid.replace("A 0 FourLeptonHardDecay", "A 1 FourLeptonHardDecay"),
            "duplicate snapshot": valid.replace(attribute, attribute + attribute, 1),
            "malformed JSON": valid.replace(attribute, "A 0 FourLeptonHardDecay not-json\n", 1),
            "wrong stage": valid.replace("pythia_event_before_decay_fsr", "pythia_process_before_shower", 1),
            "wrong units": valid.replace('"momentum_unit": "GEV"', '"momentum_unit": "MEV"', 1),
            "empty success arrays": valid.replace(attribute, native_attribute("missing_higgs").replace('"status": "missing_higgs"', '"status": "ok"'), 1),
            "nonfinite JSON": valid.replace("120.0", "NaN", 1),
            "unsupported format": hepmc_text(version=2, native_attributes=True),
        }
        for label, content in cases.items():
            with self.subTest(label=label):
                self.hepmc.write_text(content)
                with self.assertRaisesRegex(ValueError, "FourLeptonHardDecay"):
                    provenance.finalize_provenance(self.info, self.hepmc, "pythia", self.manifest, process="gg_H")
                self.assertFalse(self.manifest.exists())

    def test_explicit_native_failure_status_is_valid_transport(self):
        provenance.prepare_lhe(self.lhe, self.info)
        self.hepmc.write_text(hepmc_text(native_attributes=True).replace(native_attribute(), native_attribute("invalid_z_decay"), 1))
        result = provenance.finalize_provenance(self.info, self.hepmc, "pythia", self.manifest, process="gg_H")
        self.assertEqual(result["hard_decay_attribute_event_count"], 3)
        self.assertEqual(result["hepmc_repeated_source_events"], 1)

    def test_native_attribute_requirement_does_not_apply_to_zz_or_herwig(self):
        provenance.prepare_lhe(self.lhe, self.info)
        for process, shower in (("ZZ", "pythia"), ("gg_H", "herwig")):
            with self.subTest(process=process, shower=shower):
                result = provenance.finalize_provenance(self.info, self.hepmc, shower, self.manifest, process=process)
                self.assertNotIn("hard_decay_attribute_event_count", result)

    def test_verification_rejects_mutated_hepmc(self):
        provenance.prepare_lhe(self.lhe, self.info)
        provenance.finalize_provenance(self.info, self.hepmc, "pythia", self.manifest)
        self.hepmc.write_text(hepmc_text((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "HepMC checksum"):
            provenance.verify_provenance(self.manifest, self.hepmc)

    def test_incomplete_or_preannotated_sources_leave_no_copy_or_manifest(self):
        copy = self.root / "private.lhe"
        for content in (
            lhe_text().replace("</LesHouchesEvents>", ""),
            lhe_text().replace("</event>", "<event num>42</event num>\n</event>"),
        ):
            with self.subTest(content=content[-60:]):
                self.lhe.write_text(content)
                with self.assertRaises(ValueError):
                    provenance.prepare_lhe(self.lhe, self.info, tagged_output=copy)
                self.assertFalse(copy.exists())
                self.assertFalse(self.info.exists())

    def test_source_and_preexisting_private_files_are_never_overwritten(self):
        original = self.lhe.read_bytes()
        with self.assertRaises(ValueError):
            provenance.prepare_lhe(self.lhe, self.lhe)
        copy = self.root / "existing.lhe"
        copy.write_text("keep me")
        with self.assertRaises(FileExistsError):
            provenance.prepare_lhe(self.lhe, self.info, tagged_output=copy)
        self.assertEqual(copy.read_text(), "keep me")
        self.assertEqual(self.lhe.read_bytes(), original)


class GenerationRunnerProvenanceTest(unittest.TestCase):
    def test_both_runner_branches_publish_original_source_identity_and_clean_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = root / "Generation"
            generation.mkdir()
            for name in ("run_generation.sh", "lhe_provenance.py"):
                shutil.copyfile(GENERATION / name, generation / name)
            binaries = root / "prefix" / "bin"
            binaries.mkdir(parents=True)
            process = root / "powheg" / "gg_H"
            process.mkdir(parents=True)
            executable = process / "pwhg_main"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            (generation / "env.sh").write_text(
                f'export POWHEG_ROOT="{process.parent}"\nexport GENERATION_PREFIX="{binaries.parent}"\n'
            )
            fakegit = binaries / "git"
            fakegit.write_text("#!/bin/sh\nprintf 'fixture-commit\\n'\n")
            fakegit.chmod(0o755)
            pythia = binaries / "powheg-pythia8"
            pythia.write_text(
                "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                "source = Path(sys.argv[1]).read_text()\n"
                "assert '<event num>' not in source\n"
                f"Path(sys.argv[2]).write_text({hepmc_text(native_attributes=True)!r})\n"
            )
            pythia.chmod(0o755)
            herwig = binaries / "Herwig"
            herwig.write_text(
                "#!/usr/bin/env python3\nimport sys, re\nfrom pathlib import Path\n"
                "config = Path('herwig.in').read_text()\n"
                "assert 'set LesHouchesHandler:EventNumbering LHE' in config\n"
                "assert 'get /Herwig/Shower/ShowerHandler:SplittingGenerator' in config\n"
                "assert 'set /Herwig/Analysis/HepMC:PrintEvent 3' in config\n"
                "source_path = re.search(r'LesHouchesReader:FileName (.+)', config).group(1)\n"
                "source = Path(source_path).read_text()\n"
                "assert all(source.count(f'<event num>{i}</event num>') == 1 for i in (1,2,3))\n"
                "if sys.argv[1] == 'run':\n"
                f"    Path('events.hepmc').write_text({hepmc_text(version=2)!r})\n"
            )
            herwig.chmod(0o755)
            source = root / "original.lhe.gz"
            with gzip.open(source, "wb") as stream:
                stream.write(lhe_text().encode())
            original = source.read_bytes()
            env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
            for shower in ("pythia", "herwig"):
                with self.subTest(shower=shower):
                    output = root / shower
                    subprocess.run([
                        "bash", str(generation / "run_generation.sh"), "gg_H", shower,
                        "--lhe", str(source), "--events", "0", "--output-dir", str(output),
                    ], check=True, capture_output=True, text=True, env=env)
                    manifest = json.loads((output / "lhe-provenance.json").read_text())
                    self.assertEqual(manifest["lhe_file"], str(source.resolve()))
                    self.assertEqual(manifest["lhe_sha256"], hashlib.sha256(original).hexdigest())
                    self.assertEqual(manifest["shower"], shower)
                    self.assertEqual(manifest["process"], "gg_H")
                    self.assertEqual(manifest["hard_decay_definition"], "pythia_event_before_decay_fsr" if shower == "pythia" else "herwig_qtilde_original_decay")
                    self.assertEqual(manifest["hepmc_unique_source_events"], 2)
                    self.assertEqual(manifest["hepmc_repeated_source_events"], 1)
                    self.assertEqual(list(output.glob(".shower-lhe.*")), [])
                    self.assertIn(f"input_lhe={source}", (output / "run-metadata.txt").read_text())
                    self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
