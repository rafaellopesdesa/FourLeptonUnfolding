from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import vector

from Analysis.lhe import (
    KINEMATIC_FIELDS, LHE_BRANCH_TYPES, LHEEvent, LHEParticle, empty_columns,
    event_observables, iter_lhe_events, output_schema, source_sha256,
)
from Analysis.selection import Lepton, choose_candidate


def _particle(pdg_id, p4, *, status=1, mothers=(0, 0)):
    return LHEParticle(pdg_id, status, *mothers, 0, 0,
                       float(p4.px), float(p4.py), float(p4.pz), float(p4.E),
                       float(p4.mass), 0., 9.)


def _leptons(*, soft_forward=False, same_flavor=False):
    values = [(1., 3.4, .1) if soft_forward else (46., .2, .1),
              (42., -.4, 3.), (14., .8, 1.2), (9., -.7, -1.4)]
    pdgs = [11, -11, 11 if same_flavor else 13, -11 if same_flavor else -13]
    return tuple(_particle(pdg, vector.obj(pt=pt, eta=eta, phi=phi, mass=0.))
                 for pdg, (pt, eta, phi) in zip(pdgs, values))


def _event(particles, ordinal=1):
    return LHEEvent(ordinal, 10001, -2.5, 91.1876, .007297, .118, tuple(particles))


def _row(particle):
    values = (particle.pdg_id, particle.status, particle.mother1, particle.mother2,
              particle.color1, particle.color2, particle.px, particle.py, particle.pz,
              particle.energy, particle.mass, particle.lifetime, particle.spin)
    return " ".join(str(value) for value in values)


def _event_text(particles, *, header=None, extra=""):
    header = header or f"{len(particles)} 10001 -2.5 91.1876 .007297 .118"
    return "<event>\n" + header + "\n" + "\n".join(map(_row, particles)) + "\n" + extra + "\n</event>\n"


class LHEParserTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "events.lhe"

    def write(self, body, *, close=True):
        self.path.write_text('<LesHouchesEvents version="3.0">\n<header/>\n<init/>\n'
                             + body + ("</LesHouchesEvents>\n" if close else ""),
                             encoding="utf-8")
        return self.path

    def test_signed_header_particle_record_comments_and_xml_weights(self):
        incoming = LHEParticle(21, -1, 0, 0, 501, 502, 0., 0., 100., 100., 0., 0., 9.)
        higgs = _particle(25, vector.obj(px=3., py=4., pz=20., mass=125.), mothers=(1, 1))
        text = _event_text([incoming, higgs],
                          header="2 10001 -2.5D+00 9.11876d+01 7.297D-03 .118 # header comment",
                          extra='<rwgt><wgt id="1001">99.0</wgt></rwgt>\n<scales muf="91"/>\n# extra comment')
        text = text.replace(_row(incoming), '<!-- comment\n spanning lines -->\n' + _row(incoming) + " # particle comment")
        self.write(text + _event_text([]))
        events = list(iter_lhe_events(self.path))
        self.assertEqual([event.ordinal for event in events], [1, 2])
        first = events[0]
        self.assertEqual(first.process_id, 10001)
        self.assertEqual(first.weight, -2.5)
        self.assertAlmostEqual(first.scale, 91.1876)
        self.assertAlmostEqual(first.aqed, .007297)
        self.assertEqual(first.nup, 2)
        self.assertEqual(first.particles[0], incoming)
        self.assertEqual(first.particles[1].mothers, (1, 1))
        self.assertAlmostEqual(first.particles[1].p4.pt, 5.)
        self.assertEqual(events[1].particles, ())

    def test_gzip_and_original_byte_provenance(self):
        self.write(_event_text(_leptons()))
        compressed = self.path.with_suffix(".lhe.gz")
        with gzip.open(compressed, "wb") as stream:
            stream.write(self.path.read_bytes())
        self.assertEqual(list(iter_lhe_events(compressed)), list(iter_lhe_events(self.path)))
        self.assertEqual(source_sha256(compressed), hashlib.sha256(compressed.read_bytes()).hexdigest())
        self.assertNotEqual(source_sha256(compressed), source_sha256(self.path))

    def test_iterator_yields_complete_event_before_truncated_xml(self):
        self.write(_event_text(_leptons()), close=False)
        iterator = iter_lhe_events(self.path)
        self.assertEqual(next(iterator).ordinal, 1)
        with self.assertRaisesRegex(ValueError, "malformed LHE XML"):
            next(iterator)

    def test_malformed_headers_counts_and_particle_fields_are_rejected(self):
        particle = _leptons()[0]
        cases = [
            ("1 10001 nan 1 .01 .1", [_row(particle)], "nonfinite"),
            ("1 10001 1 inf .01 .1", [_row(particle)], "nonfinite"),
            ("1 10001 1 1 .01", [_row(particle)], "six"),
            ("-1 10001 1 1 .01 .1", [], "nonnegative"),
            ("2 10001 1 1 .01 .1", [_row(particle)], "NUP=2"),
            ("0 10001 1 1 .01 .1", [_row(particle)], "NUP=0"),
            ("1 2147483648 1 1 .01 .1", [_row(particle)], "int32"),
            ("1 10001 1 1 .01 .1", ["11 1 0 0 0 0 0 0 0 1 0 0"], "13"),
            ("1 10001 1 1 .01 .1", ["11 1 0 0 0 0 nan 0 0 1 0 0 9"], "nonfinite"),
            ("1 10001 1 1 .01 .1", ["11 1 0 2 0 0 0 0 0 1 0 0 9"], "mother"),
        ]
        for header, rows, message in cases:
            with self.subTest(header=header, rows=rows):
                self.write("<event>\n" + header + "\n" + "\n".join(rows) + "\n</event>")
                with self.assertRaisesRegex(ValueError, "LHE event 1:.*" + message):
                    list(iter_lhe_events(self.path))

    def test_event_groups_are_streamed_and_invalid_root_is_rejected(self):
        self.write('<eventgroup nreal="1">' + _event_text([]) + _event_text([]) + "</eventgroup>")
        self.assertEqual([event.ordinal for event in iter_lhe_events(self.path)], [1, 2])
        self.path.write_text("<other/>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "LesHouchesEvents"):
            list(iter_lhe_events(self.path))


class LHEObservablesTest(unittest.TestCase):
    def test_four_lepton_fields_follow_existing_pairing_conventions_without_cuts(self):
        for soft_forward, same_flavor in ((False, False), (True, False), (False, True)):
            with self.subTest(soft_forward=soft_forward, same_flavor=same_flavor):
                particles = _leptons(soft_forward=soft_forward, same_flavor=same_flavor)
                leptons = [Lepton(p.p4, "electron" if abs(p.pdg_id) == 11 else "muon",
                                  -1 if p.pdg_id > 0 else 1) for p in particles]
                if soft_forward:
                    self.assertIsNone(choose_candidate(leptons))
                candidate = choose_candidate(leptons, apply_lepton_cuts=False)
                expected = candidate.kinematics()
                result = event_observables(_event(particles, ordinal=37))
                self.assertTrue(result["lhe_available"])
                self.assertTrue(result["lhe_has_four_leptons"])
                self.assertTrue(result["lhe_kinematics_valid"])
                self.assertEqual(result["lhe_n_final_leptons"], 4)
                self.assertEqual(result["lhe_type"], candidate.event_type)
                self.assertEqual(result["lhe_event_index"], 37)
                self.assertEqual(result["lhe_weight"], -2.5)
                for field in KINEMATIC_FIELDS:
                    self.assertAlmostEqual(result[f"lhe_{field}"], getattr(expected, field))

    def test_undecayed_higgs_has_separate_kinematics_not_four_lepton_values(self):
        higgs = _particle(25, vector.obj(px=3., py=4., pz=20., mass=125.))
        result = event_observables(_event([higgs]))
        self.assertTrue(result["lhe_has_higgs"])
        self.assertTrue(result["lhe_higgs_kinematics_valid"])
        self.assertAlmostEqual(result["lhe_higgs_m"], 125.)
        self.assertAlmostEqual(result["lhe_higgs_pT"], 5.)
        self.assertAlmostEqual(result["lhe_higgs_y"], float(higgs.p4.rapidity))
        self.assertFalse(result["lhe_has_four_leptons"])
        self.assertFalse(result["lhe_kinematics_valid"])
        self.assertEqual(result["lhe_type"], -1)
        self.assertTrue(all(math.isnan(result[f"lhe_{field}"]) for field in KINEMATIC_FIELDS))

    def test_higgs_status_copies_and_ambiguous_higgs_records(self):
        p4 = vector.obj(px=3., py=4., pz=20., mass=125.)
        parent = _particle(25, p4, status=2)
        child = _particle(25, p4, mothers=(1, 1))
        self.assertTrue(event_observables(_event([parent, child]))["lhe_higgs_kinematics_valid"])
        result = event_observables(_event([parent, _particle(25, p4)]))
        self.assertTrue(result["lhe_has_higgs"])
        self.assertFalse(result["lhe_higgs_kinematics_valid"])
        self.assertTrue(math.isnan(result["lhe_higgs_m"]))

    def test_tau_hard_record_and_nonfinal_electrons_are_not_shower_decays(self):
        taus = [_particle(pdg, vector.obj(px=2., py=3., pz=1., mass=1.777)) for pdg in (15, -15, 15, -15)]
        result = event_observables(_event(taus))
        self.assertTrue(result["lhe_has_taus"])
        self.assertEqual(result["lhe_n_final_leptons"], 0)
        self.assertFalse(result["lhe_has_four_leptons"])
        direct = list(_leptons())
        direct[0] = _particle(direct[0].pdg_id, direct[0].p4, status=2)
        result = event_observables(_event(direct))
        self.assertEqual(result["lhe_n_final_leptons"], 3)
        self.assertFalse(result["lhe_has_four_leptons"])

    def test_degenerate_angles_keep_available_masses_but_validity_false(self):
        particles = [
            _particle(11, vector.obj(px=40., py=0., pz=0., E=40.)),
            _particle(-11, vector.obj(px=-40., py=0., pz=0., E=40.)),
            _particle(13, vector.obj(px=0., py=20., pz=0., E=20.)),
            _particle(-13, vector.obj(px=0., py=-20., pz=0., E=20.)),
        ]
        result = event_observables(_event(particles))
        self.assertTrue(result["lhe_has_four_leptons"])
        self.assertFalse(result["lhe_kinematics_valid"])
        self.assertAlmostEqual(result["lhe_m_ZZ"], 120.)
        self.assertAlmostEqual(result["lhe_m_Z1"], 80.)
        self.assertAlmostEqual(result["lhe_m_Z2"], 40.)
        self.assertTrue(math.isnan(result["lhe_Phi"]))

    def test_missing_schema_defaults_are_explicit_and_independent(self):
        columns = empty_columns(3)
        self.assertEqual(set(columns), set(output_schema()))
        self.assertEqual(set(event_observables(_event([]))), set(columns))
        for name, values in columns.items():
            self.assertEqual(values.dtype, LHE_BRANCH_TYPES[name])
            if values.dtype.kind == "f":
                self.assertTrue(np.isnan(values).all())
            elif name in {"lhe_event_index", "lhe_type"}:
                np.testing.assert_array_equal(values, -1)
            else:
                self.assertFalse(values.any())
        schema = output_schema()
        schema.clear()
        self.assertTrue(output_schema())
        self.assertEqual(empty_columns(0)["lhe_weight"].shape, (0,))


if __name__ == "__main__":
    unittest.main()
