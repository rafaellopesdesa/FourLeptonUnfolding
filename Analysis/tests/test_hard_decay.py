from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import pyhepmc

from Analysis.hard_decay import (
    HERWIG_DEFINITION, PYTHIA_ATTRIBUTE, PYTHIA_DEFINITION,
    extract_hard_decay, iterate_hard_decays,
)


def payload(*, angle: float = 0., flavor: int = 11) -> dict:
    x, y = 24. * math.cos(angle), 24. * math.sin(angle)
    return {
        "schema_version": 1, "source": PYTHIA_DEFINITION,
        "momentum_unit": "GEV", "status": "ok",
        "parent": [0., 0., 0., 120.],
        "bosons": [[0., 0., 36., 60.], [0., 0., -36., 60.]],
        "particles": [
            [flavor, x, y, 18., 30.], [-flavor, -x, -y, 18., 30.],
            [13, 0., 24., -18., 30.], [-13, 0., -24., -18., 30.],
        ],
    }


def graph(*, angle: float = 0., flavor: int = 11, copies: bool = True, mev: bool = False):
    data = payload(angle=angle, flavor=flavor)
    event = pyhepmc.GenEvent(pyhepmc.Units.MEV if mev else pyhepmc.Units.GEV, pyhepmc.Units.MM)
    event.event_number = 7
    factor = 1000. if mev else 1.

    def particle(pid, p4, status=11):
        return pyhepmc.GenParticle([component * factor for component in p4], pid, status)

    def vertex(parent, children):
        result = pyhepmc.GenVertex()
        result.add_particle_in(parent)
        for child in children:
            result.add_particle_out(child)
        event.add_vertex(result)
        return result

    higgs = particle(25, data["parent"])
    if copies:
        earlier = particle(25, data["parent"])
        vertex(earlier, [higgs])
    z_bosons = [particle(23, values) for values in data["bosons"]]
    h_vertex = vertex(higgs, z_bosons)
    leptons, z_vertices = [], []
    for index, z in enumerate(z_bosons):
        if copies:
            z_copy = particle(23, data["bosons"][index])
            vertex(z, [z_copy])
            z = z_copy
        pair = [particle(row[0], row[1:]) for row in data["particles"][index * 2:index * 2 + 2]]
        z_vertices.append(vertex(z, pair))
        leptons.extend(pair)
    if copies:
        # A radiating lepton's original momentum is retained. Its final copy
        # is deliberately different so choosing the stable lepton would fail.
        original = leptons[0]
        p4 = data["particles"][0][1:]
        reduced = particle(original.pid, [component * .8 for component in p4], status=1)
        photon = particle(22, [component * .2 for component in p4], status=1)
        vertex(original, [reduced, photon])
    return event, {"higgs": higgs, "h_vertex": h_vertex, "z_vertices": z_vertices, "leptons": leptons, "vertex": vertex, "particle": particle}


class NativeDecayTest(unittest.TestCase):
    def test_herwig_uses_original_copies_and_ignores_status_codes(self):
        event, _ = graph()
        record = extract_hard_decay(event, "herwig", HERWIG_DEFINITION)
        self.assertTrue(record.valid, record.status)
        self.assertEqual([pid for pid, _ in record.particles], [11, -11, 13, -13])
        self.assertAlmostEqual(record.particles[0][1].E, 30.)
        self.assertAlmostEqual(record.parent.E, 120.)

    def test_herwig_direct_tau_parent_is_kept_without_descending_to_electron(self):
        event, nodes = graph(flavor=15, copies=False)
        tau = nodes["leptons"][0]
        electron = nodes["particle"](11, [10., 0., 7.5, 12.5], status=1)
        neutrino = nodes["particle"](16, [14., 0., 10.5, 17.5], status=1)
        nodes["vertex"](tau, [electron, neutrino])
        record = extract_hard_decay(event, "herwig", HERWIG_DEFINITION)
        self.assertTrue(record.valid, record.status)
        self.assertEqual(record.particles[0][0], 15)
        self.assertAlmostEqual(record.particles[0][1].E, 30.)

    def test_mev_hepmc_is_normalized_to_gev(self):
        event, _ = graph(mev=True)
        record = extract_hard_decay(event, "herwig", HERWIG_DEFINITION)
        self.assertTrue(record.valid, record.status)
        self.assertAlmostEqual(record.parent.E, 120.)
        self.assertAlmostEqual(record.particles[0][1].px, 24.)

    def test_direct_photon_at_z_vertex_is_not_silently_removed(self):
        event, nodes = graph()
        nodes["z_vertices"][0].add_particle_out(nodes["particle"](22, [0., 0., 1., 1.], 1))
        record = extract_hard_decay(event, "herwig", HERWIG_DEFINITION)
        self.assertEqual(record.status, "non_two_body_or_radiative_z_decay")
        self.assertEqual(record.particles, ())

    def test_extra_higgs_daughter_and_radiative_copy_are_rejected(self):
        event, nodes = graph()
        nodes["h_vertex"].add_particle_out(nodes["particle"](22, [0., 0., 1., 1.], 1))
        self.assertEqual(extract_hard_decay(event, "herwig", HERWIG_DEFINITION).status, "non_zz_or_radiative_higgs_decay")
        event, nodes = graph()
        nodes["higgs"].production_vertex.add_particle_out(nodes["particle"](22, [0., 0., 1., 1.], 1))
        self.assertEqual(extract_hard_decay(event, "herwig", HERWIG_DEFINITION).status, "radiative_resonance_copy")

    def test_ambiguous_higgs_and_wrong_charge_flavor_are_rejected(self):
        event, nodes = graph()
        another = nodes["particle"](25, [0., 0., 0., 120.])
        nodes["vertex"](another, [nodes["particle"](23, [0., 0., 0., 120.])])
        self.assertEqual(extract_hard_decay(event, "herwig", HERWIG_DEFINITION).status, "ambiguous_higgs")
        for pid in (11, -13, 12):
            with self.subTest(pid=pid):
                event, nodes = graph()
                nodes["leptons"][1].pid = pid
                record = extract_hard_decay(event, "herwig", HERWIG_DEFINITION)
                self.assertFalse(record.valid)
                self.assertIn(record.status, {"non_ossf_z_decay", "noncharged_or_secondary_decay"})

    def test_momentum_nonclosure_is_explicit_and_has_no_partial_result(self):
        event, nodes = graph()
        nodes["leptons"][0].momentum = pyhepmc.FourVector(23., 0., 18., 30.)
        record = extract_hard_decay(event, "herwig", HERWIG_DEFINITION)
        self.assertEqual(record.status, "z_momentum_nonclosure")
        self.assertEqual(record.particles, ())
        self.assertIsNone(record.parent)

    def test_pythia_requires_snapshot_without_falling_back_to_good_graph(self):
        event, _ = graph()
        self.assertEqual(extract_hard_decay(event, "pythia", PYTHIA_DEFINITION).status, "missing_native_attribute")
        event.attributes[PYTHIA_ATTRIBUTE] = json.dumps(payload(angle=.25, flavor=15))
        record = extract_hard_decay(event, "pythia", PYTHIA_DEFINITION)
        self.assertTrue(record.valid, record.status)
        self.assertEqual(record.particles[0][0], 15)
        self.assertAlmostEqual(record.particles[0][1].px, 24. * math.cos(.25))

    def test_pythia_validates_schema_source_units_status_and_momenta(self):
        cases = [
            ({"schema_version": 2}, "unsupported_attribute_schema"),
            ({"source": "pythia_process_before_shower"}, "unexpected_attribute_source"),
            ({"momentum_unit": "MEV"}, "unexpected_attribute_units"),
            ({"status": "ambiguous_native_higgs"}, "ambiguous_native_higgs"),
            ({"parent": [0., 0., 0., float("nan")]}, "nonfinite_momentum"),
            ({"parent": [0., 0., 0., 121.]}, "higgs_momentum_nonclosure"),
        ]
        for replacement, expected in cases:
            with self.subTest(replacement=replacement):
                event, _ = graph()
                data = payload()
                data.update(replacement)
                event.attributes[PYTHIA_ATTRIBUTE] = json.dumps(data)
                record = extract_hard_decay(event, "pythia", PYTHIA_DEFINITION)
                self.assertEqual(record.status, expected)
                self.assertEqual(record.particles, ())
        event, _ = graph()
        event.attributes[PYTHIA_ATTRIBUTE] = "not JSON"
        self.assertEqual(extract_hard_decay(event, "pythia", PYTHIA_DEFINITION).status, "invalid_native_attribute")

    def test_unsupported_definitions_fail_as_configuration_errors(self):
        event, _ = graph()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            extract_hard_decay(event, "herwig", "herwig_dipole_after_sophty")

    def test_actual_ascii2_and_ascii3_keep_repeated_source_ids_as_distinct_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            for format_name in ("HepMC2", "HepMC3"):
                with self.subTest(format=format_name):
                    path = Path(temporary) / f"events-{format_name}.hepmc"
                    with pyhepmc.open(path, "w", format=format_name) as writer:
                        writer.write(graph(angle=0.)[0])
                        writer.write(graph(angle=.8)[0])
                    rows = list(iterate_hard_decays(path, "herwig", HERWIG_DEFINITION))
                    self.assertEqual([number for number, _ in rows], [7, 7])
                    self.assertTrue(all(record.valid for _, record in rows))
                    self.assertAlmostEqual(rows[0][1].particles[0][1].px, 24.)
                    self.assertAlmostEqual(rows[1][1].particles[0][1].px, 24. * math.cos(.8))

    def test_actual_ascii3_native_json_survives_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pythia.hepmc3"
            with pyhepmc.open(path, "w") as writer:
                for angle in (0., .8):
                    event, _ = graph()
                    event.attributes[PYTHIA_ATTRIBUTE] = json.dumps(payload(angle=angle))
                    writer.write(event)
            rows = list(iterate_hard_decays(path, "pythia", PYTHIA_DEFINITION))
            self.assertEqual([number for number, _ in rows], [7, 7])
            self.assertTrue(all(record.valid for _, record in rows))
            self.assertNotEqual(rows[0][1].particles[0][1].px, rows[1][1].particles[0][1].px)


if __name__ == "__main__":
    unittest.main()
