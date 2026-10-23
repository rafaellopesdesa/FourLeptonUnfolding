from __future__ import annotations

import math
import unittest

import numpy as np
import vector

from Analysis.born import (
    BORN_BRANCH_TYPES, KINEMATIC_FIELDS, born_observables, empty_columns,
    empty_observables, lhe_born_observables, output_schema,
)
from Analysis.lhe import LHEEvent, LHEParticle
from Analysis.selection import Lepton, choose_candidate
from Tools.born_projection import project_to_born
from Tools.four_lepton_kinematics import compute_paired_kinematics


def _quartet(first=11, second=13, *, soft_forward=False):
    values = [(1., 3.4, .1) if soft_forward else (46., .2, .1),
              (42., -.4, 3.), (14., .8, 1.2), (9., -.7, -1.4)]
    pdgs = (first, -first, second, -second)
    return tuple((pid, vector.obj(pt=pt, eta=eta, phi=phi,
                                 mass=1.77686 if abs(pid) == 15 else 0.))
                 for pid, (pt, eta, phi) in zip(pdgs, values))


def _lhe_particle(pid, p4, *, status=1):
    return LHEParticle(pid, status, 0, 0, 0, 0, float(p4.px), float(p4.py),
                       float(p4.pz), float(p4.E), float(p4.mass), 0., 9.)


def _lhe_event(particles):
    return LHEEvent(37, 10001, -2.5, 91.1876, .007297, .118, tuple(particles))


class BornObservablesTest(unittest.TestCase):
    def assertUnavailable(self, result):
        self.assertFalse(result["born_available"])
        self.assertFalse(result["born_kinematics_valid"])
        self.assertEqual(result["born_type"], -1)
        self.assertEqual(result["born_n_taus"], 0)
        self.assertTrue(all(math.isnan(result[name]) for name, dtype in BORN_BRANCH_TYPES.items()
                            if dtype.kind == "f"))

    def test_projection_removes_recoil_preserves_invariants_and_changes_beam_angle(self):
        particles = _quartet()
        original = compute_paired_kinematics(*(p4 for _, p4 in particles), z1_flavor="electron")
        before = tuple((p4.E, p4.px, p4.py, p4.pz) for _, p4 in particles)
        result = born_observables(iter(particles), source=1)
        self.assertTrue(result["born_available"])
        self.assertTrue(result["born_kinematics_valid"])
        self.assertEqual(result["born_type"], 2)
        self.assertEqual(result["born_source"], 1)
        self.assertEqual(result["born_shower_event_index"], -1)
        for name in ("m_Z1", "m_Z2", "m_ZZ", "y_ZZ", "cos_theta1", "cos_theta2", "Phi"):
            self.assertAlmostEqual(result[f"born_{name}"], getattr(original, name), places=12)
        for name in ("m_ZZ", "y_ZZ", "pT_ZZ"):
            self.assertAlmostEqual(result[f"born_input_{name}"], getattr(original, name), places=12)
        self.assertGreater(result["born_input_pT_ZZ"], 10.)
        self.assertAlmostEqual(result["born_pT_ZZ"], 0., delta=1e-12)
        self.assertGreater(abs(result["born_cos_theta_star"] - original.cos_theta_star), .05)
        self.assertEqual(before, tuple((p4.E, p4.px, p4.py, p4.pz) for _, p4 in particles))

    def test_pairing_matches_existing_analysis_before_projection_without_acceptance_cuts(self):
        for same_flavor, soft_forward in ((False, False), (True, False), (False, True)):
            with self.subTest(same_flavor=same_flavor, soft_forward=soft_forward):
                particles = _quartet(second=11 if same_flavor else 13, soft_forward=soft_forward)
                leptons = [Lepton(p4, "electron" if abs(pid) == 11 else "muon", -1 if pid > 0 else 1)
                           for pid, p4 in particles]
                if soft_forward:
                    self.assertIsNone(choose_candidate(leptons))
                candidate = choose_candidate(leptons, apply_lepton_cuts=False)
                projected = project_to_born(lepton.p4 for lepton in candidate.leptons)
                expected = compute_paired_kinematics(*projected, z1_flavor=candidate.z1_negative.flavor)
                result = born_observables(particles, source=2, shower_event_index=147)
                self.assertEqual(result["born_type"], candidate.event_type)
                self.assertEqual(result["born_shower_event_index"], 147)
                for name in KINEMATIC_FIELDS:
                    self.assertAlmostEqual(result[f"born_{name}"], getattr(expected, name), places=12)

    def test_zero_recoil_is_identity_for_all_observables(self):
        rows = [(30., 20., 10., 5.), (25., -10., 15., -4.),
                (20., 5., -7., 8.), (35., -15., -18., 12.)]
        particles = tuple((pid, vector.obj(E=e, px=x, py=y, pz=z))
                          for pid, (e, x, y, z) in zip((11, -11, 13, -13), rows))
        candidate = choose_candidate([Lepton(p4, "electron" if abs(pid) == 11 else "muon",
                                            -1 if pid > 0 else 1) for pid, p4 in particles],
                                     apply_lepton_cuts=False)
        original = candidate.kinematics()
        result = born_observables(particles, source=3, shower_event_index=5)
        self.assertTrue(result["born_kinematics_valid"])
        for name in KINEMATIC_FIELDS:
            self.assertAlmostEqual(result[f"born_{name}"], getattr(original, name), places=12)

    def test_tau_parent_channels_have_distinct_types_and_finite_kinematics(self):
        channels = {(13, 13): 0, (13, 11): 1, (11, 13): 2, (11, 11): 3,
                    (15, 15): 4, (15, 11): 5, (11, 15): 6, (15, 13): 7, (13, 15): 8}
        for (first, second), event_type in channels.items():
            with self.subTest(first=first, second=second):
                result = born_observables(_quartet(first, second), source=1)
                self.assertTrue(result["born_kinematics_valid"])
                self.assertEqual(result["born_type"], event_type)
                self.assertEqual(result["born_n_taus"], 2 * (first == 15) + 2 * (second == 15))
                self.assertAlmostEqual(result["born_pT_ZZ"], 0., delta=1e-12)
        # The generic explicit-pair math can label a hard tau. Reconstruction
        # and dressed e/mu selection intentionally retain their existing API.
        kin = compute_paired_kinematics(*(p4 for _, p4 in _quartet(15, 11)), z1_flavor="tau")
        self.assertEqual(kin.z1_flavor, "tau")
        with self.assertRaises(ValueError):
            Lepton(_quartet(15, 11)[0][1], "tau", -1)

    def test_complete_quartet_required_and_nonleptons_do_not_enter_projection(self):
        quartet = _quartet()
        cases = [(), quartet[:3], quartet + quartet[:1],
                 ((-11, quartet[0][1]), *quartet[1:])]
        for particles in cases:
            with self.subTest(particles=particles):
                result = born_observables(particles, source=2, shower_event_index=19)
                self.assertUnavailable(result)
                self.assertEqual(result["born_source"], 2)
                self.assertEqual(result["born_shower_event_index"], 19)
        result = born_observables((*quartet, (22, vector.obj(E=100., px=100., py=0., pz=0.))), source=1)
        expected = born_observables(quartet, source=1)
        self.assertEqual(result, expected)

    def test_lhe_wrapper_uses_only_direct_status_one_charged_leptons(self):
        parents = [_lhe_particle(pid, p4) for pid, p4 in _quartet(15, 11)]
        unrelated = [_lhe_particle(11, _quartet()[0][1], status=2),
                     _lhe_particle(-11, _quartet()[1][1], status=-1),
                     _lhe_particle(21, vector.obj(E=5., px=5., py=0., pz=0.))]
        result = lhe_born_observables(_lhe_event(parents + unrelated))
        self.assertTrue(result["born_kinematics_valid"])
        self.assertEqual(result["born_source"], 1)
        self.assertEqual(result["born_shower_event_index"], -1)
        self.assertEqual(result["born_n_taus"], 2)
        self.assertEqual(result["born_type"], 5)
        # An undecayed hard Higgs does not supply hard four-lepton decay angles.
        higgs = _lhe_particle(25, vector.obj(px=3., py=4., pz=20., mass=125.))
        result = lhe_born_observables(_lhe_event([higgs]))
        self.assertUnavailable(result)
        self.assertEqual(result["born_source"], 1)

    def test_lhe_wrapper_rejects_extra_colorless_particles_but_allows_qcd_recoil(self):
        leptons = [_lhe_particle(pid, p4) for pid, p4 in _quartet()]
        extra = vector.obj(E=5., px=5., py=0., pz=0.)
        for pid in (22, 12, -14, 16):
            with self.subTest(pid=pid):
                result = lhe_born_observables(_lhe_event(leptons + [_lhe_particle(pid, extra)]))
                self.assertUnavailable(result)
                self.assertEqual(result["born_source"], 1)
                self.assertEqual(result["born_shower_event_index"], -1)
        expected = lhe_born_observables(_lhe_event(leptons))
        for pid in (1, -2, 3, -4, 5, -6, 21):
            with self.subTest(pid=pid):
                result = lhe_born_observables(_lhe_event(leptons + [_lhe_particle(pid, extra)]))
                self.assertEqual(result, expected)

    def test_degenerate_angles_preserve_projected_and_input_masses(self):
        particles = [(11, vector.obj(px=40., py=0., pz=0., E=40.)),
                     (-11, vector.obj(px=-40., py=0., pz=0., E=40.)),
                     (13, vector.obj(px=0., py=20., pz=0., E=20.)),
                     (-13, vector.obj(px=0., py=-20., pz=0., E=20.))]
        result = born_observables(particles, source=1)
        self.assertTrue(result["born_available"])
        self.assertFalse(result["born_kinematics_valid"])
        self.assertAlmostEqual(result["born_m_Z1"], 80.)
        self.assertAlmostEqual(result["born_m_Z2"], 40.)
        self.assertAlmostEqual(result["born_m_ZZ"], 120.)
        self.assertAlmostEqual(result["born_input_m_ZZ"], 120.)
        self.assertAlmostEqual(result["born_y_ZZ"], 0.)
        self.assertAlmostEqual(result["born_pT_ZZ"], 0.)
        self.assertTrue(math.isnan(result["born_Phi"]))

    def test_nonfinite_and_unphysical_momenta_have_explicit_invalid_masks(self):
        quartet = _quartet()
        for bad in (dict(E=np.nan, px=0., py=0., pz=0.),
                    dict(E=1., px=np.inf, py=0., pz=0.),
                    dict(E=-100., px=0., py=0., pz=0.)):
            with self.subTest(bad=bad):
                result = born_observables(((11, bad), *quartet[1:]), source=1)
                self.assertUnavailable(result)
                self.assertEqual(result["born_source"], 1)
        collinear = [(pid, vector.obj(E=1., px=1., py=0., pz=0.)) for pid in (11, -11, 13, -13)]
        self.assertUnavailable(born_observables(collinear, source=1))

    def test_lhe_spacelike_lepton_is_rejected_even_when_quartet_is_timelike(self):
        quartet = _quartet()
        pid, first = quartet[0]
        malformed = vector.obj(E=.95 * first.E, px=first.px, py=first.py, pz=first.pz)
        self.assertLess(malformed.mass2, 0.)
        self.assertGreater(sum((p4 for _, p4 in quartet[1:]), malformed).mass2, 0.)
        particles = [_lhe_particle(pid, malformed),
                     *(_lhe_particle(pid, p4) for pid, p4 in quartet[1:])]
        result = lhe_born_observables(_lhe_event(particles))
        self.assertUnavailable(result)
        self.assertEqual(result["born_source"], 1)

        # Input rounding within the native record's 1e-7 GeV + 1e-6 E
        # tolerance should not invalidate an otherwise massless lepton.
        rounded = vector.obj(E=first.E - .5 * (1e-7 + 1e-6 * first.E),
                             px=first.px, py=first.py, pz=first.pz)
        result = born_observables(((pid, rounded), *quartet[1:]), source=1)
        self.assertTrue(result["born_available"])
        self.assertTrue(result["born_kinematics_valid"])

    def test_schema_defaults_and_provenance_validation(self):
        columns = empty_columns(3)
        self.assertEqual(set(columns), set(output_schema()))
        self.assertEqual(set(empty_observables()), set(columns))
        self.assertEqual(set(born_observables(_quartet(), source=1)), set(columns))
        for name, values in columns.items():
            self.assertEqual(values.dtype, BORN_BRANCH_TYPES[name])
            if values.dtype.kind == "f":
                self.assertTrue(np.isnan(values).all())
            elif name in {"born_type", "born_shower_event_index"}:
                np.testing.assert_array_equal(values, -1)
            else:
                self.assertFalse(values.any())
        self.assertUnavailable(born_observables([], source=0))
        schema = output_schema()
        schema.clear()
        self.assertTrue(output_schema())
        self.assertEqual(empty_columns(0)["born_pT_ZZ"].shape, (0,))
        for size in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                empty_columns(size)
        for source, index in ((-1, -1), (4, -1), (True, -1), (1, 1), (0, 1),
                              (2, -1), (3, 0), (2, 2**63), (2, 1.5)):
            with self.subTest(source=source, index=index), self.assertRaises(ValueError):
                born_observables([], source=source, shower_event_index=index)
        with self.assertRaises(ValueError):
            born_observables(_quartet(), source=0)


if __name__ == "__main__":
    unittest.main()
