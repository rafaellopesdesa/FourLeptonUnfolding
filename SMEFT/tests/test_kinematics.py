from __future__ import annotations

from itertools import combinations
import math
import unittest

import numpy as np
import vector

from SMEFT.kinematics import (
    BornKinematicsError, COMPONENT_ORDER, REQUIRED_BRANCHES, reconstruct_born_momenta,
)
from Tools.four_lepton_kinematics import compute_paired_kinematics


def _coordinates(**changes):
    result = dict(born_m_ZZ=125., born_m_Z1=81., born_m_Z2=29.,
                  born_cos_theta1=.32, born_cos_theta2=-.57, born_Phi=1.12, born_type=2)
    result.update(changes)
    return result


def _vectors(rows):
    return tuple(vector.obj(E=e, px=x, py=y, pz=z) for e, x, y, z in rows)


def _observables(rows, flavor="electron"):
    return compute_paired_kinematics(*_vectors(rows), z1_flavor=flavor)


def _mass2(momentum):
    return momentum[..., 0] ** 2 - np.sum(momentum[..., 1:] ** 2, axis=-1)


def _angle_difference(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


class BornReconstructionTest(unittest.TestCase):
    def test_roundtrip_repository_angles_and_all_channel_charge_orders(self):
        pdgs = ([13, -13, 13, -13], [13, -13, 11, -11],
                [11, -11, 13, -13], [11, -11, 11, -11])
        for event_type, expected_pdgs in enumerate(pdgs):
            for phi in (-math.pi, -2.7, -.1, 0., 1.12, math.pi):
                with self.subTest(event_type=event_type, phi=phi):
                    columns = _coordinates(born_type=event_type, born_Phi=phi)
                    momenta, particle_ids = reconstruct_born_momenta(columns)
                    self.assertEqual(momenta.shape, (4, 4))
                    self.assertEqual(momenta.dtype, np.dtype(np.float64))
                    np.testing.assert_array_equal(particle_ids, expected_pdgs)
                    np.testing.assert_allclose(_mass2(momenta), 0., atol=1e-12)
                    np.testing.assert_allclose(momenta.sum(axis=0), [125., 0., 0., 0.], atol=1e-13)
                    kin = _observables(momenta, "muon" if event_type in {0, 1} else "electron")
                    for name in ("m_ZZ", "m_Z1", "m_Z2", "cos_theta1", "cos_theta2"):
                        self.assertAlmostEqual(getattr(kin, name), columns[f"born_{name}"], places=12)
                    self.assertAlmostEqual(_angle_difference(kin.Phi, phi), 0., places=12)
                    self.assertAlmostEqual(kin.cos_theta_star, 0., places=12)
                    self.assertAlmostEqual(kin.Phi1, 0., places=12)

    def test_same_flavor_reconstruction_preserves_cross_pair_invariants_and_handedness(self):
        # Start from independently specified massless four-vectors, in a lab
        # frame with recoil and nonzero rapidity, rather than the inverse map.
        particles = tuple(vector.obj(pt=pt, eta=eta, phi=phi, mass=0.)
                          for pt, eta, phi in ((46., .2, .1), (42., -.4, 3.),
                                               (14., .8, 1.2), (9., -.7, -1.4)))
        original = np.array([[p.E, p.px, p.py, p.pz] for p in particles])
        kin = compute_paired_kinematics(*particles, z1_flavor="electron")
        columns = {f"born_{name}": getattr(kin, name)
                   for name in ("m_ZZ", "m_Z1", "m_Z2", "cos_theta1", "cos_theta2", "Phi")}
        columns["born_type"] = 3
        reconstructed, particle_ids = reconstruct_born_momenta(columns)
        np.testing.assert_array_equal(particle_ids, [11, -11, 11, -11])
        for first, second in combinations(range(4), 2):
            self.assertAlmostEqual(_mass2(original[first] + original[second]),
                                   _mass2(reconstructed[first] + reconstructed[second]), places=9)
        # A sign error in Phi can leave scalar products intact but reverses
        # the oriented four-volume relevant to CP-odd interference.
        self.assertAlmostEqual(np.linalg.det(reconstructed) / np.linalg.det(original), 1., places=12)

    def test_global_proper_rotation_does_not_change_reconstructed_decay(self):
        columns = _coordinates(born_Phi=-1.8, born_cos_theta1=-.73, born_cos_theta2=.46)
        momenta, _ = reconstruct_born_momenta(columns)
        axis = np.array([.2, .7, -.4])
        axis /= np.linalg.norm(axis)
        angle = .93
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                          [-axis[1], axis[0], 0.]])
        rotation = np.eye(3) * math.cos(angle) + (1. - math.cos(angle)) * np.outer(axis, axis)
        rotation += math.sin(angle) * cross
        rotated = momenta.copy()
        rotated[:, 1:] = momenta[:, 1:] @ rotation.T
        kin = _observables(rotated)
        self.assertGreater(abs(kin.cos_theta_star), .1)
        extracted = {f"born_{name}": getattr(kin, name)
                     for name in ("m_ZZ", "m_Z1", "m_Z2", "cos_theta1", "cos_theta2", "Phi")}
        extracted["born_type"] = 2
        reconstructed, _ = reconstruct_born_momenta(extracted)
        np.testing.assert_allclose(reconstructed, momenta, rtol=2e-13, atol=2e-13)

    def test_vectorized_broadcasting_empty_batches_prefix_and_no_mutation(self):
        columns = _coordinates(born_m_ZZ=np.array([[125.], [160.]]),
                               born_cos_theta1=np.array([-.6, .2, .7]),
                               born_type=np.array([0, 1, 3]))
        original_cosines = columns["born_cos_theta1"].copy()
        momenta, particle_ids = reconstruct_born_momenta(columns)
        self.assertEqual(momenta.shape, (2, 3, 4, 4))
        self.assertEqual(particle_ids.shape, (2, 3, 4))
        np.testing.assert_array_equal(columns["born_cos_theta1"], original_cosines)
        for row in range(2):
            for column in range(3):
                kin = _observables(momenta[row, column])
                self.assertAlmostEqual(kin.m_ZZ, columns["born_m_ZZ"][row, 0], places=11)
                self.assertAlmostEqual(kin.cos_theta1, original_cosines[column], places=12)
        alternate = {name.removeprefix("born_"): value for name, value in columns.items()}
        np.testing.assert_array_equal(reconstruct_born_momenta(alternate, prefix="")[0], momenta)
        empty, ids = reconstruct_born_momenta(_coordinates(born_type=np.empty(0)))
        self.assertEqual(empty.shape, (0, 4, 4))
        self.assertEqual(ids.shape, (0, 4))
        self.assertEqual(COMPONENT_ORDER, ("E", "px", "py", "pz"))
        self.assertEqual(set(REQUIRED_BRANCHES), set(_coordinates()))

    def test_roundoff_cosine_tolerance_and_small_pair_masses(self):
        # The inverse remains defined at a polar endpoint even though the
        # forward angle extractor cannot define its collapsed decay plane.
        momenta, _ = reconstruct_born_momenta(_coordinates(born_cos_theta1=1. + 5e-13,
                                                          born_cos_theta2=-1. - 5e-13))
        self.assertTrue(np.isfinite(momenta).all())
        np.testing.assert_allclose(_mass2(momenta), 0., atol=1e-12)
        close, _ = reconstruct_born_momenta(_coordinates(born_m_ZZ=110. + 1e-8))
        np.testing.assert_allclose(close.sum(axis=0), [110. + 1e-8, 0., 0., 0.], atol=1e-13)
        self.assertTrue(np.isfinite(close).all())
        small, _ = reconstruct_born_momenta(_coordinates(born_m_Z1=.01, born_m_Z2=.02))
        np.testing.assert_allclose(_mass2(small), 0., atol=2e-12)
        self.assertTrue(np.isfinite(small).all())
        # The angle map is periodic, including ntuple values at the +/-pi seam.
        one, _ = reconstruct_born_momenta(_coordinates())
        repeated, _ = reconstruct_born_momenta(_coordinates(born_Phi=1.12 + 2 * math.pi))
        np.testing.assert_allclose(one, repeated, atol=2e-14)

    def test_massless_model_reproduces_stored_coordinates_from_small_mass_inputs(self):
        particles = tuple(vector.obj(pt=pt, eta=eta, phi=phi, mass=.105658)
                          for pt, eta, phi in ((46., .2, .1), (42., -.4, 3.),
                                               (14., .8, 1.2), (9., -.7, -1.4)))
        kin = compute_paired_kinematics(*particles, z1_flavor="muon")
        columns = {f"born_{name}": getattr(kin, name)
                   for name in ("m_ZZ", "m_Z1", "m_Z2", "cos_theta1", "cos_theta2", "Phi")}
        columns["born_type"] = 0
        momenta, _ = reconstruct_born_momenta(columns)
        np.testing.assert_allclose(_mass2(momenta), 0., atol=2e-12)
        actual = _observables(momenta, "muon")
        for name in ("m_ZZ", "m_Z1", "m_Z2", "cos_theta1", "cos_theta2", "Phi"):
            self.assertAlmostEqual(getattr(actual, name), getattr(kin, name), places=12)

    def test_invalid_masses_angles_types_and_columns_raise_clear_errors(self):
        cases = [({"born_m_ZZ": 110.}, "threshold"),
                 ({"born_m_ZZ": 109.}, "m_ZZ >"),
                 ({"born_m_Z1": 0.}, "positive"),
                 ({"born_m_Z2": -1.}, "positive"),
                 ({"born_m_ZZ": np.nan}, "finite"),
                 ({"born_Phi": np.inf}, "finite"),
                 ({"born_cos_theta1": 1.01}, "cos_theta1"),
                 ({"born_cos_theta2": -1.01}, "cos_theta2"),
                 ({"born_type": 1.5}, "integer"),
                 ({"born_type": -1}, "integer"),
                 ({"born_type": 9}, "integer"),
                 ({"born_type": 4}, "tau"),
                 ({"born_type": 8}, "tau"),
                 ({"born_m_Z1": "invalid"}, "numeric")]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(BornKinematicsError, message):
                reconstruct_born_momenta(_coordinates(**changes))
        incomplete = _coordinates()
        del incomplete["born_Phi"]
        with self.assertRaisesRegex(BornKinematicsError, "missing.*born_Phi"):
            reconstruct_born_momenta(incomplete)
        with self.assertRaisesRegex(BornKinematicsError, "incompatible"):
            reconstruct_born_momenta(_coordinates(born_type=np.zeros(2), born_Phi=np.zeros(3)))
        with self.assertRaisesRegex(BornKinematicsError, "flat row indices 1"):
            reconstruct_born_momenta(_coordinates(born_type=np.array([1, 4, 2])))


if __name__ == "__main__":
    unittest.main()
