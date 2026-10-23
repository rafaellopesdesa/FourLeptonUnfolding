from __future__ import annotations

from itertools import combinations
import math
import unittest

import numpy as np
import vector

from Tools.born_projection import BORN_PROJECTION_METADATA, BornProjectionError, project_to_born


def _vectors(rows):
    return tuple(vector.obj(E=float(e), px=float(x), py=float(y), pz=float(z)) for e, x, y, z in rows)


def _array(momenta):
    return np.array([[p.E, p.px, p.py, p.pz] for p in momenta], dtype=np.longdouble)


def _active_boost(rows, beta):
    """Independent conventional Lorentz boost for inverse-map construction."""
    rows = np.asarray(rows, dtype=np.longdouble)
    beta = np.asarray(beta, dtype=np.longdouble)
    beta2 = beta @ beta
    gamma = 1 / np.sqrt(1 - beta2)
    dot = rows[:, 1:] @ beta
    result = rows.copy()
    result[:, 0] = gamma * (rows[:, 0] + dot)
    result[:, 1:] += (((gamma - 1) * dot / beta2 + gamma * rows[:, 0])[:, None] * beta)
    return result


def _longitudinal(rows, rapidity):
    rows = np.asarray(rows, dtype=np.longdouble)
    c, s = np.cosh(np.longdouble(rapidity)), np.sinh(np.longdouble(rapidity))
    result = rows.copy()
    result[:, 0] = rows[:, 0] * c + rows[:, 3] * s
    result[:, 3] = rows[:, 3] * c + rows[:, 0] * s
    return result


def _rest_frame():
    spatial = np.array([[20., 0., 10.], [-10., 15., -5.],
                        [5., -7., 4.], [-15., -8., -9.]], dtype=np.longdouble)
    # Small unequal masses also test preservation of individual virtualities.
    masses = np.array([.1, .2, .3, .4], dtype=np.longdouble)
    return np.column_stack([np.sqrt(np.sum(spatial * spatial, axis=1) + masses * masses), spatial])


def _example(rapidity=1.3, transverse=(25., -35.)):
    rest = _rest_frame()
    mass = rest[:, 0].sum()
    kx, ky = transverse
    mt = np.sqrt(mass * mass + kx * kx + ky * ky)
    recoil = _active_boost(rest, [kx / mt, ky / mt, 0.])
    born = _longitudinal(rest, rapidity)
    full = _longitudinal(recoil, rapidity)
    return _vectors(full), _vectors(born)


def _mass2(row):
    row = np.asarray(row, dtype=np.longdouble)
    return (row[0] + row[3]) * (row[0] - row[3]) - row[1] ** 2 - row[2] ** 2


def _rapidity(row):
    return float((np.log(row[0] + row[3]) - np.log(row[0] - row[3])) / 2)


class BornProjectionTest(unittest.TestCase):
    def test_inverse_map_roundtrip_and_all_lorentz_invariants(self):
        original, born = _example()
        before = _array(original)
        result = project_to_born(original)
        after = _array(result)
        np.testing.assert_allclose(after, _array(born), rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(_array(original), before)
        total_before, total_after = before.sum(axis=0), after.sum(axis=0)
        np.testing.assert_allclose(total_after[1:3], 0., atol=2e-13)
        self.assertAlmostEqual(_rapidity(total_after), _rapidity(total_before), places=13)
        self.assertAlmostEqual(float(_mass2(total_after)), float(_mass2(total_before)), places=9)
        for first, second in combinations(range(4), 2):
            self.assertAlmostEqual(float(_mass2(after[first] + after[second])),
                                   float(_mass2(before[first] + before[second])), places=9)
        for first, second in zip(before, after):
            self.assertAlmostEqual(float(_mass2(first)), float(_mass2(second)), places=10)

    def test_nonzero_rapidity_distinguishes_naive_transverse_boost(self):
        original, _ = _example(rapidity=1.3)
        before = _array(original)
        total = before.sum(axis=0)
        naive = _active_boost(before, [-total[1] / total[0], -total[2] / total[0], 0.])
        correct = _array(project_to_born(original))
        self.assertGreater(abs(_rapidity(naive.sum(axis=0)) - _rapidity(total)), .05)
        self.assertAlmostEqual(_rapidity(correct.sum(axis=0)), _rapidity(total), places=13)
        self.assertFalse(np.allclose(correct, naive))

    def test_identity_without_recoil_returns_copies(self):
        # Exact cancellation avoids introducing floating residual recoil.
        rows = np.array([[50., 10., 20., 30.], [50., -10., -20., 30.]])
        original = _vectors(rows)
        result = project_to_born(iter(original))
        np.testing.assert_array_equal(_array(result), _array(original))
        self.assertIsNot(result[0], original[0])
        np.testing.assert_array_equal(_array(project_to_born(result)), _array(result))

    def test_rotation_covariance_about_beam_axis(self):
        original, _ = _example()
        angle = .73
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        rotated = _array(original)
        rotated[:, 1:3] = rotated[:, 1:3] @ rotation.T
        expected = _array(project_to_born(original))
        expected[:, 1:3] = expected[:, 1:3] @ rotation.T
        actual = _array(project_to_born(_vectors(rotated)))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=2e-13)

    def test_high_rapidity_remains_stable_and_preserves_input_invariants(self):
        for rapidity in (-10., 10.):
            with self.subTest(rapidity=rapidity):
                original, born = _example(rapidity=rapidity)
                projected = _array(project_to_born(original))
                before = _array(original)
                np.testing.assert_allclose(projected, _array(born), rtol=2e-8, atol=2e-6)
                np.testing.assert_allclose(projected.sum(axis=0)[1:3], 0., atol=1e-12)
                self.assertAlmostEqual(_rapidity(projected.sum(axis=0)),
                                       _rapidity(before.sum(axis=0)), delta=2e-8)
                np.testing.assert_allclose(float(_mass2(projected.sum(axis=0))),
                                           float(_mass2(before.sum(axis=0))), rtol=2e-8)

    def test_small_recoil_is_continuous(self):
        original, born = _example(transverse=(1e-10, -2e-10))
        result = _array(project_to_born(original))
        np.testing.assert_allclose(result, _array(born), rtol=1e-13, atol=2e-13)

    def test_unphysical_or_unresolvable_totals_fail_explicitly(self):
        cases = [
            ([], "at least one"),
            ([dict(E=np.nan, px=0., py=0., pz=0.)], "finite"),
            ([dict(E=1., px=np.inf, py=0., pz=0.)], "finite"),
            ([dict(E=-1., px=0., py=0., pz=0.)], "positive"),
            ([dict(E=1., px=2., py=0., pz=0.)], "spacelike"),
            ([dict(E=1., px=0., py=0., pz=1.)], "lightlike"),
            ([dict(E=1., px=1. - 1e-15, py=0., pz=0.)], "near-lightlike"),
        ]
        for momenta, message in cases:
            with self.subTest(momenta=momenta), self.assertRaisesRegex(BornProjectionError, message):
                project_to_born(momenta)

    def test_reference_metadata_identifies_exact_mapping(self):
        self.assertEqual(BORN_PROJECTION_METADATA["arxiv_url"], "https://arxiv.org/abs/2606.11083")
        self.assertEqual(BORN_PROJECTION_METADATA["equations"], [14, 15, 16])


if __name__ == "__main__":
    unittest.main()
