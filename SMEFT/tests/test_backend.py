"""Physics and state regressions using the actual offline native backend.

The trace fixtures below were calculated independently with explicit Dirac
gamma traces and analytic SM/HB neutral-current amplitudes, without HELAS or
MadGraph code. They use MW=80.387, MZ=91.1876, GF=1.166379e-5,
GammaZ=2.4952 and Lambda=1000 GeV, with outgoing e-,e+,mu-,mu+.
The fourth point has a 0.2 GeV pair and probes the photon contribution.
"""

from __future__ import annotations

from itertools import permutations
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from SMEFT.backend import NativeHBBackend
from SMEFT.build_backend import ensure_backend, sha256, source_fingerprint
from SMEFT.kinematics import reconstruct_born_momenta


_TRACE_MOMENTA = np.array([
    [[48.12532929253381, 39.45556709768598, 0., 27.555499351577954],
     [39.45764336715122, -39.45556709768598, 0., .4047778270755935],
     [25.32417880714107, 5.004675778915091, 9.832985068021847, -22.794290014300795],
     [12.182848533173905, -5.004675778915091, -9.832985068021847, -5.165987164352756]],
    [[51.46141148614831, 43.25851332396896, 0., 27.874323258990113],
     [43.284403926754386, -43.25851332396896, 0., -1.4968795190097257],
     [21.37079157244404, 3.002805467349055, 5.899791040813109, -20.31960524795806],
     [8.973393014653261, -3.002805467349055, -5.899791040813109, -6.057838492022327]],
    [[15.751535731284038, 14.999999999999996, 0., -4.807377444503163],
     [50.39087413361323, -14.999999999999996, 0., 48.106550447414655],
     [20.813960466969064, -10.787403665554818, -14.819983878411422, -9.8600684744352],
     [38.13362966813366, 10.787403665554818, 14.819983878411422, -33.43910452847629]],
    [[49.574137184279515, 47.490499049809955, 0., 14.221377484820666],
     [49.04472963161304, -47.490499049809955, 0., 12.249000148502814],
     [2.912119315057571, -.060760572859142, .01497173289331925, -2.911446874859829],
     [23.559013869049885, .060760572859142, -.01497173289331925, -23.55893075846365]],
])
_TRACE_MSQ = np.array([
    [2.2170893049856847e-6, 2.721389865430055e-6, 2.0672098365508496e-6],
    [5.7978729474091795e-5, 7.143203499808325e-5, 6.52918806834329e-5],
    [2.3780282150707277e-7, 7.467016602150276e-8, 7.973394353013305e-7],
    [2.453625379092314e-6, .00977679077593585, .009759980234699484],
])
_TRACE_LINEAR = [.14753127612138053, .05295178395892689, -1.519471604037158,
                 3.4256536021375252]
_TRACE_QUADRATIC = [.07992936757498446, .1790868558322945, .8334719407374604,
                    3980.204950256649]


def _coordinates(**changes):
    result = dict(born_m_ZZ=125., born_m_Z1=81., born_m_Z2=29.,
                  born_cos_theta1=.32, born_cos_theta2=-.57, born_Phi=1.12,
                  born_type=np.arange(4))
    result.update(changes)
    return result


def _points():
    # Three nonsingular decay configurations, in each of the four ntuple types.
    return reconstruct_born_momenta(_coordinates(
        born_type=np.tile(np.arange(4), 3),
        born_m_ZZ=np.repeat([125., 160., 200.], 4),
        born_m_Z1=np.repeat([81., 90., 92.], 4),
        born_m_Z2=np.repeat([29., 41., 80.], 4),
        born_cos_theta1=np.repeat([.32, -.73, .11], 4),
        born_cos_theta2=np.repeat([-.57, .46, -.19], 4),
        born_Phi=np.repeat([1.12, -1.8, 2.7], 4)))


def _dot(first, second):
    return first[..., 0] * second[..., 0] - np.sum(first[..., 1:] * second[..., 1:], axis=-1)


def _sm_trace(momenta):
    """Independent closed SM current trace for distinguishable e/mu pairs."""
    mw, mz, gf, width = 80.387, 91.1876, 1.166379e-5, 2.4952
    v = 1. / math.sqrt(math.sqrt(2.) * gf)
    cosine = mw / mz
    sine = math.sqrt(1. - cosine * cosine)
    charge = 2. * mw * sine / v
    left = charge * (.5 - sine * sine) / (sine * cosine)
    right = -charge * sine / cosine
    first, second, third, fourth = np.moveaxis(momenta, 1, 0)
    pair1, pair2 = first + second, third + fourth
    denominator = ((_dot(pair1, pair1) - mz * mz) ** 2 + (mz * width) ** 2)
    denominator *= ((_dot(pair2, pair2) - mz * mz) ** 2 + (mz * width) ** 2)
    currents = ((left ** 4 + right ** 4) * _dot(first, third) * _dot(second, fourth)
                + 2. * left * left * right * right * _dot(first, fourth) * _dot(second, third))
    return 16. * (2. * mz * mz / v) ** 2 * currents / denominator


def _hb_dirac_trace(momenta, coefficient):
    """Independent spin sum, including correlated ZZ, Zgamma and gammagamma.

    With B=c_w A-s_w Z, expanding +(C_HB/Lambda^2)(H†H)B_mu_nu B^mu_nu
    gives the tensor T_mn=q2_m q1_n-(q1.q2)g_mn and a common factor
    4*C_HB*v/Lambda^2. Each neutral-current leg then contributes
    (c_w*e/s_i - s_w*g_chiral/D_Zi). This derives the relative photon/Z
    terms independently of generated vertices, propagators and helicities.
    The SM is (2*mZ^2/v)*g_x*g_y*g_mn/(D_Z1*D_Z2).
    """
    metric = np.diag([1., -1., -1., -1.])
    identity, zero = np.eye(2, dtype=complex), np.zeros((2, 2), dtype=complex)
    pauli = [np.array([[0., 1.], [1., 0.]], dtype=complex),
             np.array([[0., -1j], [1j, 0.]]), np.diag([1., -1.])]
    gamma = np.array([np.block([[identity, zero], [zero, -identity]])]
                     + [np.block([[zero, sigma], [-sigma, zero]]) for sigma in pauli])
    gamma5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    projectors = [(np.eye(4) - gamma5) / 2., (np.eye(4) + gamma5) / 2.]

    def current_trace(minus, plus, chirality):
        slash_minus = np.einsum("m,mab->ab", metric @ minus, gamma)
        slash_plus = np.einsum("m,mab->ab", metric @ plus, gamma)
        projector = projectors[chirality]
        return np.array([[np.trace(slash_minus @ gamma[mu] @ projector
                                   @ slash_plus @ gamma[nu] @ projector)
                          for nu in range(4)] for mu in range(4)])

    mw, mz, gf, width, cutoff = 80.387, 91.1876, 1.166379e-5, 2.4952, 1000.
    v = 1. / math.sqrt(math.sqrt(2.) * gf)
    cosine = mw / mz
    sine = math.sqrt(1. - cosine * cosine)
    charge = 2. * mw * sine / v
    coupling = [charge * (.5 - sine * sine) / (sine * cosine), -charge * sine / cosine]
    first_pair, second_pair = momenta[0] + momenta[1], momenta[2] + momenta[3]
    first_s, second_s = _dot(first_pair, first_pair), _dot(second_pair, second_pair)
    first_z = first_s - mz * mz + 1j * mz * width
    second_z = second_s - mz * mz + 1j * mz * width
    tensor = np.outer(metric @ second_pair, metric @ first_pair) - _dot(first_pair, second_pair) * metric
    result = 0j
    for first_chirality in range(2):
        for second_chirality in range(2):
            sm = (2. * mz * mz / v * coupling[first_chirality] * coupling[second_chirality]
                  / (first_z * second_z))
            hb = (4. * coefficient * v / cutoff ** 2
                  * (cosine * charge / first_s - sine * coupling[first_chirality] / first_z)
                  * (cosine * charge / second_s - sine * coupling[second_chirality] / second_z))
            amplitude = sm * metric + hb * tensor
            first_trace = current_trace(momenta[0], momenta[1], first_chirality)
            second_trace = current_trace(momenta[2], momenta[3], second_chirality)
            result += np.einsum("mn,rs,mr,ns->", amplitude, amplitude.conj(), first_trace, second_trace)
    if abs(result.imag) > 1e-12 * abs(result.real):
        raise AssertionError("independent spin sum has a nonzero imaginary part")
    return result.real


_COMPILER = shlex.split(os.environ.get("CXX", "c++"))


@unittest.skipUnless(_COMPILER and shutil.which(_COMPILER[0]), "a C++17 compiler is required")
class NativeHBBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="test-hb-native-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.build_dir = Path(cls.temporary.name)
        cls.backend = NativeHBBackend(build_dir=cls.build_dir)

    def test_independent_dirac_trace_absolute_amplitudes_and_coefficients(self):
        ids = np.tile([11, -11, 13, -13], (len(_TRACE_MOMENTA), 1))
        for column, coefficient in enumerate((0., 1., -1.)):
            with self.subTest(coefficient=coefficient):
                result = self.backend.matrix_elements(_TRACE_MOMENTA, ids, coefficient)
                # The near-photon point loses precision when small pair
                # invariants are formed; measured error is below 1.4e-10.
                np.testing.assert_allclose(result, _TRACE_MSQ[:, column], rtol=1e-9, atol=0.)
        linear, quadratic = self.backend.evaluate(_TRACE_MOMENTA, ids)
        np.testing.assert_allclose(linear, _TRACE_LINEAR, rtol=1e-9, atol=2e-12)
        np.testing.assert_allclose(quadratic, _TRACE_QUADRATIC, rtol=1e-9, atol=2e-12)

    def test_sm_limit_matches_independent_lorentz_dot_formula(self):
        momenta, ids = _points()
        mixed = np.isin(np.arange(len(momenta)) % 4, [1, 2])
        np.testing.assert_allclose(self.backend.matrix_elements(momenta[mixed], ids[mixed], 0.),
                                   _sm_trace(momenta[mixed]), rtol=2e-12, atol=0.)

    def test_arbitrary_coefficient_matches_derived_neutral_current_dirac_trace(self):
        ids = np.tile([11, -11, 13, -13], (len(_TRACE_MOMENTA), 1))
        for coefficient in (-.73, 0., .37, 1.9):
            with self.subTest(coefficient=coefficient):
                independent = np.array([_hb_dirac_trace(point, coefficient) for point in _TRACE_MOMENTA])
                np.testing.assert_allclose(self.backend.matrix_elements(_TRACE_MOMENTA, ids, coefficient),
                                           independent, rtol=1e-9, atol=0.)
                if coefficient == 0.:
                    np.testing.assert_allclose(independent, _sm_trace(_TRACE_MOMENTA), rtol=2e-12, atol=0.)

    def test_all_channels_arbitrary_coefficient_polynomial_and_sm_unity(self):
        momenta, ids = _points()
        linear, quadratic = self.backend.evaluate(momenta, ids)
        self.assertEqual(linear.shape, (12,))
        self.assertEqual(quadratic.shape, (12,))
        self.assertTrue(np.isrealobj(linear) and np.isrealobj(quadratic))
        self.assertTrue(np.all(np.isfinite(linear) & np.isfinite(quadratic)))
        sm = self.backend.matrix_elements(momenta, ids, 0.)
        self.assertTrue(np.all(sm > 0.))
        for coefficient in (-5.7, -1.3, -.07, 0., .31, 2.4, 9.):
            with self.subTest(coefficient=coefficient):
                direct = self.backend.matrix_elements(momenta, ids, coefficient) / sm
                expected = 1. + coefficient * linear + coefficient ** 2 * quadratic
                np.testing.assert_allclose(direct, expected, rtol=3e-12, atol=3e-12)
                if coefficient == 0.:
                    np.testing.assert_array_equal(direct, np.ones(len(momenta)))

    def test_amplitude_positivity_including_polynomial_minimum(self):
        momenta, ids = _points()
        linear, quadratic = self.backend.evaluate(momenta, ids)
        self.assertTrue(np.all(quadratic > 0.))
        minimum = 1. - linear ** 2 / (4. * quadratic)
        self.assertTrue(np.all(minimum >= -1e-12))
        sm = self.backend.matrix_elements(momenta, ids, 0.)
        for index, coefficient in enumerate(-linear / (2. * quadratic)):
            direct = self.backend.matrix_elements(momenta[index:index + 1], ids[index:index + 1],
                                                   coefficient)[0] / sm[index]
            self.assertGreaterEqual(direct, 0.)
            self.assertAlmostEqual(direct, minimum[index], places=12)

    def test_flavor_universality_and_z1_muon_pair_mapping(self):
        momenta, ids = reconstruct_born_momenta(_coordinates())
        np.testing.assert_array_equal(ids[1], [13, -13, 11, -11])
        np.testing.assert_array_equal(ids[2], [11, -11, 13, -13])
        for result in (*self.backend.evaluate(momenta, ids),
                       self.backend.matrix_elements(momenta, ids, .71)):
            self.assertAlmostEqual(result[0], result[3], delta=2e-12 * abs(result[0]))
            self.assertAlmostEqual(result[1], result[2], delta=2e-12 * abs(result[1]))
        # Reordering type1 to the generated e-,e+,mu-,mu+ order must carry
        # momenta with the PDGs; it must not reinterpret Z1 as electrons.
        order = [2, 3, 0, 1]
        canonical = self.backend.matrix_elements(momenta[1:2, order], ids[1:2, order], -.6)
        np.testing.assert_allclose(canonical, self.backend.matrix_elements(momenta[1:2], ids[1:2], -.6),
                                   rtol=2e-12, atol=0.)

    def test_all_particle_permutations_including_identical_exchange(self):
        momenta, ids = reconstruct_born_momenta(_coordinates())
        expected = (*self.backend.evaluate(momenta, ids),
                    self.backend.matrix_elements(momenta, ids, -.37))
        for order in permutations(range(4)):
            order = list(order)
            actual = (*self.backend.evaluate(momenta[:, order], ids[:, order]),
                      self.backend.matrix_elements(momenta[:, order], ids[:, order], -.37))
            with self.subTest(order=order):
                for result, reference in zip(actual, expected):
                    np.testing.assert_allclose(result, reference, rtol=5e-12, atol=1e-14)

    def test_proper_rotation_and_boost_invariance(self):
        momenta, ids = _points()
        axis = np.array([.2, .7, -.4])
        axis /= np.linalg.norm(axis)
        angle = .93
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                          [-axis[1], axis[0], 0.]])
        rotation = (np.eye(3) * math.cos(angle)
                    + (1. - math.cos(angle)) * np.outer(axis, axis)
                    + math.sin(angle) * cross)
        transformed = momenta.copy()
        transformed[..., 1:] = momenta[..., 1:] @ rotation.T
        beta = np.array([.21, -.13, .32])
        beta2 = np.dot(beta, beta)
        gamma = 1. / math.sqrt(1. - beta2)
        beta_dot_p = transformed[..., 1:] @ beta
        energy = transformed[..., 0].copy()
        transformed[..., 0] = gamma * (energy + beta_dot_p)
        transformed[..., 1:] += (((gamma - 1.) / beta2 * beta_dot_p + gamma * energy)[..., None]
                                * beta)
        for result, reference in zip(self.backend.evaluate(transformed, ids),
                                     self.backend.evaluate(momenta, ids)):
            np.testing.assert_allclose(result, reference, rtol=2e-11, atol=2e-12)
        for coefficient in (0., -.61, 2.3):
            np.testing.assert_allclose(self.backend.matrix_elements(transformed, ids, coefficient),
                                       self.backend.matrix_elements(momenta, ids, coefficient),
                                       rtol=2e-12, atol=0.)

    def test_batch_and_coefficient_call_order_does_not_change_results(self):
        momenta, ids = _points()
        expected = self.backend.evaluate(momenta, ids)
        direct = {coefficient: self.backend.matrix_elements(momenta, ids, coefficient)
                  for coefficient in (0., .8, -2.1)}
        order = np.array([9, 2, 7, 0, 11, 4, 6, 1, 8, 5, 3, 10])
        for coefficient in (-2.1, 0., .8, -2.1):
            for index in order:
                actual = self.backend.matrix_elements(momenta[index:index + 1], ids[index:index + 1],
                                                       coefficient)
                np.testing.assert_allclose(actual, direct[coefficient][index:index + 1], rtol=2e-13)
        for actual, reference in zip(self.backend.evaluate(momenta[order], ids[order]), expected):
            np.testing.assert_allclose(actual, reference[order], rtol=2e-13, atol=2e-13)

    def test_first_phase_point_cannot_prune_later_helicities(self):
        # A fresh process is essential: a second ctypes.CDLL object shares
        # static C++ state with the first one. Warm each generated channel at
        # a polar endpoint, then compare with an ordinary first phase point.
        code = r'''
import json, sys
import numpy as np
from SMEFT.backend import NativeHBBackend
from SMEFT.tests.test_backend import _coordinates, _points, reconstruct_born_momenta
backend = NativeHBBackend(build_dir=sys.argv[1])
warm, ids = reconstruct_born_momenta(_coordinates(
    born_cos_theta1=float(sys.argv[2]), born_cos_theta2=.23))
backend.matrix_elements(warm, ids, 0.)
p, ids = _points()
direct = [backend.matrix_elements(p, ids, x).tolist() for x in (.37, -1.7, 0.)]
a, b = backend.evaluate(p, ids)
print("RESULT_JSON:" + json.dumps([a.tolist(), b.tolist(), *direct]))
'''
        outputs = []
        for cosine in (.32, 1., -1.):
            completed = subprocess.run([sys.executable, "-c", code, str(self.build_dir), str(cosine)],
                                       cwd=Path(__file__).resolve().parents[2], capture_output=True,
                                       text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            encoded = next(line.split("RESULT_JSON:", 1)[1] for line in completed.stdout.splitlines()
                           if line.startswith("RESULT_JSON:"))
            outputs.append(np.array(json.loads(encoded)))
        for result in outputs[1:]:
            np.testing.assert_allclose(result, outputs[0], rtol=2e-13, atol=2e-13)

    def test_noncontiguous_inputs_empty_batch_and_no_mutation(self):
        momenta, ids = _points()
        original, original_ids = momenta.copy(), ids.copy()
        expected = self.backend.evaluate(momenta, ids)
        for result, reference in zip(self.backend.evaluate(momenta[::-1, ::-1], ids[::-1, ::-1]), expected):
            np.testing.assert_allclose(result, reference[::-1], rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(momenta, original)
        np.testing.assert_array_equal(ids, original_ids)
        for result in (*self.backend.evaluate(momenta[:0], ids[:0]),
                       self.backend.matrix_elements(momenta[:0], ids[:0], .5)):
            self.assertEqual(result.shape, (0,))

    def test_invalid_pdg_shapes_and_nonphysical_momenta_are_rejected(self):
        momenta, ids = reconstruct_born_momenta(_coordinates(born_type=np.array([2])))
        cases = [(momenta[0], ids, "shape"), (momenta, ids.astype(float), "integer"),
                 (momenta, ids[:, :3], "shape"), (momenta.astype(complex), ids, "real"),
                 (momenta, np.array([[15, -15, 11, -11]]), "electrons and muons"),
                 (momenta, np.array([[11, 11, 13, -13]]), "opposite-sign")]
        for value, message in ((np.nan, "finite"), (np.inf, "finite"),
                               (-1., "positive"), (0., "positive")):
            changed = momenta.copy()
            changed[0, 0, 0] = value
            cases.append((changed, ids, message))
        massive = momenta.copy()
        massive[0, 0, 0] += 1.
        cases.append((massive, ids, "massless"))
        collinear = np.tile([10., 10., 0., 0.], (1, 4, 1))
        cases.append((collinear, ids, "positive invariant mass"))
        for bad_momenta, bad_ids, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.backend.evaluate(bad_momenta, bad_ids)
                with self.assertRaisesRegex(ValueError, message):
                    self.backend.matrix_elements(bad_momenta, bad_ids, .3)
        for coefficient in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                self.backend.matrix_elements(momenta, ids, coefficient)
        # A rejected batch must not poison the shared generated-model state.
        self.assertTrue(np.all(np.isfinite(self.backend.evaluate(momenta, ids))))

    def test_build_cache_reuses_only_matching_content_and_records_provenance(self):
        library = ensure_backend(self.build_dir)
        timestamp = library.stat().st_mtime_ns
        with patch("SMEFT.build_backend.subprocess.run", side_effect=AssertionError("unexpected rebuild")):
            self.assertEqual(ensure_backend(self.build_dir), library)
        self.assertEqual(library.stat().st_mtime_ns, timestamp)
        metadata = self.backend.metadata
        self.assertEqual(metadata["source_sha256"], source_fingerprint())
        self.assertEqual(metadata["build"]["library_sha256"], sha256(library))
        self.assertEqual(metadata["param_card_sha256"], metadata["generated"]["param_card_sha256"])
        # Change only the reported source fingerprint, never vendored files
        # or an already-loaded binary. A distinct dlopen path is necessary
        # in long-lived notebooks after a source update.
        with patch("SMEFT.build_backend.source_fingerprint", return_value="f" * 64):
            changed = ensure_backend(self.build_dir)
            self.assertNotEqual(changed, library)
            self.assertTrue(changed.is_file())
            self.assertEqual(json.loads(changed.with_suffix(".json").read_text())["source_sha256"],
                             "f" * 64)
        self.assertEqual(ensure_backend(self.build_dir), library)


if __name__ == "__main__":
    unittest.main()
