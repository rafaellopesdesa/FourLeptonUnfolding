"""Numerical smoke and invariance tests for the four-lepton tools."""

from __future__ import annotations

import math
import unittest

import vector

from Tools.four_lepton_kinematics import compute_kinematics
from Tools.higgs_decay_width import sm_higgs_decay_density


def _massless(momentum: float, theta: float, phi: float):
    return vector.obj(
        px=momentum * math.sin(theta) * math.cos(phi),
        py=momentum * math.sin(theta) * math.sin(phi),
        pz=momentum * math.cos(theta),
        E=momentum,
    )


def _event():
    m_h, m1, m2 = 125.10, 91.0, 25.0
    momentum = math.sqrt(
        (m_h * m_h - (m1 + m2) ** 2) * (m_h * m_h - (m1 - m2) ** 2)
    ) / (2.0 * m_h)
    energy1 = math.sqrt(m1 * m1 + momentum * momentum)
    energy2 = math.sqrt(m2 * m2 + momentum * momentum)

    direction = (
        math.sin(1.1) * math.cos(0.4),
        math.sin(1.1) * math.sin(0.4),
        math.cos(1.1),
    )
    z1 = vector.obj(
        px=momentum * direction[0],
        py=momentum * direction[1],
        pz=momentum * direction[2],
        E=energy1,
    )
    z2 = vector.obj(px=-z1.px, py=-z1.py, pz=-z1.pz, E=energy2)

    e_minus = _massless(m1 / 2.0, 0.8, 0.2).boost_beta3(z1.to_beta3())
    e_plus = _massless(m1 / 2.0, math.pi - 0.8, math.pi + 0.2).boost_beta3(
        z1.to_beta3()
    )
    mu_minus = _massless(m2 / 2.0, 1.2, -0.7).boost_beta3(z2.to_beta3())
    mu_plus = _massless(m2 / 2.0, math.pi - 1.2, math.pi - 0.7).boost_beta3(
        z2.to_beta3()
    )

    # Give the H a generic laboratory boost, including transverse momentum.
    lab_beta = vector.obj(x=0.08, y=-0.04, z=0.25)
    return tuple(
        lepton.boost_beta3(lab_beta)
        for lepton in (e_minus, e_plus, mu_minus, mu_plus)
    )


class FourLeptonToolTest(unittest.TestCase):
    def test_observables_and_higgs_density(self):
        event = _event()
        kin = compute_kinematics(*event)
        self.assertAlmostEqual(kin.m_Z1, 91.0, places=9)
        self.assertAlmostEqual(kin.m_Z2, 25.0, places=9)
        self.assertAlmostEqual(kin.m_ZZ, 125.10, places=9)
        self.assertEqual(kin.z1_flavor, "electron")
        for cosine in (kin.cos_theta_star, kin.cos_theta1, kin.cos_theta2):
            self.assertGreaterEqual(cosine, -1.0)
            self.assertLessEqual(cosine, 1.0)
        for angle in (kin.Phi, kin.Phi1, kin.Psi):
            self.assertGreaterEqual(angle, -math.pi)
            self.assertLessEqual(angle, math.pi)

        density = sm_higgs_decay_density(*event)
        self.assertGreater(density.value, 0.0)
        self.assertGreater(density.angular_factor, 0.0)
        self.assertGreater(density.mass_factor, 0.0)

    def test_common_longitudinal_boost_preserves_internal_variables(self):
        event = _event()
        boosted = tuple(
            lepton.boost_beta3(vector.obj(x=0.0, y=0.0, z=-0.17))
            for lepton in event
        )
        first = compute_kinematics(*event)
        second = compute_kinematics(*boosted)
        for field in (
            "cos_theta_star",
            "cos_theta1",
            "cos_theta2",
            "Phi",
            "Phi1",
            "Psi",
            "m_Z1",
            "m_Z2",
            "m_ZZ",
            "pT_ZZ",
        ):
            self.assertAlmostEqual(getattr(first, field), getattr(second, field), places=10)


if __name__ == "__main__":
    unittest.main()
