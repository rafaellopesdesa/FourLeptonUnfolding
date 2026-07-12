from __future__ import annotations

import math
import unittest

import vector

from Analysis.selection import Lepton, select_event


def massless(pt: float, eta: float, phi: float):
    return vector.obj(pt=pt, eta=eta, phi=phi, mass=0.0)


class SelectionTest(unittest.TestCase):
    def test_mixed_flavor_pairing_and_type(self):
        leptons = [
            Lepton(massless(46.0, 0.1, 0.0), "electron", -1),
            Lepton(massless(46.0, -0.1, math.pi), "electron", 1),
            Lepton(massless(18.0, 0.3, 1.0), "muon", -1),
            Lepton(massless(18.0, -0.3, 1.0 + math.pi), "muon", 1),
        ]
        result = select_event(leptons)
        self.assertIsNotNone(result.candidate)
        self.assertTrue(result.selected)
        self.assertEqual(result.candidate.event_type, 2)  # leading Z is ee
        self.assertAlmostEqual(result.candidate.z1.mass, 92.46, delta=0.2)

    def test_failed_event_is_retained_as_a_candidate(self):
        leptons = [
            Lepton(massless(46.0, 0.1, 0.0), "muon", -1),
            Lepton(massless(46.0, -0.1, math.pi), "muon", 1),
            Lepton(massless(8.0, 0.3, 1.0), "electron", -1),
            Lepton(massless(8.0, -0.3, 1.0 + math.pi), "electron", 1),
        ]
        result = select_event(leptons)
        self.assertIsNotNone(result.candidate)
        self.assertFalse(result.selected)


if __name__ == "__main__":
    unittest.main()
