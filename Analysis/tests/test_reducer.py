from __future__ import annotations

import math
import unittest

import awkward as ak
import vector

from Analysis.build_analysis_tree import reduce_chunk


def p4(pt: float, eta: float, phi: float):
    momentum = vector.obj(pt=pt, eta=eta, phi=phi, mass=0.0)
    return momentum.E, momentum.px, momentum.py, momentum.pz


class ReducerTest(unittest.TestCase):
    def test_one_row_per_event_and_weight(self):
        electron_vectors = [p4(46.0, 0.1, 0.0), p4(46.0, -0.1, 2.9)]
        muon_vectors = [p4(18.0, 0.3, 1.0), p4(18.0, -0.3, 1.0 + math.pi)]
        arrays = {
            "Event.Number": ak.Array([[17]]),
            "Event.Weight": ak.Array([[2.5]]),
            "Particle.PID": ak.Array([[23, 11, -11, 13, -13]]),
            "Particle.M1": ak.Array([[-1, 0, 0, 0, 0]]),
            "Particle.M2": ak.Array([[-1, 0, 0, 0, 0]]),
            "DressedElectron.PID": ak.Array([[11, -11]]),
            "DressedElectron.M1": ak.Array([[0, 0]]),
            "DressedElectron.M2": ak.Array([[0, 0]]),
            "DressedElectron.E": ak.Array([[v[0] for v in electron_vectors]]),
            "DressedElectron.Px": ak.Array([[v[1] for v in electron_vectors]]),
            "DressedElectron.Py": ak.Array([[v[2] for v in electron_vectors]]),
            "DressedElectron.Pz": ak.Array([[v[3] for v in electron_vectors]]),
            "DressedMuon.PID": ak.Array([[13, -13]]),
            "DressedMuon.M1": ak.Array([[0, 0]]),
            "DressedMuon.M2": ak.Array([[0, 0]]),
            "DressedMuon.E": ak.Array([[v[0] for v in muon_vectors]]),
            "DressedMuon.Px": ak.Array([[v[1] for v in muon_vectors]]),
            "DressedMuon.Py": ak.Array([[v[2] for v in muon_vectors]]),
            "DressedMuon.Pz": ak.Array([[v[3] for v in muon_vectors]]),
            "RecoElectron.PT": ak.Array([[46.0, 46.0]]),
            "RecoElectron.Eta": ak.Array([[0.1, -0.1]]),
            "RecoElectron.Phi": ak.Array([[0.0, 2.9]]),
            "RecoElectron.Charge": ak.Array([[-1, 1]]),
            "RecoMuon.PT": ak.Array([[18.0, 18.0]]),
            "RecoMuon.Eta": ak.Array([[0.3, -0.3]]),
            "RecoMuon.Phi": ak.Array([[1.0, 1.0 + math.pi]]),
            "RecoMuon.Charge": ak.Array([[-1, 1]]),
        }
        output = reduce_chunk(
            arrays,
            first_event_id=100,
            four_lepton_mass_window=(105.0, 160.0),
        )
        self.assertEqual(output["event_id"].tolist(), [100])
        self.assertEqual(output["event_number"].tolist(), [17])
        self.assertEqual(output["weight"].tolist(), [2.5])
        self.assertTrue(output["fiducial"][0])
        self.assertTrue(output["reconstructed"][0])
        self.assertEqual(output["type"][0], 2)
        self.assertAlmostEqual(output["truth_m_Z1"][0], 91.8, delta=0.5)


if __name__ == "__main__":
    unittest.main()
