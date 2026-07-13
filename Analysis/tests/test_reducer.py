from __future__ import annotations

import math
import unittest

import awkward as ak
import vector

from Analysis.build_analysis_tree import (
    INPUT_BRANCHES,
    _prompt_mask,
    available_branch_names,
    reduce_chunk,
)


def p4(pt: float, eta: float, phi: float):
    momentum = vector.obj(pt=pt, eta=eta, phi=phi, mass=0.0)
    return momentum.E, momentum.px, momentum.py, momentum.pz


class ReducerTest(unittest.TestCase):
    def test_prompt_ancestry_stops_at_incoming_parton(self):
        # lepton <- Z <- H <- gluon <- beam proton
        particle_pid = [2212, 21, 25, 23, 11]
        particle_m1 = [-1, 0, 1, 2, 3]
        particle_m2 = [-1, 0, 1, 2, 3]
        self.assertEqual(
            _prompt_mask([3], [3], particle_pid, particle_m1, particle_m2),
            [True],
        )

    def test_hadron_decay_lepton_is_rejected(self):
        # lepton <- B hadron <- b quark
        particle_pid = [5, 511, 11]
        particle_m1 = [-1, 0, 1]
        particle_m2 = [-1, 0, 1]
        self.assertEqual(
            _prompt_mask([1], [1], particle_pid, particle_m1, particle_m2),
            [False],
        )

    def test_cyclic_generator_ancestry_terminates(self):
        # Some generator status-copy records can contain A -> B -> A cycles.
        particle_pid = [23, 11]
        particle_m1 = [1, 0]
        particle_m2 = [1, 0]
        self.assertEqual(
            _prompt_mask([0], [0], particle_pid, particle_m1, particle_m2),
            [True],
        )

    def test_nested_delphes_branches_are_compared_by_leaf_name(self):
        class NestedTree:
            def keys(self, *, recursive, full_paths):
                self.arguments = (recursive, full_paths)
                if full_paths:
                    return [f"{name.split('.', 1)[0]}/{name}" for name in INPUT_BRANCHES]
                return list(INPUT_BRANCHES)

        tree = NestedTree()
        self.assertEqual(available_branch_names(tree), set(INPUT_BRANCHES))
        self.assertEqual(tree.arguments, (True, False))

    def test_one_row_per_event_and_weight(self):
        electron_vectors = [p4(46.0, 0.1, 0.0), p4(46.0, -0.1, 2.9)]
        muon_vectors = [p4(18.0, 0.3, 1.0), p4(18.0, -0.3, 1.0 + math.pi)]
        arrays = {
            "Event.Number": ak.Array([[17]]),
            "Event.Weight": ak.Array([[2.5]]),
            "Event.CrossSection": ak.Array([[0.125]]),
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
        self.assertEqual(output["cross_section_pb"].tolist(), [0.125])
        self.assertTrue(output["fiducial"][0])
        self.assertTrue(output["reconstructed"][0])
        self.assertEqual(output["type"][0], 2)
        self.assertAlmostEqual(output["truth_m_Z1"][0], 91.8, delta=0.5)


if __name__ == "__main__":
    unittest.main()
