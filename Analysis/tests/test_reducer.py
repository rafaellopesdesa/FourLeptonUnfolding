from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import awkward as ak
import vector

from Analysis.build_analysis_tree import (
    INPUT_BRANCHES,
    SelectionDiagnostics,
    _empty_output,
    _fill_event_types,
    _prompt_mask,
    available_branch_names,
    diagnostic_lines,
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
            _prompt_mask(
                [11], [3], [3], particle_pid, [0.0] * 5, particle_m1, particle_m2
            ),
            [True],
        )

    def test_hadron_decay_lepton_is_rejected(self):
        # lepton <- B hadron <- b quark
        particle_pid = [5, 511, 11]
        particle_m1 = [-1, 0, 1]
        particle_m2 = [-1, 0, 1]
        self.assertEqual(
            _prompt_mask(
                [11], [1], [1], particle_pid, [0.0] * 3, particle_m1, particle_m2
            ),
            [False],
        )

    def test_lepton_without_w_or_z_ancestor_is_rejected(self):
        # A non-hadronic origin alone is insufficient for the four-lepton
        # fiducial definition.
        particle_pid = [25, 11]
        particle_m1 = [-1, 0]
        particle_m2 = [-1, 0]
        self.assertEqual(
            _prompt_mask(
                [11], [0], [0], particle_pid, [0.0] * 2, particle_m1, particle_m2
            ),
            [False],
        )

    def test_z_to_tau_to_lepton_is_accepted(self):
        particle_pid = [23, 15, 11]
        particle_m1 = [-1, 0, 1]
        particle_m2 = [-1, 0, 1]
        self.assertEqual(
            _prompt_mask(
                [11], [1], [1], particle_pid, [0.0] * 3, particle_m1, particle_m2
            ),
            [True],
        )

    def test_hard_virtual_photon_decay_is_accepted(self):
        particle_pid = [22, 11]
        particle_mass = [20.0, 0.0]
        particle_m1 = [-1, 0]
        particle_m2 = [-1, 0]
        self.assertEqual(
            _prompt_mask(
                [11],
                [0],
                [0],
                particle_pid,
                particle_mass,
                particle_m1,
                particle_m2,
            ),
            [True],
        )

    def test_low_mass_photon_conversion_is_rejected(self):
        # Even if the photon ultimately came from a Z, a conversion electron
        # is not a prompt H4l lepton.
        particle_pid = [23, 22, 11]
        particle_mass = [91.2, 0.0, 0.0]
        particle_m1 = [-1, 0, 1]
        particle_m2 = [-1, 0, 1]
        self.assertEqual(
            _prompt_mask(
                [11],
                [1],
                [1],
                particle_pid,
                particle_mass,
                particle_m1,
                particle_m2,
            ),
            [False],
        )

    def test_hadron_between_lepton_and_boson_is_rejected(self):
        # This deliberately unphysical topology exercises the rule that a
        # hadron in the decay path makes the lepton nonprompt.
        particle_pid = [23, 511, 11]
        particle_m1 = [-1, 0, 1]
        particle_m2 = [-1, 0, 1]
        self.assertEqual(
            _prompt_mask(
                [11], [1], [1], particle_pid, [0.0] * 3, particle_m1, particle_m2
            ),
            [False],
        )

    def test_hadron_upstream_of_boson_is_rejected_like_delphes(self):
        # Candidate-origin parity with LeptonDressing::HasHadronAncestor:
        # the hadron veto examines the complete ancestry, even beyond the
        # otherwise acceptable Z decay.
        particle_pid = [511, 23, 11]
        particle_m1 = [-1, 0, 1]
        particle_m2 = [-1, 0, 1]
        self.assertEqual(
            _prompt_mask(
                [11],
                [1],
                [1],
                particle_pid,
                [5.28, 91.2, 0.0],
                particle_m1,
                particle_m2,
            ),
            [False],
        )

    def test_cyclic_generator_ancestry_terminates(self):
        # Some generator status-copy records can contain A -> B -> A cycles.
        particle_pid = [23, 11]
        particle_m1 = [1, 0]
        particle_m2 = [1, 0]
        self.assertEqual(
            _prompt_mask(
                [11], [0], [0], particle_pid, [0.0] * 2, particle_m1, particle_m2
            ),
            [True],
        )

    def test_mother_indices_are_not_an_inclusive_range(self):
        # M1 and M2 point to two Z copies. The B hadron between their array
        # indices is unrelated and must never enter the ancestry traversal.
        particle_pid = [23, 511, 23]
        particle_m1 = [-1, -1, -1]
        particle_m2 = [-1, -1, -1]
        self.assertEqual(
            _prompt_mask(
                [11], [0], [2], particle_pid, [0.0] * 3, particle_m1, particle_m2
            ),
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

    def test_selection_diagnostics_report_unfolding_metrics_by_channel(self):
        diagnostics = SelectionDiagnostics()

        def result(selected: bool, event_type: int | None):
            candidate = (
                None if event_type is None else SimpleNamespace(event_type=event_type)
            )
            return SimpleNamespace(selected=selected, candidate=candidate)

        diagnostics.add(result(True, 0), result(True, 0), weight=2.0)
        diagnostics.add(result(True, 1), result(False, None), weight=1.0)
        diagnostics.add(result(False, None), result(True, 2), weight=3.0)

        self.assertEqual(diagnostics.overall.fiducial, 2)
        self.assertEqual(diagnostics.overall.reconstructed, 2)
        self.assertEqual(diagnostics.overall.both, 1)
        self.assertEqual(diagnostics.overall.reconstructed_only, 1)
        self.assertEqual(diagnostics.overall_weighted.fiducial, 3.0)
        self.assertEqual(diagnostics.overall_weighted.reconstructed, 5.0)
        self.assertEqual(diagnostics.overall_weighted.both, 2.0)
        self.assertEqual(diagnostics.overall_weighted.reconstructed_only, 3.0)
        rendered = "\n".join(diagnostic_lines(diagnostics))
        self.assertIn("C_count", rendered)
        self.assertIn("C_weight", rendered)
        self.assertRegex(rendered, r"all\s+2\s+2\s+1\s+1\s+1\.0000\s+0\.5000\s+0\.5000")
        self.assertRegex(rendered, r"2e2mu\s+0\s+1\s+0\s+1\s+n/a\s+n/a\s+1\.0000")
        self.assertRegex(
            rendered,
            r"all\s+3\s+5\s+2\s+3\s+1\.6667\s+0\.6667\s+0\.6000",
        )

    def test_event_types_keep_truth_and_reco_pairing_migrations_separate(self):
        output = _empty_output(2, 0)
        truth = SimpleNamespace(
            selected=True, candidate=SimpleNamespace(event_type=1)
        )
        reco = SimpleNamespace(
            selected=True, candidate=SimpleNamespace(event_type=2)
        )
        no_truth = SimpleNamespace(selected=False, candidate=None)
        reco_only = SimpleNamespace(
            selected=True, candidate=SimpleNamespace(event_type=3)
        )

        _fill_event_types(output, 0, truth, reco)
        _fill_event_types(output, 1, no_truth, reco_only)

        self.assertEqual(output["truth_type"].tolist(), [1, -1])
        self.assertEqual(output["reco_type"].tolist(), [2, 3])
        # The compatibility alias must be safe for reconstructed pseudo-data.
        self.assertEqual(output["type"].tolist(), [2, 3])

    def test_one_row_per_event_and_weight(self):
        electron_vectors = [p4(46.0, 0.1, 0.0), p4(46.0, -0.1, 2.9)]
        muon_vectors = [p4(18.0, 0.3, 1.0), p4(18.0, -0.3, 1.0 + math.pi)]
        arrays = {
            "Event.Number": ak.Array([[17]]),
            "Event.Weight": ak.Array([[2.5]]),
            "Event.CrossSection": ak.Array([[0.125]]),
            "Particle.PID": ak.Array([[23, 11, -11, 13, -13]]),
            "Particle.E": ak.Array([[91.1876, 0.0, 0.0, 0.0, 0.0]]),
            "Particle.Px": ak.Array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            "Particle.Py": ak.Array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            "Particle.Pz": ak.Array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
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
            "RecoElectronNoIso.PT": ak.Array([[46.0, 46.0]]),
            "RecoMuonNoIso.PT": ak.Array([[18.0, 18.0]]),
            "RecoElectron.PT": ak.Array([[46.0, 46.0]]),
            "RecoElectron.Eta": ak.Array([[0.1, -0.1]]),
            "RecoElectron.Phi": ak.Array([[0.0, 2.9]]),
            "RecoElectron.Charge": ak.Array([[-1, 1]]),
            "RecoMuon.PT": ak.Array([[18.0, 18.0]]),
            "RecoMuon.Eta": ak.Array([[0.3, -0.3]]),
            "RecoMuon.Phi": ak.Array([[1.0, 1.0 + math.pi]]),
            "RecoMuon.Charge": ak.Array([[-1, 1]]),
        }
        diagnostics = SelectionDiagnostics()
        output = reduce_chunk(
            arrays,
            first_event_id=100,
            four_lepton_mass_window=(105.0, 160.0),
            diagnostics=diagnostics,
        )
        self.assertEqual(output["event_id"].tolist(), [100])
        self.assertEqual(output["event_number"].tolist(), [17])
        self.assertEqual(output["weight"].tolist(), [2.5])
        self.assertEqual(output["cross_section_pb"].tolist(), [0.125])
        self.assertTrue(output["fiducial"][0])
        self.assertTrue(output["reconstructed"][0])
        self.assertEqual(output["truth_type"][0], 2)
        self.assertEqual(output["reco_type"][0], 2)
        self.assertEqual(output["type"][0], 2)
        self.assertAlmostEqual(output["truth_m_Z1"][0], 91.8, delta=0.5)
        self.assertEqual(diagnostics.overall.both, 1)
        self.assertEqual(diagnostics.by_channel[2].fiducial, 1)


if __name__ == "__main__":
    unittest.main()
