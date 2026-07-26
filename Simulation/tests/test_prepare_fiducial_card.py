from __future__ import annotations

import unittest

from Simulation.prepare_fiducial_card import _module_block, prepare_card


CARD = """
set ExecutionPath {
  ParticlePropagator
  GenJetFinder
  JetEnergyScale
  TauTagging
  UniqueObjectFinder
  TreeWriter
}
#################################
# Propagate particles in cylinder
#################################
module Efficiency ElectronEfficiency {
  set EfficiencyFormula { (pt <= 10.0) * (0.0) + (abs(eta) <= 2.5) * (pt > 10.0) * (0.9) }
}
module Efficiency MuonEfficiency {
  set EfficiencyFormula { (pt <= 10.0) * (0.0) + (pt > 10.0) * (0.9) }
}
module FastJetFinder GenJetFinder {
  set ParameterR 0.6
  set JetPTMin 20.0
}
#########################
# Gen Missing ET merger
########################
module FastJetFinder FastJetFinder {
  set ParameterR 0.6
  set JetPTMin 20.0
}
module EnergyScale JetEnergyScale {
  set InputArray FastJetFinder/jets
  set OutputArray jets
}
#####################################################
# Find uniquely identified photons/electrons/tau/jets
#####################################################
module UniqueObjectFinder UniqueObjectFinder {
  add InputArray JetEnergyScale/jets jets
}
module TreeWriter TreeWriter {
  add Branch Delphes/allParticles Particle GenParticle
  add Branch GenJetFinder/jets GenJet Jet
  add Branch UniqueObjectFinder/electrons Electron Electron
  add Branch UniqueObjectFinder/muons Muon Muon
}
""".lstrip()


class PrepareFiducialCardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = prepare_card(CARD)

    def test_builds_exact_prompt_dressing_definition(self):
        result = self.result
        self.assertIn("Delphes/stableParticles StableParticle GenParticle", result)
        self.assertIn("module LeptonDressing TruthLeptonDressing", result)
        self.assertIn("set DeltaRMax 0.1", result)
        self.assertIn("set DressingPTMin 0.0", result)
        self.assertIn("set RequireNoHadronAncestor true", result)
        self.assertIn("set RequireNoHadronAncestorCandidate true", result)
        self.assertIn("set RequireBosonAncestorCandidate true", result)
        self.assertIn("set AllowTauDecayCandidate true", result)
        self.assertIn("set VirtualPhotonMinMass 5.0", result)
        self.assertIn("set UniqueAssignment true", result)
        self.assertIn(
            "DressedElectronFilter/electrons DressedElectron GenParticle", result
        )
        self.assertIn("DressedMuonFilter/muons DressedMuon GenParticle", result)
        self.assertLess(
            result.index("TruthLeptonDressing\n"), result.index("ParticlePropagator\n")
        )
        self.assertEqual(result.count("StableParticle GenParticle"), 1)

    def test_builds_separate_reco_id_and_isolation_response(self):
        result = self.result
        self.assertIn("module Efficiency H4lElectronRecoID", result)
        self.assertIn("module Efficiency H4lElectronIsolation", result)
        self.assertIn("module Efficiency H4lMuonRecoID", result)
        self.assertIn("module Efficiency H4lMuonIsolation", result)
        self.assertIn(
            "H4lElectronRecoID/electrons RecoElectronNoIso Electron", result
        )
        self.assertIn(
            "H4lElectronIsolation/electrons RecoElectron Electron", result
        )
        self.assertIn("H4lMuonRecoID/muons RecoMuonNoIso Muon", result)
        self.assertIn("H4lMuonIsolation/muons RecoMuon Muon", result)
        self.assertGreaterEqual(result.count("set UseMomentumVector true"), 6)
        self.assertIn("(pt > 4.0  && pt <= 5.0)", result)
        self.assertIn("(abs(eta) < 2.5)", result)
        self.assertNotIn("abs(phi)", result.lower())

    def test_does_not_repurpose_the_generic_atlas_leptons(self):
        # Standard Delphes branches remain useful as an independent diagnostic.
        # The H4l response has dedicated modules and must not silently rewrite
        # the generic card's 10 GeV object model.
        self.assertIn("pt <= 10.0", self.result)
        self.assertIn("pt > 10.0", self.result)

    def test_configures_h4l_jet_radius_and_final_acceptance(self):
        result = self.result
        self.assertNotIn("set ParameterR 0.6", result)
        self.assertEqual(result.count("set ParameterR 0.4"), 2)
        self.assertIn("module Efficiency H4lGenJetAcceptance", result)
        self.assertIn("module Efficiency H4lRecoJetAcceptance", result)
        self.assertIn("(pt > 30.0) * (abs(eta) < 4.5)", result)
        self.assertIn(
            "add InputArray H4lRecoJetAcceptance/jets jets", result
        )
        self.assertIn("add Branch H4lGenJetAcceptance/jets GenJet Jet", result)

    def test_efficiency_formulas_are_bounded_on_response_grid(self):
        module_names = (
            "H4lElectronRecoID",
            "H4lElectronIsolation",
            "H4lMuonRecoID",
            "H4lMuonIsolation",
        )
        pts = (0.0, 4.0, 4.001, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 30.0, 100.0)
        etas = (0.0, 0.1, 0.8, 1.4, 1.51, 2.0, 2.499, 2.5, 2.6)
        for name in module_names:
            _, _, block = _module_block(
                self.result, f"module Efficiency {name} {{"
            )
            marker = "set EfficiencyFormula {"
            formula_start = block.index(marker) + len(marker)
            formula = block[formula_start:].split("\n  }\n", 1)[0].strip()
            python_formula = f"({formula.replace('&&', ' and ')})"
            self.assertNotIn("phi", python_formula.lower())
            for pt in pts:
                for eta in etas:
                    value = float(
                        eval(  # noqa: S307 - fixed formula generated in this test
                            python_formula,
                            {"__builtins__": {}, "abs": abs},
                            {"pt": pt, "eta": eta},
                        )
                    )
                    self.assertGreaterEqual(value, 0.0, (name, pt, eta))
                    self.assertLessEqual(value, 1.0, (name, pt, eta))
                    if pt <= 4.0 or abs(eta) >= 2.5:
                        self.assertEqual(value, 0.0, (name, pt, eta))

    def test_efficiency_formulas_are_continuous_at_pt_knots(self):
        knots_by_module = {
            "H4lElectronRecoID": (4.0, 5.0, 7.0, 10.0, 15.0, 30.0),
            "H4lElectronIsolation": (
                4.0,
                5.0,
                7.0,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
            ),
            "H4lMuonRecoID": (4.0, 5.0, 6.0, 8.0, 10.0),
            "H4lMuonIsolation": (4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0),
        }
        for name, knots in knots_by_module.items():
            _, _, block = _module_block(
                self.result, f"module Efficiency {name} {{"
            )
            marker = "set EfficiencyFormula {"
            formula_start = block.index(marker) + len(marker)
            formula = block[formula_start:].split("\n  }\n", 1)[0].strip()
            python_formula = f"({formula.replace('&&', ' and ')})"
            for knot in knots:
                values = [
                    float(
                        eval(  # noqa: S307 - fixed generated formula
                            python_formula,
                            {"__builtins__": {}, "abs": abs},
                            {"pt": knot + offset, "eta": 1.0},
                        )
                    )
                    for offset in (-1.0e-6, 1.0e-6)
                ]
                self.assertAlmostEqual(
                    values[0], values[1], delta=2.0e-6, msg=(name, knot, values)
                )


if __name__ == "__main__":
    unittest.main()
