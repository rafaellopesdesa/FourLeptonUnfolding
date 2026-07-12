from __future__ import annotations

import unittest

from Simulation.prepare_fiducial_card import prepare_card


CARD = """
module Efficiency ElectronEfficiency {
  set EfficiencyFormula { (pt <= 10.0) * (0.0) + (pt > 10.0) * (0.9) }
}
module Efficiency MuonEfficiency {
  set EfficiencyFormula { (pt <= 10.0) * (0.0) + (pt > 10.0) * (0.9) }
}
module TreeWriter TreeWriter {
  add Branch Delphes/allParticles Particle GenParticle
  add Branch UniqueObjectFinder/electrons Electron Electron
  add Branch UniqueObjectFinder/muons Muon Muon
}
""".lstrip()


class PrepareFiducialCardTest(unittest.TestCase):
    def test_adds_truth_and_loose_reco_content(self):
        result = prepare_card(CARD)
        self.assertIn("Delphes/stableParticles StableParticle GenParticle", result)
        self.assertIn("ElectronEfficiency/electrons RecoElectron Electron", result)
        self.assertIn("MuonEfficiency/muons RecoMuon Muon", result)
        self.assertNotIn("pt <= 10.0", result)
        self.assertNotIn("pt > 10.0", result)
        self.assertEqual(result.count("StableParticle GenParticle"), 1)


if __name__ == "__main__":
    unittest.main()
