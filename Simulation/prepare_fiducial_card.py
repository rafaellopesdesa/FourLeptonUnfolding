#!/usr/bin/env python3
"""Build the H->ZZ->4l response card from Delphes's bundled ATLAS card.

The installed Delphes card remains untouched.  This script modifies only the
per-job resolved copy used by ``run_simulation.sh``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _module_block(text: str, declaration: str) -> tuple[int, int, str]:
    start = text.find(declaration)
    if start < 0:
        raise ValueError(f"card does not contain expected module: {declaration}")

    lines = text[start:].splitlines(keepends=True)
    depth = 0
    length = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        length += len(line)
        if depth == 0:
            return start, start + length, text[start : start + length]
    raise ValueError(f"unterminated module: {declaration}")


def _insert_after(text: str, anchor: str, addition: str) -> str:
    if addition.strip() in text:
        return text
    if text.count(anchor) != 1:
        raise ValueError(f"expected exactly one card line matching: {anchor.strip()}")
    return text.replace(anchor, anchor + addition, 1)


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one card fragment matching: {old.strip()}")
    return text.replace(old, new, 1)


def _replace_in_module(
    text: str, declaration: str, old: str, new: str
) -> str:
    start, end, block = _module_block(text, declaration)
    if block.count(old) != 1:
        raise ValueError(
            f"{declaration} does not contain exactly one expected fragment: {old.strip()}"
        )
    return text[:start] + block.replace(old, new, 1) + text[end:]


def _add_h4l_lepton_modules(text: str) -> str:
    execution_anchor = "set ExecutionPath {\n"
    execution_modules = (
        "  TruthLeptonFilter\n"
        "  TruthPhotonFilter\n"
        "  TruthLeptonDressing\n"
        "  DressedElectronFilter\n"
        "  DressedMuonFilter\n"
        "  H4lElectronMomentumSmearing\n"
        "  H4lElectronRecoID\n"
        "  H4lElectronIsolation\n"
        "  H4lMuonMomentumSmearing\n"
        "  H4lMuonRecoID\n"
        "  H4lMuonIsolation\n"
    )
    text = _insert_after(text, execution_anchor, execution_modules)

    module_anchor = (
        "#################################\n"
        "# Propagate particles in cylinder\n"
        "#################################\n"
    )
    modules = r"""
######################################################
# H->ZZ->4l particle-level and reconstruction response
######################################################

# Start from stable, post-FSR electrons and muons.  LeptonDressing below
# applies the ancestry requirements before exporting the dressed collection.
module PdgCodeFilter TruthLeptonFilter {
  set InputArray Delphes/stableParticles
  set OutputArray leptons
  set Invert true
  add PdgCode {11}
  add PdgCode {-11}
  add PdgCode {13}
  add PdgCode {-13}
}

module PdgCodeFilter TruthPhotonFilter {
  set InputArray Delphes/stableParticles
  set OutputArray photons
  set Invert true
  add PdgCode {22}
}

module LeptonDressing TruthLeptonDressing {
  set CandidateInputArray TruthLeptonFilter/leptons
  set DressingInputArray TruthPhotonFilter/photons
  set ParticleInputArray Delphes/allParticles
  set OutputArray dressedLeptons
  set DeltaRMax 0.1
  set DressingPTMin 0.0
  set RequireNoHadronAncestor true
  set RequireNoHadronAncestorCandidate true
  set RequireBosonAncestorCandidate true
  set AllowTauDecayCandidate true
  set VirtualPhotonMinMass 5.0
  set UniqueAssignment true
}

module PdgCodeFilter DressedElectronFilter {
  set InputArray TruthLeptonDressing/dressedLeptons
  set OutputArray electrons
  set Invert true
  add PdgCode {11}
  add PdgCode {-11}
}

module PdgCodeFilter DressedMuonFilter {
  set InputArray TruthLeptonDressing/dressedLeptons
  set OutputArray muons
  set Invert true
  add PdgCode {13}
  add PdgCode {-13}
}

# The dedicated H4l reconstructed collections begin from the same prompt,
# dressed particles used at truth level.  This supplies the simplified FSR
# recovery requested for the response study, while the standard Delphes
# Electron and Muon branches remain available independently.
module MomentumSmearing H4lElectronMomentumSmearing {
  set InputArray DressedElectronFilter/electrons
  set OutputArray electrons
  set UseMomentumVector true
  set ResolutionFormula {                  (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.03^2 + pt^2*1.3e-3^2) +
                         (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.05^2 + pt^2*1.7e-3^2) +
                         (abs(eta) > 1.5 && abs(eta) <  2.5) * (pt > 0.1) * sqrt(0.15^2 + pt^2*3.1e-3^2)}
}

# Approximate H4l Loose reconstruction+identification efficiency.  The
# 5--7 GeV bin and 2.47--2.5 edge are explicit extrapolations.  The eta
# multipliers describe broad detector regions only; there is no phi model.
module Efficiency H4lElectronRecoID {
  set InputArray H4lElectronMomentumSmearing/electrons
  set OutputArray electrons
  set UseMomentumVector true
  set EfficiencyFormula {
    (abs(eta) < 2.5) *
    ( (pt <= 4.0)                         * (0.00) +
      (pt > 4.0  && pt <= 5.0)           * (0.85*(pt - 4.0)) +
      (pt > 5.0  && pt <= 7.0)           * (0.85 + (0.90 - 0.85)*(pt - 5.0)/2.0) +
      (pt > 7.0  && pt <= 10.0)          * (0.90 + (0.92 - 0.90)*(pt - 7.0)/3.0) +
      (pt > 10.0 && pt <= 15.0)          * (0.92 + (0.95 - 0.92)*(pt - 10.0)/5.0) +
      (pt > 15.0 && pt <= 30.0)          * (0.95 + (0.96 - 0.95)*(pt - 15.0)/15.0) +
      (pt > 30.0)                         * (0.96) ) *
    ( (abs(eta) <= 0.8)                  * (1.02) +
      (abs(eta) > 0.8  && abs(eta) <= 1.37) * (1.00) +
      (abs(eta) > 1.37 && abs(eta) <= 1.52) * (0.94) +
      (abs(eta) > 1.52 && abs(eta) <= 2.0)  * (0.99) +
      (abs(eta) > 2.0  && abs(eta) <  2.5)  * (0.96) )
  }
}

# Loose_VarRad from the public Run-2 electron performance is used as a
# source-backed proxy for the H4l loose isolation efficiency.  It is applied
# stochastically because the no-pile-up Delphes cone sum is not a Run-2
# particle-flow isolation model.
module Efficiency H4lElectronIsolation {
  set InputArray H4lElectronRecoID/electrons
  set OutputArray electrons
  set UseMomentumVector true
  set EfficiencyFormula {
    (abs(eta) < 2.5) *
    ( (pt <= 4.0)                         * (0.00) +
      (pt > 4.0  && pt <= 5.0)           * (0.68*(pt - 4.0)) +
      (pt > 5.0  && pt <= 7.0)           * (0.68 + (0.77 - 0.68)*(pt - 5.0)/2.0) +
      (pt > 7.0  && pt <= 10.0)          * (0.77 + (0.84 - 0.77)*(pt - 7.0)/3.0) +
      (pt > 10.0 && pt <= 15.0)          * (0.84 + (0.91 - 0.84)*(pt - 10.0)/5.0) +
      (pt > 15.0 && pt <= 20.0)          * (0.91 + (0.95 - 0.91)*(pt - 15.0)/5.0) +
      (pt > 20.0 && pt <= 25.0)          * (0.95 + (0.97 - 0.95)*(pt - 20.0)/5.0) +
      (pt > 25.0 && pt <= 30.0)          * (0.97 + (0.985 - 0.97)*(pt - 25.0)/5.0) +
      (pt > 30.0)                         * (0.985) )
  }
}

module MomentumSmearing H4lMuonMomentumSmearing {
  set InputArray DressedMuonFilter/muons
  set OutputArray muons
  set UseMomentumVector true
  set ResolutionFormula {                  (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.01^2 + pt^2*1.0e-4^2) +
                         (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.015^2 + pt^2*1.5e-4^2) +
                         (abs(eta) > 1.5 && abs(eta) <  2.5) * (pt > 0.1) * sqrt(0.025^2 + pt^2*3.5e-4^2)}
}

# Loose-muon reconstruction+ID is close to unity in H4l kinematics.  Small
# eta factors average over local detector structures without introducing phi.
module Efficiency H4lMuonRecoID {
  set InputArray H4lMuonMomentumSmearing/muons
  set OutputArray muons
  set UseMomentumVector true
  set EfficiencyFormula {
    (abs(eta) < 2.5) *
    ( (pt <= 4.0)                         * (0.00) +
      (pt > 4.0  && pt <= 5.0)           * (0.96*(pt - 4.0)) +
      (pt > 5.0  && pt <= 6.0)           * (0.96 + (0.98 - 0.96)*(pt - 5.0)) +
      (pt > 6.0  && pt <= 8.0)           * (0.98 + (0.985 - 0.98)*(pt - 6.0)/2.0) +
      (pt > 8.0  && pt <= 10.0)          * (0.985 + (0.99 - 0.985)*(pt - 8.0)/2.0) +
      (pt > 10.0)                         * (0.99) ) *
    ( (abs(eta) <= 0.1)                  * (0.995) +
      (abs(eta) > 0.1 && abs(eta) <= 1.0) * (1.000) +
      (abs(eta) > 1.0 && abs(eta) <= 1.3) * (0.995) +
      (abs(eta) > 1.3 && abs(eta) <= 2.0) * (1.000) +
      (abs(eta) > 2.0 && abs(eta) <  2.5) * (0.995) )
  }
}

# PflowLoose prompt-muon efficiencies, parameterised from the public Run-2
# performance.  This is separate from reconstruction+ID to prevent accidental
# double counting.
module Efficiency H4lMuonIsolation {
  set InputArray H4lMuonRecoID/muons
  set OutputArray muons
  set UseMomentumVector true
  set EfficiencyFormula {
    (abs(eta) < 2.5) *
    ( (pt <= 4.0)                         * (0.00) +
      (pt > 4.0  && pt <= 5.0)           * (0.72*(pt - 4.0)) +
      (pt > 5.0  && pt <= 6.0)           * (0.72 + (0.80 - 0.72)*(pt - 5.0)) +
      (pt > 6.0  && pt <= 8.0)           * (0.80 + (0.88 - 0.80)*(pt - 6.0)/2.0) +
      (pt > 8.0  && pt <= 10.0)          * (0.88 + (0.92 - 0.88)*(pt - 8.0)/2.0) +
      (pt > 10.0 && pt <= 15.0)          * (0.92 + (0.96 - 0.92)*(pt - 10.0)/5.0) +
      (pt > 15.0 && pt <= 20.0)          * (0.96 + (0.985 - 0.96)*(pt - 15.0)/5.0) +
      (pt > 20.0 && pt <= 30.0)          * (0.985 + (0.995 - 0.985)*(pt - 20.0)/10.0) +
      (pt > 30.0)                         * (0.995) )
  }
}

"""
    if "module LeptonDressing TruthLeptonDressing {" not in text:
        if text.count(module_anchor) != 1:
            raise ValueError("could not locate the ParticlePropagator section")
        text = text.replace(module_anchor, modules + module_anchor, 1)
    return text


def _configure_jets(text: str) -> str:
    # H4l uses anti-kt R=0.4.  Keep the finder at 20 GeV as a technical
    # preselection and impose the published 30 GeV object threshold after the
    # detector response, so upward migrations are not silently removed.
    for name in ("GenJetFinder", "FastJetFinder"):
        text = _replace_in_module(
            text,
            f"module FastJetFinder {name} {{",
            "set ParameterR 0.6",
            "set ParameterR 0.4",
        )

    text = _insert_after(
        text,
        "  GenJetFinder\n",
        "  H4lGenJetAcceptance\n",
    )
    text = _insert_after(
        text,
        "  TauTagging\n",
        "  H4lRecoJetAcceptance\n",
    )

    module_anchor = (
        "#########################\n"
        "# Gen Missing ET merger\n"
        "########################\n"
    )
    gen_acceptance = r"""
######################################
# H4l particle-level jet acceptance
######################################

module Efficiency H4lGenJetAcceptance {
  set InputArray GenJetFinder/jets
  set OutputArray jets
  set UseMomentumVector true
  set EfficiencyFormula { (pt > 30.0) * (abs(eta) < 4.5) }
}

"""
    if "module Efficiency H4lGenJetAcceptance {" not in text:
        if text.count(module_anchor) != 1:
            raise ValueError("could not locate the GenMissingET section")
        text = text.replace(module_anchor, gen_acceptance + module_anchor, 1)

    unique_anchor = (
        "#####################################################\n"
        "# Find uniquely identified photons/electrons/tau/jets\n"
        "#####################################################\n"
    )
    reco_acceptance = r"""
################################
# H4l reconstructed jet acceptance
################################

module Efficiency H4lRecoJetAcceptance {
  set InputArray JetEnergyScale/jets
  set OutputArray jets
  set UseMomentumVector true
  set EfficiencyFormula { (pt > 30.0) * (abs(eta) < 4.5) }
}

"""
    if "module Efficiency H4lRecoJetAcceptance {" not in text:
        if text.count(unique_anchor) != 1:
            raise ValueError("could not locate the UniqueObjectFinder section")
        text = text.replace(unique_anchor, reco_acceptance + unique_anchor, 1)

    text = _replace_once(
        text,
        "  add InputArray JetEnergyScale/jets jets\n",
        "  add InputArray H4lRecoJetAcceptance/jets jets\n",
    )
    text = _replace_once(
        text,
        "  add Branch GenJetFinder/jets GenJet Jet\n",
        "  add Branch H4lGenJetAcceptance/jets GenJet Jet\n",
    )
    return text


def _add_output_branches(text: str) -> str:
    text = _insert_after(
        text,
        "  add Branch Delphes/allParticles Particle GenParticle\n",
        "\n"
        "  # Explicit post-shower status-1 particles (bare, before dressing).\n"
        "  add Branch Delphes/stableParticles StableParticle GenParticle\n",
    )
    text = _insert_after(
        text,
        "  add Branch Delphes/stableParticles StableParticle GenParticle\n",
        "  # Prompt W/Z/gamma*-origin leptons dressed with eligible status-1 photons.\n"
        "  add Branch DressedElectronFilter/electrons DressedElectron GenParticle\n"
        "  add Branch DressedMuonFilter/muons DressedMuon GenParticle\n",
    )
    text = _insert_after(
        text,
        "  add Branch UniqueObjectFinder/electrons Electron Electron\n",
        "  # H4l response objects before and after the isolation efficiency.\n"
        "  add Branch H4lElectronRecoID/electrons RecoElectronNoIso Electron\n"
        "  add Branch H4lElectronIsolation/electrons RecoElectron Electron\n",
    )
    text = _insert_after(
        text,
        "  add Branch UniqueObjectFinder/muons Muon Muon\n",
        "  add Branch H4lMuonRecoID/muons RecoMuonNoIso Muon\n"
        "  add Branch H4lMuonIsolation/muons RecoMuon Muon\n",
    )
    return text


def prepare_card(text: str) -> str:
    """Return the H4l response variant of a bundled Delphes ATLAS card."""

    text = _add_h4l_lepton_modules(text)
    text = _configure_jets(text)
    text = _add_output_branches(text)

    return (
        "# FourLeptonUnfolding H->ZZ->4l response card.\n"
        "# Derived at run time from Delphes's bundled ATLAS card.\n"
        "# RecoElectron/RecoMuon include smearing, Loose reco+ID, and loose isolation.\n"
        "# Identical truth/reco kinematic and event cuts are applied downstream.\n\n"
        + text
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_card", type=Path)
    parser.add_argument("output_card", type=Path)
    args = parser.parse_args()

    source = args.input_card.read_text(encoding="utf-8")
    args.output_card.write_text(prepare_card(source), encoding="utf-8")


if __name__ == "__main__":
    main()
