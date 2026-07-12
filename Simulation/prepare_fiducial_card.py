#!/usr/bin/env python3
"""Make a loose H->4l fiducial-study card from the bundled ATLAS card.

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


def _lower_lepton_threshold(
    text: str, module_name: str, old_threshold: str = "10.0", new_threshold: str = "5.0"
) -> str:
    declaration = f"module Efficiency {module_name} {{"
    start, end, block = _module_block(text, declaration)
    low_expression = f"pt <= {old_threshold}"
    high_expression = f"pt > {old_threshold}"
    if low_expression not in block or high_expression not in block:
        raise ValueError(
            f"{module_name} does not have the expected {old_threshold} GeV threshold"
        )
    block = block.replace(low_expression, f"pt <= {new_threshold}")
    block = block.replace(high_expression, f"pt > {new_threshold}")
    return text[:start] + block + text[end:]


def _extend_eta_acceptance(text: str, module_name: str) -> str:
    declaration_by_name = {
        "ElectronTrackingEfficiency": "module Efficiency ElectronTrackingEfficiency {",
        "MuonTrackingEfficiency": "module Efficiency MuonTrackingEfficiency {",
        "ElectronMomentumSmearing": "module MomentumSmearing ElectronMomentumSmearing {",
        "MuonMomentumSmearing": "module MomentumSmearing MuonMomentumSmearing {",
        "ElectronEfficiency": "module Efficiency ElectronEfficiency {",
    }
    declaration = declaration_by_name[module_name]
    start, end, block = _module_block(text, declaration)
    if "2.5" not in block:
        raise ValueError(f"{module_name} does not have the expected eta=2.5 boundary")
    block = block.replace("2.5", "2.7")
    return text[:start] + block + text[end:]


def _insert_after(text: str, anchor: str, addition: str) -> str:
    if addition.strip() in text:
        return text
    if text.count(anchor) != 1:
        raise ValueError(f"expected exactly one card line matching: {anchor.strip()}")
    return text.replace(anchor, anchor + addition, 1)


def _add_truth_dressing_modules(text: str) -> str:
    execution_anchor = "set ExecutionPath {\n"
    execution_modules = (
        "  TruthLeptonFilter\n"
        "  TruthPhotonFilter\n"
        "  TruthLeptonDressing\n"
        "  DressedElectronFilter\n"
        "  DressedMuonFilter\n"
    )
    text = _insert_after(text, execution_anchor, execution_modules)

    module_anchor = (
        "#################################\n"
        "# Propagate particles in cylinder\n"
        "#################################\n"
    )
    modules = r"""
##########################################
# Particle-level lepton dressing for H->4l
##########################################

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

"""
    if "module LeptonDressing TruthLeptonDressing {" not in text:
        if text.count(module_anchor) != 1:
            raise ValueError("could not locate the ParticlePropagator section")
        text = text.replace(module_anchor, modules + module_anchor, 1)
    return text


def prepare_card(text: str) -> str:
    """Return the fiducial-study variant of a Delphes ATLAS card."""

    text = _lower_lepton_threshold(text, "ElectronEfficiency")
    text = _lower_lepton_threshold(text, "MuonEfficiency")
    for module_name in (
        "ElectronTrackingEfficiency",
        "MuonTrackingEfficiency",
        "ElectronMomentumSmearing",
        "MuonMomentumSmearing",
        "ElectronEfficiency",
    ):
        text = _extend_eta_acceptance(text, module_name)
    text = _add_truth_dressing_modules(text)

    text = _insert_after(
        text,
        "  add Branch Delphes/allParticles Particle GenParticle\n",
        "\n"
        "  # Explicit post-shower status-1 truth particles (no photon dressing).\n"
        "  add Branch Delphes/stableParticles StableParticle GenParticle\n",
    )
    text = _insert_after(
        text,
        "  add Branch Delphes/stableParticles StableParticle GenParticle\n",
        "  # Fiducial truth leptons dressed with non-hadronic status-1 photons.\n"
        "  add Branch DressedElectronFilter/electrons DressedElectron GenParticle\n"
        "  add Branch DressedMuonFilter/muons DressedMuon GenParticle\n",
    )
    text = _insert_after(
        text,
        "  add Branch UniqueObjectFinder/electrons Electron Electron\n",
        "  # Loose, pre-isolation reconstructed objects for response studies.\n"
        "  add Branch ElectronEfficiency/electrons RecoElectron Electron\n",
    )
    text = _insert_after(
        text,
        "  add Branch UniqueObjectFinder/muons Muon Muon\n",
        "  add Branch MuonEfficiency/muons RecoMuon Muon\n",
    )

    return (
        "# FourLeptonUnfolding fiducial-study card.\n"
        "# Derived at run time from Delphes's bundled ATLAS card.\n"
        "# Analysis-level lepton pT, isolation, pairing, and mass cuts belong downstream.\n\n"
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
