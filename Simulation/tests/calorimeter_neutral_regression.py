"""Executable regression: neutral particles with no tracks must not crash Delphes.

Run inside Slurm with the ROOT environment loaded. Tests the actual reader and
calorimeter, and checks the fixed output energy rather than source-code strings.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import awkward as ak
import numpy as np
import uproot

CARD = '''set RandomSeed 7
set ExecutionPath {ParticlePropagator ECal TreeWriter}
module ParticlePropagator ParticlePropagator {
  set InputArray Delphes/stableParticles
  set OutputArray stableParticles
  set ChargedHadronOutputArray chargedHadrons
  set ElectronOutputArray electrons
  set MuonOutputArray muons
  set Radius 1.15
  set HalfLength 3.51
  set Bz 2.0
}
module SimpleCalorimeter ECal {
  set ParticleInputArray ParticlePropagator/stableParticles
  set TrackInputArray ParticlePropagator/chargedHadrons
  set TowerOutputArray towers
  set EFlowTrackOutputArray eflowTracks
  set EFlowTowerOutputArray eflowPhotons
  set IsEcal true
  set EnergyMin 0.0
  set EnergySignificanceMin 0.0
  set SmearTowerCenter false
  add EtaPhiBins {-5.0 -1.0 0.0 1.0 5.0} {-3.141592653589793 -1.0 0.0 1.0 3.141592653589793}
  add EnergyFraction {0} {1.0}
  set ResolutionFormula {0.0}
}
module TreeWriter TreeWriter {
  add Branch ECal/towers Tower Tower
  add Branch ECal/eflowTracks Track Track
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--fixed', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    card = args.output / 'neutral-only.tcl'
    card.write_text(CARD)
    hepmc = args.output / 'neutral-only.hepmc'
    photons = [(10., 0., 5.), (-3., 4., -2.)]
    energies = [math.sqrt(sum(x*x for x in p)) for p in photons]
    lines = ['HepMC::Version 2.06.09', 'HepMC::IO_GenEvent-START_EVENT_LISTING',
             'E 0 0 1 0 0 0 -1 1 0 0 0 1 1.0', 'U GEV MM', 'V -1 0 0 0 0 0 0 2 0']
    for i, ((px,py,pz), energy) in enumerate(zip(photons, energies), 1):
        lines.append(f'P {i} 22 {px} {py} {pz} {energy:.17g} 0 1 0 0 0 0')
    lines.append('HepMC::IO_GenEvent-END_EVENT_LISTING')
    hepmc.write_text('\n'.join(lines) + '\n')
    result = {}
    for label, binary in [('original', args.original), ('fixed', args.fixed)]:
        output = args.output / f'{label}.root'
        log = args.output / f'{label}.log'
        environment = dict(os.environ)
        environment['LD_LIBRARY_PATH'] = str(binary.parent) + ':' + environment.get('LD_LIBRARY_PATH', '')
        with log.open('w') as stream:
            run = subprocess.run([str(binary), str(card), str(output), str(hepmc)],
                                 stdout=stream, stderr=subprocess.STDOUT, env=environment)
        result[label] = {'returncode': run.returncode, 'log': str(log)}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['original']['returncode'] != 0, 'Original must reproduce the memory failure'
    assert 'SimpleCalorimeter::Process' in (args.output / 'original.log').read_text()
    assert result['fixed']['returncode'] == 0, 'Fixed reader must process the same event'
    with uproot.open(args.output / 'fixed.root') as stream:
        tree = stream['Delphes']
        assert tree.num_entries == 1
        assert ak.to_list(ak.num(tree['Track.PT'].array())) == [0]
        measured = float(ak.sum(tree['Tower.E'].array()))
        np.testing.assert_allclose(measured, sum(energies), rtol=2e-6, atol=1e-6)
    result.update(passed=True, events=1, tracks=0, expected_energy=sum(energies), measured_energy=measured)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
