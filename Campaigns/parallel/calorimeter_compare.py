"""Rerun retained HepMC with the fixed build and compare every analysis branch."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import uproot

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
BASE = STUDY / 'runs/calorimeter_diagnosis_20261114'
RUNTIME = BASE / 'runtime'


def values(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def compare(left, right):
    differences = {}
    with uproot.open(left) as a, uproot.open(right) as b:
        a, b = a['Analysis'], b['Analysis']
        if a.num_entries != b.num_entries or set(a.keys()) != set(b.keys()):
            raise ValueError('Tree entries/schema changed')
        entries = a.num_entries
        branches = len(a.keys())
        for name in a.keys():
            x, y = a[name].array(library='np'), b[name].array(library='np')
            match = (x == y)
            if x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
                match |= np.isnan(x) & np.isnan(y)
            if not np.all(match):
                differences[name] = {'unequal_values': int(np.count_nonzero(~match))}
    return {'entries': entries, 'branches': branches, 'differences': differences,
            'all_branch_values_identical': not differences}


def main(name):
    record = json.loads((BASE / 'submission.json').read_text())
    case = next(item for item in record['cases'] if item['name'] == name)
    config = Path(case['config'])
    if hashlib.sha256(config.read_bytes()).hexdigest() != case['config_sha256']:
        raise ValueError('Diagnostic config changed')
    settings = {k: shlex.split(v)[0] for k,v in values(config).items()}
    directory = config.parent
    original_jobs = list((directory / 'jobs').glob('job_*'))
    if len(original_jobs) != 1:
        raise ValueError('Expected exactly one original attempt')
    original = original_jobs[0]
    status = values(original / 'slurm-status.txt')
    if status['status'] not in {'complete', 'failed'}:
        raise ValueError('Original attempt is not terminal')
    generation = Path(status['work_directory']) / 'generation'
    if not generation.is_relative_to(directory) or not generation.is_dir():
        raise ValueError('Retained generation is missing or outside the case')
    if not (BASE / 'BUILD_SUCCESS').exists():
        raise ValueError('Fixed build has not passed')
    fixed = directory / 'fixed'
    fixed.mkdir(exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [str(RUNTIME / 'Simulation/run_simulation.sh'), str(generation),
               '--process', case['process'], '--output-root', str(fixed / 'simulation'),
               '--higgs-br', settings['HIGGS_BR'], '--muon-scale', case['scale']]
    subprocess.run(command, check=True, env=environment)
    suffix = '' if case['scale'] == 'nominal' else '_muon_scale_' + case['scale']
    delphes = fixed / 'simulation' / ('generation' + suffix) / 'delphes.root'
    analysis = fixed / 'analysis.root'
    subprocess.run([sys.executable, str(RUNTIME / 'Analysis/build_analysis_tree.py'), str(delphes),
                    '--output', str(analysis), '--mass-region', settings['MASS_REGION'],
                    '--require-lhe', '--require-born'], check=True, env=environment)
    subprocess.run([sys.executable, str(RUNTIME / 'Analysis/validate_analysis_output.py'), str(analysis),
                    '--expected-entries', settings['EVENTS_PER_JOB'], '--require-lhe', '--require-born'],
                   check=True, env=environment)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'case': case,
              'original_status': status, 'fixed_analysis': str(analysis),
              'fixed_analysis_sha256': hashlib.file_digest(analysis.open('rb'), 'sha256').hexdigest()}
    if status['status'] == 'complete':
        result['original_vs_fixed'] = compare(Path(status['analysis_output']), analysis)
    else:
        text = (original / 'diagnostics/delphes.log').read_text()
        result['reproduced_calorimeter_crash'] = ('segmentation violation' in text and
                                                 'SimpleCalorimeter::Process' in text)
        if not result['reproduced_calorimeter_crash']:
            raise RuntimeError('Original failed for a different reason')
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    if 'original_vs_fixed' in result and not result['original_vs_fixed']['all_branch_values_identical']:
        raise RuntimeError('Fixed simulation changed analysis values; inspect comparison.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
