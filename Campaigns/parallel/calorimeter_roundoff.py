"""Distinguish cross-node floating-point rounding from a detector change."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import uproot

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
DIAG = STUDY / 'runs/calorimeter_diagnosis_20261114'
CASE = DIAG / 'cases/gg_H_pythia_down_000000'
ANGLES = {'born_cos_theta_star', 'born_cos_theta1', 'born_cos_theta2',
          'born_Phi', 'born_Phi1', 'born_Psi'}
BOUND = 8 * np.finfo(np.float64).eps


def assess(original, fixed):
    details = {}
    with uproot.open(original) as a, uproot.open(fixed) as b:
        a, b = a['Analysis'], b['Analysis']
        if a.num_entries != 10000 or b.num_entries != 10000 or set(a.keys()) != set(b.keys()):
            raise ValueError('Schema/count mismatch')
        for name in a.keys():
            x, y = a[name].array(library='np'), b[name].array(library='np')
            equal = x == y
            if x.dtype.kind in 'fc':
                equal |= np.isnan(x) & np.isnan(y)
            if np.all(equal):
                continue
            if name not in ANGLES or x.dtype != np.float64 or y.dtype != np.float64:
                raise ValueError('Unexpected changed branch: ' + name)
            if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                raise ValueError('Finite-value mask changed: ' + name)
            maximum = float(np.nanmax(np.abs(x-y)))
            if maximum > BOUND:
                raise ValueError('Difference exceeds eight double-precision epsilons: ' + name)
            details[name] = {'unequal_values': int(np.count_nonzero(~equal)),
                             'maximum_absolute_difference': maximum}
    return details


def main():
    evidence = json.loads((CASE / 'comparison.json').read_text())
    old_analysis = Path(evidence['original_status']['analysis_output'])
    fixed_analysis = Path(evidence['fixed_analysis'])
    differences = assess(old_analysis, fixed_analysis)
    outputs = CASE / 'same_node'
    outputs.mkdir(exist_ok=False)
    old = Path(evidence['original_status']['work_directory']) / 'simulation/generation_muon_scale_down/delphes.root'
    fixed = CASE / 'fixed/simulation/generation_muon_scale_down/delphes.root'
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for label, source in [('original', old), ('fixed', fixed)]:
        subprocess.run([sys.executable, str(DIAG / 'runtime/Analysis/build_analysis_tree.py'),
                        str(source), '--output', str(outputs / (label + '.root')),
                        '--mass-region', 'extended', '--require-lhe', '--require-born'],
                       env=environment, check=True)
    spec = importlib.util.spec_from_file_location('calorimeter_compare', DIAG / 'control/calorimeter_compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compared = module.compare(outputs / 'original.root', outputs / 'fixed.root')
    if not compared['all_branch_values_identical']:
        raise RuntimeError('Original and fixed detector outputs differ on the same analysis node')
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'case': CASE.name,
              'node': os.environ.get('SLURMD_NODENAME'), 'job_id': os.environ.get('SLURM_JOB_ID'),
              'cross_node_differences': differences, 'absolute_bound': BOUND,
              'same_node_comparison': compared, 'passed': True}
    (CASE / 'roundoff-certificate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
