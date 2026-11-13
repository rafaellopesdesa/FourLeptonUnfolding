"""Exercise the training readers on real pilot MC without fitting a network."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import uproot

from Systematics.train import read_sample
from omnifold.workflow import _read_sample

torch.set_num_threads(1)
study = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
base = study / 'runs/pilot/diagnostics/1634795'
records = []
for shower, scale in [('pythia', 'nominal'), ('pythia', 'up'),
                      ('pythia', 'down'), ('herwig', 'nominal')]:
    path = base / f'gg_H_{shower}_{scale}.root'
    for region in ('signal', 'extended'):
        for fiducial in (False, True):
            sample = read_sample(path, shower=shower, scale=scale, region=region,
                                 fiducial=fiducial)
            assert sample.features.shape == (len(sample.weights), 7)
            assert sample.processes == {'gg_H'} and sample.luminosity_fb == 312.
            records.append({'sample': path.name, 'reader': 'systematic',
                            'region': region, 'fiducial': fiducial,
                            'entries': len(sample.weights), 'sumw': float(sample.weights.sum()),
                            'split_counts': {str(int(k)): int(v) for k, v in
                                             zip(*np.unique(sample.splits, return_counts=True))}})
    if scale == 'nominal':
        kind = 'mc' if shower == 'pythia' else 'truth'
        sample = _read_sample(path, kind=kind, weight_branch='weight', seed=19110907)
        assert sample.truth.shape == (len(sample.weights), 7)
        records.append({'sample': path.name, 'reader': 'omnifold', 'kind': kind,
                        'entries': len(sample.weights), 'matched': int(sample.matched.sum()),
                        'sumw': float(sample.weights.sum())})
    with uproot.open(path) as source:
        assert source['Analysis'].num_entries == 1000
result = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'checks': records,
          'neural_training_performed': False}
destination = study / 'runs/pilot' / f'training-inputs-{os.environ["SLURM_JOB_ID"]}.json'
with destination.open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(result, indent=2))
