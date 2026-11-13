"""Validate the bounded pilot, persist shards, and record measured acceptance."""
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import uproot
from Analysis.validate_analysis_output import validate_analysis_output

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
PROJECT = Path('/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(process):
    if process not in {'gg_H', 'ZZ'}:
        raise ValueError('Expected gg_H or ZZ')
    records = []
    for shower, scale in [('pythia', 'nominal'), ('pythia', 'up'),
                          ('pythia', 'down'), ('herwig', 'nominal')]:
        pattern = f'{process}_{shower}_*_pilot_20261113_job_*.root'
        paths = sorted((STUDY / 'runs/pilot/shards' / scale).glob(pattern))
        record = {'process': process, 'shower': shower, 'scale': scale,
                  'files': [], 'errors': []}
        if len(paths) != 1:
            record['errors'].append(f'Expected one shard; found {len(paths)}')
        for path in paths:
            try:
                checked = validate_analysis_output(path, expected_entries=1000,
                                                    require_lhe=True, require_born=True)
                with uproot.open(path) as source:
                    arrays = source['Analysis'].arrays(
                        ['reconstructed', 'fiducial', 'reco_m_ZZ', 'weight',
                         'born_available', 'born_kinematics_valid', 'born_n_taus'], library='np')
                selected = arrays['reconstructed'].astype(bool)
                signal = selected & (arrays['reco_m_ZZ'] > 115) & (arrays['reco_m_ZZ'] < 130)
                weights = arrays['weight']
                item = {'path': str(path), 'validation': asdict(checked),
                        'bytes': path.stat().st_size, 'sha256': digest(path),
                        'reconstructed': int(selected.sum()),
                        'reconstructed_signal_window': int(signal.sum()),
                        'fiducial': int(arrays['fiducial'].sum()),
                        'negative_weight_entries': int((weights < 0).sum()),
                        'reco_sumw': float(weights[selected].sum()),
                        'reco_sum_abs_w': float(np.abs(weights[selected]).sum()),
                        'reco_sumw2': float(np.square(weights[selected]).sum()),
                        'born_available_reco': int((selected & arrays['born_available']).sum()),
                        'born_valid_reco': int((selected & arrays['born_kinematics_valid']).sum()),
                        'born_tau_reco': int((selected & (arrays['born_n_taus'] > 0)).sum())}
                destination = PROJECT / 'samples/pilot_20261113' / scale / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                used = int(subprocess.check_output(['du', '-sb', str(PROJECT)]).split()[0])
                if used + item['bytes'] > 2_000_000_000_000:
                    raise RuntimeError('Project storage target would be exceeded')
                if destination.exists():
                    if digest(destination) != item['sha256']:
                        raise ValueError(f'Existing persisted shard differs: {destination}')
                else:
                    # Only a bounded 1,000-event compact shard is copied here.
                    temporary = destination.with_suffix('.partial')
                    if temporary.exists():
                        raise FileExistsError(temporary)
                    shutil.copyfile(path, temporary)
                    if digest(temporary) != item['sha256']:
                        raise ValueError('Copied shard checksum mismatch')
                    temporary.rename(destination)
                item['persisted_path'] = str(destination)
                record['files'].append(item)
            except Exception as error:
                record['errors'].append(f'{path.name}: {type(error).__name__}: {error}')
        records.append(record)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'process': process,
              'samples': records, 'physics_closure_performed': False}
    output = STUDY / f'runs/pilot/{process}-summary.json'
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    history = STUDY / f'runs/pilot/{process}-summary-{os.environ.get("SLURM_JOB_ID", "interactive")}.json'
    with history.open('x') as stream:
        stream.write(payload)
    output.write_text(payload)
    # Serialize concurrent process summaries without overwriting the narrative.
    with (STUDY / 'docs/.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (STUDY / 'docs/CURRENT_STATUS.md').open('a') as status:
            status.write(f'\nPilot update {result["updated_utc"]}: {process}; '
                         f'{sum(bool(r["errors"]) for r in records)} configurations with errors. '
                         f'Measured details: `runs/pilot/{process}-summary.json`.\n')
    print(json.dumps(result, indent=2))
    return int(any(r['errors'] for r in records))


if __name__ == '__main__':
    raise SystemExit(summarize(sys.argv[1]))
