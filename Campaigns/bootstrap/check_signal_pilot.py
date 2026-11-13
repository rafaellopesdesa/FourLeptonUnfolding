"""Normalize the bounded signal pilot and inspect the confirmed HB scan.

This is a finite-sample diagnostic, not unfolding or a coefficient fit.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import uproot

from Analysis.merge_analysis_outputs import merge_sample
from SMEFT.application import decorate_file
from SMEFT.backend import NativeHBBackend

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
PROJECT = Path('/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    job_id = os.environ['SLURM_JOB_ID']
    output = STUDY / 'runs/pilot/diagnostics' / job_id
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((STUDY / 'runs/pilot/gg_H-summary.json').read_text())
    if any(record['errors'] or len(record['files']) != 1 for record in summary['samples']):
        raise ValueError('All four signal configurations must pass shard validation')
    records, identities = [], {}
    backend = NativeHBBackend(build_dir=STUDY / 'install/build/smeft-pilot')
    for sample in summary['samples']:
        item = sample['files'][0]
        source = Path(item['path'])
        if digest(source) != item['sha256']:
            raise ValueError('Pilot source changed since validation')
        name = f"gg_H_{sample['shower']}_{sample['scale']}"
        normalized = output / f'{name}.root'
        stats = merge_sample([source], normalized, step_size='16 MB',
                             cross_section_override_pb=None, luminosity_fb=312., overwrite=False)
        with uproot.open(normalized) as root:
            arrays = root['Analysis'].arrays(library='np')
        weights = arrays['weight']
        selected = arrays['reconstructed'].astype(bool)
        signal = selected & (arrays['reco_m_ZZ'] > 115) & (arrays['reco_m_ZZ'] < 130)
        np.testing.assert_allclose(weights.sum(), stats.cross_section_pb * 312000., rtol=1e-10)
        np.testing.assert_allclose(arrays['weight_shape'].sum(), 1000., rtol=1e-10)
        ids = np.column_stack([arrays['lhe_source_id'], arrays['lhe_event_index']])
        if sample['shower'] == 'pythia':
            identities[sample['scale']] = ids
        record = {'sample': name, 'cross_section_pb': stats.cross_section_pb,
                  'inclusive_expected_events_312fb': float(weights.sum()),
                  'selected_expected_events_312fb': float(weights[selected].sum()),
                  'signal_window_expected_events_312fb': float(weights[signal].sum()),
                  'selected_mc_effective_entries': float(weights[selected].sum()**2 / np.square(weights[selected]).sum()),
                  'signed_poisson_variance_over_net_selected': float(np.abs(weights[selected]).sum() / weights[selected].sum()),
                  'normalized_file': str(normalized), 'normalized_sha256': digest(normalized)}
        if sample['scale'] == 'nominal':
            decorated = output / f'{name}_hb_scan.root'
            metadata = decorate_file(normalized, decorated, [-.1, 0., .1], backend=backend,
                                     lambda_tev=1., unsupported='unity', overwrite=False)
            with uproot.open(decorated) as root:
                scan = root['Analysis'].arrays(['smeft_hb_ratio', 'smeft_hb_status',
                                                'smeft_hb_valid'], library='np')
            ratios = scan['smeft_hb_ratio']
            np.testing.assert_array_equal(ratios[:, 1], np.ones(len(weights)))
            if not np.all(np.isfinite(ratios)) or np.any(ratios < 0):
                raise ValueError('Invalid squared-amplitude ratio')
            record['hb_scan'] = {
                'coefficients': [-.1, 0., .1], 'lambda_tev': 1.,
                'unsupported_policy': 'unity; explicitly counted, not evaluated',
                'status_counts_all': metadata['status_counts'],
                'status_counts_selected': {str(int(code)): int((selected & (scan['smeft_hb_status'] == code)).sum())
                                           for code in np.unique(scan['smeft_hb_status'])},
                'ratio_quantiles_selected': np.quantile(ratios[selected], [0, .01, .5, .99, 1], axis=0).tolist(),
                'quantile_levels': [0, .01, .5, .99, 1],
                'inclusive_expected_events': np.sum(weights[:, None] * ratios, axis=0).tolist(),
                'selected_expected_events': np.sum(weights[selected, None] * ratios[selected], axis=0).tolist(),
                'signal_window_expected_events': np.sum(weights[signal, None] * ratios[signal], axis=0).tolist(),
                'decorated_file': str(decorated), 'decorated_sha256': digest(decorated)}
        records.append(record)
    for scale in ('up', 'down'):
        np.testing.assert_array_equal(identities['nominal'], identities[scale])
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'job_id': job_id,
              'luminosity_fb': 312., 'pythia_hard_event_identities_match': True,
              'samples': records, 'physics_closure_performed': False,
              'limitations': ['1000 generated events per configuration; no precision acceptance claim',
                             'Tau signal rows retained unchanged under an explicit diagnostic policy',
                             'Zero selected negative weights in a small pilot does not establish positivity at production scale',
                             'Merged and SMEFT-decorated MC are weighted templates, not Poisson pseudo-data']}
    (output / 'signal-checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    destination = PROJECT / 'samples/pilot_20261113/diagnostics' / job_id
    if destination.exists():
        raise FileExistsError(destination)
    # Six compact 1,000-event files plus JSON; reject unexpected output growth.
    if sum(path.stat().st_size for path in output.iterdir()) > 20_000_000:
        raise RuntimeError('Bounded pilot diagnostic exceeded 20 MB')
    shutil.copytree(output, destination)
    for path in output.iterdir():
        if digest(path) != digest(destination / path.name):
            raise RuntimeError('Persisted pilot diagnostic checksum mismatch')
    with (STUDY / 'docs/.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (STUDY / 'docs/CURRENT_STATUS.md').open('a') as status:
            status.write(f'\nSignal normalization/HB pilot checks passed in Slurm {job_id}; '
                         f'details: `runs/pilot/diagnostics/{job_id}/signal-checks.json`.\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
