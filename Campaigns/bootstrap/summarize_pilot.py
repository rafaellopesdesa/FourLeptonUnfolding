"""Validate a bounded campaign, persist shards, and record measured acceptance."""
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
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


def summarize(process, run_area='pilot', campaign_tag='pilot_20261113', expected_entries=1000,
              expected_shards=1):
    if process not in {'gg_H', 'ZZ'}:
        raise ValueError('Expected gg_H or ZZ')
    if not all(re.fullmatch(r'[a-zA-Z0-9_]+', value) for value in (run_area, campaign_tag)):
        raise ValueError('Run area and campaign tag must be simple identifiers')
    if expected_entries not in (1000, 10000):
        raise ValueError('Only bounded 1000/10000-event pilots are supported')
    if expected_shards not in (1, 100, 200):
        raise ValueError('Expected one pilot shard or a bounded 100/200-shard production wave')
    base = STUDY / 'runs' / run_area
    # Inventory once per process summary, then account for each new compact
    # copy. Rewalking all previously saved shards for every file scales poorly.
    project_used = int(subprocess.check_output(['du', '-sb', str(PROJECT)]).split()[0])
    records = []
    for shower, scale in [('pythia', 'nominal'), ('pythia', 'up'),
                          ('pythia', 'down'), ('herwig', 'nominal')]:
        patterns = [f'{process}_{shower}_*_{campaign_tag}_job_*.root']
        if scale != 'nominal':
            # Direct immutable-runtime submissions retain the scale suffix in
            # CAMPAIGN_NAME as well as in the final task label.
            patterns.append(f'{process}_{shower}_*_{campaign_tag}_muon_scale_{scale}_job_*.root')
        paths = sorted({path for pattern in patterns
                        for path in (base / 'shards' / scale).glob(pattern)})
        record = {'process': process, 'shower': shower, 'scale': scale,
                  'files': [], 'errors': []}
        if len(paths) != expected_shards:
            record['errors'].append(f'Expected {expected_shards} shards; found {len(paths)}')
        for path in paths:
            try:
                checked = validate_analysis_output(path, expected_entries=expected_entries,
                                                    require_lhe=True, require_born=True)
                with uproot.open(path) as source:
                    sources = json.loads(str(source['lhe_metadata']))['sources']
                    if len(sources) != 1:
                        raise ValueError('Expected one hard-event source per unmerged shard')
                    provenance = sources[0]
                    arrays = source['Analysis'].arrays(
                        ['reconstructed', 'fiducial', 'reco_m_ZZ', 'weight',
                         'born_available', 'born_kinematics_valid', 'born_n_taus'], library='np')
                selected = arrays['reconstructed'].astype(bool)
                signal = selected & (arrays['reco_m_ZZ'] > 115) & (arrays['reco_m_ZZ'] < 130)
                weights = arrays['weight']
                item = {'path': str(path), 'validation': asdict(checked),
                        'hard_source_events': int(provenance['lhe_event_count']),
                        'hard_source_sha256': provenance['lhe_sha256'],
                        'showered_events': checked.entries,
                        'repeated_showered_sources': int(provenance['hepmc_repeated_source_events']),
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
                destination = PROJECT / 'samples' / campaign_tag / scale / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if project_used + item['bytes'] > 2_000_000_000_000:
                    raise RuntimeError('Project storage target would be exceeded')
                if destination.exists():
                    if digest(destination) != item['sha256']:
                        raise ValueError(f'Existing persisted shard differs: {destination}')
                else:
                    # Only a bounded compact pilot shard is copied here.
                    temporary = destination.with_suffix('.partial')
                    if temporary.exists():
                        raise FileExistsError(temporary)
                    shutil.copyfile(path, temporary)
                    if digest(temporary) != item['sha256']:
                        raise ValueError('Copied shard checksum mismatch')
                    temporary.rename(destination)
                    project_used += item['bytes']
                item['persisted_path'] = str(destination)
                record['files'].append(item)
            except Exception as error:
                record['errors'].append(f'{path.name}: {type(error).__name__}: {error}')
        record['totals'] = {name: sum(item[name] for item in record['files'])
                            for name in ('reconstructed', 'reconstructed_signal_window',
                                         'fiducial', 'negative_weight_entries', 'bytes',
                                         'showered_events', 'repeated_showered_sources')}
        hard_sources = {item['hard_source_sha256']: item['hard_source_events'] for item in record['files']}
        record['totals']['distinct_hard_sources'] = len(hard_sources)
        record['totals']['hard_source_events'] = sum(hard_sources.values())
        records.append(record)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'process': process,
              'samples': records, 'physics_closure_performed': False}
    result['expected_entries_per_shard'] = expected_entries
    result['expected_shards_per_configuration'] = expected_shards
    result['count_semantics'] = 'Showered/compact rows are distinct from supplied POWHEG hard events; hard sources are deduplicated by SHA256 within each configuration.'
    output = base / f'{process}-summary.json'
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    history = base / f'{process}-summary-{os.environ.get("SLURM_JOB_ID", "interactive")}.json'
    with history.open('x') as stream:
        stream.write(payload)
    output.write_text(payload)
    # Serialize concurrent process summaries without overwriting the narrative.
    with (STUDY / 'docs/.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (STUDY / 'docs/CURRENT_STATUS.md').open('a') as status:
            status.write(f'\nPilot update {result["updated_utc"]}: {process}; '
                         f'{sum(bool(r["errors"]) for r in records)} configurations with errors. '
                         f'Measured details: `runs/{run_area}/{process}-summary.json`.\n')
            for record in records:
                status.write(f'- {record["shower"]} {record["scale"]}: '
                             f'{record["totals"]["reconstructed"]} selected / '
                             f'{record["totals"]["showered_events"]} showered rows from '
                             f'{record["totals"]["hard_source_events"]} supplied hard events; '
                             f'{record["totals"]["repeated_showered_sources"]} repeated source occurrences.\n')
    print(json.dumps({'process': process, 'manifest': str(output), 'samples': [
        {k: r[k] for k in ('shower', 'scale', 'totals', 'errors')} for r in records]}, indent=2))
    return int(any(r['errors'] for r in records))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        raise SystemExit(summarize(sys.argv[1]))
    if len(sys.argv) == 5:
        raise SystemExit(summarize(*sys.argv[1:4], int(sys.argv[4])))
    if len(sys.argv) == 6:
        raise SystemExit(summarize(*sys.argv[1:4], int(sys.argv[4]), int(sys.argv[5])))
    raise SystemExit('Expected PROCESS [RUN_AREA CAMPAIGN_TAG ENTRIES [SHARDS]]')
