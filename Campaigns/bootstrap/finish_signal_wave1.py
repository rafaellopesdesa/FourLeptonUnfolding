"""Recover known failed shards once, using immutable code, before final summary."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
BASE = STUDY / 'runs/production_signal_wave1'
SNAPSHOT = STUDY / 'runs/runtime_snapshots/e07de718c4d3967c8efd8f752f5ed1f282be9305'
CAMPAIGNS = [
    ('gg_H_pythia_production_wave1_20261113', 30000001, ''),
    ('gg_H_pythia_production_wave1_20261113_muon_scale_up', 30000001, '_muon_scale_up'),
    ('gg_H_pythia_production_wave1_20261113_muon_scale_down', 30000001, '_muon_scale_down'),
    ('gg_H_herwig_production_wave1_20261113', 40000001, ''),
]


def settings(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def plan(allow_pending=False):
    result = []
    for name, seed, suffix in CAMPAIGNS:
        campaign = BASE / 'campaigns' / name
        failed = []
        for index in range(100):
            job = campaign / 'jobs' / f'job_{index:06d}_seed{seed+index}{suffix}'
            status = job / 'slurm-status.txt'
            if not status.exists():
                if allow_pending: continue
                raise RuntimeError(f'Missing terminal job status: {job}')
            data = settings(status)
            output = Path(data['analysis_output'])
            if not output.resolve().is_relative_to((BASE / 'shards').resolve()):
                raise ValueError('Unexpected output path')
            if (job / 'SUCCESS').exists() and output.is_file():
                continue
            if data['status'] == 'running' and allow_pending:
                continue
            if output.exists() or not (job / 'FAILED').exists() or data['status'] != 'failed':
                raise RuntimeError(f'Inconsistent terminal state: {job}')
            parent = data['slurm_array_job_id']
            task = data['slurm_array_task_id']
            candidates = [campaign / 'logs' / f'{prefix}-{parent}_{task}.out'
                          for prefix in ('events', 'recovery')]
            wrapper = next((p for p in candidates if p.is_file()), None)
            wrapper_text = wrapper.read_text() if wrapper else ''
            native = job / 'diagnostics/generation-herwig.log'
            native_text = native.read_text() if native.is_file() else ''
            if 'error reading input file: Stale file handle' in wrapper_text:
                cause = 'runtime_script_replaced_on_nfs'
            elif 'More events requested than available in LesHouchesReader' in native_text:
                cause = 'herwig_finite_lhe_exhaustion'
            elif 'RuntimeError: HepMC changed while validating its source identities' in wrapper_text:
                # Retry from the same seed with the full integrity check still
                # enabled. This labels the observed failure, not its root cause.
                cause = 'hepmc_changed_during_validation'
            else:
                raise RuntimeError(f'Unknown failure requires investigation: {job}')
            failed.append({'index': index, 'job_directory': str(job),
                           'wrapper_log': str(wrapper) if wrapper else None, 'cause': cause})
        result.append({'campaign': name, 'failed': failed})
    return result


def main():
    if '--plan-only' in sys.argv:
        print(json.dumps(plan(allow_pending=True), indent=2))
        return
    metadata = json.loads((SNAPSHOT / 'SNAPSHOT.json').read_text())
    if metadata['tree'] != '20efbbe8af1efa630df413398da70c7c1fe7361b':
        raise RuntimeError('Unexpected immutable code snapshot')
    records = plan()
    record = BASE / 'completion_recovery' / os.environ['SLURM_JOB_ID']
    record.mkdir(parents=True, exist_ok=False)
    (record / 'plan.json').write_text(json.dumps(records, indent=2) + '\n')
    submitted = []
    for item in records:
        if not item['failed']:
            continue
        campaign = BASE / 'campaigns' / item['campaign']
        saved = record / item['campaign']
        saved.mkdir()
        for failure in item['failed']:
            source = Path(failure['job_directory'])
            shutil.copytree(source, saved / source.name)
            if failure['wrapper_log']:
                shutil.copyfile(failure['wrapper_log'], saved / Path(failure['wrapper_log']).name)
        config = saved / 'campaign.env'
        payload, count = re.subn(r'(?m)^REPO_ROOT=.*$',
            'REPO_ROOT=' + shlex.quote(str(SNAPSHOT)), (campaign / 'campaign.env').read_text())
        if count != 1: raise ValueError('Missing unique REPO_ROOT')
        if any(f['cause'] == 'hepmc_changed_during_validation' for f in item['failed']):
            # Keep evidence if the unexplained integrity failure recurs.
            payload, count = re.subn(r'(?m)^KEEP_INTERMEDIATES=.*$',
                                    'KEEP_INTERMEDIATES=1', payload)
            if count != 1: raise ValueError('Missing unique KEEP_INTERMEDIATES')
        config.write_text(payload)
        config.chmod(0o444)
        indices = ','.join(str(f['index']) for f in item['failed'])
        job_id = subprocess.check_output([
            'sbatch', '--parsable', '--account=pi_rclsa_umass_edu', '--partition=cpu',
            '--qos=normal', '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=3G',
            '--time=01:00:00', '--array=' + indices + '%4', '--export=NONE',
            '--job-name=wave1_completion', '--output=' + str(campaign / 'logs/recovery-%A_%a.out'),
            str(SNAPSHOT / 'BatchSubmit/unity_generation_job.sh'), 'events', str(config), '0'],
            text=True).strip().split(';')[0]
        if not job_id.isdigit(): raise ValueError('Invalid submitted job ID')
        item['recovery_job_id'] = job_id
        submitted.append(job_id)
        (record / 'submitted.json').write_text(json.dumps(records, indent=2) + '\n')
        # Replace the summary dependency before this controller can finish.
        subprocess.run(['scontrol', 'update', 'JobId=1636701',
                        'Dependency=afterany:' + ':'.join([os.environ['SLURM_JOB_ID'], *submitted])], check=True)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'campaigns': records,
              'code_snapshot': str(SNAPSHOT), 'automatic_recovery_rounds': 1}
    (record / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
