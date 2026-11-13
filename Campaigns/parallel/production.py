"""Bounded two-process production, gated by local-disk pilots and storage checks."""
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
PROJECT = Path('/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
SNAPSHOT = STUDY / 'runs/runtime_snapshots/e07de718c4d3967c8efd8f752f5ed1f282be9305'
BASE = STUDY / 'runs/production_parallel_wave2'
TAG = 'production_parallel_wave2_20261113'
SHARDS = 200
EVENTS = 10000
CONCURRENCY = 25  # Four configurations per physics process: 100 each, 200 total.
PYTHON = STUDY / 'repository/Analysis/.pixi/envs/default/bin/python'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def scalar_config(path):
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split('=', 1)
        values = shlex.split(value)
        if len(values) != 1:
            raise ValueError('Expected a quoted scalar configuration')
        result[key] = values[0]
    return result


def check_storage():
    result = {}
    for root, limit, reserve in [(STUDY, 300_000_000_000, 50_000_000_000),
                                 (PROJECT, 2_000_000_000_000, 10_000_000_000)]:
        used = int(subprocess.check_output(['du', '-sb', str(root)]).split()[0])
        if used + reserve >= limit or shutil.disk_usage(root).free < reserve:
            raise RuntimeError(f'Insufficient storage headroom: {root}')
        result[str(root)] = {'bytes': used, 'reserved_headroom_bytes': reserve,
                             'soft_limit_bytes': limit}
    return result


def submit(arguments, command):
    result = subprocess.check_output([
        'sbatch', '--parsable', '--account=pi_rclsa_umass_edu', '--partition=cpu',
        '--qos=normal', '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--export=NONE',
        *arguments, '--wrap=' + shlex.join([str(a) for a in command])], text=True).strip().split(';')[0]
    if not result.isdigit():
        raise ValueError('Invalid submitted Slurm job ID')
    return result


def submit_array(plan, sample, indices, phase, hold=False):
    args = ['--mem=4G', '--time=01:00:00', '--constraint=localscratch',
            '--array=' + indices + '%' + str(CONCURRENCY),
            '--job-name=fourlepton_' + phase,
            '--output=' + str(Path(sample['directory']) / f'logs/{phase}-%A_%a.out')]
    if hold:
        args.append('--hold')
    return submit(args, [PYTHON, Path(plan['control']) / 'local_worker.py', sample['config']])


def append_status(message):
    with (STUDY / 'docs/.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (STUDY / 'docs/CURRENT_STATUS.md').open('a') as stream:
            stream.write('\n' + datetime.now(timezone.utc).isoformat() + ': ' + message + '\n')


def prepare():
    if BASE.exists():
        raise FileExistsError('Wave directory already exists; inspect its manifest before resuming')
    metadata = json.loads((SNAPSHOT / 'SNAPSHOT.json').read_text())
    if metadata['tree'] != '20efbbe8af1efa630df413398da70c7c1fe7361b':
        raise ValueError('Unexpected immutable runtime')
    for relative, checksum in metadata['environment_sha256'].items():
        if sha(SNAPSHOT / relative) != checksum:
            raise ValueError('Runtime environment changed')
    storage = check_storage()
    for process, summary in [('gg_H', STUDY / 'runs/production_signal_wave1/gg_H-summary.json'),
                              ('ZZ', STUDY / 'runs/pilot10k/ZZ-summary.json')]:
        data = json.loads(summary.read_text())
        if len(data['samples']) != 4:
            raise ValueError('Incomplete prerequisite summary')
        for sample in data['samples']:
            if sample['errors'] or sample['totals']['reconstructed'] <= 0:
                raise ValueError('Prerequisite sample failed validation')
            for item in sample['files']:
                if sha(Path(item['persisted_path'])) != item['sha256']:
                    raise ValueError('Persisted prerequisite checksum mismatch')
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    control = BASE / 'control'
    control.mkdir()
    for name in ['production.py', 'local_worker.py']:
        shutil.copyfile(Path(__file__).parent / name, control / name)
    shutil.copyfile(STUDY / 'repository/Campaigns/bootstrap/summarize_pilot.py', control / 'summarize_pilot.py')
    for path in control.iterdir():
        path.chmod(0o444)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'tag': TAG,
            'shards_per_configuration': SHARDS, 'events_per_shard': EVENTS,
            'concurrency_per_configuration': CONCURRENCY, 'concurrency_per_process': 100,
            'total_concurrency': 200, 'control': str(control), 'runtime': str(SNAPSHOT),
            'storage': storage, 'control_sha256': {p.name: sha(p) for p in control.iterdir()},
            'samples': [], 'phase': 'prepared'}
    for process in ['gg_H', 'ZZ']:
        for shower, scale in [('pythia', 'nominal'), ('pythia', 'up'), ('pythia', 'down'), ('herwig', 'nominal')]:
            suffix = '' if scale == 'nominal' else '_muon_scale_' + scale
            if process == 'gg_H':
                prior = STUDY / 'runs/production_signal_wave1/campaigns' / f'gg_H_{shower}_production_wave1_20261113{suffix}'
                seed = 50_000_001 if shower == 'pythia' else 60_000_001
            else:
                prior = STUDY / 'runs/pilot10k/campaigns' / f'ZZ_{shower}_pilot10k_20261113{suffix}'
                seed = 70_000_001 if shower == 'pythia' else 80_000_001
            settings = scalar_config(prior / 'campaign.env')
            name = f'{process}_{shower}_{TAG}{suffix}'
            directory = BASE / 'campaigns' / name
            (directory / 'inputs').mkdir(parents=True)
            (directory / 'logs').mkdir()
            card = directory / 'inputs/powheg.input'
            shutil.copyfile(settings['RUN_CARD'], card)
            settings.update(REPO_ROOT=str(SNAPSHOT), CAMPAIGN_NAME=name, CAMPAIGN_DIR=str(directory),
                            JOBS=str(SHARDS), EVENTS_PER_JOB=str(EVENTS), TOTAL_EVENTS=str(SHARDS * EVENTS),
                            BASE_SEED=str(seed), REUSED_GRID='1', RUN_CARD=str(card), KEEP_INTERMEDIATES='0',
                            SCRATCH_ROOT='/localscratch/rclsa_umass_edu/FourLeptonStudy/production_parallel_wave2',
                            ANALYSIS_OUTPUT_ROOT=str(BASE / 'shards' / scale),
                            CREATED_UTC=datetime.now(timezone.utc).isoformat())
            config = directory / 'campaign.env'
            config.write_text(''.join(f'{k}={shlex.quote(v)}\n' for k, v in settings.items()))
            config.chmod(0o444)
            plan['samples'].append({'process': process, 'shower': shower, 'scale': scale,
                                    'name': name, 'directory': str(directory), 'config': str(config),
                                    'config_sha256': sha(config), 'base_seed': seed})
    save(BASE / 'plan.json', plan)
    submitted = []
    try:
        for sample in plan['samples']:
            sample['pilot_job_id'] = submit_array(plan, sample, '0', 'local_pilot', hold=True)
            submitted.append(sample['pilot_job_id'])
            save(BASE / 'plan.json', plan)
        gate = submit(['--hold', '--mem=2G', '--time=00:20:00', '--job-name=fourlepton_wave2_gate',
                       '--dependency=afterany:' + ':'.join(submitted),
                       '--output=' + str(BASE / 'logs/gate-%j.log')],
                      [PYTHON, control / 'production.py', 'gate'])
        submitted.append(gate)
        plan.update(phase='local_pilots_submitted', gate_job_id=gate)
        save(BASE / 'plan.json', plan)
    except BaseException:
        if submitted:
            subprocess.run(['scancel', *submitted], check=True)
        raise
    subprocess.run(['scontrol', 'release', ','.join(submitted)], check=True)
    append_status('Local-disk pilots submitted for eight configurations; bulk gate ' + gate +
                  ' will permit at most 100 gg_H and 100 ZZ jobs after validation.')
    print(json.dumps(plan, indent=2))


def task_directory(sample, index):
    suffix = '' if sample['scale'] == 'nominal' else '_muon_scale_' + sample['scale']
    return Path(sample['directory']) / 'jobs' / f'job_{index:06d}_seed{sample["base_seed"] + index}{suffix}'


def completed(sample, index):
    job = task_directory(sample, index)
    status = job / 'slurm-status.txt'
    if not status.exists():
        return False
    data = dict(line.split('=', 1) for line in status.read_text().splitlines() if '=' in line)
    output = Path(data['analysis_output'])
    if not output.resolve().is_relative_to((BASE / 'shards').resolve()):
        raise ValueError('Unexpected output location')
    return data['status'] == 'complete' and (job / 'SUCCESS').is_file() and output.is_file()


def gate():
    plan = json.loads((BASE / 'plan.json').read_text())
    if plan['phase'] != 'local_pilots_submitted':
        raise ValueError('Unexpected gate state')
    for sample in plan['samples']:
        if not completed(sample, 0):
            raise RuntimeError('Local pilot failed: ' + sample['name'])
        output_metadata = dict(line.split('=', 1) for line in
                               (task_directory(sample, 0) / 'output-metadata.txt').read_text().splitlines()
                               if '=' in line)
        if sha(Path(sample['config'])) != sample['config_sha256']:
            raise RuntimeError('Pilot campaign configuration changed')
        if sha(Path(output_metadata['analysis_output'])) != output_metadata['analysis_sha256']:
            raise RuntimeError('Pilot compact output changed')
        usage = list(task_directory(sample, 0).glob('local-storage-*.json'))
        if len(usage) != 1:
            raise RuntimeError('Missing unique local-storage measurement')
        measured = json.loads(usage[0].read_text())
        if measured['guard'] or measured['child_exit_code'] or measured['remaining_workspace_paths']:
            raise RuntimeError('Local staging/cleanup failed')
        if measured['sampled_peak_bytes'] <= 0 or measured['sampled_peak_bytes'] > 5_000_000_000:
            raise RuntimeError('Unexpected pilot local-storage usage')
        sample['pilot_local_storage'] = measured
    plan['storage_before_bulk'] = check_storage()
    submitted = []
    try:
        for sample in plan['samples']:
            sample['bulk_job_id'] = submit_array(plan, sample, f'1-{SHARDS-1}', 'bulk', hold=True)
            submitted.append(sample['bulk_job_id'])
            save(BASE / 'plan.json', plan)
        finish = submit(['--hold', '--mem=1G', '--time=00:20:00', '--job-name=fourlepton_wave2_finish',
                         '--dependency=afterany:' + ':'.join(submitted),
                         '--output=' + str(BASE / 'logs/finish-%j.log')],
                        [PYTHON, Path(plan['control']) / 'production.py', 'finish'])
        submitted.append(finish)
        summary = submit(['--hold', '--mem=2G', '--time=01:00:00', '--job-name=fourlepton_wave2_summary',
                          '--dependency=afterany:' + finish, '--output=' + str(BASE / 'logs/summary-%j.log')],
                         [PYTHON, Path(plan['control']) / 'production.py', 'summary'])
        submitted.append(summary)
        plan.update(phase='bulk_submitted', finish_job_id=finish, summary_job_id=summary)
        save(BASE / 'plan.json', plan)
    except BaseException:
        if submitted:
            subprocess.run(['scancel', *submitted], check=True)
        raise
    subprocess.run(['scontrol', 'release', ','.join(submitted)], check=True)
    append_status('All eight local pilots passed; 1,592 remaining 10k tasks submitted with a '
                  '200-job total cap (100 per physics process). Summary ' + summary + '.')
    print(json.dumps(plan, indent=2))


def finish():
    plan = json.loads((BASE / 'plan.json').read_text())
    if plan['phase'] != 'bulk_submitted':
        raise ValueError('Unexpected recovery state')
    failed = []
    for sample in plan['samples']:
        failures = []
        for index in range(SHARDS):
            if completed(sample, index):
                continue
            job = task_directory(sample, index)
            status = job / 'slurm-status.txt'
            if not status.exists():
                raise RuntimeError('Missing task status: ' + str(job))
            data = dict(line.split('=', 1) for line in status.read_text().splitlines() if '=' in line)
            wrapper = Path(sample['directory']) / 'logs' / f'bulk-{sample["bulk_job_id"]}_{index}.out'
            text = wrapper.read_text() if wrapper.exists() else ''
            if data['status'] != 'failed' or not (job / 'FAILED').is_file():
                raise RuntimeError('Nonterminal or inconsistent task: ' + str(job))
            messages = ['HepMC changed while validating its source identities',
                        'More events requested than available in LesHouchesReader']
            native = job / 'diagnostics/generation-herwig.log'
            if native.exists():
                text += native.read_text()
            if not any(message in text for message in messages):
                raise RuntimeError('Unrecognized failure requires investigation: ' + str(job))
            failures.append(index)
        if failures:
            failed.append((sample, failures))
    if sum(len(indices) for _, indices in failed) > 32:
        raise RuntimeError('More than 32 failures: investigate before recovery')
    submitted = []
    for sample, indices in failed:
        saved = BASE / 'recovery' / sample['name']
        saved.mkdir(parents=True, exist_ok=False)
        for index in indices:
            job = task_directory(sample, index)
            shutil.copytree(job, saved / job.name)
        job_id = submit_array(plan, sample, ','.join(map(str, indices)), 'recovery')
        submitted.append(job_id)
        sample['recovery_job_id'] = job_id
        sample['recovery_tasks'] = indices
        save(BASE / 'plan.json', plan)
        subprocess.run(['scontrol', 'update', 'JobId=' + plan['summary_job_id'],
                        'Dependency=afterany:' + ':'.join([os.environ['SLURM_JOB_ID'], *submitted])], check=True)
    plan['phase'] = 'recovery_submitted' if submitted else 'awaiting_summary'
    save(BASE / 'plan.json', plan)
    print(json.dumps({'recovery_jobs': submitted, 'one_recovery_round_only': True}, indent=2))


def summary():
    plan = json.loads((BASE / 'plan.json').read_text())
    sys.path.insert(0, str(SNAPSHOT))
    spec = importlib.util.spec_from_file_location('wave_summary', Path(plan['control']) / 'summarize_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failures = sum(module.summarize(process, BASE.name, TAG, EVENTS, SHARDS) for process in ['gg_H', 'ZZ'])
    plan['phase'] = 'complete' if failures == 0 else 'summary_failed'
    plan['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(BASE / 'plan.json', plan)
    return int(failures > 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'gate', 'finish', 'summary'])
    args = parser.parse_args()
    result = globals()[args.action]()
    raise SystemExit(result or 0)
