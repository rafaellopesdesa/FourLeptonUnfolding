"""Run the existing full chain on node-local disk and retain small usage records."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys


def read_config(path):
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split('=', 1)
        values = shlex.split(value)
        if len(values) != 1:
            raise ValueError(f'Invalid scalar setting: {key}')
        result[key] = values[0]
    return result


def main(config_path):
    config = read_config(config_path)
    task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    job_id = os.environ['SLURM_JOB_ID']
    if not 0 <= task < int(config['JOBS']):
        raise ValueError('Task outside the campaign')
    scratch = Path(config['SCRATCH_ROOT'])
    if not scratch.is_relative_to('/localscratch') or config['KEEP_INTERMEDIATES'] != '0':
        raise ValueError('This worker requires disposable /localscratch intermediates')
    mount = json.loads(subprocess.check_output([
        'findmnt', '-J', '-T', '/localscratch', '-o', 'TARGET,SOURCE,FSTYPE'], text=True))['filesystems'][0]
    if mount['target'] != '/localscratch' or mount['fstype'] not in {'ext4', 'xfs', 'btrfs', 'zfs'}:
        raise RuntimeError('A real node-local disk mount is required')
    os.umask(0o077)
    scratch.mkdir(parents=True, exist_ok=True)
    if scratch.resolve() != scratch or scratch.stat().st_uid != os.getuid():
        raise ValueError('Unexpected local workspace ownership or symlink')
    initial_free = shutil.disk_usage(scratch).free
    if initial_free < 60_000_000_000:
        raise RuntimeError('Less than 60 GB free on node-local scratch')
    suffix = '' if config['MUON_SCALE'] == 'nominal' else '_muon_scale_' + config['MUON_SCALE']
    seed = int(config['BASE_SEED']) + task
    job = Path(config['CAMPAIGN_DIR']) / 'jobs' / f'job_{task:06d}_seed{seed}{suffix}'
    job.mkdir(parents=True, exist_ok=True)
    metadata = job / f'local-storage-{job_id}.json'
    worker = Path(config['REPO_ROOT']) / 'BatchSubmit/unity_generation_job.sh'
    command = [str(worker), 'events', str(config_path), '0']
    started = datetime.now(timezone.utc).isoformat()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    child = subprocess.Popen(command, start_new_session=True, env=environment)
    interrupted = None

    def forward(signum, frame):
        nonlocal interrupted
        interrupted = signum
        if child.poll() is None:
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    peak = 0
    minimum_free = initial_free
    guard = None
    pattern = f'four-lepton-{job_id}-{task:06d}.*'
    while child.poll() is None:
        active = list(scratch.glob(pattern))
        current = 0
        for path in active:
            measured = subprocess.run(['du', '-sb', str(path)], text=True, capture_output=True)
            if measured.returncode == 0:
                current += int(measured.stdout.split()[0])
        peak = max(peak, current)
        minimum_free = min(minimum_free, shutil.disk_usage(scratch).free)
        if current > 8_000_000_000 or minimum_free < 20_000_000_000:
            guard = 'local_workspace_above_8GB' if current > 8_000_000_000 else 'free_space_below_20GB'
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    result = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'job_id': job_id, 'task_id': task, 'node': os.environ.get('SLURMD_NODENAME'),
              'filesystem': mount, 'scratch_root': str(scratch),
              'sampled_peak_bytes': peak, 'sampling_interval_seconds': 10,
              'initial_free_bytes': initial_free, 'minimum_free_bytes': minimum_free,
              'remaining_workspace_paths': [str(p) for p in scratch.glob(pattern)],
              'child_exit_code': child.returncode, 'guard': guard, 'signal': interrupted}
    metadata.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if guard or interrupted or result['remaining_workspace_paths']:
        return 1
    return 0 if child.returncode == 0 else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Expected campaign.env')
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
