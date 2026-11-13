"""Launch one bounded background timing pilot from the validated immutable code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
PROJECT = Path('/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
SNAPSHOT = STUDY / 'runs/runtime_snapshots/e07de718c4d3967c8efd8f752f5ed1f282be9305'
BASE = STUDY / 'runs/pilot10k'


def main():
    metadata = json.loads((SNAPSHOT / 'SNAPSHOT.json').read_text())
    if metadata['tree'] != '20efbbe8af1efa630df413398da70c7c1fe7361b':
        raise RuntimeError('Unexpected code snapshot')
    pilot = json.loads((STUDY / 'runs/pilot/ZZ-summary.json').read_text())
    if len(pilot['samples']) != 4:
        raise ValueError('Expected all four validated ZZ configurations')
    for sample in pilot['samples']:
        if sample['errors'] or len(sample['files']) != 1 or sample['totals']['reconstructed'] <= 0:
            raise ValueError('Pilot validation failed')
        item = sample['files'][0]
        with Path(item['persisted_path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('Persisted pilot checksum mismatch')
    for root, limit, reserve in [(STUDY, 300_000_000_000, 10_000_000_000),
                                  (PROJECT, 2_000_000_000_000, 1_000_000_000)]:
        used = int(subprocess.check_output(['du', '-sb', str(root)]).split()[0])
        if used + reserve > limit:
            raise RuntimeError('Insufficient storage headroom')
    submitted = []
    record = BASE / 'ZZ-submission.json'
    if record.exists():
        raise FileExistsError(record)
    for shower, scale in [('pythia', 'nominal'), ('pythia', 'up'),
                          ('pythia', 'down'), ('herwig', 'nominal')]:
        suffix = '' if scale == 'nominal' else '_muon_scale_' + scale
        old_name = f'ZZ_{shower}_pilot_20261113{suffix}'
        old = STUDY / 'runs/pilot/campaigns' / old_name
        settings = {}
        for line in (old / 'campaign.env').read_text().splitlines():
            key, value = line.split('=', 1)
            settings[key] = shlex.split(value)[0]
        name = f'ZZ_{shower}_pilot10k_20261113{suffix}'
        campaign = BASE / 'campaigns' / name
        campaign.mkdir(exist_ok=False)
        (campaign / 'logs').mkdir()
        (campaign / 'inputs').mkdir()
        card = campaign / 'inputs/powheg.input'
        shutil.copyfile(settings['RUN_CARD'], card)
        settings.update(REPO_ROOT=str(SNAPSHOT), CAMPAIGN_NAME=name,
                        CAMPAIGN_DIR=str(campaign), EVENTS_PER_JOB='10000', TOTAL_EVENTS='10000',
                        BASE_SEED='12002001' if shower == 'herwig' else '2002001',
                        REUSED_GRID='1', RUN_CARD=str(card),
                        ANALYSIS_OUTPUT_ROOT=str(BASE / 'shards' / scale),
                        SCRATCH_ROOT=str(BASE / 'scratch'),
                        CREATED_UTC=datetime.now(timezone.utc).isoformat())
        config = campaign / 'campaign.env'
        config.write_text(''.join(f'{k}={shlex.quote(v)}\n' for k, v in settings.items()))
        config.chmod(0o444)
        job = subprocess.check_output([
            'sbatch', '--parsable', '--account=pi_rclsa_umass_edu', '--partition=cpu',
            '--qos=normal', '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=3G',
            '--time=01:00:00', '--array=0%1', '--export=NONE', '--job-name=ZZ_10k',
            '--output=' + str(campaign / 'logs/events-%A_%a.out'),
            str(SNAPSHOT / 'BatchSubmit/unity_generation_job.sh'), 'events', str(config), '0'],
            text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Invalid Slurm job ID')
        submitted.append({'shower': shower, 'scale': scale, 'job_id': job,
                          'config': str(config)})
        record.write_text(json.dumps({'jobs': submitted, 'code_snapshot': str(SNAPSHOT)}, indent=2) + '\n')
    summary = subprocess.check_output([
        'sbatch', '--parsable', '--dependency=afterany:' + ':'.join(r['job_id'] for r in submitted),
        str(STUDY / 'repository/Campaigns/bootstrap/pilot_summary.sbatch'),
        'ZZ', 'pilot10k', 'pilot10k_20261113', '10000'], text=True).strip().split(';')[0]
    result = {'jobs': submitted, 'summary_job_id': summary, 'code_snapshot': str(SNAPSHOT)}
    record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
