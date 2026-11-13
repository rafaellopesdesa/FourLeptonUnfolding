"""Attach checksum-protected work-copy cleanup after the successful wave summary."""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
BASE = STUDY / 'runs/production_parallel_wave2'
PYTHON = STUDY / 'repository/Analysis/.pixi/envs/default/bin/python'


def main():
    with (BASE / '.archive-submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = BASE / 'archive-submission.json'
        if record.exists():
            print(record.read_text())
            return
        plan = json.loads((BASE / 'plan.json').read_text())
        summary = plan['summary_job_id']
        if not summary.isdigit() or plan['phase'] not in {'bulk_submitted', 'recovery_submitted', 'awaiting_summary', 'complete'}:
            raise ValueError('Summary has not been scheduled')
        source = Path(__file__).parent / 'archive_work_copies.py'
        target = Path(plan['control']) / 'archive_work_copies.py'
        payload = source.read_bytes()
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError('Archive code snapshot differs')
        else:
            shutil.copyfile(source, target)
            target.chmod(0o444)
        job_id = subprocess.check_output([
            'sbatch', '--parsable', '--account=pi_rclsa_umass_edu', '--partition=cpu',
            '--qos=normal', '--cpus-per-task=1', '--mem=2G', '--time=00:30:00', '--export=NONE',
            '--job-name=fourlepton_wave2_archive', '--dependency=afterok:' + summary,
            '--output=' + str(BASE / 'logs/archive-%j.log'),
            '--wrap=' + shlex.join([str(PYTHON), str(target)])], text=True).strip().split(';')[0]
        if not job_id.isdigit():
            raise ValueError('Invalid archive job ID')
        data = {'job_id': job_id, 'summary_job_id': summary, 'code': str(target),
                'code_sha256': hashlib.sha256(payload).hexdigest()}
        record.write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
