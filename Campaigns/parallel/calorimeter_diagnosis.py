"""Submit isolated, same-seed calorimeter reproductions; never repair production in place."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
BASE = STUDY / 'runs/calorimeter_diagnosis_20261114'
WAVE = STUDY / 'runs/production_parallel_wave2'


def main():
    record = BASE / 'submission.json'
    if record.exists():
        raise FileExistsError('Diagnostic run already submitted; inspect its record')
    plan = json.loads((WAVE / 'plan.json').read_text())
    jobs = []
    data = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'upstream_fix': '7f2b717c96671e78b6724b27aceea2580c9ce73a',
            'purpose': 'Two failing seeds and eight successful controls; original software, retained inputs',
            'production_outputs_changed': False, 'cases': []}

    def save():
        record.write_text(json.dumps(data, indent=2) + '\n')

    try:
        build = subprocess.check_output(['sbatch', '--parsable', '--hold',
            str(Path(__file__).with_name('calorimeter_build.sbatch'))], text=True).strip().split(';')[0]
        if not build.isdigit():
            raise ValueError('Invalid build job ID')
        jobs.append(build)
        data['build_job_id'] = build
        save()
        for sample in plan['samples']:
            indices = [0]
            if sample['process'] == 'gg_H' and sample['scale'] == 'nominal':
                indices.append(43 if sample['shower'] == 'pythia' else 45)
            source = Path(sample['config'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != sample['config_sha256']:
                raise ValueError('Original config changed')
            settings = {}
            for line in source.read_text().splitlines():
                key, value = line.split('=', 1)
                settings[key] = shlex.split(value)[0]
            for index in indices:
                name = f'{sample["process"]}_{sample["shower"]}_{sample["scale"]}_{index:06d}'
                case = BASE / 'cases' / name
                case.mkdir(parents=True, exist_ok=False)
                config = case / 'campaign.env'
                amended = dict(settings, CAMPAIGN_DIR=str(case), KEEP_INTERMEDIATES='1',
                               ANALYSIS_OUTPUT_ROOT=str(case / 'compact'))
                config.write_text(''.join(f'{k}={shlex.quote(v)}\n' for k,v in amended.items()))
                config.chmod(0o444)
                worker = Path(settings['REPO_ROOT']) / 'BatchSubmit/unity_generation_job.sh'
                wrapper = 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; exec ' + shlex.join([str(worker), 'events', str(config), '0'])
                job = subprocess.check_output(['sbatch', '--parsable', '--hold',
                    '--account=pi_rclsa_umass_edu', '--partition=cpu', '--qos=normal',
                    '--cpus-per-task=1', '--mem=4G', '--time=00:30:00', '--export=NONE',
                    '--job-name=fourlepton_calo_repro', '--array=' + str(index),
                    '--output=' + str(case / 'original-%A_%a.log'), '--wrap=' + wrapper],
                    text=True).strip().split(';')[0]
                if not job.isdigit():
                    raise ValueError('Invalid diagnostic job ID')
                jobs.append(job)
                data['cases'].append({'name': name, 'process': sample['process'],
                    'shower': sample['shower'], 'scale': sample['scale'], 'index': index,
                    'seed': sample['base_seed'] + index, 'original_config': str(source),
                    'original_config_sha256': sample['config_sha256'],
                    'config': str(config), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                    'job_id': job, 'expected_original': 'success' if index == 0 else 'simulation_crash'})
                save()
    except BaseException:
        if jobs:
            subprocess.run(['scancel', *jobs], check=True)
        raise
    subprocess.run(['scontrol', 'release', ','.join(jobs)], check=True)
    data['released_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
