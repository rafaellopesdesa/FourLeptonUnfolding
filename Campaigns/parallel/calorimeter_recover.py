"""Repair only two diagnosed shards after all calorimeter comparisons pass."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
DIAG = STUDY / 'runs/calorimeter_diagnosis_20261114'
BASE = STUDY / 'runs/production_parallel_wave2'


def main():
    record = BASE / 'calorimeter-recovery.json'
    if record.exists():
        raise FileExistsError('Recovery already recorded; inspect before resubmission')
    regression = json.loads((DIAG / 'neutral-regression/result.json').read_text())
    if regression.get('passed') is not True:
        raise RuntimeError('Executable neutral-only regression did not pass')
    submission = json.loads((DIAG / 'submission.json').read_text())
    controls = failures = 0
    for case in submission['cases']:
        directory = Path(case['config']).parent
        result = json.loads((directory / 'comparison.json').read_text())
        if case['expected_original'] == 'success':
            comparison = result['original_vs_fixed']
            if not comparison['all_branch_values_identical'] or comparison['entries'] != 10000:
                raise RuntimeError('A successful control changed under the correction')
            controls += 1
        else:
            if result.get('reproduced_calorimeter_crash') is not True:
                raise RuntimeError('Original failure was not reproduced')
            failures += 1
        fixed = Path(result['fixed_analysis'])
        if hashlib.file_digest(fixed.open('rb'), 'sha256').hexdigest() != result['fixed_analysis_sha256']:
            raise RuntimeError('A fixed diagnostic output changed')
        # Same hard and showered inputs as the original production, including
        # both failures: neither replacement seeds nor event filtering is allowed.
        current_jobs = list((directory / 'jobs').glob('job_*'))
        if len(current_jobs) != 1:
            raise ValueError('Ambiguous diagnostic attempt')
        original_job = Path(case['original_config']).parent / 'jobs' / current_jobs[0].name
        current = json.loads((current_jobs[0] / 'diagnostics/generation-lhe-provenance.json').read_text())
        original = json.loads((original_job / 'diagnostics/generation-lhe-provenance.json').read_text())
        for key in ['lhe_sha256', 'hepmc_sha256', 'hepmc_event_count']:
            if current[key] != original[key]:
                raise RuntimeError('Diagnostic input differs from original production: ' + case['name'])
    if (controls, failures) != (8, 2):
        raise RuntimeError('Expected eight controls and two reproduced crashes')
    spec = importlib.util.spec_from_file_location('frozen_production', BASE / 'control/production.py')
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    plan = json.loads((BASE / 'plan.json').read_text())
    if plan['phase'] != 'summary_failed':
        raise RuntimeError('Unexpected wave state')
    storage = production.check_storage()
    snapshot = json.loads((DIAG / 'runtime/SNAPSHOT.json').read_text())
    for absolute, checksum in snapshot['detector_sha256'].items():
        if production.sha(Path(absolute)) != checksum:
            raise RuntimeError('Fixed detector binary/manifest changed')
    for relative, checksum in snapshot['environment_sha256'].items():
        if production.sha(DIAG / 'runtime' / relative) != checksum:
            raise RuntimeError('Fixed environment changed')
    recovery = BASE / 'recovery/calorimeter_20261114'
    recovery.mkdir(parents=True, exist_ok=False)
    data = {'created_utc': datetime.now(timezone.utc).isoformat(), 'validated_controls': controls,
            'reproduced_crashes': failures, 'storage': storage, 'runtime_snapshot': snapshot,
            'original_summary_job_id': plan['summary_job_id'], 'tasks': []}
    submitted = []
    try:
        for shower, index in [('pythia', 43), ('herwig', 45)]:
            sample = next(s for s in plan['samples'] if s['process'] == 'gg_H' and
                          s['shower'] == shower and s['scale'] == 'nominal')
            if production.completed(sample, index):
                raise RuntimeError('Refusing to replace a completed shard')
            job = production.task_directory(sample, index)
            saved = recovery / sample['name']
            saved.mkdir()
            shutil.copytree(job, saved / job.name)
            settings = production.scalar_config(Path(sample['config']))
            settings['REPO_ROOT'] = str(DIAG / 'runtime')
            config = saved / 'fixed-campaign.env'
            config.write_text(''.join(f'{k}={shlex.quote(v)}\n' for k,v in settings.items()))
            config.chmod(0o444)
            amended = dict(sample, config=str(config))
            job_id = production.submit_array(plan, amended, str(index), 'calorimeter_recovery', hold=True)
            submitted.append(job_id)
            data['tasks'].append({'sample': sample['name'], 'index': index,
                'seed': sample['base_seed'] + index, 'job_id': job_id,
                'original_config': sample['config'], 'original_config_sha256': sample['config_sha256'],
                'fixed_config': str(config), 'fixed_config_sha256': production.sha(config),
                'original_diagnostics': str(saved / job.name)})
            production.save(record, data)
        summary = production.submit(['--hold', '--mem=2G', '--time=01:00:00',
            '--job-name=fourlepton_calo_summary', '--dependency=afterany:' + ':'.join(submitted),
            '--output=' + str(BASE / 'logs/calorimeter-summary-%j.log')],
            [production.PYTHON, BASE / 'control/production.py', 'summary'])
        submitted.append(summary)
        archive = production.submit(['--hold', '--mem=2G', '--time=00:30:00',
            '--job-name=fourlepton_calo_archive', '--dependency=afterok:' + summary,
            '--output=' + str(BASE / 'logs/calorimeter-archive-%j.log')],
            [production.PYTHON, BASE / 'control/archive_work_copies.py'])
        submitted.append(archive)
        data.update(summary_job_id=summary, archive_job_id=archive)
        production.save(record, data)
        plan.update(phase='calorimeter_recovery_submitted', summary_job_id=summary,
                    calorimeter_recovery_record=str(record))
        production.save(BASE / 'plan.json', plan)
    except BaseException:
        if submitted:
            subprocess.run(['scancel', *submitted], check=True)
        raise
    subprocess.run(['scontrol', 'release', ','.join(submitted)], check=True)
    production.append_status('Calorimeter fix passed a neutral-only regression, two exact-input crash '
                             'reproductions and eight identical-output controls. Two same-seed repairs '
                             'submitted, followed by summary ' + summary + ' and archive ' + archive + '.')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
