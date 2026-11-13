"""Submission gates must preserve caps, atomic release, and failed-job evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('parallel_production', ROOT / 'production.py')
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


class ProductionGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.patch_base = patch.object(production, 'BASE', self.base)
        self.patch_base.start()
        self.addCleanup(self.patch_base.stop)
        self.plan = {'phase': 'local_pilots_submitted', 'control': str(self.base / 'control'),
                     'samples': []}
        for process in ['gg_H', 'ZZ']:
            for shower, scale in [('pythia', 'nominal'), ('pythia', 'up'), ('pythia', 'down'), ('herwig', 'nominal')]:
                name = f'{process}_{shower}_{scale}'
                directory = self.base / name
                directory.mkdir()
                config = directory / 'campaign.env'
                config.write_text('unchanged fixture\n')
                sample = {'name': name, 'process': process, 'shower': shower, 'scale': scale,
                          'base_seed': 50000001, 'directory': str(directory),
                          'config': str(config), 'config_sha256': production.sha(config)}
                job = production.task_directory(sample, 0)
                job.mkdir(parents=True)
                output = self.base / 'shards' / (name + '.root')
                output.parent.mkdir(exist_ok=True)
                output.write_bytes(b'checksum-verified fixture')
                (job / 'slurm-status.txt').write_text('status=complete\nanalysis_output=' + str(output) + '\n')
                (job / 'SUCCESS').touch()
                (job / 'output-metadata.txt').write_text('analysis_output=' + str(output) +
                                                       '\nanalysis_sha256=' + production.sha(output) + '\n')
                (job / 'local-storage-1.json').write_text(json.dumps({
                    'guard': None, 'child_exit_code': 0, 'remaining_workspace_paths': [],
                    'sampled_peak_bytes': 3_600_000_000}))
                self.plan['samples'].append(sample)
        production.save(self.base / 'plan.json', self.plan)

    def test_failed_pilot_prevents_any_bulk_submission(self):
        production.task_directory(self.plan['samples'][0], 0).joinpath('SUCCESS').unlink()
        with patch.object(production, 'submit') as submit:
            with self.assertRaisesRegex(RuntimeError, 'Local pilot failed'):
                production.gate()
            submit.assert_not_called()

    def test_mutated_pilot_is_not_accepted(self):
        config = Path(self.plan['samples'][0]['config'])
        config.write_text('changed fixture\n')
        with patch.object(production, 'submit') as submit:
            with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                production.gate()
            submit.assert_not_called()

    def test_bulk_caps_and_dependencies_are_saved_before_release(self):
        submissions = []
        releases = []

        def submit(arguments, command):
            submissions.append((arguments, command))
            return str(100 + len(submissions))

        def run(command, **kwargs):
            releases.append(command)
            saved = json.loads((self.base / 'plan.json').read_text())
            self.assertEqual(saved['phase'], 'bulk_submitted')
            self.assertEqual(saved['summary_job_id'], '110')
            self.assertTrue(all('bulk_job_id' in r for r in saved['samples']))

        with patch.object(production, 'check_storage', return_value={}), \
             patch.object(production, 'submit', side_effect=submit), \
             patch.object(production.subprocess, 'run', side_effect=run), \
             patch.object(production, 'append_status'):
            production.gate()
        self.assertEqual(len(submissions), 10)
        for arguments, command in submissions[:8]:
            self.assertIn('--array=1-199%25', arguments)
            self.assertIn('--constraint=localscratch', arguments)
            self.assertIn('--hold', arguments)
            self.assertIn('--mem=4G', arguments)
        self.assertEqual(4 * production.CONCURRENCY, 100)
        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0][:2], ['scontrol', 'release'])
        self.assertIn('--dependency=afterany:101:102:103:104:105:106:107:108', submissions[8][0])
        self.assertIn('--dependency=afterany:109', submissions[9][0])

    def test_partial_submission_cancels_only_its_held_jobs(self):
        with patch.object(production, 'check_storage', return_value={}), \
             patch.object(production, 'submit', side_effect=['101', '102', RuntimeError('scheduler unavailable')]), \
             patch.object(production.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'scheduler unavailable'):
                production.gate()
        run.assert_called_once_with(['scancel', '101', '102'], check=True)

    def test_unknown_failure_never_triggers_an_automatic_retry(self):
        self.plan['phase'] = 'bulk_submitted'
        self.plan['samples'] = self.plan['samples'][:1]
        sample = self.plan['samples'][0]
        sample['bulk_job_id'] = '101'
        production.save(self.base / 'plan.json', self.plan)
        job = production.task_directory(sample, 1)
        job.mkdir(parents=True)
        (job / 'slurm-status.txt').write_text('status=failed\nanalysis_output=' + str(self.base / 'shards/missing.root') + '\n')
        (job / 'FAILED').touch()
        with patch.object(production, 'submit_array') as submit:
            with self.assertRaisesRegex(RuntimeError, 'Unrecognized failure'):
                production.finish()
            submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
