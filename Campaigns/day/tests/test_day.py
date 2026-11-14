"""Exercise resource caps, atomic release, deadlines, and failure stops."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('production_day',Path(__file__).resolve().parents[1]/'production_day.py')
day=importlib.util.module_from_spec(spec);spec.loader.exec_module(day)


class DayTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);(self.root/'runs').mkdir()
        self.directory=self.root/'day';self.directory.mkdir();(self.directory/'logs').mkdir()
        self.control=self.directory/'control';self.control.mkdir()
        self.config={'tag':'test_day','control':str(self.control),'runtime':str(self.root/'runtime'),
            'control_sha256':{},'launch_deadline_utc':'2026-11-15T00:00:00+00:00','maximum_waves':8,
            'monitor_until_utc':'2026-11-15T04:00:00+00:00','templates':[],
            'prerequisite_plan':str(self.directory/'prerequisite.json'),'prerequisite_summaries':[]}
        day.save(Path(self.config['prerequisite_plan']),{'phase':'complete'})
        for process in ['gg_H','ZZ']:
            samples=[]
            for shower,scale in [('pythia','nominal'),('pythia','up'),('pythia','down'),('herwig','nominal')]:
                name=f'{process}_{shower}_{scale}'
                card=self.directory/(name+'.input');card.write_text('fixture physics card\n')
                config=self.directory/(name+'.env');config.write_text('RUN_CARD='+str(card)+'\n')
                self.config['templates'].append({'process':process,'shower':shower,'scale':scale,
                    'config':str(config),'config_sha256':day.sha(config),'run_card_sha256':day.sha(card)})
                samples.append({'process':process,'shower':shower,'scale':scale,'errors':[],
                                'totals':{'reconstructed':100}})
            manifest=self.directory/(process+'.json');day.save(manifest,{'samples':samples})
            self.config['prerequisite_summaries'].append(str(manifest))
        day.save(self.directory/'config.json',self.config)
        day.save(self.directory/'state.json',{'phase':'waiting_for_repair','completed_waves':[]})
        for name,value in [('STUDY',self.root)]:
            p=patch.object(day,name,value);p.start();self.addCleanup(p.stop)
        for p in [patch.object(day,'now',return_value=datetime(2026,11,14,12,tzinfo=timezone.utc)),
                  patch.object(day,'storage',return_value={}),
                  patch.object(day.Campaign,'status'),patch.object(day.Campaign,'check_runtime')]:
            p.start();self.addCleanup(p.stop)
        self.campaign=day.Campaign(self.directory)

    def test_caps_and_all_dependencies_recorded_before_release(self):
        submissions=[]
        def submit(args,command):
            submissions.append((args,command));return str(100+len(submissions))
        def release(jobs):
            plan=json.loads((self.campaign.wave(1)/'plan.json').read_text())
            self.assertEqual(len(plan['samples']),8)
            self.assertEqual(plan['advance_job_id'],'112')
            self.assertEqual(self.campaign.state()['phase'],'producing')
            self.assertEqual(len(jobs),12)
        with patch.object(day,'submit',side_effect=submit),patch.object(day,'release',side_effect=release) as freed:
            self.campaign.launch()
        self.assertEqual(len(submissions),12);freed.assert_called_once()
        for arguments,_ in submissions[:8]:
            self.assertIn('--array=0-199%25',arguments)
            self.assertIn('--constraint=localscratch',arguments)
            self.assertIn('--mem=4G',arguments);self.assertIn('--hold',arguments)
        self.assertIn('--dependency=afterany:101:102:103:104:105:106:107:108',submissions[8][0])
        self.assertIn('--dependency=afterok:110:111',submissions[-1][0])

    def test_partial_submission_cancels_only_own_held_jobs(self):
        with patch.object(day,'submit',side_effect=['101','102',RuntimeError('scheduler full')]),\
             patch.object(day,'cancel') as cancelled,patch.object(day,'release') as release:
            with self.assertRaisesRegex(RuntimeError,'scheduler full'):self.campaign.launch()
        cancelled.assert_called_once_with(['101','102']);release.assert_not_called()

    def test_deadline_and_stop_file_prevent_submissions(self):
        with patch.object(day,'now',return_value=datetime(2026,11,15,tzinfo=timezone.utc)),\
             patch.object(day,'submit') as submitted:
            self.campaign.launch();submitted.assert_not_called()
        self.assertEqual(self.campaign.state()['phase'],'finished')
        day.save(self.directory/'state.json',{'phase':'between_waves','completed_waves':[]})
        (self.directory/'STOP').touch()
        with patch.object(day,'submit') as submitted:self.campaign.launch();submitted.assert_not_called()

    def test_maximum_wave_count_is_bounded(self):
        day.save(self.directory/'state.json',{'phase':'between_waves','completed_waves':list(range(1,9))})
        with patch.object(day,'submit') as submitted:self.campaign.launch();submitted.assert_not_called()
        self.assertEqual(self.campaign.state()['phase'],'finished')

    def test_fresh_wave_seed_ranges_and_correlated_scales(self):
        waves=[self.campaign.make_wave(i,{}) for i in [1,2]]
        for plan in waves:
            for process in ['gg_H','ZZ']:
                seeds={s['base_seed'] for s in plan['samples'] if s['process']==process and s['shower']=='pythia'}
                self.assertEqual(len(seeds),1)
        seeds1={s['base_seed'] for s in waves[0]['samples']}
        seeds2={s['base_seed'] for s in waves[1]['samples']}
        self.assertEqual(len(seeds1),4);self.assertEqual(len(seeds2),4)
        self.assertTrue(all(abs(a-b)>=day.SHARDS for a in seeds1 for b in seeds2))

    def test_unknown_failure_stops_without_retry(self):
        plan=self.campaign.make_wave(1,{})
        with patch.object(self.campaign,'inspect',side_effect=RuntimeError('Unknown failure')),\
             patch.object(day,'submit') as submitted:
            with self.assertRaisesRegex(RuntimeError,'Unknown failure'):self.campaign.finish(1,0)
        submitted.assert_not_called();self.assertEqual(self.campaign.state()['phase'],'stopped')

    def test_changed_prerequisite_prevents_launch(self):
        day.save(Path(self.config['prerequisite_plan']),{'phase':'summary_failed'})
        with patch.object(day,'submit') as submitted:
            with self.assertRaisesRegex(RuntimeError,'Repaired production'):self.campaign.launch()
        submitted.assert_not_called()

    def test_monitor_stops_at_its_own_deadline(self):
        with patch.object(day,'now',return_value=datetime(2026,11,15,4,tzinfo=timezone.utc)),\
             patch.object(day,'submit') as submitted:
            self.campaign.monitor();submitted.assert_not_called()


if __name__=='__main__':unittest.main()
