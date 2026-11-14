"""Continue bounded production waves until a UTC deadline, checking every stage."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import html
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
PYTHON = STUDY / 'repository/Analysis/.pixi/envs/default/bin/python'
CONCURRENCY = 25
SHARDS = 200
EVENTS = 10000


def now():
    return datetime.now(timezone.utc)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def scalar(path):
    return {k: shlex.split(v)[0] for k,v in
            (line.split('=',1) for line in Path(path).read_text().splitlines())}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def submit(arguments, command):
    environment = ['/usr/bin/env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                   'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1']
    result = subprocess.check_output(['sbatch', '--parsable', '--account=pi_rclsa_umass_edu',
        '--partition=cpu', '--qos=normal', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
        '--export=NONE', *arguments, '--wrap=' + shlex.join(environment + [str(x) for x in command])],
        text=True).strip().split(';')[0]
    if not result.isdigit():
        raise ValueError('Invalid Slurm job ID')
    return result


def cancel(jobs):
    if jobs:
        subprocess.run(['scancel', *jobs], check=True)


def release(jobs):
    subprocess.run(['scontrol', 'release', ','.join(jobs)], check=True)


def storage():
    result = {}
    for path, limit, reserve in [(STUDY, 300_000_000_000, 50_000_000_000),
                                 (PROJECT, 2_000_000_000_000, 10_000_000_000)]:
        used = int(subprocess.check_output(['du', '-sb', str(path)]).split()[0])
        if used + reserve >= limit or shutil.disk_usage(path).free < reserve:
            raise RuntimeError('Insufficient storage headroom: ' + str(path))
        result[str(path)] = {'bytes': used, 'soft_limit': limit, 'reserve': reserve}
    return result


class Campaign:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = json.loads((self.directory / 'config.json').read_text())
        self.control = Path(self.config['control'])
        self.runtime = Path(self.config['runtime'])
        for name, checksum in self.config['control_sha256'].items():
            if sha(self.control / name) != checksum:
                raise ValueError('Frozen control changed: ' + name)

    def state(self):
        return json.loads((self.directory / 'state.json').read_text())

    def update(self, state, **changes):
        state.update(changes, updated_utc=now().isoformat())
        save(self.directory / 'state.json', state)
        self.status(state)

    def status(self, state):
        begin, end = '<!-- DAY_PRODUCTION_BEGIN -->', '<!-- DAY_PRODUCTION_END -->'
        lines = [begin, '## Production scheduled for 14 November', '',
                 'Updated: ' + state['updated_utc'], '',
                 'State: **' + state['phase'] + '**. ' + state.get('message',''),
                 'New waves stop at ' + self.config['launch_deadline_utc'] +
                 '; running jobs finish. Maximum eight waves, 16M showered rows each.',
                 'Concurrency cap: 100 gg_H + 100 ZZ. One CPU and 4 GiB per task.',
                 'Validated waves completed: ' + str(len(state.get('completed_waves', []))) + '.',
                 'Control record: `' + str(self.directory / 'state.json') + '`.', '']
        if state.get('live'):
            lines += ['Latest worker snapshot: `' + json.dumps(state['live']) + '`.', '']
        if state.get('cumulative_selected'):
            lines += ['| Configuration | Saved selected events (excludes pilots) |', '|---|---:|']
            lines += [f'| {k} | {v} |' for k,v in sorted(state['cumulative_selected'].items())]
            lines.append('')
        lines += [end, '']
        block = '\n'.join(lines)
        with (STUDY / 'docs/.status.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = STUDY / 'docs/CURRENT_STATUS.md'
            text = path.read_text()
            if begin in text:
                left = text.index(begin); right = text.index(end, left) + len(end)
                text = text[:left] + block + text[right:]
            else:
                split = text.index('\n') + 1
                text = text[:split] + '\n' + block + '\n' + text[split:]
            path.write_text(text)
            (STUDY / 'report/docs/CURRENT_STATUS.md').write_text(text)
            save(STUDY / 'report/docs/evidence/day-production-20261114.json', state)
            page=STUDY/'report/docs/index.html'
            content=page.read_text()
            first,last='<!-- DAY_SLIDE_BEGIN -->','<!-- DAY_SLIDE_END -->'
            if first in content and last in content:
                rows=[]
                for process in ['gg_H','ZZ']:
                    counts=[v for k,v in state.get('cumulative_selected',{}).items() if k.startswith(process+' ')]
                    if counts:
                        rows.append(f'<tr><td>{process}</td><td>{min(counts):,}–{max(counts):,}</td><td>≥5M</td></tr>')
                slide=(first+'\n<section><p class="eyebrow">Production / 14 November</p>'
                    '<h2>Automatic waves through the day</h2><p><strong>'+html.escape(state['phase'])+
                    '</strong>: '+html.escape(state.get('message','')[:450])+'</p>'
                    '<table><thead><tr><th>Process</th><th>Saved selected / configuration</th><th>Target</th></tr></thead>'
                    '<tbody>'+''.join(rows)+'</tbody></table>'
                    '<p>At most 100 gg_H + 100 ZZ jobs. Each wave is validated, saved and checked before the next starts.</p>'
                    '<p class="small">New waves stop at midnight UTC; running jobs finish. Updated '+html.escape(state['updated_utc'])+
                    '. Local status refreshes every 30 minutes; GitHub is a published snapshot. '
                    '<a href="CURRENT_STATUS.md">Details and job records →</a></p></section>\n'+last)
                left=content.index(first);right=content.index(last,left)+len(last)
                page.write_text(content[:left]+slide+content[right:])

    def command(self, action, *args):
        return [PYTHON, self.control / 'production_day.py', self.directory, action, *map(str,args)]

    def stop_reason(self, state):
        if (self.directory / 'STOP').exists():
            return 'STOP file present; no further wave submitted'
        if now() >= datetime.fromisoformat(self.config['launch_deadline_utc']):
            return 'UTC launch deadline reached'
        if len(state.get('completed_waves', [])) >= self.config['maximum_waves']:
            return 'Maximum number of waves reached'
        return None

    def check_runtime(self):
        if sha(self.runtime / 'SNAPSHOT.json') != self.config['runtime_snapshot_sha256']:
            raise ValueError('Runtime snapshot manifest changed')
        frozen = json.loads((self.runtime / 'SNAPSHOT.json').read_text())
        for path, checksum in frozen['detector_sha256'].items():
            if sha(path) != checksum:
                raise ValueError('Detector installation changed')
        for relative, checksum in frozen['environment_sha256'].items():
            if sha(self.runtime / relative) != checksum:
                raise ValueError('Runtime environment changed')

    def wave(self, index):
        return STUDY / 'runs' / f'{self.config["tag"]}_w{index:02d}'

    def make_wave(self, index, measured):
        base = self.wave(index)
        base.mkdir(exist_ok=False)
        (base / 'logs').mkdir()
        plan = {'index': index, 'tag': base.name, 'control': str(self.control),
                'shards_per_configuration': SHARDS, 'events_per_shard': EVENTS,
                'concurrency_per_process': 100, 'phase': 'prepared', 'samples': [],
                'created_utc': now().isoformat(), 'storage': measured}
        for source in self.config['templates']:
            if sha(source['config']) != source['config_sha256']:
                raise ValueError('Source campaign changed')
            settings = scalar(source['config'])
            suffix = '' if source['scale'] == 'nominal' else '_muon_scale_' + source['scale']
            name = f'{source["process"]}_{source["shower"]}_{base.name}{suffix}'
            directory = base / 'campaigns' / name
            (directory / 'inputs').mkdir(parents=True)
            (directory / 'logs').mkdir()
            card = directory / 'inputs/powheg.input'
            if sha(settings['RUN_CARD']) != source['run_card_sha256']:
                raise ValueError('Source run card changed')
            shutil.copyfile(settings['RUN_CARD'], card)
            group = {('gg_H','pythia'): 1, ('gg_H','herwig'): 2,
                     ('ZZ','pythia'): 3, ('ZZ','herwig'): 4}[(source['process'],source['shower'])]
            seed = group * 100_000_000 + index * 1_000_000 + 1
            settings.update(REPO_ROOT=str(self.runtime), CAMPAIGN_DIR=str(directory),
                CAMPAIGN_NAME=name, BASE_SEED=str(seed), JOBS=str(SHARDS), EVENTS_PER_JOB=str(EVENTS),
                TOTAL_EVENTS=str(SHARDS * EVENTS), RUN_CARD=str(card), KEEP_INTERMEDIATES='0',
                ANALYSIS_OUTPUT_ROOT=str(base / 'shards' / source['scale']),
                SCRATCH_ROOT='/localscratch/rclsa_umass_edu/FourLeptonStudy/' + self.config['tag'],
                CREATED_UTC=now().isoformat())
            config = directory / 'campaign.env'
            config.write_text(''.join(f'{k}={shlex.quote(v)}\n' for k,v in settings.items()))
            config.chmod(0o444)
            plan['samples'].append({k: source[k] for k in ['process','shower','scale']} |
                {'name':name, 'directory':str(directory), 'config':str(config),
                 'config_sha256':sha(config), 'base_seed':seed})
        save(base / 'plan.json', plan)
        return plan

    def array(self, plan, sample, indices, phase):
        return submit(['--hold','--mem=4G','--time=01:00:00','--constraint=localscratch',
            '--array='+indices+'%'+str(CONCURRENCY),'--job-name=fourlepton_day_'+phase,
            '--output='+str(Path(sample['directory']) / f'logs/{phase}-%A_%a.log')],
            [PYTHON, self.control / 'local_worker.py', sample['config']])

    def launch(self):
        state = self.state()
        if state['phase'] not in {'waiting_for_repair','between_waves'}:
            raise RuntimeError('Cannot launch from phase '+state['phase'])
        reason = self.stop_reason(state)
        if reason:
            self.update(state, phase='finished', message=reason)
            return
        self.check_runtime()
        if not state.get('completed_waves'):
            prerequisite = json.loads(Path(self.config['prerequisite_plan']).read_text())
            if prerequisite['phase'] != 'complete':
                raise RuntimeError('Repaired production has not passed final validation')
            totals = {}
            for manifest in self.config['prerequisite_summaries']:
                for sample in json.loads(Path(manifest).read_text())['samples']:
                    if sample['errors']:
                        raise RuntimeError('Prerequisite sample contains errors')
                    key = ' '.join(sample[k] for k in ['process','shower','scale'])
                    totals[key] = totals.get(key,0) + sample['totals']['reconstructed']
            state['cumulative_selected'] = totals
        index = len(state.get('completed_waves',[])) + 1
        plan = self.make_wave(index, storage())
        base = self.wave(index)
        submitted = []
        try:
            for sample in plan['samples']:
                sample['bulk_job_id'] = self.array(plan,sample,'0-199','bulk')
                submitted.append(sample['bulk_job_id'])
                save(base / 'plan.json',plan)
            finish = submit(['--hold','--mem=2G','--time=00:20:00','--job-name=fourlepton_day_finish',
                '--dependency=afterany:'+':'.join(submitted),'--output='+str(base/'logs/finish-%j.log')],
                self.command('finish',index,0))
            submitted.append(finish)
            summaries = {}
            for process in ['gg_H','ZZ']:
                job = submit(['--hold','--mem=2G','--time=01:00:00','--job-name=fourlepton_day_summary',
                    '--dependency=afterany:'+finish,'--output='+str(base/f'logs/summary-{process}-%j.log')],
                    self.command('summarize',index,process))
                summaries[process] = job; submitted.append(job)
            advance = submit(['--hold','--mem=2G','--time=00:30:00','--job-name=fourlepton_day_advance',
                '--dependency=afterok:'+':'.join(summaries.values()),'--output='+str(base/'logs/advance-%j.log')],
                self.command('advance',index))
            submitted.append(advance)
            plan.update(phase='submitted',finish_job_id=finish,summary_job_ids=summaries,advance_job_id=advance)
            save(base / 'plan.json',plan)
            self.update(state,phase='producing',active_wave=index,active_plan=str(base/'plan.json'),
                        message=f'Wave {index}: 1,600 tasks submitted; caps 100 per process.',live={})
        except BaseException:
            cancel(submitted)
            raise
        release(submitted)

    def task(self, sample, index):
        suffix = '' if sample['scale']=='nominal' else '_muon_scale_'+sample['scale']
        return Path(sample['directory'])/'jobs'/f'job_{index:06d}_seed{sample["base_seed"]+index}{suffix}'

    def inspect(self, plan, sample, index):
        job = self.task(sample,index)
        status = dict(line.split('=',1) for line in (job/'slurm-status.txt').read_text().splitlines() if '=' in line)
        if status['status']=='complete' and (job/'SUCCESS').exists():
            meta = dict(line.split('=',1) for line in (job/'output-metadata.txt').read_text().splitlines())
            path = Path(meta['analysis_output'])
            if not path.resolve().is_relative_to((self.wave(plan['index'])/'shards').resolve()):
                raise ValueError('Output outside current wave')
            if meta['campaign_config_sha256']!=sample['config_sha256'] or sha(path)!=meta['analysis_sha256']:
                raise ValueError('Output/configuration checksum mismatch')
            disk = json.loads((job/f'local-storage-{status["slurm_job_id"]}.json').read_text())
            if disk['guard'] or disk['child_exit_code'] or disk['remaining_workspace_paths']:
                raise RuntimeError('Worker scratch/cleanup guard failed')
            return True
        if status['status']!='failed' or not (job/'FAILED').exists():
            raise RuntimeError('Nonterminal/inconsistent worker state')
        logs = list((Path(sample['directory'])/'logs').glob(f'*{status["slurm_array_job_id"]}_{index}.log'))
        logs += list((job/'diagnostics').glob('generation-herwig.log'))
        text = '\n'.join(p.read_text(errors='replace') for p in logs)
        known = ['HepMC changed while validating its source identities',
                 'More events requested than available in LesHouchesReader']
        if not any(message in text for message in known):
            raise RuntimeError('Unknown failure: '+str(job))
        return False

    def finish(self,index,attempt):
        base=self.wave(index);plan=json.loads((base/'plan.json').read_text());failed=[]
        try:
            for sample in plan['samples']:
                if sha(sample['config'])!=sample['config_sha256']:
                    raise ValueError('Campaign config changed')
                indices=[i for i in range(SHARDS) if not self.inspect(plan,sample,i)]
                if indices: failed.append((sample,indices))
            if failed and (attempt or sum(len(i) for _,i in failed)>32):
                raise RuntimeError('Recovery limit reached')
            if not failed:
                plan['phase']='workers_validated';save(base/'plan.json',plan)
                self.update(self.state(),phase='validating',message=f'Wave {index}: all 1,600 workers validated.')
                return
            submitted=[]
            try:
                for sample,indices in failed:
                    saved=base/'recovery'/sample['name'];saved.mkdir(parents=True,exist_ok=False)
                    for i in indices:shutil.copytree(self.task(sample,i),saved/self.task(sample,i).name)
                    job=self.array(plan,sample,','.join(map(str,indices)),'recovery')
                    submitted.append(job);sample['recovery_job_id']=job;sample['recovery_indices']=indices
                    save(base/'plan.json',plan)
                check=submit(['--hold','--mem=2G','--time=00:20:00','--job-name=fourlepton_day_recheck',
                    '--dependency=afterany:'+':'.join(submitted),'--output='+str(base/'logs/recheck-%j.log')],
                    self.command('finish',index,1))
                submitted.append(check)
                for summary in plan['summary_job_ids'].values():
                    subprocess.run(['scontrol','update','JobId='+summary,'Dependency=afterany:'+check],check=True)
                plan.update(phase='recovering',recheck_job_id=check);save(base/'plan.json',plan)
                self.update(self.state(),phase='recovering',message=f'Wave {index}: one bounded same-seed retry round.')
            except BaseException:
                cancel(submitted);raise
            release(submitted)
        except BaseException as error:
            plan.update(phase='failed',error=str(error));save(base/'plan.json',plan)
            self.update(self.state(),phase='stopped',message=str(error))
            raise

    def summarize(self,index,process):
        sys.path.insert(0,str(self.runtime))
        summarizer=module(self.control/'summarize_pilot.py','day_summary')
        base=self.wave(index)
        result=summarizer.summarize(process,base.name,base.name,EVENTS,SHARDS)
        if result:
            self.update(self.state(),phase='stopped',message=f'Wave {index} {process} summary failed; valid shards retained.')
        return result

    def advance(self,index):
        base=self.wave(index);plan=json.loads((base/'plan.json').read_text());state=self.state()
        if plan['phase']!='workers_validated':
            raise RuntimeError('Worker validation did not authorize a new wave')
        totals={}
        for process in ['gg_H','ZZ']:
            summary=json.loads((base/f'{process}-summary.json').read_text())
            for s in summary['samples']:
                if s['errors'] or len(s['files'])!=SHARDS:
                    raise RuntimeError('Incomplete summary prevents advancement')
                totals[' '.join(s[k] for k in ['process','shower','scale'])]=s['totals']['reconstructed']
        plan['phase']='complete';save(base/'plan.json',plan)
        archive=module(self.control/'archive_work_copies.py','day_archive');archive.BASE=base
        archived=archive.archive()
        if index in state.get('completed_waves',[]):
            raise RuntimeError('Wave already accounted')
        for key,value in totals.items():
            state['cumulative_selected'][key]+=value
        state.setdefault('completed_waves',[]).append(index)
        self.update(state,phase='between_waves',message=f'Wave {index} saved and duplicate work files removed.',
                    latest_archive=archived,live={})
        reason=self.stop_reason(state)
        if reason:
            self.update(state,phase='finished',message=reason);return
        job=submit(['--hold','--mem=2G','--time=00:30:00','--job-name=fourlepton_day_launch',
            '--dependency=afterok:'+os.environ['SLURM_JOB_ID'],
            '--output='+str(self.directory/'logs/launch-%j.log')],self.command('launch'))
        self.update(state,next_launch_job_id=job)
        release([job])

    def monitor(self):
        state=self.state()
        if (state['phase'] in {'stopped','finished'} or
                now() >= datetime.fromisoformat(self.config['monitor_until_utc'])):
            return
        live={}
        if state.get('active_plan'):
            plan=json.loads(Path(state['active_plan']).read_text())
            queue=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i %T'],text=True)
            rows=[line.split() for line in queue.splitlines()]
            for process in ['gg_H','ZZ']:
                samples=[s for s in plan['samples'] if s['process']==process]
                parents={s[k] for s in samples for k in ['bulk_job_id','recovery_job_id'] if k in s}
                counts=Counter(v[1] for v in rows if len(v)==2 and v[0].split('_')[0] in parents)
                live[process]={'scheduler':dict(counts),'worker_success_markers':sum(
                    1 for s in samples for _ in Path(s['directory']).glob('jobs/*/SUCCESS'))}
        # The monitor does not rewrite state.json, avoiding races with controllers.
        snapshot=dict(state,live=live,updated_utc=now().isoformat());self.status(snapshot)
        job=submit(['--mem=1G','--time=00:05:00','--begin=now+30minutes',
                    '--job-name=fourlepton_day_monitor','--output='+str(self.directory/'logs/monitor-%j.log')],
                   self.command('monitor'))
        save(self.directory/'monitor.json',{'job_id':job,'updated_utc':now().isoformat(),'live':live})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('action')
    parser.add_argument('arguments',nargs='*');args=parser.parse_args();campaign=Campaign(args.directory)
    converted=[int(x) if x.isdigit() else x for x in args.arguments]
    if args.action not in {'launch','finish','summarize','advance','monitor'}:
        raise SystemExit('Unknown action')
    try:
        result=getattr(campaign,args.action)(*converted)
    except BaseException as error:
        if args.action!='monitor':
            campaign.update(campaign.state(),phase='stopped',message=f'{args.action}: {error}')
        raise
    raise SystemExit(result or 0)
