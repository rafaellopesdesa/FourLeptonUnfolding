from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

GENERATION = Path(__file__).resolve().parents[1]


class HardEventBufferTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.generation = self.root / 'Generation'
        self.generation.mkdir()
        for name in ('run_generation.sh', 'lhe_provenance.py'):
            shutil.copyfile(GENERATION / name, self.generation / name)
        self.prefix = self.root / 'prefix'
        (self.prefix / 'bin').mkdir(parents=True)
        self.powheg = self.root / 'powheg'
        (self.powheg / 'gg_H').mkdir(parents=True)
        self.card = self.root / 'card.input'
        self.card.write_text('ebeam1 6800\nebeam2 6800\nnumevts 3\niseed 7\n')
        self.output = self.root / 'output'
        (self.generation / 'env.sh').write_text(
            f'GENERATION_PREFIX={shlex.quote(str(self.prefix))}\n'
            f'POWHEG_ROOT={shlex.quote(str(self.powheg))}\n'
            f'export PATH={shlex.quote(str(self.prefix / "bin"))}:'
            f'{shlex.quote(str(Path(sys.executable).parent))}:"$PATH"\n')
        self.executable(self.prefix / 'bin/git', '#!/bin/sh\nprintf "%040d\\n" 0\n')
        self.executable(self.powheg / 'gg_H/pwhg_main', '''
            #!/usr/bin/env python3
            from pathlib import Path
            values = dict(line.split()[:2] for line in Path('powheg.input').read_text().splitlines() if line.strip())
            n = int(values['numevts'])
            event = '<event>\\n1 1 1 125 0.007 0.1\\n25 1 0 0 0 0 0 0 0 125 125 0 9\\n</event>\\n'
            Path('pwgevents.lhe').write_text('<LesHouchesEvents version="3.0">\\n<init>\\n2212 2212 6800 6800 0 0 0 0 -4 1\\n1 0 1 1\\n</init>\\n' + event*n + '</LesHouchesEvents>\\n')
        ''')
        self.executable(self.prefix / 'bin/Herwig', '''
            #!/usr/bin/env python3
            from pathlib import Path
            import os, re, sys
            card = Path('herwig.in').read_text()
            assert 'set LesHouchesReader:AllowedToReOpen No' in card
            if sys.argv[1] == 'read':
                Path('FourLeptonLHERun.run').touch()
                raise SystemExit(0)
            n = int(sys.argv[sys.argv.index('-N')+1])
            source = Path(re.search(r'set LesHouchesReader:FileName (.*)',card).group(1))
            count = source.read_text().count('<event>\\n')
            ids = [i for i in range(1,count+1) if not (os.environ.get('REJECT_SECOND')=='1' and i==2)][:n]
            if len(ids) != n: raise SystemExit(19)
            out = Path(re.search(r'set /Herwig/Analysis/HepMC:Filename (.*)',card).group(1))
            out.write_text('HepMC::Version 2.06.11\\nHepMC::IO_GenEvent-START_EVENT_LISTING\\n' + ''.join(f'E {i} 0 0\\n' for i in ids) + 'HepMC::IO_GenEvent-END_EVENT_LISTING\\n')
        ''')

    @staticmethod
    def executable(path, code):
        path.write_text(textwrap.dedent(code).lstrip())
        path.chmod(0o755)

    def run_generation(self, *extra, reject=False):
        env = dict(os.environ, REJECT_SECOND='1' if reject else '0')
        return subprocess.run(['bash', str(self.generation / 'run_generation.sh'),
            'gg_H', 'herwig', '--events', '3', '--seed', '7', '--run-card', str(self.card),
            '--output-dir', str(self.output), *extra], env=env, text=True, capture_output=True)

    def test_reserve_keeps_requested_outputs_and_explicit_skipped_source_id(self):
        result = self.run_generation('--hard-events', '4', reject=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.output / 'lhe-provenance.json').read_text())
        self.assertEqual(manifest['lhe_event_count'], 4)
        self.assertEqual(manifest['hepmc_event_count'], 3)
        self.assertEqual(manifest['hepmc_unique_source_events'], 3)
        self.assertEqual(manifest['hepmc_repeated_source_events'], 0)
        self.assertIn('E 4 0 0', (self.output / 'events.hepmc').read_text())
        self.assertIn('requested_hard_events=4', (self.output / 'run-metadata.txt').read_text())

    def test_default_retains_equal_hard_and_shower_counts(self):
        result = self.run_generation()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.output / 'lhe-provenance.json').read_text())
        self.assertEqual((manifest['lhe_event_count'], manifest['hepmc_event_count']), (3, 3))

    def test_exhausted_source_still_fails(self):
        result = self.run_generation(reject=True)
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertFalse((self.output / 'lhe-provenance.json').exists())

    def test_invalid_counts_fail_before_generation(self):
        for count in ('0', '-1', '2', '2.5'):
            with self.subTest(count=count):
                result = self.run_generation('--hard-events', count)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--hard-events must', result.stderr)
                self.assertFalse(self.output.exists())

    def test_external_lhe_cannot_request_new_hard_events(self):
        source = self.root / 'external.lhe'
        source.touch()
        result = self.run_generation('--lhe', str(source), '--hard-events', '4')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('cannot be combined', result.stderr)


if __name__ == '__main__':
    unittest.main()
