import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('archive_work', Path(__file__).resolve().parents[1] / 'archive_work_copies.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.base, self.project = root / 'work', root / 'project'
        self.base.mkdir()
        for name, value in [('BASE', self.base), ('PROJECT', self.project)]:
            context = patch.object(archive, name, value)
            context.start()
            self.addCleanup(context.stop)
        (self.base / 'plan.json').write_text(json.dumps({'phase': 'complete', 'tag': 'wave', 'shards_per_configuration': 1}))
        self.sources, self.destinations = [], []
        for process in ['gg_H', 'ZZ']:
            source = self.base / 'shards' / (process + '.root')
            destination = self.project / 'samples/wave' / source.name
            source.parent.mkdir(parents=True, exist_ok=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b'validated compact data')
            destination.write_bytes(source.read_bytes())
            record = {'path': str(source), 'persisted_path': str(destination), 'sha256': archive.digest(source)}
            (self.base / f'{process}-summary.json').write_text(json.dumps({'samples': [{'errors': [], 'files': [record]}]}))
            self.sources.append(source)
            self.destinations.append(destination)

    def test_bad_persisted_copy_prevents_all_removals(self):
        self.destinations[-1].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Persisted copy checksum mismatch'):
            archive.archive()
        self.assertTrue(all(p.exists() for p in self.sources))

    def test_verified_cleanup_preserves_persistent_data_and_is_repeatable(self):
        result = archive.archive()
        self.assertEqual(result['removed_work_files'], 2)
        self.assertTrue(all(not p.exists() for p in self.sources))
        self.assertTrue(all(p.read_bytes() == b'validated compact data' for p in self.destinations))
        self.assertEqual(archive.archive()['removed_work_files'], 0)


if __name__ == '__main__':
    unittest.main()
