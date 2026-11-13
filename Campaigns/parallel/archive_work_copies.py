"""Remove only verified duplicate compact shards after complete-wave publication."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

STUDY = Path('/work/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
PROJECT = Path('/project/pi_rclsa_umass_edu/rclsa_umass_edu/FourLeptonStudy')
BASE = STUDY / 'runs/production_parallel_wave2'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive():
    with (BASE / '.archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = json.loads((BASE / 'plan.json').read_text())
        if plan['phase'] != 'complete':
            raise RuntimeError('The entire wave must pass validation before archiving')
        pending = []
        verified = []
        for process in ['gg_H', 'ZZ']:
            result = json.loads((BASE / f'{process}-summary.json').read_text())
            for sample in result['samples']:
                if sample['errors'] or len(sample['files']) != plan['shards_per_configuration']:
                    raise RuntimeError('Incomplete summary cannot authorize cleanup')
                for item in sample['files']:
                    source = Path(item['path'])
                    destination = Path(item['persisted_path'])
                    if source.is_symlink() or destination.is_symlink():
                        raise ValueError('Unexpected symlink')
                    if not source.resolve().is_relative_to((BASE / 'shards').resolve()):
                        raise ValueError('Source outside this wave')
                    if not destination.resolve().is_relative_to((PROJECT / 'samples' / plan['tag']).resolve()):
                        raise ValueError('Destination outside the published wave')
                    if digest(destination) != item['sha256']:
                        raise ValueError('Persisted copy checksum mismatch; no files removed')
                    verified.append(str(destination))
                    if source.exists():
                        if digest(source) != item['sha256']:
                            raise ValueError('Work copy checksum mismatch; no files removed')
                        pending.append((source, source.stat().st_size))
        # Check every retained copy before removing any redundant work file.
        for path, _ in pending:
            path.unlink()
        record = {'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'removed_work_files': len(pending), 'removed_work_bytes': sum(n for _, n in pending),
                  'verified_persisted_files': len(verified), 'persistent_directory': str(PROJECT / 'samples' / plan['tag'])}
        (BASE / 'archive-result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
        return record


if __name__ == '__main__':
    archive()
