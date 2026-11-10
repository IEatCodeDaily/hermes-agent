"""Stdlib-only transport validation. No application imports or execution."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile


def inspect_archive(archive, manifest):
    expected = {entry['path']: entry for entry in manifest['files']}
    seen = set()
    with tarfile.open(archive) as source:
        for member in source:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts, member.name
            assert member.name not in seen, 'duplicate member'
            seen.add(member.name)
            entry = expected[member.name]
            filtered = tarfile.data_filter(member, '/fixture-owned-empty-destination')
            assert filtered.mode == int(entry['mode'], 8) == member.mode, member.name
            data = source.extractfile(member).read()
            assert len(data) == entry['size'], member.name
            assert hashlib.sha256(data).hexdigest() == entry['sha256'], member.name
    assert seen == set(expected), 'archive path population mismatch'
    return len(seen)


def extract_verified(archive, destination, manifest):
    """For a separately approved future transport; exact fresh destination only."""
    inspect_archive(archive, manifest)
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter='data')
    for entry in manifest['files']:
        path = destination / entry['path']
        assert path.is_file() and not path.is_symlink(), entry['path']
        assert stat.S_IMODE(path.stat().st_mode) == int(entry['mode'], 8), entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']


def roundtrip_check(archive, manifest, scratch_parent):
    """Authored check only during execution HOLD; caller owns scratch cleanup."""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix='transport-roundtrip-', dir=scratch_parent))
    extract_verified(archive, root / 'extracted', manifest)
    return root


if __name__ == '__main__':
    import sys
    archive, manifest_file = map(Path, sys.argv[1:])
    manifest = json.loads(manifest_file.read_text())
    print(json.dumps({'static_archive_files_verified': inspect_archive(archive, manifest),
                      'filesystem_roundtrip': 'NOT RUN', 'application_imports': False}))
