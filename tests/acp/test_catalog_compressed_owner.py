from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from acp_adapter import catalog


def test_compressed_root_resolves_tip_owner():
    class DB:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get_compression_lineage(self, sid): return ['root', 'tip']
    with TemporaryDirectory() as directory, patch.object(catalog, 'SessionDB', DB):
        path = Path(directory) / 'state.db'
        path.touch()
        for owners, expected in [({'tip': 'tip'}, 'tip'), ({'root': 'owner', 'tip': 'owner'}, 'owner'), ({}, 'root')]:
            with patch.object(catalog, '_local_owners', return_value=owners):
                assert catalog.logical_session_id(path, 'root') == expected


if __name__ == '__main__':
    test_compressed_root_resolves_tip_owner()
    print('PASS: adopted tip, canonical owner, unadopted lineage')
