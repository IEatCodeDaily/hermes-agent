"""Imported-path candidate checks; copy to tests/tools in an isolated Talos snapshot."""
import json
from types import SimpleNamespace

import pytest
from agent import prompt_builder as pb, skill_utils as su
from tools import skills_tool as st


@pytest.fixture
def library(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    (home / 'skills').mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(st, 'SKILLS_DIR', home / 'skills')
    monkeypatch.chdir(tmp_path)
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, 'discover_plugins', lambda: None)
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: SimpleNamespace(list_plugin_skill_metadata=lambda: []))
    (home / 'config.yaml').write_text('skills:\n  external_dirs: []\n')
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()
    st._SKILLS_CACHE.clear()
    yield home
    st._SKILLS_CACHE.clear()
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def put(root, rel, name, description='Security review', extra=''):
    path = root / rel / 'SKILL.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'---\nname: {name}\ndescription: {description}\n{extra}---\nBody\n')
    return path


def test_full_fallback_requires_both_tools():
    rows = {'ops': [('restore', 'Recover a backup')]}
    for tools in [None, set(), {'skills_list'}, {'skill_view'}]:
        assert 'Recover a backup' in pb._render_skills_index(rows, {}, None, tools)
    compact = pb._render_skills_index(rows, {}, None, {'skills_list', 'skill_view'})
    assert 'Recover a backup' not in compact
    assert 'NO arguments at all' in compact and 'no query and no category' in compact
    assert 'has_more' in compact and 'fails' in compact
    # Callers may pass lists, not sets: the gate must not raise on membership testing.
    assert 'Recover a backup' not in pb._render_skills_index(rows, {}, None, ['skills_list', 'skill_view'])
    # Unknown tool context (None) fails open to the historical posture demotion...
    unknown = pb._render_skills_index(rows, {}, frozenset({'ops'}), None)
    assert 'demo [names only]' in unknown or 'ops [names only]' in unknown
    # ...while a KNOWN tool set missing the recovery pair retains descriptions.
    assert 'Recover a backup' in pb._render_skills_index(rows, {}, frozenset({'ops'}), {'skills_list'})


def test_actual_scanner_categories(library, tmp_path, monkeypatch):
    project = tmp_path / 'project'
    (project / '.git').mkdir(parents=True)
    brain, external = tmp_path / 'brain', tmp_path / 'external'
    put(project / '.hermes/skills', 'ops/restore', 'project-restore')
    put(library / 'skills', 'mlops/training/train', 'train')
    put(brain, 'ops/snapshot', 'snapshot')
    put(external, 'ops/recover', 'recover')
    (library / 'config.yaml').write_text(
        f'skills:\n  trusted_project_dirs: ["{project}"]\n  create_dir: {brain}\n  external_dirs: ["{external}"]\n')
    monkeypatch.chdir(project)
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()
    rows = {r['name']: r for r in st.find_all_skills()}
    assert set(rows) == {'project-restore', 'train', 'snapshot', 'recover'}
    assert rows['project-restore']['category'] == 'ops'
    assert rows['train']['category'] == 'mlops/training'
    assert rows['snapshot']['category'] == rows['recover']['category'] == 'ops'
    assert {r['name'] for r in json.loads(st.skills_list(category='ops'))['skills']} == {'project-restore', 'snapshot', 'recover'}


def test_plugin_environment_app_gates(library, monkeypatch):
    import hermes_cli.plugins as plugins
    meta = {'name': 'example:restore', 'description': 'Recover backup', 'category': 'plugin', 'frontmatter': {}}
    original = json.loads(json.dumps(meta))
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: SimpleNamespace(list_plugin_skill_metadata=lambda: [meta]))
    monkeypatch.setattr(st, '_is_skill_disabled', lambda *a: False)
    monkeypatch.setattr(st, 'skill_matches_platform', lambda fm: True)
    for env, apps in [(False, True), (True, False), (True, True)]:
        monkeypatch.setattr(st, 'skill_matches_environment', lambda fm: env)
        monkeypatch.setattr(st, 'skill_matches_apps', lambda fm: apps)
        result = json.loads(st.skills_list(query='backup'))
        assert result['count'] == int(env and apps)
    assert meta == original


def test_plugin_failure_is_not_complete_absence(library, monkeypatch):
    import hermes_cli.plugins as plugins

    def fail():
        raise RuntimeError('injected plugin failure')

    for populated in [False, True]:
        if populated:
            put(library / 'skills', 'ops/backup', 'backup', 'Backup')
        for stage in ['discover', 'metadata']:
            monkeypatch.setattr(plugins, 'discover_plugins', fail if stage == 'discover' else lambda: None)
            monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: SimpleNamespace(list_plugin_skill_metadata=fail))
            for args in [{}, {'query': 'missing'}]:
                result = json.loads(st.skills_list(**args))
                assert result['success'] is False
                assert 'discovery' in result['error'].lower()


def test_registered_dispatch_and_recovery(library):
    from tools.registry import registry
    put(library / 'skills', 'ops/review', 'review', 'Security review')
    entry = registry.get_entry('skills_list')
    schema = registry.get_definitions({'skills_list'})[0]['function']
    assert {'query', 'limit'} <= schema['parameters']['properties'].keys()
    result = json.loads(entry.handler({'query': 'security', 'limit': 1}))
    assert [r['name'] for r in result['skills']] == ['review']
    assert json.loads(st.skills_list(query='security code review'))['count'] == 0
    fallback = json.loads(st.skills_list())
    assert fallback == {'success': True, 'skills': [{'name': 'review', 'description': 'Security review', 'category': 'ops'}],
                        'categories': ['ops'], 'count': 1, 'hint': 'Use skill_view(name) to see full content, tags, and linked files'}
    for query, limit in [('x' * 201, 20), ('x', 0), ('x', 101), ('x', True)]:
        assert json.loads(st.skills_list(query=query, limit=limit))['success'] is False
    # Blank query must behave exactly like an omitted query (models auto-fill empty strings).
    blank = json.loads(st.skills_list(query='', limit=100))
    assert blank['success'] is True and blank == json.loads(st.skills_list(limit=100))


def test_empty_and_truncation(library):
    assert json.loads(st.skills_list(query='backup'))['total_matches'] == 0
    for i in range(3):
        put(library / 'skills', f'ops/b{i}', f'b{i}', 'backup')
    result = json.loads(st.skills_list(query='backup', limit=1))
    assert result['total_matches'] == 3 and result['has_more']
    assert {r['name'] for r in json.loads(st.skills_list())['skills']} == {'b0', 'b1', 'b2'}


def test_exact_identifiers_and_oneshot(monkeypatch):
    from agent.skills_index_delta import index_entries
    from agent import oneshot_footprint
    rows = {'coding': [('a/review', 'Review'), ('b/review', 'Review'),
                       ('plugin:review', 'Review'), ('hermes-agent', 'Hermes help')]}
    expected = {name for name, _ in rows['coding']}
    for oneshot in [False, True]:
        monkeypatch.setattr(oneshot_footprint, 'is_single_query_session', lambda: oneshot)
        for tools in [None, set(), {'skills_list'}, {'skill_view'}, {'skills_list', 'skill_view'}]:
            prompt = pb._render_skills_index(rows, {}, frozenset({'coding'}), tools)
            assert set(index_entries(prompt)) == expected
            assert prompt == pb._render_skills_index(rows, {}, frozenset({'coding'}), tools)
            # The help pointer must not duplicate a name the names-only row already lists.
            assert sum('hermes-agent' in line for line in prompt.splitlines()) == 1


def test_description_recall_recovery_matrix(library):
    from agent.skills_index_delta import index_entries
    for name, description in [('review', 'Security review'), ('restore', 'Recover backup'),
                              ('motion', 'Smooth UI animation')]:
        put(library / 'skills', f'ops/{name}', name, description)
    # Lexical search is not semantic: missing words/synonyms/punctuation require full recovery.
    for query, required in [('security code review', 'review'), ('audit', 'review'),
                            ('restore!', 'restore'), ('motion design', 'motion'),
                            ('backup', 'restore'), ('animation', 'motion')]:
        result = json.loads(st.skills_list(query=query, limit=1))
        matches = {r['name'] for r in result['skills']}
        if not matches or result['has_more']:
            matches = {r['name'] for r in json.loads(st.skills_list())['skills']}
        assert required in matches
    prompt = pb.build_skills_system_prompt(available_tools={'skills_list', 'skill_view'})
    assert set(index_entries(prompt)) == {'review', 'restore', 'motion'}
    assert 'NO arguments at all' in prompt and 'no query and no category' in prompt

