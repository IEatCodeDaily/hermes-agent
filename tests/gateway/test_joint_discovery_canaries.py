"""Additional discovery smoke on a bounded fixture library, never host skills."""
import json
from types import SimpleNamespace

import pytest

from agent import prompt_builder as pb, skill_utils as su
from tools import skills_tool as st


@pytest.fixture
def library(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    skills = home / 'skills'
    for name, description in [('backup', 'PostgreSQL backup'), ('hermes-agent', 'Hermes help')]:
        folder = skills / 'ops' / name
        folder.mkdir(parents=True)
        (folder / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: {description}\n---\nFixture guidance.\n')
    (home / 'config.yaml').write_text('skills:\n  external_dirs: []\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(st, 'SKILLS_DIR', skills)
    monkeypatch.chdir(tmp_path)
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, 'discover_plugins', lambda: None)
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: SimpleNamespace(list_plugin_skill_metadata=lambda: []))
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()
    st._SKILLS_CACHE.clear()
    yield
    st._SKILLS_CACHE.clear()
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def test_prompt_builder_list_and_unknown_tools(library):
    rows = {'ops': [('x', 'X desc')]}
    compact = pb._render_skills_index(rows, {}, frozenset({'ops'}), ['skills_list', 'skill_view'])
    assert '[names only]' in compact and 'X desc' not in compact
    unknown = pb._render_skills_index(rows, {}, frozenset({'ops'}), None)
    assert '[names only]' in unknown
    prompt = pb.build_skills_system_prompt(available_tools={'skills_list', 'skill_view'})
    assert '[names only]' in prompt
    assert 'NO arguments at all' in prompt
    assert prompt.count('hermes-agent') == 1


def test_blank_and_zero_match_fallback(library):
    full = json.loads(st.skills_list())
    assert len(full['skills']) == 2
    assert json.loads(st.skills_list(query='', category=''))['skills'] == full['skills']
    assert json.loads(st.skills_list(query='zzz-none'))['total_matches'] == 0
    assert json.loads(st.skills_list())['skills'] == full['skills']
    assert json.loads(st.skills_list(query='x' * 201))['success'] is False
