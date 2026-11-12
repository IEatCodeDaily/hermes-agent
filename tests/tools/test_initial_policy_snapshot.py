"""Initial assembly shares refresh filtering without moving existing schema slots."""

from copy import deepcopy
from types import SimpleNamespace

from agent.agent_init import _load_tools


def test_initial_control_replacement_preserves_unrelated_order(monkeypatch):
    import model_tools
    from tools.delegate_tool import DELEGATE_CONTROL_SCHEMA, DELEGATE_TASK_SCHEMA

    editor = "mcp__editor_t3_code_fixture__delegate_task"
    definitions = [{"type": "function", "function": {"name": "read_file"}},
                   {"type": "function", "function": deepcopy(DELEGATE_TASK_SCHEMA)},
                   {"type": "function", "function": {"name": editor}}]
    before = deepcopy(definitions)
    monkeypatch.setattr("hermes_cli.left_core_migration.recover_at_startup", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: definitions)
    agent = SimpleNamespace(platform="acp", quiet_mode=True)
    _load_tools(agent, None, None)
    assert agent.tools == [before[0], {"type": "function", "function": DELEGATE_CONTROL_SCHEMA}, before[2]]
    assert definitions == before
    assert agent.valid_tool_names == {"read_file", "delegate_task", editor}
    assert "delegation" in agent.disabled_toolsets
