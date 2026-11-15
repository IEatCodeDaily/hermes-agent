"""Native spawn policy at real dispatch/catalog seams; no model or child runs."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def isolated_source(monkeypatch):
    from gateway import session_context as sc
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    token = sc._SESSION_SOURCE.set("")
    # Force the process-level gateway latch OFF: this suite must be order-independent
    # no matter which earlier test file in the same pytest process bound a session.
    monkeypatch.setattr(sc, "_session_context_engaged", False)
    try:
        yield
    finally:
        sc._SESSION_SOURCE.reset(token)


def parent(platform="acp"):
    return SimpleNamespace(platform=platform, session_id="owned", enabled_toolsets=["delegation"],
                           disabled_toolsets=None, quiet_mode=True)


@pytest.mark.parametrize("seam", ["direct", "registry", "agent"])
@pytest.mark.parametrize("platform,source", [("acp", "cli"), ("cli", "acp")])
def test_spawn_refused_before_config_or_children(monkeypatch, seam, platform, source):
    from gateway import session_context as sc
    from tools import delegate_tool as dt
    from tools.registry import registry
    from run_agent import AIAgent
    sc._SESSION_SOURCE.set(source)
    p = parent(platform)
    config = Mock(side_effect=AssertionError("spawn reached config"))
    monkeypatch.setattr(dt, "_load_config", config)
    args = {"tasks": [{"goal": "do not run"}]}
    if seam == "direct":
        raw = dt.delegate_task(**args, parent_agent=p)
    elif seam == "registry":
        raw = registry.dispatch("delegate_task", args, parent_agent=p)
    else:
        raw = AIAgent._dispatch_delegate_task(p, args)
    assert "disabled by policy" in raw
    config.assert_not_called()


def test_shared_child_constructor_refuses_before_runtime(monkeypatch):
    from tools import delegate_tool as dt
    runtime = Mock(side_effect=AssertionError("child constructor reached config"))
    monkeypatch.setattr(dt, "_get_max_spawn_depth", runtime)
    with pytest.raises(ValueError, match="disabled by policy"):
        dt._build_child_agent(0, "do not run", None, None, None, 1, 1, parent())
    runtime.assert_not_called()


def test_session_policy_is_pinned_and_platform_cannot_be_overridden(monkeypatch):
    from gateway import session_context as sc
    from toolsets import native_delegate_refusal
    p = parent("cli")
    sc._SESSION_SOURCE.set("acp")
    assert native_delegate_refusal(p)
    sc._SESSION_SOURCE.set("cli")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    assert native_delegate_refusal(p)
    # Other sessions in the same process must not inherit this session's deny.
    ordinary = parent("cli")
    assert native_delegate_refusal(ordinary) is None
    sc._SESSION_SOURCE.set("acp")
    assert native_delegate_refusal(ordinary) is None
    assert native_delegate_refusal(parent("acp"))


def test_owned_controls_survive_all_dispatch_seams(monkeypatch):
    from tools import delegate_tool as dt
    from tools import delegate_tool_registry as dr
    from tools.registry import registry
    from run_agent import AIAgent
    p = parent()
    child, foreign = Mock(), Mock()
    child._live_transcript_path = foreign._live_transcript_path = None
    child._delegate_parent_ref = foreign._delegate_parent_ref = None
    child.steer.return_value = True
    child.interrupt.return_value = True
    records = {
        "owned-child": {"subagent_id": "owned-child", "owner_agent_session_id": "owned",
                        "agent": child, "accepting_steer": True},
        "foreign-child": {"subagent_id": "foreign-child", "owner_agent_session_id": "foreign",
                          "agent": foreign, "accepting_steer": True},
    }
    monkeypatch.setattr(dr, "_active_subagents", records)
    for dispatch in (lambda a: dt.delegate_task(**a, parent_agent=p),
                     lambda a: registry.dispatch("delegate_task", a, parent_agent=p),
                     lambda a: AIAgent._dispatch_delegate_task(p, a)):
        roster = json.loads(dispatch({"action": "list"}))
        assert [r["subagent_id"] for r in roster["subagents"]] == ["owned-child"]
        assert not child.interrupt.called
        assert json.loads(dispatch({"action": "steer", "subagent_id": "owned-child", "message": "continue"}))["status"] == "queued"
        assert "error" in json.loads(dispatch({"action": "steer", "subagent_id": "foreign-child", "message": "no"}))
        assert "error" in json.loads(dispatch({"action": "stop", "subagent_id": "foreign-child"}))
    foreign.steer.assert_not_called()
    foreign.interrupt.assert_not_called()
    for dispatch in (lambda a: dt.delegate_task(**a, parent_agent=p),
                     lambda a: registry.dispatch("delegate_task", a, parent_agent=p),
                     lambda a: AIAgent._dispatch_delegate_task(p, a)):
        assert json.loads(dispatch({"action": "stop", "subagent_id": "owned-child"}))["status"] == "interrupt_requested"
    assert set(records) == {"owned-child", "foreign-child"}


def test_standalone_spawn_reaches_unchanged_validation(monkeypatch):
    from tools import delegate_tool as dt
    monkeypatch.setattr(dt, "_load_config", dict)
    result = json.loads(dt.delegate_task(tasks=[], parent_agent=parent("cli")))
    assert "disabled by policy" not in result.get("error", "")
    assert "error" in result


@pytest.mark.parametrize("seam", ["direct", "none_parent", "registry", "agent"])
def test_gateway_hosted_process_denies_spawn_on_every_platform(monkeypatch, seam):
    """The process latch is the policy: any platform, and parent_agent=None, are refused."""
    from gateway import session_context as sc
    from tools import delegate_tool as dt
    from tools.registry import registry
    from run_agent import AIAgent
    monkeypatch.setattr(sc, "_session_context_engaged", True)
    config = Mock(side_effect=AssertionError("spawn reached config"))
    monkeypatch.setattr(dt, "_load_config", config)
    args = {"tasks": [{"goal": "do not run"}]}
    calls = {
        "direct": lambda: dt.delegate_task(**args, parent_agent=parent("cli")),
        "none_parent": lambda: dt.delegate_task(**args, parent_agent=None),
        "registry": lambda: registry.dispatch("delegate_task", args, parent_agent=parent("cli")),
        "agent": lambda: AIAgent._dispatch_delegate_task(parent("cli"), args),
    }
    raw = calls[seam]()
    assert "disabled by policy" in raw
    config.assert_not_called()


def test_gateway_hosted_child_constructor_refuses(monkeypatch):
    from gateway import session_context as sc
    from tools import delegate_tool as dt
    monkeypatch.setattr(sc, "_session_context_engaged", True)
    runtime = Mock(side_effect=AssertionError("child constructor reached config"))
    monkeypatch.setattr(dt, "_get_max_spawn_depth", runtime)
    with pytest.raises(ValueError, match="disabled by policy"):
        dt._build_child_agent(0, "do not run", None, None, None, 1, 1, parent("cli"))
    runtime.assert_not_called()


def test_gateway_hosted_schema_is_control_only(monkeypatch):
    from gateway import session_context as sc
    from toolsets import native_delegate_control_tools, native_delegate_refusal
    monkeypatch.setattr(sc, "_session_context_engaged", True)
    p = parent("cli")
    assert native_delegate_refusal(p)
    assert native_delegate_refusal(None)
    tools = native_delegate_control_tools(p, [{"type": "function", "function": {"name": "read_file"}}])
    schema = next(d["function"] for d in tools if d["function"]["name"] == "delegate_task")
    assert set(schema["parameters"]["properties"]) == {"action", "subagent_id", "message"}
    assert schema["parameters"]["properties"]["action"]["enum"] == ["list", "steer", "stop"]


def test_discovery_control_only_and_deferred_catalog_cannot_redisclose_spawn(monkeypatch):
    import model_tools
    from tools import delegate_tool as dt, tool_search as ts
    from tools.registry import registry
    from toolsets import session_disabled_toolsets
    from agent.agent_init import _load_tools
    monkeypatch.setattr("hermes_cli.left_core_migration.recover_at_startup", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(ts, "load_config", lambda: ts.ToolSearchConfig.from_raw({"defer": ["delegate_task"]}))
    monkeypatch.setattr(ts, "load_config_readonly", ts.load_config)
    editor = "mcp__editor_t3_code_fixture__delegate_task"
    registry.register(name=editor, toolset="mcp-policy-fixture", schema={"name": editor, "description": "editor delegation", "parameters": {"type": "object", "properties": {}}}, handler=lambda a, **kw: "editor-ok")
    registry.register_toolset_alias("delegation", "mcp-policy-fixture")
    try:
        p = parent()
        _load_tools(p, ["delegation", "mcp-policy-fixture"], None)
        schema = next(d["function"] for d in p.tools if d["function"]["name"] == "delegate_task")
        assert schema["parameters"]["required"] == ["action"]
        assert set(schema["parameters"]["properties"]) == {"action", "subagent_id", "message"}
        assert schema["parameters"]["properties"]["action"]["enum"] == ["list", "steer", "stop"]
        assert "delegation" in p.disabled_toolsets
        raw = model_tools.get_tool_definitions(enabled_toolsets=["delegation", "mcp-policy-fixture"], disabled_toolsets=p.disabled_toolsets, quiet_mode=True, skip_tool_search_assembly=True)
        assert "delegate_task" not in {d["function"]["name"] for d in raw}
        assert editor in {d["function"]["name"] for d in raw}
        described, _ = model_tools._dispatch_bridge_tool("tool_describe", {"names": ["delegate_task"]}, p.enabled_toolsets + ["mcp-policy-fixture"], p.disabled_toolsets)
        assert "Spawn subagents" not in described
        assert "\"tasks\"" not in described
        assert registry.dispatch(editor, {}, parent_agent=p) == "editor-ok"
        from tools.mcp_tool_agent import _drop_session_tools, _resolve_refresh_toolsets
        stale = [{"type": "function", "function": dt.DELEGATE_TASK_SCHEMA},
                 {"type": "function", "function": {"name": editor}}]
        refreshed, names = _drop_session_tools(p, stale, {"delegate_task", editor})
        assert editor in names
        assert next(d["function"] for d in refreshed if d["function"]["name"] == "delegate_task") == schema
        assert "delegation" in _resolve_refresh_toolsets(p, None, [])[1]
        assert "delegation" not in (session_disabled_toolsets(None, "cli") or [])
    finally:
        registry.deregister(editor)


@pytest.mark.parametrize("stale_spawn,late_tool", [(False, True), (True, False)])
def test_public_refresh_preserves_control_slot_and_enforces_same_name_policy(monkeypatch, stale_spawn, late_tool):
    from copy import deepcopy
    import model_tools
    from tools import delegate_tool as dt
    from tools import mcp_tool_agent as ma
    from tools.registry import registry

    def definition(name, description="original"):
        return {"type": "function", "function": {"name": name, "description": description}}

    control = {"type": "function", "function": deepcopy(dt.DELEGATE_CONTROL_SCHEMA)}
    p = parent()
    p.tools = [definition("read_file"),
               {"type": "function", "function": deepcopy(dt.DELEGATE_TASK_SCHEMA)} if stale_spawn else deepcopy(control)]
    before = deepcopy(p.tools)
    p.valid_tool_names = {"read_file", "delegate_task"}
    editor = "mcp__editor_t3_code_fixture__delegate_task"
    rebuilt = [definition("read_file", "changed")]
    if late_tool:
        rebuilt.append(definition(editor))
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: rebuilt)
    monkeypatch.setattr(registry, "get_all_entries", lambda: [SimpleNamespace(name=n) for n in ["read_file", "delegate_task", editor]])
    monkeypatch.setattr(ma, "_reinject_post_build_tools", lambda *a: set())
    monkeypatch.setattr(ma, "_reinject_authorized_dynamic_tools", lambda *a: None)
    monkeypatch.setattr(ma, "persist_agent_tool_names", lambda *a: None)
    added = ma.refresh_agent_mcp_tools(p, preserve_prefix=True)
    assert p.tools[:2] == [before[0], control]
    assert p.valid_tool_names == {d["function"]["name"] for d in p.tools}
    assert added == ({editor} if late_tool else set())
    if late_tool:
        assert p.tools[:len(before)] == before
        assert p.tools[2] == definition(editor)


def test_orchestrator_cannot_regain_policy_disabled_delegation():
    from tools.delegate_tool_toolsets import _resolve_child_toolsets
    from toolsets import native_delegate_refusal
    p = parent()
    assert native_delegate_refusal(p)
    enabled, disabled = _resolve_child_toolsets(p, ["delegation"], "orchestrator")
    assert "delegation" not in enabled
    assert "delegation" in disabled
