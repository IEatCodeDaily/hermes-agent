"""MCP machine-readable results survive ACP display formatting."""
import json

import pytest

from acp_adapter.tools import build_tool_complete


@pytest.mark.parametrize("payload", [
    {"htmlRender": {"attachmentId": "chart-html", "title": "Chart", "height": 300}},
    {"threadId": "thread-1", "messageId": "message-1"},
])
@pytest.mark.parametrize("structured", [False, True])
def test_mcp_completion_preserves_client_metadata(payload, structured):
    envelope = {"result": json.dumps(payload)}
    if structured:
        envelope["structuredContent"] = payload
    update = build_tool_complete("tc-mcp", "mcp__editor_test__html_render", json.dumps(envelope))
    raw = update.model_dump(by_alias=True, exclude_none=True)["rawOutput"]
    assert json.loads(raw["content"][0]["text"]) == payload
    if structured:
        assert raw["structuredContent"] == payload
    assert update.content  # readable summaries stay intact


def test_mcp_completion_preserves_error_verdict():
    update = build_tool_complete("tc-error", "mcp__editor_test__html_render",
                                 json.dumps({"error": "failed"}), is_error=True)
    assert update.status == "failed"
    assert update.raw_output["isError"] is True


@pytest.mark.parametrize("name,server,tool", [
    ("mcp__editor_t3_code_214ffdcc__html_render", "t3_code", "html_render"),
    ("mcp__editor_other_1234abcd__html_render", "other", "html_render"),
    ("mcp__grafana__search_dashboards", "grafana", "search_dashboards"),
])
def test_mcp_start_exposes_original_server_identity(name, server, tool):
    from acp_adapter.tools import build_tool_start

    update = build_tool_start("tc-identity", name, {"query": "test"})
    assert update.field_meta == {"serverId": server, "toolName": tool}
    assert update.raw_input == {"query": "test"}
    assert name in update.title


def test_non_mcp_structured_results_remain_compact():
    update = build_tool_complete("tc-file", "read_file", json.dumps({"content": "hello"}))
    assert update.raw_output is None


@pytest.mark.asyncio
async def test_gateway_and_replay_preserve_mcp_attachment():
    from unittest.mock import AsyncMock

    from acp_adapter.gateway_server import GatewayACPAgent
    from acp_adapter.server import _history_replay_updates

    name = "mcp__editor_test__html_render"
    payload = {"htmlRender": {"attachmentId": "chart-html", "title": "Chart", "height": 300}}
    result = json.dumps({"result": json.dumps(payload)})
    agent = GatewayACPAgent()
    agent._snapshots["s"] = {}
    agent._conn = AsyncMock()
    for kind in ("tool.start", "tool.complete"):
        await agent._project({"session_id": "s", "type": kind, "admission_id": "a",
                              "payload": {"tool_call_id": "call-1", "tool_name": name,
                                          "args": {}, "result": result, "is_error": False}})
    live = agent._conn.session_update.await_args_list[-1].kwargs["update"]
    history = [
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
         "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "tool_name": name, "content": result},
    ]
    replay = list(_history_replay_updates(history))[-1]
    assert live.raw_output == replay.raw_output
    assert json.loads(live.raw_output["content"][0]["text"]) == payload
