"""Execution-owned trace identities, terminal cleanup and dispatch timing (no API calls)."""
import asyncio
import contextvars
import threading
from types import SimpleNamespace

import pytest

from agent.monitoring.events import AgentToolExecEvent, AgentTurnEndEvent, pseudonymize_id
from gateway import agent_trace_emit as tracing


@pytest.fixture
def events(monkeypatch):
    collected = []
    monkeypatch.setattr(tracing, "_emit", collected.append)
    return collected


def _agent():
    return SimpleNamespace(session_id="session-old", provider="custom", model="request-model")


def test_prompt_occupancy_is_not_usage(events):
    tracing.emit_turn_end("s", "t", "completed", last_prompt_tokens=123)
    event = events[-1]
    assert event.input_tokens is None
    assert event.prompt_occupancy_tokens == 123
    assert event.model_id is None
    assert not event.model_class


@pytest.mark.parametrize("occupancy", [None, 0, -1, True, "123"])
def test_unmeasured_prompt_occupancy_is_omitted(events, occupancy):
    tracing.emit_turn_end("s", "t", "completed", last_prompt_tokens=occupancy, model_id="unknown")
    assert events[-1].prompt_occupancy_tokens is None
    assert events[-1].model_id is None


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted", "superseded"])
def test_captured_identity_and_terminal_result(events, outcome):
    agent = _agent()
    with tracing.trace_execution(agent, "session-old", "turn-old") as trace:
        agent.session_id = "rotated"
        agent._current_turn_id = "successor"
        trace.result = {outcome: True, "model": "active-model", "provider": "custom",
                        "last_prompt_tokens": 17, "session_input_tokens": 999}
    ends = [e for e in events if isinstance(e, AgentTurnEndEvent)]
    assert len(ends) == 1
    assert ends[0].status == outcome
    assert ends[0].session_ref == pseudonymize_id("session-old")
    assert ends[0].turn_ref == pseudonymize_id("turn-old")
    assert ends[0].input_tokens is None
    assert ends[0].prompt_occupancy_tokens == 17
    assert ends[0].model_id == "active-model"
    assert ends[0].model_class == "custom"


@pytest.mark.parametrize("error,status", [(asyncio.CancelledError(), "cancelled"),
                                         (KeyboardInterrupt(), "cancelled"),
                                         (RuntimeError("private error"), "failed")])
def test_baseexception_cleanup_once(events, error, status):
    with pytest.raises(type(error)):
        with tracing.trace_execution(_agent(), "s", "t"):
            raise error
    ends = [e for e in events if isinstance(e, AgentTurnEndEvent)]
    assert len(ends) == 1
    assert ends[0].status == status
    assert ends[0].error_class == type(error).__name__
    assert "private error" not in repr(events)


@pytest.mark.parametrize("error,status", [(RuntimeError("private"), "failed"),
                                         (asyncio.CancelledError(), "cancelled")])
def test_tool_exception_has_one_terminal_with_type_only(events, error, status):
    with tracing.trace_execution(_agent(), "s", "t") as trace:
        with pytest.raises(type(error)):
            with tracing.trace_tool("call", "web_search"):
                raise error
        trace.result = {"completed": True}
    tools = [e for e in events if isinstance(e, AgentToolExecEvent)]
    assert [e.phase for e in tools] == ["start", "end"]
    assert tools[-1].status == status
    assert tools[-1].error_class == type(error).__name__
    assert tools[-1].ts_ns >= tools[0].ts_ns
    assert "private" not in repr(events)


def test_late_worker_end_cannot_reparent_or_duplicate(events):
    started, release = threading.Event(), threading.Event()
    agent = _agent()
    with tracing.trace_execution(agent, "s-old", "t-old") as trace:
        worker_context = contextvars.copy_context()
        def work():
            with tracing.trace_tool("call", "web_search") as tool:
                started.set()
                assert release.wait(5)
                tool.result = "ok"
        worker = threading.Thread(target=worker_context.run, args=(work,))
        worker.start()
        assert started.wait(5)
        trace.result = {"interrupted": True}
    with tracing.trace_execution(agent, "s-new", "t-new") as successor:
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        successor.result = {"completed": True}
    tools = [e for e in events if isinstance(e, AgentToolExecEvent)]
    assert [e.phase for e in tools] == ["start", "end"]
    assert tools[-1].status == "interrupted"
    assert all(e.turn_ref == pseudonymize_id("t-old") for e in tools)
    old_end = next(e for e in events if isinstance(e, AgentTurnEndEvent) and e.turn_ref == pseudonymize_id("t-old"))
    assert tools[-1].ts_ns == old_end.ts_ns


def test_telemetry_failure_cannot_change_tool_or_turn_outcome(events, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("telemetry unavailable")
    monkeypatch.setattr(tracing._ExecutionTrace, "_tool_event", broken)
    monkeypatch.setattr(tracing, "emit_turn_end", broken)
    with tracing.trace_execution(_agent(), "s", "t") as trace:
        with tracing.trace_tool("call", "web_search") as tool:
            tool.result = {"ok": True}
        trace.result = {"completed": True}
    with pytest.raises(ValueError, match="original"):
        with tracing.trace_execution(_agent(), "s", "error"):
            raise ValueError("original")


@pytest.mark.asyncio
async def test_worker_factory_captures_generation_before_executor_context(events, monkeypatch):
    from gateway import session_runtime_workers as workers
    monkeypatch.setattr(workers, "track_turn_worker", lambda *args: None)
    current = [True]
    worker = SimpleNamespace(is_current=lambda: current[0])
    async def executor(fn):
        return await asyncio.to_thread(fn)
    def execute():
        with tracing.trace_execution(_agent(), "host-session", "host-turn") as trace:
            current[0] = False
            trace.result = {"completed": True}
    workers.start_turn_worker(SimpleNamespace(_run_in_executor_with_context=executor), worker, [], execute)
    await worker.executor_task
    assert next(e for e in events if isinstance(e, AgentTurnEndEvent)).status == "superseded"
    # Task-local fence must not leak into a subsequent standalone execution.
    with tracing.trace_execution(_agent(), "standalone", "standalone") as trace:
        trace.result = {"completed": True}
    assert [e.status for e in events if isinstance(e, AgentTurnEndEvent)] == ["superseded", "completed"]


def test_generation_fence_marks_execution_superseded(events):
    current = [True]
    with tracing.bind_execution_owner(lambda: current[0]):
        with tracing.trace_execution(_agent(), "s", "t") as trace:
            current[0] = False
            trace.result = {"completed": True}
    assert next(e for e in events if isinstance(e, AgentTurnEndEvent)).status == "superseded"


def test_actual_dispatch_starts_before_execution_and_block_has_no_span(events, monkeypatch):
    from agent import tool_executor as te
    monkeypatch.setattr(te, "_pre_tool_block", lambda a, r: (None, r.args))
    monkeypatch.setattr(te, "_begin_tool_execution", lambda *a: None)
    monkeypatch.setattr(te, "_run_with_activity_heartbeat", lambda a, n, fn: fn())
    monkeypatch.setattr("agent.terminal_approval_batch.prepare_current_terminal", lambda ref: None)
    agent = _agent()
    agent.tool_progress_callback = None
    agent._tool_guardrails = SimpleNamespace(before_call=lambda *a: SimpleNamespace(allows_execution=True))
    def execute(args):
        tools = [e for e in events if isinstance(e, AgentToolExecEvent)]
        assert [e.phase for e in tools] == ["start"]
        return {"error": "do not export this"}
    with tracing.trace_execution(agent, "s", "t") as trace:
        for blocked in ["policy refusal", None]:
            state = te._ManagedToolResult(None, {}, [], False, False)
            te._dispatch_authorized_once(agent, state, te._ToolCallRef("web_search", {}, "task", "call", []),
                                         execute=execute, scope_block=blocked, display_index=None,
                                         begin_execution=None, authorization_gate=None)
            if blocked:
                assert not any(isinstance(e, AgentToolExecEvent) for e in events)
        trace.result = {"completed": True}
    tools = [e for e in events if isinstance(e, AgentToolExecEvent)]
    assert [e.phase for e in tools] == ["start", "end"]
    assert tools[-1].status == "failed"
    assert tools[-1].error_class == "tool_result_error"
    assert "do not export this" not in repr(events)


@pytest.mark.parametrize("headers,model,fallback,display,expected", [
    ({"x-litellm-model-api-base": "https://proxy.invalid/v1"}, "request-model", False,
     "https://proxy.invalid/v1", "request-model"),
    ({"x-litellm-model-api-base": "opaque-endpoint"}, "unknown", False, "opaque-endpoint", None),
    ({"x-litellm-model-id": "deployment-eu-01", "x-litellm-model-api-base": "opaque-endpoint"},
     "request-model", False, "deployment-eu-01", "request-model"),
    ({"x-litellm-model-id": "deployment-eu-01"}, "unknown", False, "deployment-eu-01", None),
    ({}, "request-model", False, None, "request-model"),
    ({}, "unknown", False, None, None),
    ({}, " UNKNOWN ", False, None, None),
    ({}, "  ", False, None, None),
    ({}, None, False, None, None),
    ({"x-litellm-model-id": " UNKNOWN "}, "request-model", False, "UNKNOWN", "request-model"),
    ({"x-litellm-model-id": "unknown"}, "unknown", False, "unknown", None),
    ({"x-litellm-model-api-base": "opaque-endpoint"}, "active-model", True,
     "opaque-endpoint", "active-model"),
    ({}, "active-model", True, "active-model", "active-model"),
])
def test_header_result_export_model_provenance(events, headers, model, fallback, display, expected):
    from agent.served_model import install_served_model_capture, result_model_fields
    from agent.monitoring.agent_traces import AgentTracePlane
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    agent = _agent()
    agent.model = model
    agent._fallback_activated = fallback
    agent._primary_runtime = {"model": "primary-model"}
    client = SimpleNamespace(_client=SimpleNamespace(event_hooks={}))
    install_served_model_capture(agent, client)
    hook, = client._client.event_hooks["response"]
    hook(SimpleNamespace(status_code=200, headers=headers))
    fields = result_model_fields(agent)  # same result builder expanded by turn_finalizer
    assert fields["served_model"] == display
    assert fields["requested_model"] == ("primary-model" if fallback and not headers else model)
    with tracing.trace_execution(agent, "s", "t") as trace:
        trace.result = {"completed": True, "model": agent.model, **fields}
    end = next(e for e in events if isinstance(e, AgentTurnEndEvent))
    assert end.model_id == expected
    assert end.input_tokens is end.output_tokens is end.prompt_occupancy_tokens is None

    sink = InMemorySpanExporter()
    config = {"monitoring": {"agent_traces": {"enabled": True},
                             "export": {"otlp": {"enabled": True, "endpoint": "http://127.0.0.1:1"}}}}
    plane = AgentTracePlane(config, sink_factory=lambda: sink)
    plane.start()
    try:
        plane([event.to_dict() for event in events])
        plane._provider.force_flush(timeout_millis=2000)
        turn, = [span for span in sink.get_finished_spans() if span.name == "agent.turn"]
        assert turn.attributes.get("gen_ai.request.model") == expected
        assert not any(key.startswith("gen_ai.usage.") for key in turn.attributes)
    finally:
        plane.shutdown()


@pytest.mark.parametrize("model_fields,expected", [
    ({"model": "active-model"}, "active-model"),
    ({"model": "unknown"}, None),
    ({"model": " UNKNOWN "}, None),
    ({"model": "  "}, None),
    ({"model": None}, None),
    ({}, None),
])
def test_final_model_ignores_legacy_deployment_and_display_fields(events, model_fields, expected):
    agent = _agent()
    agent.last_served_model_id = "old-deployment"
    with tracing.trace_execution(agent, "s", "t") as trace:
        trace.result = {"completed": True, "served_model": "looks-like-a-model",
                        "served_model_id": "old-deployment", **model_fields}
    assert events[-1].model_id == expected


@pytest.mark.parametrize("model,expected", [
    ("active-fallback", "active-fallback"), (None, None), ("", None), (" UNKNOWN ", None),
])
def test_same_agent_next_turn_without_capture_uses_only_active_result_model(events, model, expected):
    from agent.served_model import install_served_model_capture, result_model_fields

    agent = _agent()
    client = SimpleNamespace(_client=SimpleNamespace(event_hooks={}))
    install_served_model_capture(agent, client)
    hook, = client._client.event_hooks["response"]
    with tracing.trace_execution(agent, "s", "old-turn") as trace:
        hook(SimpleNamespace(status_code=200, headers={"x-litellm-model-id": "old-proxy-deployment"}))
        trace.result = {"completed": True, "model": agent.model, **result_model_fields(agent)}
    assert agent.last_served_model == "old-proxy-deployment"
    # Legacy state may remain on a reused agent; neither it nor the UI label is current provenance.
    agent.last_served_model_id = "old-proxy-deployment"
    agent._primary_runtime = {"model": agent.model}
    agent.model, agent.provider, agent._fallback_activated = model, "anthropic", True
    with tracing.trace_execution(agent, "s", "next-turn") as trace:
        # No response hook on this turn (native fallback or interrupted before a response).
        fields = result_model_fields(agent)
        assert fields == {"requested_model": model, "served_model": "old-proxy-deployment"}
        trace.result = {"interrupted": True, "model": agent.model, **fields}
    ends = [e for e in events if isinstance(e, AgentTurnEndEvent)]
    assert [e.turn_ref for e in ends] == [pseudonymize_id("old-turn"), pseudonymize_id("next-turn")]
    assert ends[-1].status == "interrupted"
    assert ends[-1].model_id == expected
