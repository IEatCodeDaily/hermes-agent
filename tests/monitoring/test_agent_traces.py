"""Agent-trace plane tests (slice 1, review F3): in-process span assertions.

Precondition (review F8): run with PYTHONPATH including a pip --target dir
holding pytest, pytest-asyncio and opentelemetry-api/sdk/exporter at 1.45.1 —
the target dir shadows any venv opentelemetry-api, avoiding version skew.
Example on fxbuilder: /tmp/otel-spike-deps.
"""

from __future__ import annotations

import pytest

from agent.monitoring.agent_traces import AgentTracePlane, agent_traces_enabled
from agent.monitoring.events import (
    AgentChildEvent,
    AgentSessionStartEvent,
    AgentToolExecEvent,
    AgentTurnEndEvent,
    AgentTurnStartEvent,
    pseudonymize_id,
)

CONFIG = {
    "monitoring": {
        "agent_traces": {"enabled": True, "ttl_seconds": 3600},
        "export": {"otlp": {"enabled": True, "endpoint": "http://127.0.0.1:1"}},
    }
}

DISABLED_CONFIG = {
    "monitoring": {
        "agent_traces": {"enabled": False},
        "export": {"otlp": {"enabled": True, "endpoint": "http://127.0.0.1:1"}},
    }
}


def _plane_and_sink():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    sink = InMemorySpanExporter()
    plane = AgentTracePlane(CONFIG, sink_factory=lambda: sink)
    plane.start()
    return plane, sink


def _span_tree(sink):
    """Finished spans keyed by span name, preserving parent links."""
    spans = sink.get_finished_spans()
    by_name = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    return spans, by_name


def test_pseudonymize_id_stable_and_opaque():
    first = pseudonymize_id("agent:main:telegram:dm:1234567890")
    assert first == pseudonymize_id("agent:main:telegram:dm:1234567890")
    assert first.startswith("sha256:")
    assert "1234567890" not in first  # raw phone/chat id never egresses
    assert len(first) == len("sha256:") + 24
    assert pseudonymize_id(None) == pseudonymize_id("unknown")


def test_span_tree_hierarchy_and_exactly_once():
    plane, sink = _plane_and_sink()
    s, t = pseudonymize_id("sess-1"), pseudonymize_id("turn-1")
    plane([AgentSessionStartEvent(session_ref=s, origin="acp").to_dict()])
    plane(
        [
            AgentTurnStartEvent(
                session_ref=s, turn_ref=t, model_class="primary", model_id="requested-model"
            ).to_dict()
        ]
    )
    plane(
        [
            AgentToolExecEvent(
                phase="start",
                session_ref=s,
                turn_ref=t,
                tool_call_id="tc-abc123def456",
                tool_name="browser",
            ).to_dict()
        ]
    )
    plane(
        [
            AgentToolExecEvent(
                phase="end",
                session_ref=s,
                turn_ref=t,
                tool_call_id="tc-abc123def456",
                tool_name="browser",
                duration_ms=42,
            ).to_dict()
        ]
    )
    plane(
        [
            AgentChildEvent(
                phase="start",
                parent_session_ref=s,
                parent_turn_ref=t,
                child_session_ref=pseudonymize_id("child-1"),
            ).to_dict()
        ]
    )
    plane(
        [
            AgentChildEvent(
                phase="end",
                parent_session_ref=s,
                parent_turn_ref=t,
                child_session_ref=pseudonymize_id("child-1"),
                status="completed",
            ).to_dict()
        ]
    )
    plane(
        [
            AgentTurnEndEvent(
                session_ref=s,
                turn_ref=t,
                status="completed",
                duration_ms=100,
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=2,
                cache_write_tokens=3,
            ).to_dict()
        ]
    )
    plane.shutdown()

    spans, by_name = _span_tree(sink)
    assert len(spans) == 4  # session, turn, tool, child — one each, no duplicates
    session = by_name["agent.session"][0]
    turn = by_name["agent.turn"][0]
    tool = by_name["agent.tool"][0]
    child = by_name["agent.child"][0]
    assert turn.parent.span_id == session.context.span_id
    assert tool.parent.span_id == turn.context.span_id
    assert child.parent.span_id == turn.context.span_id
    turn_attrs = dict(turn.attributes)
    assert turn_attrs["hermes.model_class"] == "primary"
    assert turn_attrs["hermes.status"] == "completed"
    assert turn_attrs["hermes.input_tokens"] == 10
    assert turn_attrs["hermes.cache_read_tokens"] == 2
    assert dict(tool.attributes)["hermes.tool_call_id"] == "tc-abc123def456"
    # GenAI semconv dual-emit: additive on the same spans; conversation id is
    # the pseudonymized session_ref; success spans stay status-UNSET (no OK).
    session_attrs = dict(session.attributes)
    assert session_attrs["gen_ai.operation.name"] == "invoke_agent"
    assert session_attrs["gen_ai.agent.name"] == "hermes"
    assert session_attrs["gen_ai.conversation.id"] == s
    assert turn_attrs["gen_ai.operation.name"] == "chat"
    assert turn_attrs["gen_ai.request.model"] == "requested-model"
    assert turn_attrs["gen_ai.conversation.id"] == s
    assert turn_attrs["gen_ai.usage.input_tokens"] == 10
    assert turn_attrs["gen_ai.usage.output_tokens"] == 5
    assert turn_attrs["gen_ai.usage.cache_read.input_tokens"] == 2
    assert turn_attrs["gen_ai.usage.cache_write.input_tokens"] == 3
    tool_attrs = dict(tool.attributes)
    assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
    assert tool_attrs["gen_ai.tool.name"] == "browser"
    assert tool_attrs["gen_ai.tool.call.id"] == "tc-abc123def456"
    assert dict(child.attributes)["gen_ai.operation.name"] == "invoke_agent"
    from opentelemetry.trace import StatusCode

    for span in (turn, tool, child):
        assert span.status.status_code is StatusCode.UNSET
    assert session.status.status_code is StatusCode.ERROR  # shutdown abandonment
    # Raw session id never appears anywhere in exported attributes.
    blob = str([dict(span.attributes) for span in spans])
    assert "sess-1" not in blob and "turn-1" not in blob


def test_turn_end_closes_leaked_tool_span():
    """Emitter drop-oldest can lose a tool end event; turn end must not leak it open (F2)."""
    plane, sink = _plane_and_sink()
    s, t = pseudonymize_id("sess-2"), pseudonymize_id("turn-2")
    plane([AgentSessionStartEvent(session_ref=s, origin="acp").to_dict()])
    plane([AgentTurnStartEvent(session_ref=s, turn_ref=t).to_dict()])
    plane(
        [
            AgentToolExecEvent(
                phase="start",
                session_ref=s,
                turn_ref=t,
                tool_call_id="tc-leak12345678",
                tool_name="shell",
            ).to_dict()
        ]
    )
    plane(
        [AgentTurnEndEvent(session_ref=s, turn_ref=t, status="interrupted").to_dict()]
    )
    plane.shutdown()

    _, by_name = _span_tree(sink)
    tool = by_name["agent.tool"][0]
    attrs = dict(tool.attributes)
    assert attrs["hermes.status"] == "interrupted"
    assert "hermes.abandoned" not in attrs
    turn = by_name["agent.turn"][0]
    turn_attrs = dict(turn.attributes)
    assert turn_attrs["hermes.status"] == "interrupted"
    # interrupted is not failed/abandoned: no error.type, status stays UNSET.
    assert "error.type" not in turn_attrs
    from opentelemetry.trace import StatusCode

    assert turn.status.status_code is StatusCode.UNSET


def test_ttl_sweep_abandons_open_spans():
    from agent.monitoring import agent_traces as mod

    plane, sink = _plane_and_sink()
    s, t = pseudonymize_id("sess-3"), pseudonymize_id("turn-3")
    plane([AgentSessionStartEvent(session_ref=s, origin="acp").to_dict()])
    plane([AgentTurnStartEvent(session_ref=s, turn_ref=t).to_dict()])
    plane._ttl_ns = 1  # everything is now expired
    plane._sweep()  # _sweep takes the plane lock itself; do not hold it here
    plane.shutdown()

    spans, by_name = _span_tree(sink)
    assert dict(by_name["agent.turn"][0].attributes)["hermes.abandon_reason"] == "ttl"
    assert dict(by_name["agent.session"][0].attributes)["hermes.abandoned"] is True
    assert len(spans) == 2


def test_w3c_parent_adoption():
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    provider = TracerProvider()
    external = provider.get_tracer("external").start_span("upstream")
    plane, sink = _plane_and_sink()
    s = pseudonymize_id("sess-4")
    plane.adopt_w3c_parent(
        s,
        NonRecordingSpan(
            SpanContext(
                trace_id=external.get_span_context().trace_id,
                span_id=external.get_span_context().span_id,
                is_remote=True,
                trace_flags=TraceFlags(1),
            )
        ),
    )
    plane([AgentSessionStartEvent(session_ref=s, origin="acp").to_dict()])
    plane.shutdown()

    _, by_name = _span_tree(sink)
    session = by_name["agent.session"][0]
    assert session.context.trace_id == external.get_span_context().trace_id
    assert (
        session.parent is not None
        and session.parent.span_id == external.get_span_context().span_id
    )


def test_non_agent_events_ignored_and_config_gates():
    plane, sink = _plane_and_sink()
    plane(
        [
            {"event": "gateway_health", "name": "tick"},
            {"event": "cron_execution", "status": "ok"},
        ]
    )
    assert sink.get_finished_spans() == ()  # F9: no cross-plane leakage
    plane.shutdown()

    assert agent_traces_enabled(DISABLED_CONFIG) is False
    assert (
        agent_traces_enabled({"monitoring": {"agent_traces": {"enabled": True}}})
        is False
    )  # no otlp gate

    from agent.monitoring import agent_traces as mod

    assert mod.start_agent_trace_plane(DISABLED_CONFIG) is None


def test_duplicate_session_start_is_idempotent():
    plane, sink = _plane_and_sink()
    s = pseudonymize_id("sess-5")
    plane([AgentSessionStartEvent(session_ref=s, origin="acp").to_dict()])
    plane([AgentSessionStartEvent(session_ref=s, origin="acp").to_dict()])
    plane.shutdown()
    assert len(sink.get_finished_spans()) == 1


def test_failed_turn_error_type_and_zero_cache_gate():
    """GenAI semconv: failed turn → error.type + ERROR; zero cache usage omitted."""
    plane, sink = _plane_and_sink()
    s, t = pseudonymize_id("sess-6"), pseudonymize_id("turn-6")
    plane(
        [
            AgentTurnStartEvent(
                session_ref=s, turn_ref=t, model_class="primary", model_id="requested-model"
            ).to_dict()
        ]
    )
    plane(
        [
            AgentTurnEndEvent(
                session_ref=s,
                turn_ref=t,
                status="failed",
                error_class="RateLimitError",
                input_tokens=7,
                output_tokens=0,
                cache_read_tokens=0,
            ).to_dict()
        ]
    )
    plane.shutdown()

    from opentelemetry.trace import StatusCode

    _, by_name = _span_tree(sink)
    turn = by_name["agent.turn"][0]
    attrs = dict(turn.attributes)
    assert attrs["hermes.status"] == "failed"
    assert attrs["hermes.error_class"] == "RateLimitError"
    assert attrs["error.type"] == "RateLimitError"
    assert turn.status.status_code is StatusCode.ERROR
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 0  # present-zero still emitted
    assert "gen_ai.usage.cache_read.input_tokens" not in attrs  # zero cache gated
    assert attrs["gen_ai.conversation.id"] == s
    assert "sess-6" not in str(attrs)  # pseudonymized conversation id only


@pytest.mark.parametrize("kind", ["agent.tool", "agent.child", "agent.turn"])
def test_terminal_timestamp_redaction_and_allowlist(kind):
    from agent.monitoring.redaction import redact_bounded
    from opentelemetry.trace import StatusCode

    plane, sink = _plane_and_sink()
    secret = "person@example.com " + "x" * 600
    try:
        span = plane._start_span(kind, None, {"error_class": secret}, 100)
        plane._close_span(span, {"error_class": secret, "prompt": secret}, 200)
        finished = sink.get_finished_spans()[0]
        assert finished.end_time == 200
        assert finished.attributes["hermes.error_class"] == redact_bounded(secret)
        assert finished.attributes["error.type"] == redact_bounded(secret)
        assert "hermes.prompt" not in finished.attributes
        assert finished.status.status_code is StatusCode.ERROR
    finally:
        plane.shutdown()


@pytest.mark.parametrize("event_kind", ["agent_tool_exec", "agent_child"])
def test_unknown_exact_turn_identity_is_rejected(event_kind):
    plane, sink = _plane_and_sink()
    try:
        plane.handle({"event": "agent_turn_start", "session_ref": "s", "turn_ref": "t"})
        if event_kind == "agent_tool_exec":
            event = dict(event=event_kind, session_ref="s", turn_ref="old", tool_call_id="tc", tool_name="shell")
        else:
            event = dict(event=event_kind, parent_session_ref="s", parent_turn_ref="old", child_session_ref="child")
        plane.handle({**event, "phase": "start"})
        plane.handle({**event, "phase": "end"})
        plane.handle({"event": "agent_turn_end", "session_ref": "s", "turn_ref": "old"})
        assert not plane._sessions["s"]["turns"]["t"]["tools"]
        assert not plane._sessions["s"]["turns"]["t"]["children"]
        assert not sink.get_finished_spans()
    finally:
        plane.shutdown()


def test_producer_end_timestamps_and_abandoned_error():
    from opentelemetry.trace import StatusCode

    plane, sink = _plane_and_sink()
    try:
        plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref="t", ts_ns=100))
        for call_id in ("ended", "lost"):
            plane.handle(dict(event="agent_tool_exec", phase="start", session_ref="s", turn_ref="t", tool_call_id=call_id, ts_ns=110))
        plane.handle(dict(event="agent_tool_exec", phase="end", session_ref="s", turn_ref="t", tool_call_id="ended", ts_ns=120))
        plane.handle(dict(event="agent_child", phase="start", parent_session_ref="s", parent_turn_ref="t", child_session_ref="child", ts_ns=110))
        plane.handle(dict(event="agent_child", phase="end", parent_session_ref="s", parent_turn_ref="t", child_session_ref="child", ts_ns=130))
        plane.handle(dict(event="agent_turn_end", session_ref="s", turn_ref="t", status="completed", ts_ns=140))
        _, by_name = _span_tree(sink)
        assert [span.end_time for span in by_name["agent.tool"]] == [120, 140]
        assert by_name["agent.tool"][1].status.status_code is StatusCode.ERROR
        assert by_name["agent.tool"][1].attributes["error.type"] == "abandoned"
        assert by_name["agent.child"][0].end_time == 130
        assert by_name["agent.turn"][0].end_time == 140
    finally:
        plane.shutdown()


def test_registry_budgets_evict_oldest_and_close_descendants(monkeypatch):
    from agent.monitoring import agent_traces as mod
    from opentelemetry.trace import StatusCode

    for key in ("_MAX_SESSIONS", "_MAX_TURNS_PER_SESSION", "_MAX_TOOLS_PER_TURN", "_MAX_CHILDREN_PER_TURN", "_MAX_PENDING_PARENTS"):
        monkeypatch.setattr(mod, key, 2, raising=False)
    plane, sink = _plane_and_sink()
    try:
        for i in range(3):
            plane.adopt_w3c_parent(str(i), object())
        assert list(plane._w3c_parent) == ["1", "2"]
        for i in range(3):
            plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref=str(i)))
        assert list(plane._sessions["s"]["turns"]) == ["1", "2"]
        for i in range(3):
            plane.handle(dict(event="agent_tool_exec", phase="start", session_ref="s", turn_ref="2", tool_call_id=str(i)))
            plane.handle(dict(event="agent_child", phase="start", parent_session_ref="s", parent_turn_ref="2", child_session_ref=str(i)))
        turn = plane._sessions["s"]["turns"]["2"]
        assert list(turn["tools"]) == ["1", "2"]
        assert list(turn["children"]) == ["1", "2"]
        for s in ("new1", "new2"):
            plane.handle(dict(event="agent_turn_start", session_ref=s, turn_ref="t"))
        assert list(plane._sessions) == ["new1", "new2"]
        spans = sink.get_finished_spans()
        assert len(spans) == 10  # three turns, three tools, three children, one session
        assert all(span.attributes["hermes.abandon_reason"] == "capacity" for span in spans)
        assert all(span.status.status_code is StatusCode.ERROR for span in spans)
    finally:
        plane.shutdown()


def test_ttl_uses_receipt_inactivity_and_expires_pending_parents(monkeypatch):
    from agent.monitoring import agent_traces as mod

    now = [0]
    monkeypatch.setattr(mod.time, "monotonic_ns", lambda: now[0])
    plane, sink = _plane_and_sink()
    plane._ttl_ns = 100
    try:
        plane.handle(dict(event="agent_session_start", session_ref="s", ts_ns=1))
        plane.adopt_w3c_parent("pending", object())
        now[0] = 80
        plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref="old", ts_ns=2))
        now[0] = 160
        plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref="active", ts_ns=3))
        now[0] = 200
        plane._sweep()
        assert "s" in plane._sessions
        assert list(plane._sessions["s"]["turns"]) == ["active"]
        assert not plane._w3c_parent
        assert len(sink.get_finished_spans()) == 1
        now[0] = 261
        plane._sweep()
        assert not plane._sessions
    finally:
        plane.shutdown()


def test_shutdown_owns_emitter_rejects_captured_batch_and_resets_singleton(monkeypatch):
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(mod, "_PLANE", None)
    emitter = MonitoringEmitter(enabled=False)
    sink = InMemorySpanExporter()
    plane = mod.start_agent_trace_plane(CONFIG, emitter=emitter, sink_factory=lambda: sink)
    try:
        plane.adopt_w3c_parent("pending", object())
        assert emitter.stats()["subscribers"] == 1
        plane.shutdown()
        plane.shutdown()
        assert emitter.stats()["subscribers"] == 0
        assert mod._PLANE is None
        assert not plane._w3c_parent
        assert not plane._sweeper.is_alive()
        plane([dict(event="agent_turn_start", session_ref="late", turn_ref="t")])
        plane.adopt_w3c_parent("late", object())
        assert not plane._sessions and not plane._w3c_parent
        replacement = mod.start_agent_trace_plane(CONFIG, emitter=emitter, sink_factory=InMemorySpanExporter)
        assert replacement is not plane
        replacement.shutdown()
    finally:
        plane.shutdown()
        emitter.close()


def test_failed_subscription_cleans_up_started_plane(monkeypatch):
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    class FailingEmitter(MonitoringEmitter):
        def subscribe(self, callback):
            self.plane = callback
            super().subscribe(callback)
            raise RuntimeError("subscription failed")

    monkeypatch.setattr(mod, "_PLANE", None)
    emitter = FailingEmitter(enabled=False)
    try:
        with pytest.raises(RuntimeError, match="subscription failed"):
            mod.start_agent_trace_plane(CONFIG, emitter=emitter, sink_factory=InMemorySpanExporter)
        assert emitter.plane._shutdown_done.wait(2)
        assert not emitter.plane._sweeper.is_alive()
        assert emitter.stats()["subscribers"] == 0
        assert mod._PLANE is None
    finally:
        emitter.plane.shutdown()
        emitter.close()


@pytest.mark.parametrize("started_before_error", [False, True])
def test_sweeper_start_failure_retires_provider_off_loop(monkeypatch, started_before_error):
    import asyncio
    import threading
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    owner = threading.get_ident()
    closed = []
    real_start = threading.Thread.start
    real_join = threading.Thread.join
    planes = []

    class Sink(InMemorySpanExporter):
        def shutdown(self):
            closed.append(threading.get_ident())
            super().shutdown()

    def start(thread):
        if thread.name == "agent-trace-ttl":
            planes.append(mod._PLANE)
            if started_before_error:
                real_start(thread)
            raise RuntimeError("sweeper start failed")
        return real_start(thread)

    def join(thread, *args, **kwargs):
        assert thread.ident is not None, "never join an allocated, unstarted sweeper"
        return real_join(thread, *args, **kwargs)

    monkeypatch.setattr(mod, "_PLANE", None)
    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(threading.Thread, "join", join)
    bus = MonitoringEmitter(enabled=False)

    async def scenario():
        with pytest.raises(RuntimeError, match="sweeper start failed"):
            mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=Sink)
        await planes[0].shutdown_async()

    try:
        asyncio.run(scenario())
        plane = planes[0]
        assert len(closed) == 1 and closed[0] != owner
        assert plane._shutdown_done.is_set()
        assert not plane._sweeper.is_alive()
        assert mod._PLANE is None
    finally:
        bus.close()


@pytest.mark.parametrize("failure", ["span", "join", "flush"])
def test_cleanup_errors_attempt_later_steps_and_keep_replacement_fenced(monkeypatch, failure):
    import threading
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(mod, "_PLANE", None)
    bus = MonitoringEmitter(enabled=False)
    plane = mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=InMemorySpanExporter)
    attempted = []
    finished = threading.Event()
    real_shutdown = plane._provider.shutdown
    real_flush = plane._provider.force_flush
    real_join = plane._sweeper.join
    real_worker = plane._shutdown_worker

    class Span:
        name = "agent.tool"

        def __init__(self, label):
            self.label = label

        def set_attribute(self, *args):
            pass

        def set_status(self, *args):
            pass

        def end(self, **kwargs):
            attempted.append(self.label)
            if failure == "span" and self.label == "bad":
                raise RuntimeError("span end failed")

    plane._sessions["s"] = {
        "session_span": Span("session"),
        "turns": {"t": {"tools": {"a": Span("bad"), "b": Span("tool")},
                         "children": {"c": Span("child")}, "span": Span("turn"), "ended": False}},
    }

    def join():
        attempted.append("join")
        real_join()
        if failure == "join":
            raise RuntimeError("join failed after thread exited")

    def flush(**kwargs):
        attempted.append("flush")
        if failure == "flush":
            raise RuntimeError("flush failed")
        return real_flush(**kwargs)

    def shutdown():
        attempted.append("shutdown")
        real_shutdown()

    def worker():
        try:
            real_worker()
        finally:
            finished.set()

    monkeypatch.setattr(plane._sweeper, "join", join)
    monkeypatch.setattr(plane._provider, "force_flush", flush)
    monkeypatch.setattr(plane._provider, "shutdown", shutdown)
    monkeypatch.setattr(plane, "_shutdown_worker", worker)
    try:
        plane._begin_shutdown()
        assert finished.wait(2)
        assert attempted == ["bad", "tool", "child", "turn", "session", "join", "flush", "shutdown"]
        assert not plane._sweeper.is_alive()
        assert bus.stats()["subscribers"] == 0
        assert mod._PLANE is plane
        assert not plane._shutdown_done.is_set()
        assert mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=InMemorySpanExporter) is None
    finally:
        bus.close()


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize("stage", ["exporter", "processor", "attach", "attached", "tracer"])
@pytest.mark.parametrize("cleanup_error", [None, "provider", "processor", "exporter"])
def test_partial_sdk_construction_is_owned_until_off_loop_cleanup(monkeypatch, injected, stage, cleanup_error):
    import asyncio
    import threading
    from agent.monitoring import agent_traces as mod
    from agent.monitoring import otlp_exporter as oe
    from agent.monitoring.emitter import MonitoringEmitter

    owner = threading.get_ident()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    made, closed, cleanup_threads = [], [], []
    real_worker = mod.AgentTracePlane._shutdown_worker

    class Exporter:
        def __init__(self):
            if stage == "exporter":
                raise RuntimeError("construction failed")
            made.append("exporter")

        def shutdown(self):
            closed.append("exporter")
            cleanup_threads.append(threading.get_ident())
            if cleanup_error == "exporter":
                raise RuntimeError("exporter shutdown failed")

    class Processor:
        def __init__(self, exporter):
            if stage == "processor":
                raise RuntimeError("construction failed")
            self.exporter = exporter
            self.stopped = False
            made.append("processor")

        def shutdown(self):
            # Like the SDK batch processor, shutdown is idempotent even if
            # add_span_processor attached us before reporting failure.
            if not self.stopped:
                self.stopped = True
                closed.append("processor")
                cleanup_threads.append(threading.get_ident())
                if cleanup_error == "processor":
                    raise RuntimeError("processor shutdown failed before exporter")
                self.exporter.shutdown()

    class Provider:
        def __init__(self, **kwargs):
            self.processors = []
            made.append("provider")

        def add_span_processor(self, processor):
            if stage in ("attached", "tracer"):
                self.processors.append(processor)
            if stage != "tracer":
                raise RuntimeError("construction failed")

        def get_tracer(self, *args):
            raise RuntimeError("construction failed")

        def force_flush(self, **kwargs):
            pass

        def shutdown(self):
            closed.append("provider")
            cleanup_threads.append(threading.get_ident())
            entered.set()
            assert release.wait(10)
            if cleanup_error == "provider":
                raise RuntimeError("provider shutdown failed before children")
            for processor in self.processors:
                processor.shutdown()

    def worker(plane):
        try:
            real_worker(plane)
        finally:
            finished.set()

    from types import SimpleNamespace
    sdk = {"TracerProvider": Provider, "BatchSpanProcessor": Processor,
           "SimpleSpanProcessor": Processor, "Resource": SimpleNamespace(create=lambda attrs: attrs)}
    monkeypatch.setattr(mod, "_PLANE", None)
    monkeypatch.setattr(mod, "_SHUTDOWN_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(mod.AgentTracePlane, "_shutdown_worker", worker)
    monkeypatch.setattr(oe, "_require_sdk", lambda *args, **kwargs: sdk)
    monkeypatch.setattr(oe, "_resource_attributes", lambda cfg: {})
    monkeypatch.setattr(oe, "build_exporter", lambda cfg: Exporter())
    bus = MonitoringEmitter(enabled=False)

    async def scenario():
        with pytest.raises(RuntimeError, match="construction failed"):
            mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=Exporter if injected else None)
        plane = mod._PLANE
        assert plane is not None
        from tests.waiting import async_wait_until
        await async_wait_until(entered.is_set, timeout=2, what="partial provider cleanup entered")
        await plane.shutdown_async()
        assert entered.is_set(), "partial provider must be reachable by cleanup"
        assert not finished.is_set()
        assert mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=Exporter) is None
        release.set()
        return plane

    try:
        plane = asyncio.run(scenario())
        assert finished.wait(2)
        assert set(closed) == set(made)
        assert cleanup_threads and all(t != owner for t in cleanup_threads)
        if cleanup_error in made:
            assert mod._PLANE is plane
            assert not plane._shutdown_done.is_set()
            assert mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=Exporter) is None
        else:
            assert plane._shutdown_done.is_set()
            assert mod._PLANE is None
    finally:
        release.set()
        assert finished.wait(2)
        bus.close()


def test_model_id_and_occupancy_are_not_fabricated_usage():
    plane, sink = _plane_and_sink()
    try:
        plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref="t", model_class="primary"))
        plane.handle(dict(event="agent_turn_end", session_ref="s", turn_ref="t", status="completed", model_id="served-model", model_class="fallback", prompt_occupancy_tokens=42))
        plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref="empty", model_class="unknown", model_id="unknown"))
        plane.handle(dict(event="agent_turn_end", session_ref="s", turn_ref="empty", status="completed", prompt_occupancy_tokens=0))
        _, by_name = _span_tree(sink)
        first, empty = by_name["agent.turn"]
        assert first.attributes["hermes.model_class"] == "fallback"
        assert first.attributes["gen_ai.request.model"] == "served-model"
        assert first.attributes["hermes.prompt_occupancy_tokens"] == 42
        assert not any(key.startswith("gen_ai.usage.") for key in first.attributes)
        assert "gen_ai.request.model" not in empty.attributes
        assert "hermes.model_class" not in empty.attributes
        assert "hermes.prompt_occupancy_tokens" not in empty.attributes
    finally:
        plane.shutdown()


@pytest.mark.parametrize("outcome", [False, "exception", "cancelled"])
@pytest.mark.parametrize("running", [False, True])
def test_gateway_failed_start_releases_trace_handle(outcome, running):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from gateway.run_startup import GatewayStartupMixin

    plane = Mock(shutdown_async=AsyncMock())

    async def start_impl():
        if outcome == "exception":
            raise RuntimeError("startup failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return outcome

    async def flush():
        pass

    runner = SimpleNamespace(_agent_trace_plane=plane, _running=running,
                             _start_impl=start_impl, _start_flush_runtime_status=flush)
    try:
        asyncio.run(GatewayStartupMixin.start(runner))
    except (RuntimeError, asyncio.CancelledError):
        pass
    plane.shutdown_async.assert_awaited_once_with()
    assert runner._agent_trace_plane is None


@pytest.mark.parametrize("failed_start", [False, True])
def test_gateway_terminal_cleanup_releases_trace_handle(monkeypatch, tmp_path, failed_start):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    import sys
    from gateway.run_shutdown import GatewayShutdownMixin
    from gateway import status

    async def flush(**kwargs):
        return True

    # Only these facade seams are used by this phase; no full gateway boot needed.
    run = SimpleNamespace(_hermes_home=tmp_path, _planned_restart_notification_path=Mock(),
                          _shutdown_gateway_health_export=Mock())
    monkeypatch.setitem(sys.modules, "gateway.run", run)
    monkeypatch.setattr(status, "flush_runtime_status_async", flush)
    plane = Mock(shutdown_async=AsyncMock())
    runner = SimpleNamespace(_agent_trace_plane=plane, _startup_cleanup_only=failed_start,
                             _restart_requested=False, _signal_initiated_shutdown=False,
                             _exit_reason=None, _update_runtime_status=Mock())
    ctx = SimpleNamespace(timed_out=False, active_agents={}, elapsed=lambda: 0)
    asyncio.run(GatewayShutdownMixin._stop_persist_exit_state(runner, ctx))
    plane.shutdown_async.assert_awaited_once_with()
    assert runner._agent_trace_plane is None
    run._shutdown_gateway_health_export.assert_called_once_with(runner)


def test_real_emitter_dispatch_preserves_terminal_timestamp(monkeypatch):
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(mod, "_PLANE", None)
    emitter = MonitoringEmitter(enabled=False)
    sink = InMemorySpanExporter()
    plane = mod.start_agent_trace_plane(CONFIG, emitter=emitter, sink_factory=lambda: sink)
    try:
        emitter.emit(AgentTurnStartEvent(session_ref="s", turn_ref="t", ts_ns=100))
        emitter.emit(AgentTurnEndEvent(session_ref="s", turn_ref="t", status="failed", ts_ns=200))
        emitter.flush()
        turn = sink.get_finished_spans()[0]
        assert turn.name == "agent.turn"
        assert turn.start_time == 100 and turn.end_time == 200
        assert turn.attributes["error.type"] == "failed"
    finally:
        plane.shutdown()
        emitter.close()


@pytest.mark.parametrize("value", [False, -1, "42", 0, None])
def test_unavailable_occupancy_is_omitted(value):
    plane, sink = _plane_and_sink()
    try:
        plane.handle(dict(event="agent_turn_start", session_ref="s", turn_ref="t"))
        plane.handle(dict(event="agent_turn_end", session_ref="s", turn_ref="t", prompt_occupancy_tokens=value))
        assert "hermes.prompt_occupancy_tokens" not in sink.get_finished_spans()[0].attributes
    finally:
        plane.shutdown()


@pytest.mark.parametrize("cancel_owner", [False, True])
@pytest.mark.parametrize("stall", ["handler", "exporter"])
def test_async_shutdown_keeps_loop_live_and_fences_until_late_cleanup(monkeypatch, stall, cancel_owner):
    import asyncio
    import threading
    from types import SimpleNamespace
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from gateway.agent_trace_emit import _shutdown_agent_trace_plane
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    entered, release = threading.Event(), threading.Event()

    class StalledExporter(InMemorySpanExporter):
        def shutdown(self):
            entered.set()
            assert release.wait(10)
            super().shutdown()

    monkeypatch.setattr(mod, "_PLANE", None)
    monkeypatch.setattr(mod, "_SHUTDOWN_TIMEOUT_SECONDS", 5.0 if cancel_owner else 0.1)
    bus = MonitoringEmitter(enabled=False)
    sink = StalledExporter() if stall == "exporter" else InMemorySpanExporter()
    plane = mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=lambda: sink)
    if stall == "handler":
        original = plane._on_turn_start

        def stalled(event):
            entered.set()
            assert release.wait(10)
            original(event)

        monkeypatch.setattr(plane, "_on_turn_start", stalled)
        bus.emit(AgentTurnStartEvent(session_ref="s", turn_ref="t"))
        assert entered.wait(2)

    async def check():
        runner = SimpleNamespace(_agent_trace_plane=plane)
        progress = asyncio.Event()
        # Scheduling shutdown first makes an inline wait starve the next loop callback.
        task = asyncio.create_task(_shutdown_agent_trace_plane(runner))
        asyncio.get_running_loop().call_soon(progress.set)
        await asyncio.wait_for(progress.wait(), 2)
        if cancel_owner:
            from tests.waiting import async_wait_until
            await async_wait_until(entered.is_set, timeout=2, what="cleanup entered stall")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, 2)
        from tests.waiting import async_wait_until
        await async_wait_until(entered.is_set, timeout=2, what="cleanup entered stall")
        assert not release.is_set()
        assert not plane._shutdown_done.is_set()
        assert mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=InMemorySpanExporter) is None
        release.set()
        await asyncio.wait_for(asyncio.to_thread(plane._shutdown_done.wait, 2), 3)
        assert plane._shutdown_done.is_set() and mod._PLANE is None
        assert not plane._sweeper.is_alive() and not plane._sessions
        replacement = mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=InMemorySpanExporter)
        assert replacement is not None and replacement is not plane
        await replacement.shutdown_async()

    try:
        asyncio.run(check())
    finally:
        release.set()
        plane.shutdown()
        bus.close()


@pytest.mark.parametrize("shared_health", [False, True])
def test_actual_stop_drains_queued_terminal_before_detach_and_restart(monkeypatch, tmp_path, shared_health):
    import asyncio
    import sys
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agent.monitoring import agent_traces as mod
    from agent.monitoring.emitter import MonitoringEmitter
    from gateway.run_shutdown import GatewayShutdownMixin
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(mod, "_PLANE", None)
    bus = MonitoringEmitter(enabled=False)
    sink = InMemorySpanExporter()
    plane = mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=lambda: sink)
    # Hold the dispatcher after a genuinely consumed start. The terminal stays queued
    # until the real stop owner begins cleanup; there is no test-side pre-flush.
    entered, release = threading.Event(), threading.Event()
    original = plane._on_turn_start

    def stalled(event):
        original(event)
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(plane, "_on_turn_start", stalled)
    health = []
    if shared_health:
        bus.subscribe(lambda batch: health.extend(batch))
    bus.emit(AgentTurnStartEvent(session_ref="s", turn_ref="t", ts_ns=100))
    assert entered.wait(2)
    phases = []

    async def phase(name, *args):
        if name == "begin":
            import time
            args[-1].started_at = time.monotonic()
        phases.append(name)

    async def drain(*args):
        phases.append("drain")
        for phase, ts_ns in [("start", 110), ("end", 180)]:
            bus.emit(AgentToolExecEvent(
                phase=phase, session_ref="s", turn_ref="t",
                tool_call_id="tc-123456789abc", tool_name="terminal",
                status="completed" if phase == "end" else None, ts_ns=ts_ns,
            ))
        bus.emit(AgentTurnEndEvent(session_ref="s", turn_ref="t", status="completed", ts_ns=200))

    original_flush = bus.flush

    def release_during_production_drain(timeout=2.0):
        release.set()
        return original_flush(timeout=timeout)

    monkeypatch.setattr(bus, "flush", release_during_production_drain)

    async def persist(runner, ctx):
        phases.append("persist")
        await GatewayShutdownMixin._stop_persist_exit_state(runner, ctx)

    async def settle(*args):
        phases.append("settle")

    from gateway import run_runtime, status

    async def flush_status(**kwargs):
        return True

    monkeypatch.setattr(status, "flush_runtime_status_async", flush_status)
    monkeypatch.setattr(run_runtime, "settle_gateway_runtime", settle)
    facade = SimpleNamespace(
        _hermes_home=tmp_path, _planned_restart_notification_path=Mock(),
        _shutdown_gateway_health_export=Mock(),
        GatewayRunner=SimpleNamespace(
            _stop_begin_teardown=lambda *a: phase("begin", *a),
            _stop_drain_active_work=drain,
            _stop_finalize_agents_and_adapters=lambda *a: phase("finalize", *a),
            _stop_release_runtime_state=lambda *a: phase("release", *a),
            _stop_quiesce_and_close_session_dbs=lambda *a: phases.append("quiesce"),
            _stop_persist_exit_state=persist,
            _shutdown_interrupt_reason=lambda *a: "shutdown",
        ),
    )
    monkeypatch.setitem(sys.modules, "gateway.run", facade)
    runner = SimpleNamespace(
        _agent_trace_plane=plane, _restart_drain_timeout=30,
        _restart_requested=False, _signal_initiated_shutdown=False,
        _exit_reason=None, _update_runtime_status=Mock(),
    )
    try:
        asyncio.run(GatewayShutdownMixin._stop_impl(runner))
        assert phases == ["begin", "drain", "settle", "finalize", "release", "quiesce", "persist"]
        turn = next(span for span in sink.get_finished_spans() if span.name == "agent.turn")
        assert turn.end_time == 200 and turn.attributes["hermes.status"] == "completed"
        assert "hermes.abandoned" not in turn.attributes
        tool = next(span for span in sink.get_finished_spans() if span.name == "agent.tool")
        assert tool.end_time == 180 and tool.attributes["hermes.status"] == "completed"
        assert "hermes.abandoned" not in tool.attributes
        facade._shutdown_gateway_health_export.assert_called_once_with(runner)
        assert bus.stats()["subscribers"] == int(shared_health)
        bus.emit({"event": "gateway_health", "name": "after_trace_stop"})
        bus.flush()
        if shared_health:
            assert health[-1]["name"] == "after_trace_stop"
        next_sink = InMemorySpanExporter()
        replacement = mod.start_agent_trace_plane(CONFIG, emitter=bus, sink_factory=lambda: next_sink)
        assert replacement is not plane
        bus.emit(AgentTurnStartEvent(session_ref="new", turn_ref="new", ts_ns=300))
        bus.emit(AgentTurnEndEvent(session_ref="new", turn_ref="new", status="completed", ts_ns=400))
        replacement.shutdown()
        turns = [span for span in next_sink.get_finished_spans() if span.name == "agent.turn"]
        assert len(turns) == 1 and turns[0].start_time == 300
        assert turns[0].attributes["hermes.status"] == "completed"
    finally:
        release.set()
        plane.shutdown()
        bus.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
