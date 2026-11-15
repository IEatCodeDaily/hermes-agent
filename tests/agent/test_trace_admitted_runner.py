"""Trace production at the common durable-admitted ACP/gateway runner, without model I/O."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.turn_facade import TurnFacadeMixin
from agent.monitoring.events import AgentToolExecEvent, AgentTurnStartEvent, AgentTurnEndEvent, pseudonymize_id
from gateway import agent_trace_emit as tracing


class Agent(TurnFacadeMixin):
    session_id = "admitted-session"
    platform = "acp"
    provider = "custom"
    model = "request-model"

    def _conversation_root_id(self):
        return self.session_id

    def _reset_activity_labels_after_turn(self):
        pass


@pytest.mark.parametrize("terminal", ["completed", "interrupted", "cancelled", "failed", "refused"])
def test_common_runner_emits_only_admitted_execution(monkeypatch, terminal):
    from agent import conversation_loop, relay_runtime, turn_facade_lease
    from hermes_cli.observability import relay_shared_metrics  # register against the real coordinator first
    from agent.turn_context import _bind_turn_identity
    monkeypatch.setattr(relay_shared_metrics, "start_task_run", lambda **k: None)
    monkeypatch.setattr(relay_shared_metrics, "finish_task_run", lambda **k: None)

    events, stages, admitted_turns, relay_sessions, admitted_contexts = [], [], [], [], []
    monkeypatch.setattr(tracing, "_emit", events.append)
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", lambda *a: ("", ""))
    monkeypatch.setattr("agent.prompt_cache_scope.declared_conversation_scope_safe", lambda a: "scope")
    relay = SimpleNamespace(relay_enabled=False)
    monkeypatch.setattr(relay_runtime, "SESSION_COORDINATOR", SimpleNamespace(
        acquire_conversation=lambda **k: relay_sessions.append(k["session_id"]) or object(),
        begin_turn=lambda *a, **k: relay,
        finish_logical_calls=lambda *a, **k: None, end_turn=lambda *a, **k: None,
        release_conversation=lambda *a: None,
    ))
    lease = SimpleNamespace(start=lambda: stages.append("lease-start"), stop_refresher=lambda: None,
                            join_threads=lambda: None, clear_interrupt=lambda: None, release=lambda: None)
    def admit(*args, **kwargs):
        assert events == []
        stages.append("admitted")
        admitted_turns.append(kwargs["relay_turn_id"])
        admitted_contexts.append(kwargs["task_context"])
        # Contended durable admission resolves the resume tip before it returns.
        args[0].session_id = kwargs["task_context"]["session_id"] = "resolved-session"
        return SimpleNamespace(lease=lease, conversation_history=[],
                               early_result={"interrupted": True, "messages": []} if terminal == "refused" else None)
    monkeypatch.setattr(turn_facade_lease, "admit_durable_turn_lease", admit)
    def execute(agent, *a, **k):
        assert stages == ["admitted", "lease-start"]
        start = next(e for e in events if isinstance(e, AgentTurnStartEvent))
        assert relay_sessions == ["resolved-session"]
        assert start.session_ref == pseudonymize_id("resolved-session")
        assert start.turn_ref == pseudonymize_id(admitted_turns[0])
        # Recovery/rotation and mutable pending fields cannot change the captured identity.
        agent.session_id = admitted_contexts[0]["session_id"] = "rotated"
        agent._relay_pending_turn_id = "wrong-pending-turn"
        _, turn = _bind_turn_identity(agent, "task", None, None, None, None)
        assert turn == admitted_turns[0]
        assert tracing.execution_identity(agent) == ("resolved-session", turn)
        with tracing.trace_tool("admitted-call", "web_search") as tool:
            agent.session_id = "rotated-again"
            agent._current_turn_id = "wrong-current-turn"
            tool.result = {"ok": True}
        if terminal == "cancelled":
            raise asyncio.CancelledError()
        if terminal == "failed":
            raise RuntimeError("private-message")
        return {terminal: True, "served_model_id": "served-model", "provider": "custom"}
    monkeypatch.setattr(conversation_loop, "run_conversation", execute)
    agent = Agent()
    if terminal in {"cancelled", "failed"}:
        with pytest.raises(asyncio.CancelledError if terminal == "cancelled" else RuntimeError):
            agent.run_conversation("prompt", task_id="task")
    else:
        agent.run_conversation("prompt", task_id="task")
    ends = [e for e in events if isinstance(e, AgentTurnEndEvent)]
    if terminal == "refused":
        assert not events
    else:
        assert len(ends) == 1
        assert ends[0].status == terminal
        assert ends[0].session_ref == pseudonymize_id("resolved-session")
        assert ends[0].turn_ref == pseudonymize_id(admitted_turns[0])
        tools = [e for e in events if isinstance(e, AgentToolExecEvent)]
        assert [e.phase for e in tools] == ["start", "end"]
        assert all((e.session_ref, e.turn_ref) == (ends[0].session_ref, ends[0].turn_ref) for e in tools)
        assert tracing.execution_identity(agent) is None
        assert "private-message" not in repr(events)
