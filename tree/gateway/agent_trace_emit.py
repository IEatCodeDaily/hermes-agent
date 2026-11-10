"""agent-trace emission helpers: fire-and-forget, never raise into gateway code.

All producers route through here so the monitoring plane stays optional:
if the emitter or the agent_traces module is unavailable, these no-op.
Correlation ids are pseudonymized HERE (design F1) — callers pass raw ids.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def _shutdown_agent_trace_plane(runner) -> None:
    from gateway.run_shutdown import _log_suppressed

    plane = getattr(runner, "_agent_trace_plane", None)
    runner._agent_trace_plane = None
    if plane is not None:
        with _log_suppressed(logging.DEBUG, "agent-trace OTLP plane shutdown failed", exc_info=True):
            await plane.shutdown_async()


def _emit(event: Any) -> None:
    try:
        from agent.monitoring import emit

        emit(event)
    except Exception:
        logger.debug("agent-trace emit dropped", exc_info=True)


def emit_session_start(session_key: str, origin: str) -> None:
    try:
        from agent.monitoring.events import AgentSessionStartEvent, pseudonymize_id

        _emit(
            AgentSessionStartEvent(
                session_ref=pseudonymize_id(session_key), origin=origin
            )
        )
    except Exception:
        logger.debug("agent-trace session_start dropped", exc_info=True)


def _known_label(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown" else None


def emit_turn_start(
    session_key: str, turn_id: str, model_class: str = "", *, model_id: Optional[str] = None,
) -> None:
    try:
        from agent.monitoring.events import AgentTurnStartEvent, pseudonymize_id

        _emit(AgentTurnStartEvent(
            session_ref=pseudonymize_id(session_key), turn_ref=pseudonymize_id(turn_id),
            model_class=_known_label(model_class) or "", model_id=_known_label(model_id),
        ))
    except Exception:
        logger.debug("agent-trace turn_start dropped", exc_info=True)


def emit_turn_end(
    session_key: str, turn_id: str, status: str, *, duration_ms: Optional[int] = None,
    model_class: str = "", model_id: Optional[str] = None,
    last_prompt_tokens: Optional[int] = None, error_class: Optional[str] = None,
    ts_ns: Optional[int] = None,
) -> None:
    try:
        from agent.monitoring.events import AgentTurnEndEvent, pseudonymize_id

        # Finalizer uses zero as an unavailable-reading sentinel, not a measured empty prompt.
        occupancy = last_prompt_tokens if type(last_prompt_tokens) is int and last_prompt_tokens > 0 else None
        _emit(AgentTurnEndEvent(
            session_ref=pseudonymize_id(session_key), turn_ref=pseudonymize_id(turn_id),
            status=status, duration_ms=duration_ms, model_class=_known_label(model_class) or "",
            model_id=_known_label(model_id), prompt_occupancy_tokens=occupancy,
            error_class=error_class, ts_ns=ts_ns if ts_ns is not None else time.time_ns(),
        ))
    except Exception:
        logger.debug("agent-trace turn_end dropped", exc_info=True)


_EXECUTION: ContextVar[Optional[_ExecutionTrace]] = ContextVar("agent_trace_execution", default=None)
_RUN_CURRENT: ContextVar[Any] = ContextVar("agent_trace_run_current", default=None)


@contextmanager
def bind_execution_owner(is_current):
    """Carry the gateway's captured generation fence into its executor context."""
    token = _RUN_CURRENT.set(is_current)
    try:
        yield
    finally:
        _RUN_CURRENT.reset(token)


def execution_identity(agent: Any) -> Optional[tuple[str, str]]:
    """Only the admitted execution's immutable identity, never a mutable agent fallback."""
    execution = _EXECUTION.get()
    return execution.identity if execution is not None and execution.agent is agent else None


def _exception_status(error: BaseException) -> str:
    return "cancelled" if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) else (
        "interrupted" if isinstance(error, InterruptedError) else "failed"
    )


class _ToolTrace:
    def __init__(self, call_id: str, name: str) -> None:
        self.call_id, self.name = call_id, name
        self.started = time.monotonic_ns()
        self.result: Any = None


class _ExecutionTrace:
    """One owner plus inherited worker contexts; the lock orders terminal races exactly once."""

    def __init__(self, agent: Any, session_id: str, turn_id: str) -> None:
        self.agent = agent
        self.identity = (session_id, turn_id)
        self.started = time.monotonic_ns()
        self.is_current = _RUN_CURRENT.get()
        self.result: Any = None
        self.lock = threading.Lock()
        self.closed = False
        self.tools: dict[int, _ToolTrace] = {}

    def _tool_event(self, tool: _ToolTrace, phase: str, *, ts_ns: int,
                    status: Optional[str] = None, error_class: Optional[str] = None) -> None:
        from agent.monitoring.events import AgentToolExecEvent, pseudonymize_id

        _emit(AgentToolExecEvent(
            phase=phase, session_ref=pseudonymize_id(self.identity[0]),
            turn_ref=pseudonymize_id(self.identity[1]), tool_call_id=tool.call_id,
            tool_name=tool.name, ts_ns=ts_ns, status=status, error_class=error_class,
            duration_ms=(time.monotonic_ns() - tool.started) // 1_000_000 if phase == "end" else None,
        ))

    def start_tool(self, tool: _ToolTrace) -> None:
        with self.lock:
            if not self.closed:
                self.tools[id(tool)] = tool
                self._tool_event(tool, "start", ts_ns=time.time_ns())

    def end_tool(self, tool: _ToolTrace, status: str, error_class: Optional[str]) -> None:
        with self.lock:
            if self.tools.pop(id(tool), None) is not None:
                self._tool_event(tool, "end", ts_ns=time.time_ns(), status=status, error_class=error_class)

    def finish(self, error: Optional[BaseException]) -> None:
        result = self.result if isinstance(self.result, dict) else {}
        status = _exception_status(error) if error is not None else next(
            (key for key in ("superseded", "interrupted", "cancelled", "failed") if result.get(key)),
            "failed" if result.get("error") else "completed",
        )
        if self.is_current is not None and not self.is_current():
            status = "superseded"
        with self.lock:
            if self.closed:
                return
            self.closed = True
            terminal_ns = time.time_ns()
            for tool in self.tools.values():
                self._tool_event(tool, "end", ts_ns=terminal_ns,
                                 status="abandoned" if status == "completed" else status)
            self.tools.clear()
            emit_turn_end(
                *self.identity, status, ts_ns=terminal_ns,
                duration_ms=(time.monotonic_ns() - self.started) // 1_000_000,
                model_class=result.get("provider") or "",
                # Observe only the final active model, not an exact served deployment.
                # UI labels and retained header IDs may belong to an earlier route/turn.
                model_id=_known_label(result.get("model")),
                last_prompt_tokens=result.get("last_prompt_tokens"),
                error_class=type(error).__name__ if error is not None else ("turn_failed" if status == "failed" else None),
            )


@contextmanager
def trace_execution(agent: Any, session_id: str, turn_id: str):
    """Admitted synchronous execution only; neither admission wait nor delivery is inside."""
    trace = _ExecutionTrace(agent, session_id, turn_id)
    token = _EXECUTION.set(trace)
    error = None
    try:
        emit_session_start(session_id, str(getattr(agent, "platform", "") or "unknown"))
        emit_turn_start(session_id, turn_id, getattr(agent, "provider", "") or "",
                        model_id=getattr(agent, "model", None))
        yield trace
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            with suppress(Exception):  # Never mask the original execution exception.
                trace.finish(error)
        finally:
            # Copied worker/callback contexts can outlive a turn: retain IDs, not agent/history.
            trace.agent = trace.result = trace.is_current = None
            _EXECUTION.reset(token)


@contextmanager
def trace_tool(call_id: str, tool_name: str):
    """Wrap only real dispatch; inherited context keeps late workers on their original owner."""
    trace = _EXECUTION.get()
    tool = _ToolTrace(call_id, tool_name)
    status, error_class = "completed", None
    if trace is not None:
        with suppress(Exception):  # Optional telemetry cannot prevent real dispatch.
            trace.start_tool(tool)
    try:
        yield tool
        if trace is not None:
            with suppress(Exception):
                from agent.display import _detect_tool_failure

                data = tool.result
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (ValueError, TypeError):
                        pass
                explicit = data.get("status") if isinstance(data, dict) else None
                if explicit in {"cancelled", "interrupted", "superseded"}:
                    status = explicit
                elif (isinstance(data, dict) and (data.get("error") or data.get("success") is False)
                      or _detect_tool_failure(tool_name, tool.result)[0]):
                    status, error_class = "failed", "tool_result_error"
    except BaseException as exc:
        status, error_class = _exception_status(exc), type(exc).__name__
        raise
    finally:
        if trace is not None:
            with suppress(Exception):
                trace.end_tool(tool, status, error_class)
