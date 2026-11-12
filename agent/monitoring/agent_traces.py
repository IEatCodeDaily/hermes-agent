"""Agent-trace plane: turn-lifecycle spans over the existing monitoring bus.

Subscribes to ``agent_*`` monitoring events with producer-pseudonymized ids.
This slice wires session, turn and executed-tool spans; child handling and
validated-parent registration are helper capabilities only. ACP W3C ingress
and child lifecycle producers are not wired. A TTL sweep closes abandoned
spans when queue drops lose terminals. Exports use the same
``monitoring.export.otlp`` config as gateway health.

Config (both gates required, OFF by default):
    monitoring.agent_traces.enabled: true
    monitoring.agent_traces.ttl_seconds: 3600
    monitoring.export.otlp.{enabled,endpoint,headers_env}

Charter (F4): usage attrs are emitted only when supplied as measured per-turn
accounting; prompt occupancy is separate, never synthesized as input usage. No prompts, completions, tool args
or secrets enter this plane: attributes are a fixed allowlist per kind and
string values pass ``redact_bounded``. GenAI semconv attributes (``gen_ai.*``,
Development stability) are dual-emitted additively from ``_GENAI_MAP`` —
content stays OFF.

Tests inject ``sink_factory`` returning an ``InMemorySpanExporter`` (F3);
the OTLP exporter is only constructed when no sink is injected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional

from . import otlp_exporter

logger = logging.getLogger(__name__)

_AGENT_EVENT_KINDS = frozenset(
    {
        "agent_session_start",
        "agent_turn_start",
        "agent_turn_end",
        "agent_tool_exec",
        "agent_child",
    }
)
_DEFAULT_TTL_SECONDS = 3600
_SWEEP_INTERVAL_SECONDS = 60
_SHUTDOWN_TIMEOUT_SECONDS = 2.0
# Fixed budgets bound memory even when arrival volume exceeds the TTL window.
_MAX_SESSIONS = 128
_MAX_TURNS_PER_SESSION = 8
_MAX_TOOLS_PER_TURN = 32
_MAX_CHILDREN_PER_TURN = 16
_MAX_PENDING_PARENTS = 128

# Fixed attribute allowlist per span kind; everything else never egresses.
_ATTR_ALLOWLIST: dict[str, Iterable[str]] = {
    "agent.session": ("origin", "abandoned", "abandon_reason"),
    "agent.turn": (
        "model_class",
        "model_id",
        "prompt_occupancy_tokens",
        "status",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "error_class",
        "abandoned",
        "abandon_reason",
    ),
    "agent.tool": (
        "tool_call_id",
        "tool_name",
        "duration_ms",
        "error_class",
        "status",
        "abandoned",
        "abandon_reason",
    ),
    "agent.child": ("child_session_ref", "status", "error_class", "abandoned", "abandon_reason"),
}

# GenAI semantic-convention dual-emit table (single place this mapping lives).
# GenAI semconv is Development-stability as of 2026-11-08: these attributes are
# ADDITIVE alongside the stable hermes.* allowlist above (span renames are a
# later, separate cut). Entries are either "<hermes event key>": "<gen_ai
# attribute>" (same value dual-emitted; cache usage attrs only when non-zero)
# or "<span kind>": {fixed gen_ai attributes for that kind}.
# gen_ai.conversation.id is set in _start_span from the pseudonymized
# session_ref (never allowlisted into hermes.*). Content stays OFF.
_GENAI_MAP: dict[str, Any] = {
    "model_id": "gen_ai.request.model",
    "tool_name": "gen_ai.tool.name",
    "tool_call_id": "gen_ai.tool.call.id",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_write_tokens": "gen_ai.usage.cache_write.input_tokens",
    "agent.session": {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "hermes",
    },
    "agent.turn": {"gen_ai.operation.name": "chat"},
    "agent.tool": {"gen_ai.operation.name": "execute_tool"},
    "agent.child": {"gen_ai.operation.name": "invoke_agent"},
}


def _genai_wanted(key: str, value: Any) -> bool:
    """Cache usage attrs are Recommended-when-non-zero; other gen_ai attrs when present."""
    return not (key.startswith("cache_") and value == 0)


def agent_traces_enabled(config: dict[str, Any]) -> bool:
    """True only when agent_traces AND the OTLP exporter gates are both on."""
    try:
        agent = ((config or {}).get("monitoring") or {}).get("agent_traces") or {}
        return bool(agent.get("enabled")) and bool(otlp_exporter.is_enabled(config))
    except Exception:
        logger.debug("agent-trace configuration unavailable", exc_info=True)
        return False


def _bounded(attrs: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """``hermes.<key>`` (plus dual-emitted gen_ai.*) for present keys; strings redacted+bounded."""
    from .redaction import redact_bounded

    out: dict[str, Any] = {}
    for key in keys:
        value = attrs.get(key)
        if key in ("model_id", "model_class") and (
            not isinstance(value, str) or not value.strip() or value.strip().lower() == "unknown"
        ):
            continue
        if key == "prompt_occupancy_tokens" and (type(value) is not int or value <= 0):
            continue
        if value is not None:
            if not isinstance(value, (str, bool, int, float)):
                continue
            processed = redact_bounded(value) if isinstance(value, str) else value
            out[f"hermes.{key}"] = processed
            genai_key = _GENAI_MAP.get(key)
            if genai_key is not None and _genai_wanted(key, value):
                out[genai_key] = processed
    error_type = out.get("hermes.error_class")
    if not error_type and "abandoned" in keys and attrs.get("abandoned") is True:
        error_type = "abandoned"
    if not error_type and out.get("hermes.status") in ("failed", "abandoned"):
        error_type = out["hermes.status"]
    if error_type:
        out["error.type"] = error_type
    return out


class AgentTracePlane:
    """Consumes agent_* monitoring events and maintains the open-span tree.

    ``sink_factory()`` must return a SpanExporter; tests inject
    ``InMemorySpanExporter``. Without it the OTLP provider from
    ``otlp_exporter._make_provider`` is used (bounded BatchSpanProcessor).
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        sink_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._config = config
        self._sink_factory = sink_factory
        self._lock = threading.Lock()
        agent_cfg = ((config or {}).get("monitoring") or {}).get("agent_traces") or {}
        self._ttl_ns = (
            int(
                agent_cfg.get("ttl_seconds", _DEFAULT_TTL_SECONDS)
                or _DEFAULT_TTL_SECONDS
            )
            * 1_000_000_000
        )
        # Open-span registry: session_ref -> {session_span, turns:{turn_ref -> ...}}.
        self._sessions: dict[str, dict[str, Any]] = {}
        # W3C ingress: validated parent span registered before session start.
        self._w3c_parent: dict[str, Any] = {}
        self._trace = None
        self._provider = None
        self._sdk_resources: list[list[Any]] = []
        self._tracer = None
        self._started = False
        self._initializing = False
        self._closing = False
        self._shutdown_deadline = 0.0
        self._shutdown_done = threading.Event()
        self._emitter = None
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None

    def __call__(self, batch: list[dict[str, Any]]) -> None:
        """Emitter subscriber entry: one agent_* event per handle() call (F9 filter)."""
        for event in batch:
            if (event.get("event") or "") in _AGENT_EVENT_KINDS:
                self.handle(event)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._started or self._stop.is_set():
            return
        from opentelemetry import (
            trace as otel_trace,
        )  # opentelemetry-api, present with sdk

        self._trace = otel_trace
        if self._sink_factory is not None:
            sdk = otlp_exporter._require_sdk(
                ("TracerProvider", "SimpleSpanProcessor", "Resource")
            )
            self._provider = sdk["TracerProvider"](resource=sdk["Resource"].create({}))
            provider_chain = [self._provider]
            self._sdk_resources.append(provider_chain)
            exporter = self._sink_factory()
            processor_chain = [exporter]
            self._sdk_resources.append(processor_chain)
            processor = sdk["SimpleSpanProcessor"](exporter)
            processor_chain.insert(0, processor)
            self._provider.add_span_processor(processor)
            provider_chain.extend(processor_chain)
            self._sdk_resources.remove(processor_chain)
        else:
            self._provider, _ = otlp_exporter._make_provider(
                self._config, _owned=self._sdk_resources
            )
        self._tracer = self._provider.get_tracer("hermes.gateway.agent_traces", "1.0.0")
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="agent-trace-ttl", daemon=True
        )
        self._sweeper.start()
        self._started = True

    def _begin_shutdown(self) -> float:
        # No plane/exporter lock on the owner loop. This lock protects bookkeeping
        # only; the singleton remains fenced until the daemon has really retired.
        # ponytail: Python cannot kill a wedged SDK thread; keep one daemon and
        # refuse replacement until it exits instead of multiplying leaked planes.
        with _PLANE_LOCK:
            if not self._closing:
                self._closing = True
                self._shutdown_deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
                threading.Thread(
                    target=self._shutdown_worker, name="agent-trace-shutdown", daemon=True
                ).start()
            return self._shutdown_deadline

    def shutdown(self) -> None:
        """Bounded synchronous owner wait (CLI/tests); cleanup may finish later."""
        deadline = self._begin_shutdown()
        self._shutdown_done.wait(max(0.0, deadline - time.monotonic()))

    async def shutdown_async(self) -> None:
        """One total owner budget, including drain, handler locks and SDK cleanup."""
        import asyncio

        deadline = self._begin_shutdown()
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def poll():
            if done.done():
                return
            if self._shutdown_done.is_set() or time.monotonic() >= deadline:
                done.set_result(None)
            else:
                loop.call_later(min(0.01, max(0.0, deadline - time.monotonic())), poll)

        poll()
        await done

    def _shutdown_worker(self) -> None:
        global _PLANE
        retired = True

        def attempt(action, *args, **kwargs):
            nonlocal retired
            try:
                action(*args, **kwargs)
            except BaseException:
                retired = False
                logger.debug("agent-trace cleanup failed; replacement remains fenced", exc_info=True)

        # Producers have quiesced at the gateway hook. Keep intake live until
        # all already accepted/in-flight events reach their original consumer.
        if self._emitter is not None:
            attempt(self._emitter.flush, timeout=None)
        with self._lock:
            self._stop.set()
            self._started = False
        if self._emitter is not None:
            attempt(self._emitter.unsubscribe, self)
            self._emitter = None
        with self._lock:
            # A broken span must not skip its siblings, sweeper or SDK roots.
            abandoned = {"abandoned": True, "abandon_reason": "shutdown"}
            ts_ns = time.time_ns()
            for state in self._sessions.values():
                for turn in state["turns"].values():
                    for span in (*turn["tools"].values(), *turn["children"].values()):
                        attempt(self._close_span, span, abandoned, ts_ns)
                    if turn.get("span") is not None and not turn.get("ended"):
                        attempt(self._close_span, turn["span"], abandoned, ts_ns)
                attempt(self._close_span, state["session_span"], abandoned, ts_ns)
            self._sessions.clear()
            self._w3c_parent.clear()
        # Thread.ident remains set after exit, but is None if start never ran.
        # This also joins a thread whose start wrapper raised after launching it.
        if self._sweeper is not None and self._sweeper.ident is not None:
            attempt(self._sweeper.join)
        if self._provider is not None:
            attempt(
                self._provider.force_flush,
                timeout_millis=max(1, int((self._shutdown_deadline - time.monotonic()) * 1000)),
            )
        if not otlp_exporter._shutdown_resources(self._sdk_resources):
            retired = False
        if retired:
            self._sdk_resources.clear()
            with _PLANE_LOCK:
                if _PLANE is self:
                    _PLANE = None
                self._shutdown_done.set()

    # -- event intake ---------------------------------------------------------

    def handle(self, event: Any) -> None:
        """Consume one monitoring event dict (pre-filtered to agent kinds)."""
        handler = {
            "agent_session_start": self._on_session_start,
            "agent_turn_start": self._on_turn_start,
            "agent_turn_end": self._on_turn_end,
            "agent_tool_exec": self._on_tool_exec,
            "agent_child": self._on_child,
        }.get(event.get("event"))
        if handler is None:
            return
        try:
            with self._lock:
                if self._stop.is_set() or not self._started:
                    return
                handler(event)
        except Exception:
            # Never raise into the emitter thread; the span tree degrades.
            logger.debug("agent-trace handling failed", exc_info=True)

    def adopt_w3c_parent(self, session_ref: str, parent_span: Any) -> None:
        """Register a validated W3C parent span for a not-yet-started session."""
        with self._lock:
            if self._stop.is_set() or not self._started or session_ref in self._sessions:
                return
            self._w3c_parent.pop(session_ref, None)
            if len(self._w3c_parent) >= _MAX_PENDING_PARENTS:
                self._w3c_parent.pop(next(iter(self._w3c_parent)))
            self._w3c_parent[session_ref] = (parent_span, time.monotonic_ns())

    # -- span helpers ---------------------------------------------------------

    def _start_span(
        self, name: str, parent_span: Any, attrs: dict[str, Any], ts_ns: int
    ) -> Any:
        context = (
            self._trace.set_span_in_context(parent_span)
            if parent_span is not None
            else None
        )
        out = _bounded(attrs, _ATTR_ALLOWLIST.get(name, ()))
        for genai_key, value in _GENAI_MAP.get(name, {}).items():
            out[genai_key] = value
        if attrs.get("session_ref"):
            from .redaction import redact_bounded

            out["gen_ai.conversation.id"] = redact_bounded(attrs["session_ref"])
        return self._tracer.start_span(
            name,
            context=context,
            start_time=ts_ns,
            attributes=out,
        )

    @staticmethod
    def _close_span(
        span: Any, extra: Optional[dict[str, Any]] = None, ts_ns: Optional[int] = None
    ) -> None:
        out = _bounded(extra or {}, _ATTR_ALLOWLIST.get(span.name, ()))
        for key, value in out.items():
            span.set_attribute(key, value)
        if "error.type" in out:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR))
        span.end(end_time=ts_ns if ts_ns is not None else time.time_ns())

    def _ensure_session(
        self, session_ref: str, origin: str, ts_ns: int
    ) -> dict[str, Any]:
        """Get or lazily create session state (emitter may have dropped the start)."""
        state = self._sessions.get(session_ref)
        if state is not None:
            state["last_activity_ns"] = time.monotonic_ns()
            return state
        if len(self._sessions) >= _MAX_SESSIONS:
            oldest = self._sessions.pop(next(iter(self._sessions)))
            self._abandon_session(oldest, "capacity")
        pending = self._w3c_parent.pop(session_ref, None)
        parent = pending[0] if pending is not None and time.monotonic_ns() - pending[1] <= self._ttl_ns else None
        span = self._start_span(
            "agent.session",
            parent,
            {"origin": origin, "session_ref": session_ref},
            ts_ns,
        )
        state = {"session_span": span, "turns": {}, "last_activity_ns": time.monotonic_ns()}
        self._sessions[session_ref] = state
        return state

    @staticmethod
    def _abandon_session(state: dict[str, Any], reason: str) -> None:
        ts_ns = time.time_ns()
        for turn in state["turns"].values():
            AgentTracePlane._abandon_turn(turn, reason, ts_ns)
        state["turns"].clear()
        AgentTracePlane._close_span(
            state["session_span"], {"abandoned": True, "abandon_reason": reason}, ts_ns
        )

    @staticmethod
    def _abandon_turn(turn: dict[str, Any], reason: str, ts_ns: Optional[int] = None) -> None:
        """Force-end one turn's still-open tool/child spans and the turn span."""
        for tool in turn["tools"].values():
            AgentTracePlane._close_span(
                tool, {"abandoned": True, "abandon_reason": reason}, ts_ns
            )
        turn["tools"].clear()
        for child in turn["children"].values():
            AgentTracePlane._close_span(
                child, {"abandoned": True, "abandon_reason": reason}, ts_ns
            )
        turn["children"].clear()
        span = turn.get("span")
        if span is not None and not turn.get("ended"):
            AgentTracePlane._close_span(
                span, {"abandoned": True, "abandon_reason": reason}, ts_ns
            )
            turn["ended"] = True

    # -- handlers -------------------------------------------------------------

    def _on_session_start(self, ev: dict) -> None:
        self._ensure_session(
            ev["session_ref"], ev.get("origin", "unknown"), ev.get("ts_ns") or time.time_ns()
        )

    def _on_turn_start(self, ev: dict) -> None:
        session_ref = ev["session_ref"]
        ts_ns = ev.get("ts_ns") or time.time_ns()
        state = self._ensure_session(session_ref, ev.get("origin", "unknown"), ts_ns)
        turn_ref = ev["turn_ref"]
        if turn_ref in state["turns"]:
            state["turns"][turn_ref]["last_activity_ns"] = time.monotonic_ns()
            return
        if len(state["turns"]) >= _MAX_TURNS_PER_SESSION:
            oldest = state["turns"].pop(next(iter(state["turns"])))
            self._abandon_turn(oldest, "capacity")
        state["turns"][turn_ref] = {
            "span": self._start_span(
                "agent.turn",
                state.get("session_span"),
                {
                    "model_class": ev.get("model_class"),
                    "model_id": ev.get("model_id"),
                    "session_ref": session_ref,
                },
                ts_ns,
            ),
            "ended": False,
            "tools": {},
            "children": {},
            "last_activity_ns": time.monotonic_ns(),
        }

    def _on_tool_exec(self, ev: dict) -> None:
        state = self._sessions.get(ev["session_ref"])
        if state is None:
            return
        turn = state["turns"].get(ev["turn_ref"])
        if turn is None:
            return
        state["last_activity_ns"] = turn["last_activity_ns"] = time.monotonic_ns()
        call_id = ev["tool_call_id"]
        if ev.get("phase") == "start":
            if call_id not in turn["tools"]:
                self._make_span_room(turn["tools"], _MAX_TOOLS_PER_TURN)
                turn["tools"][call_id] = self._start_span(
                    "agent.tool",
                    turn["span"],
                    {
                        "tool_call_id": call_id,
                        "tool_name": ev.get("tool_name", "unknown"),
                    },
                    ev.get("ts_ns") or time.time_ns(),
                )
            return
        tool = turn["tools"].pop(call_id, None)
        if tool is not None:
            self._close_span(tool, ev, ev.get("ts_ns"))

    def _on_turn_end(self, ev: dict) -> None:
        state = self._sessions.get(ev["session_ref"])
        if state is None:
            return
        turn = state["turns"].get(ev["turn_ref"])
        if turn is None or turn.get("ended"):
            return
        state["last_activity_ns"] = time.monotonic_ns()
        ts_ns = ev.get("ts_ns") or time.time_ns()
        # Close dropped terminal events at the execution owner's boundary.
        outcome = ev.get("status")
        extra = {"abandoned": True, "abandon_reason": "turn_end"}
        if outcome in ("failed", "interrupted", "cancelled", "superseded"):
            extra = {"status": outcome, "error_class": ev.get("error_class")}
        for tool in turn["tools"].values():
            self._close_span(tool, extra, ts_ns)
        turn["tools"].clear()
        for child in turn["children"].values():
            self._close_span(child, extra, ts_ns)
        turn["children"].clear()
        self._close_span(turn["span"], ev, ts_ns)
        turn["ended"] = True
        state["turns"].pop(ev["turn_ref"], None)

    def _on_child(self, ev: dict) -> None:
        state = self._sessions.get(ev["parent_session_ref"])
        if state is None:
            return
        turn = state["turns"].get(ev["parent_turn_ref"])
        if turn is None:
            return
        state["last_activity_ns"] = turn["last_activity_ns"] = time.monotonic_ns()
        child_ref = ev["child_session_ref"]
        if ev.get("phase") == "start":
            if child_ref not in turn["children"]:
                self._make_span_room(turn["children"], _MAX_CHILDREN_PER_TURN)
                turn["children"][child_ref] = self._start_span(
                    "agent.child",
                    turn["span"],
                    {"child_session_ref": child_ref},
                    ev.get("ts_ns") or time.time_ns(),
                )
            return
        child = turn["children"].pop(child_ref, None)
        if child is not None:
            self._close_span(child, ev, ev.get("ts_ns"))

    def _make_span_room(self, registry: dict[str, Any], limit: int) -> None:
        if len(registry) >= limit:
            span = registry.pop(next(iter(registry)))
            self._close_span(span, {"abandoned": True, "abandon_reason": "capacity"})

    # -- TTL sweep ------------------------------------------------------------

    def _sweep_loop(self) -> None:
        while not self._stop.wait(_SWEEP_INTERVAL_SECONDS):
            try:
                self._sweep()
            except Exception:
                logger.debug("agent-trace TTL sweep failed", exc_info=True)

    def _sweep(self) -> None:
        now = time.monotonic_ns()
        with self._lock:
            for session_ref, (_, adopted_ns) in list(self._w3c_parent.items()):
                if now - adopted_ns > self._ttl_ns:
                    self._w3c_parent.pop(session_ref, None)
            for session_ref, state in list(self._sessions.items()):
                if now - state["last_activity_ns"] > self._ttl_ns:
                    self._abandon_session(state, "ttl")
                    self._sessions.pop(session_ref, None)
                    continue
                for turn_ref, turn in list(state["turns"].items()):
                    if now - turn["last_activity_ns"] > self._ttl_ns:
                        self._abandon_turn(turn, "ttl")
                        state["turns"].pop(turn_ref, None)


_PLANE: Optional[AgentTracePlane] = None
_PLANE_LOCK = threading.RLock()


def start_agent_trace_plane(
    config: dict[str, Any],
    *,
    emitter: Any = None,
    sink_factory: Optional[Callable[[], Any]] = None,
) -> Optional[AgentTracePlane]:
    """Idempotently start the agent-trace plane on the monitoring bus.

    Returns None when disabled (either gate false). Mirrors the gateway-health
    export startup contract; both planes subscribe with explicit event_filter
    so no event is double-exported (F9).
    """
    global _PLANE
    if not agent_traces_enabled(config):
        return None
    with _PLANE_LOCK:
        if _PLANE is not None:
            return None if _PLANE._closing or _PLANE._initializing else _PLANE
        if emitter is None:
            from .emitter import get_emitter

            emitter = get_emitter()
        plane = AgentTracePlane(config, sink_factory=sink_factory)
        plane._emitter = emitter
        plane._initializing = True
        _PLANE = plane  # reserve replacement fence, never hold it across SDK work
    try:
        plane.start()
        emitter.subscribe(plane)  # batch callable with internal kind filter
    except BaseException:
        plane._begin_shutdown()
        raise
    finally:
        plane._initializing = False
    return plane
