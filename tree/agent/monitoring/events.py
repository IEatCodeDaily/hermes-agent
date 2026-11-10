"""Typed gateway monitoring events.

Content-free service-health and redacted diagnostic events for the gateway
daemon — the only shapes the monitoring plane emits: no prompts, messages,
tool args/results, session history, or aggregate usage analytics. Field order
is wire order (``asdict``); never reorder.

``agent_*`` events carry execution metadata and named prompt occupancy. Optional
usage fields are reserved for measured accounting; current producers leave them
unset rather than mislabel occupancy or session totals as turn usage. All correlation ids in
``agent_*`` events MUST be pseudonymized by the producer via
``pseudonymize_id`` before construction: raw chat ids, phone numbers and ACP
UUIDs never enter the monitoring plane.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar, Dict, Optional


class _MonitoringEvent:
    __slots__ = ()
    EVENT: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.EVENT, **asdict(self)}


@dataclass(slots=True)
class GatewayHealthEvent(_MonitoringEvent):
    """Content-free gateway health snapshot or lifecycle event."""

    EVENT: ClassVar[str] = "gateway_health"
    name: str
    gateway_state: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    exit_reason: Optional[str] = None
    restart_requested: Optional[bool] = None
    active_agents: int = 0
    gateway_busy: bool = False
    gateway_drainable: bool = False
    platform_count: int = 0
    fatal_platform_count: int = 0
    profile: Optional[str] = None
    install_id: Optional[str] = None
    version: Optional[str] = None
    supervision_mode: Optional[str] = None
    pid: Optional[int] = None
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(slots=True)
class GatewayDiagnosticEvent(_MonitoringEvent):
    """Redacted gateway diagnostic event for operator-owned observability."""

    EVENT: ClassVar[str] = "gateway_diagnostic"
    name: str
    subsystem: str
    error_class: str = "unknown"
    error_code: Optional[str] = None
    platform: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    profile: Optional[str] = None
    version: Optional[str] = None
    severity: str = "warning"
    ts_ns: int = field(default_factory=time.time_ns)
    source_logger: Optional[str] = None


@dataclass(slots=True)
class CronExecutionEvent(_MonitoringEvent):
    """Content-free durable cron execution lifecycle projection."""

    EVENT: ClassVar[str] = "cron_execution"
    status: str
    job_key: str
    source: str = "unknown"
    duration_ms: Optional[int] = None
    delivery_outcome: Optional[str] = None
    error_class: Optional[str] = None
    ts_ns: int = field(default_factory=time.time_ns)


__all__ = [
    "AgentChildEvent",
    "AgentSessionStartEvent",
    "AgentToolExecEvent",
    "AgentTurnEndEvent",
    "AgentTurnStartEvent",
    "CronExecutionEvent",
    "GatewayDiagnosticEvent",
    "GatewayHealthEvent",
    "pseudonymize_id",
]


def pseudonymize_id(raw: Any) -> str:
    """Stable opaque correlation key: ``sha256:<24hex>`` — raw ids never leave the host.

    Producer-side (F1): every ``agent_*`` event hashes its correlation ids here
    BEFORE construction, so raw Telegram chat ids, phone numbers and ACP UUIDs
    never reach the emitter, the redactor or the OTLP wire, while identical ids
    still map to identical keys (span trees stay correlated). Mirrors the
    ``_safe_instance_id`` pattern in ``gateway_health.py``.
    """
    import hashlib

    value = str(raw or "unknown").encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(value).hexdigest()[:24]}"


@dataclass(slots=True)
class AgentSessionStartEvent(_MonitoringEvent):
    """A new agent session became traceable (ACP attach, gateway platform, API)."""

    EVENT: ClassVar[str] = "agent_session_start"
    session_ref: str  # pseudonymized — never a raw session key/chat id/UUID
    origin: str = "unknown"  # acp | gateway | api | cron — platform CLASS only
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(slots=True)
class AgentTurnStartEvent(_MonitoringEvent):
    """A turn (admission claim) started executing for a session."""

    EVENT: ClassVar[str] = "agent_turn_start"
    session_ref: str
    turn_ref: str  # pseudonymized admission_id + generation
    model_class: str = ""  # provider-class label only, no base_url/keys
    ts_ns: int = field(default_factory=time.time_ns)
    model_id: Optional[str] = None  # actual model identifier, never inferred from class


@dataclass(slots=True)
class AgentToolExecEvent(_MonitoringEvent):
    """A tool execution started or finished inside a turn."""

    EVENT: ClassVar[str] = "agent_tool_exec"
    phase: str  # start | end
    session_ref: str
    turn_ref: str
    tool_call_id: str  # tc-<12hex> — survives redaction intact (verified)
    tool_name: str
    duration_ms: Optional[int] = None
    error_class: Optional[str] = None
    ts_ns: int = field(default_factory=time.time_ns)
    status: Optional[str] = None  # completed | failed | cancelled | interrupted | abandoned


@dataclass(slots=True)
class AgentTurnEndEvent(_MonitoringEvent):
    """Execution terminal; occupancy is not aggregate provider token usage."""

    EVENT: ClassVar[str] = "agent_turn_end"
    session_ref: str
    turn_ref: str
    status: str  # completed | failed | cancelled | interrupted
    duration_ms: Optional[int] = None
    model_class: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    error_class: Optional[str] = None
    ts_ns: int = field(default_factory=time.time_ns)
    model_id: Optional[str] = None
    prompt_occupancy_tokens: Optional[int] = None


@dataclass(slots=True)
class AgentChildEvent(_MonitoringEvent):
    """delegate_task child session linked to its parent turn."""

    EVENT: ClassVar[str] = "agent_child"
    phase: str  # start | end
    parent_session_ref: str
    parent_turn_ref: str
    child_session_ref: str  # pseudonymized child session
    status: str = "running"
    ts_ns: int = field(default_factory=time.time_ns)
