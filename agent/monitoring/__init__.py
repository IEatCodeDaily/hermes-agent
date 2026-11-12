"""Hermes gateway monitoring: service health + redacted diagnostics over OTLP.

``emitter`` is the in-process event bus: producers hand typed events to a
fire-and-forget queue that never blocks or raises into gateway code, and OTLP
subscribers consume them off the hot path. Nothing is persisted locally.
Out of scope: trajectory capture, aggregate usage analytics, any
content-bearing signal (those belong to the NeMo Relay integration).

``agent_traces`` wires session, turn and executed-tool spans with
producer-pseudonymized correlation ids. Child handling and validated-parent
registration are helper capabilities; child producers and ACP W3C ingress
are not wired. Producers expose prompt occupancy separately and do not
synthesize per-turn usage from session counters. Measured usage fields are
consumer capabilities only, not proof of wired provider accounting.
"""

from __future__ import annotations

from . import agent_traces, emitter, events

emit = emitter.emit
get_emitter = emitter.get_emitter

__all__ = ["agent_traces", "emit", "emitter", "events", "get_emitter"]
