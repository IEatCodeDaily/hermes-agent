"""Trace startup config I/O leaves the gateway loop; plane adoption stays on it."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from gateway.run_startup import GatewayStartupMixin


def test_trace_startup_loads_config_off_loop(monkeypatch, tmp_path):
    from agent.monitoring import agent_traces, gateway_health_export
    from gateway import status
    from hermes_cli import config, security_advisories

    owner = threading.get_ident()
    loads = []
    plane = object()
    raw = {"monitoring": {"agent_traces": {"enabled": True}}}

    def load():
        loads.append(threading.get_ident())
        return raw

    def start(value):
        assert value is raw
        assert threading.get_ident() == owner
        return plane

    monkeypatch.setattr(config, "load_config", load)
    monkeypatch.setattr(status, "write_runtime_status", lambda **kw: True)
    monkeypatch.setattr(gateway_health_export, "start_gateway_health_export", lambda value: None)
    monkeypatch.setattr(agent_traces, "start_agent_trace_plane", start)
    monkeypatch.setattr(security_advisories, "detect_compromised", list)
    monkeypatch.setattr(security_advisories, "gateway_log_message", lambda value: None)
    runner = SimpleNamespace(config=SimpleNamespace(sessions_dir=tmp_path),
                             _start_loop_liveness_guards=Mock(),
                             _start_log_systemd_timing_alignment=Mock(),
                             _start_log_retired_session_reset=Mock(), _log_agent_budget=Mock())
    asyncio.run(GatewayStartupMixin._start_log_startup_environment(runner))
    assert runner._agent_trace_plane is plane
    assert len(loads) == 2
    assert loads[-1] != owner
