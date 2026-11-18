"""Blue-green handover primitives: env contract, gates, standby, guard ordering."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys

import pytest

from gateway import handover, systemd_notify


def _clear_env(monkeypatch):
    for key in (handover.HANDOVER_ENV, handover.PARENT_PID_ENV, handover.PARENT_START_ENV,
                "NOTIFY_SOCKET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(handover, "_parsed_context", None)


def test_env_roundtrip_and_scrub(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(handover.HANDOVER_ENV, "1")
    monkeypatch.setenv(handover.PARENT_PID_ENV, str(os.getpid() + 12345))
    monkeypatch.delenv(handover.PARENT_START_ENV, raising=False)
    ctx = handover.handover_from_env()
    assert ctx is not None and ctx.parent_pid == os.getpid() + 12345
    assert ctx.parent_start_ticks is None
    # Parsed-and-scrubbed: the authority must not leak to this process's own children, and
    # bootstrap's second parse returns the cached context.
    assert os.environ.get(handover.HANDOVER_ENV) is None
    assert handover.handover_from_env() is ctx
    env = handover.successor_env()
    assert env[handover.HANDOVER_ENV] == "1"


def test_env_rejects_self_and_missing(monkeypatch):
    _clear_env(monkeypatch)
    assert handover.handover_from_env() is None
    monkeypatch.setattr(handover, "_parsed_context", None)
    monkeypatch.setenv(handover.HANDOVER_ENV, "1")
    monkeypatch.setenv(handover.PARENT_PID_ENV, str(os.getpid()))
    assert handover.handover_from_env() is None  # predecessor cannot be ourselves
    monkeypatch.setattr(handover, "_parsed_context", None)
    monkeypatch.setenv(handover.PARENT_PID_ENV, "not-a-pid")
    assert handover.handover_from_env() is None


def test_enabled_gate(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert handover.enabled(False) is False  # no NOTIFY_SOCKET -> cannot switch MainPID
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/user/1/notify")
    assert handover.enabled(True) is True
    assert handover.enabled(False) is False


def test_notify_mainpid_payload(tmp_path):
    sock_path = tmp_path / "notify.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(sock_path))
    server.setblocking(False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(systemd_notify, "_notify_socket", lambda: str(sock_path))
        assert systemd_notify.notify_mainpid(4242) is True
    assert server.recv(4096) == b"MAINPID=4242"
    server.close()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(systemd_notify, "_notify_socket", lambda: "")
        assert systemd_notify.notify_mainpid(1) is False


def _write_standby_marker(tmp_path, successor_pid, parent_pid):
    marker = tmp_path / f"gateway.handover-standby.{successor_pid}.{parent_pid}.tmp"
    marker.write_text(json.dumps({"successor_pid": successor_pid, "parent_pid": parent_pid}))
    return marker


def test_wait_successor_standby_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(handover, "_POLL_INTERVAL_S", 0.05)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    # Green keys the marker by BLUE's pid (this process); a marker keyed by the child's own
    # pid is what the original poll looked for, so blue never saw a real standby.
    _write_standby_marker(tmp_path, proc.pid, os.getpid())
    try:
        loop = asyncio.new_event_loop()
        try:
            ready = loop.run_until_complete(
                asyncio.wait_for(handover.wait_successor_standby(proc, timeout=3.0), 6))
            assert ready is True
        finally:
            loop.close()
    finally:
        proc.kill()
        proc.wait()


def test_wait_successor_standby_rejects_draining_blue(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(handover, "_POLL_INTERVAL_S", 0.02)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        loop = asyncio.new_event_loop()
        try:
            # No marker for THIS successor: blue's own identify/answer can never satisfy it.
            ready = loop.run_until_complete(handover.wait_successor_standby(proc, timeout=0.3))
            assert ready is False
        finally:
            loop.close()
    finally:
        proc.kill()
        proc.wait()


def test_wait_successor_standby_dead_child(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(handover, "_POLL_INTERVAL_S", 0.05)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    loop = asyncio.new_event_loop()
    try:
        # A successor that already exited fails fast (no full-timeout wait).
        ready = loop.run_until_complete(
            asyncio.wait_for(handover.wait_successor_standby(proc, timeout=30.0), 5))
        assert ready is False
    finally:
        loop.close()


def test_standby_marker_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parent = os.getpid() + 321
    assert handover.poll_standby_marker(parent) is None  # nothing written yet
    assert handover.write_standby_marker(parent) is True
    assert handover.poll_standby_marker(parent) == os.getpid()
    handover.clear_standby_marker(parent)
    assert handover.poll_standby_marker(parent) is None


def test_terminate_successor_fallback():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    handover.terminate_successor(child)
    assert child.poll() is not None
    already = subprocess.Popen([sys.executable, "-c", "pass"])
    already.wait()
    handover.terminate_successor(already)  # no-op on an exited child


def test_stand_by_blocks_until_predecessor_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    monkeypatch.setattr(handover, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(handover, "BOOT_FEED_S", 0.0)
    blue = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    ctx = handover.HandoverContext(parent_pid=blue.pid, parent_start_ticks=None)
    seen = []
    real_alive = handover.predecessor_alive

    def alive(c):
        seen.append(handover.poll_standby_marker(blue.pid))  # marker visible to blue while waiting
        if blue.poll() is not None:
            return False
        return real_alive(c)
    monkeypatch.setattr(handover, "predecessor_alive", alive)
    handover.stand_by(ctx)
    assert blue.poll() is not None  # returned only after blue exited
    assert os.getpid() in seen
    assert handover.poll_standby_marker(blue.pid) is None  # cleared on takeover


def test_run_gateway_stands_by_before_single_owner_guards(monkeypatch):
    """Regression: the successor hit 'already serves profile' because the standby wait ran
    inside start_gateway, after run_gateway's host-attach guard had already refused it."""
    from hermes_cli import gateway as cli_gateway
    order = []
    ctx = handover.HandoverContext(parent_pid=os.getpid() + 1, parent_start_ticks=None)
    monkeypatch.setattr(handover, "handover_from_env", lambda: ctx)
    monkeypatch.setattr(handover, "stand_by", lambda c: order.append("stand_by"))
    monkeypatch.setattr(cli_gateway, "_guard_official_docker_root_gateway", lambda: None)

    def attach(**_kw):
        order.append("attach_guard")
        raise SystemExit(0)
    monkeypatch.setattr(cli_gateway, "_attach_to_host_gateway_or_guard", attach)
    with pytest.raises(SystemExit):
        cli_gateway.run_gateway()
    assert order == ["stand_by", "attach_guard"]


def test_restart_wait_allows_acp_reattach_reads():
    from types import SimpleNamespace

    from gateway.drain_control import ws_method_refused_while_draining
    draining = SimpleNamespace(_draining=True)
    for method in ("config.get", "session.stats", "session.resume", "session.info"):
        assert not ws_method_refused_while_draining(draining, method), method
    assert ws_method_refused_while_draining(draining, "config.set")


def test_predecessor_alive_pid_reuse_guard(monkeypatch):
    # A recycled pid with mismatched start ticks is not our predecessor.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    try:
        ctx = handover.HandoverContext(parent_pid=child.pid, parent_start_ticks=1)
        monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 999)
        assert handover.predecessor_alive(ctx) is False
    finally:
        child.kill()
        child.wait()
