"""Blue-green handover primitives: env contract, gates, readiness probe, deferred claim."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import types

import pytest

from gateway import handover, systemd_notify


class _AgentCfg:
    blue_green_restart = True


class _Cfg:
    agent = _AgentCfg()


class _CfgOff:
    agent = types.SimpleNamespace(blue_green_restart=False)


def _clear_env(monkeypatch):
    for key in (handover.HANDOVER_ENV, handover.PARENT_PID_ENV, handover.PARENT_START_ENV,
                "NOTIFY_SOCKET"):
        monkeypatch.delenv(key, raising=False)


def test_env_roundtrip(monkeypatch):
    monkeypatch.setenv(handover.HANDOVER_ENV, "1")
    monkeypatch.setenv(handover.PARENT_PID_ENV, str(os.getpid() + 12345))
    monkeypatch.delenv(handover.PARENT_START_ENV, raising=False)
    ctx = handover.handover_from_env()
    assert ctx is not None and ctx.parent_pid == os.getpid() + 12345
    assert ctx.parent_start_ticks is None
    env = handover.successor_env()
    assert env[handover.HANDOVER_ENV] == "1"
    # successor_env() stamps the CALLER as the parent; from this process that is self -> rejected.
    assert handover.handover_from_env() is None


def test_env_rejects_self_and_missing(monkeypatch):
    _clear_env(monkeypatch)
    assert handover.handover_from_env() is None
    monkeypatch.setenv(handover.HANDOVER_ENV, "1")
    monkeypatch.setenv(handover.PARENT_PID_ENV, str(os.getpid()))
    assert handover.handover_from_env() is None  # predecessor cannot be ourselves
    monkeypatch.setenv(handover.PARENT_PID_ENV, "not-a-pid")
    assert handover.handover_from_env() is None


def test_enabled_gate(monkeypatch):
    _clear_env(monkeypatch)
    assert handover.enabled(_Cfg()) is False  # no NOTIFY_SOCKET -> cannot switch MainPID
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/user/1/notify")
    assert handover.enabled(_Cfg()) is True
    assert handover.enabled(_CfgOff()) is False


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


def test_wait_successor_ready_pid_match(tmp_path):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock_path = tmp_path / "gateway.sock"
    server.bind(str(sock_path))
    server.listen(1)

    successor_pid = os.getpid() + 777

    async def serve_once():
        conn, _ = server.accept()
        with conn:
            conn.recv(4096)
            conn.sendall(json.dumps({"ok": True, "result": {"pid": successor_pid}}).encode() + b"\n")

    loop = asyncio.new_event_loop()
    try:
        serve_task = loop.run_in_executor(None, serve_once)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(handover, "_POLL_INTERVAL_S", 0.05)
            mp.setattr("gateway.control_socket.resolve_client_socket_path", lambda home: sock_path)
            ready = loop.run_until_complete(
                asyncio.wait_for(handover.wait_successor_ready(successor_pid, timeout=5.0), 8))
        assert ready is True
        serve_task.result()
    finally:
        loop.close()
        server.close()


def test_wait_successor_ready_timeout(tmp_path, monkeypatch):
    sock_path = tmp_path / "gateway.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    monkeypatch.setattr("gateway.control_socket.resolve_client_socket_path", lambda home: sock_path)
    monkeypatch.setattr(handover, "_POLL_INTERVAL_S", 0.02)
    loop = asyncio.new_event_loop()
    try:
        # Nobody answers with the successor's pid -> bounded False.
        ready = loop.run_until_complete(handover.wait_successor_ready(os.getpid() + 999, timeout=0.2))
        assert ready is False
    finally:
        loop.close()
        server.close()


def test_arm_after_predecessor_exit(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    ctx = handover.HandoverContext(parent_pid=child.pid, parent_start_ticks=None)
    fired = []
    monkeypatch.setattr(handover, "_POLL_INTERVAL_S", 0.02)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            asyncio.wait_for(handover.arm_after_predecessor_exit(ctx, lambda: fired.append(1)), 10))
    finally:
        loop.close()
    child.wait()
    assert fired == [1]


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


def _patch_claim_status(monkeypatch, recorded_pid, recorded_alive, writes):
    import gateway.status as status
    monkeypatch.setattr(status, "get_running_pid", lambda *a, **k: recorded_pid)
    if recorded_alive:
        monkeypatch.setattr(status, "get_process_start_time", lambda pid: 12345)

        def _kill(pid, sig):
            return None
    else:
        def _kill(pid, sig):
            raise ProcessLookupError()
    monkeypatch.setattr(os, "kill", _kill)
    monkeypatch.setattr(status, "write_pid_file", lambda: writes.append(1))
    monkeypatch.setattr(status, "remove_pid_file", lambda: writes.append("removed"))
    import hermes_cli.gateway_multiplex_mode as gmm
    monkeypatch.setattr(gmm, "publish_pending_multiplex_decision", lambda: None)
    import gateway.run as run_mod
    monkeypatch.setattr(run_mod, "_claim_host_gateway_role", lambda force=False: None)


def test_claim_skips_live_owner(monkeypatch):
    writes: list = []
    _patch_claim_status(monkeypatch, recorded_pid=4321, recorded_alive=True, writes=writes)
    ctx = handover.HandoverContext(parent_pid=os.getpid() + 555, parent_start_ticks=None)
    handover._claim_pid_record(ctx)
    assert writes == []  # a live recorded gateway blocks the claim


def test_claim_replaces_stale_owner(monkeypatch):
    writes: list = []
    _patch_claim_status(monkeypatch, recorded_pid=4321, recorded_alive=False, writes=writes)
    ctx = handover.HandoverContext(parent_pid=os.getpid() + 555, parent_start_ticks=None)
    handover._claim_pid_record(ctx)
    assert writes == ["removed", 1]


def test_claim_when_record_free(monkeypatch):
    writes: list = []
    _patch_claim_status(monkeypatch, recorded_pid=None, recorded_alive=False, writes=writes)
    ctx = handover.HandoverContext(parent_pid=os.getpid() + 555, parent_start_ticks=None)
    handover._claim_pid_record(ctx)
    assert writes == [1]
