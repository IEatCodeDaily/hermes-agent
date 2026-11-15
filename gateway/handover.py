"""Blue-green gateway restart handover.

Successor-first restart: blue (the running gateway) spawns green (``gateway run``) as a
child in the same supervisor cgroup. Green boots with platform-lock takeover authority,
binds the control socket name once it is ready (new sessions and relays land on green
while blue keeps serving its established connections), and points systemd's MainPID at
itself via ``sd_notify(MAINPID=...)`` so ``Restart=always`` stays quiet when blue exits.

Ownership rules that keep the overlap safe:

- The PID record (``gateway.pid``, O_EXCL) stays with blue until blue exits; green claims
  it from a deferred task that polls for the predecessor's exit.
- Messaging/cron single-ownership needs no extra deferral: green boots with
  ``_platform_lock_takeover_on_start`` (the same authority ``--replace`` uses), so its
  adapter cold-connects fence blue's pollers out of each platform lock.
- If green never becomes ready, blue falls back to the classic drain-exit restart; a green
  that crashes after taking MainPID still converges, because systemd then respawns the
  unit normally.

Requires a ``Type=notify`` unit (``NOTIFY_SOCKET`` present) so the MainPID switch works;
without it the handover is refused and restarts keep the classic drain-exit shape.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

HANDOVER_ENV = "HERMES_GATEWAY_HANDOVER"
PARENT_PID_ENV = "HERMES_GATEWAY_HANDOVER_PARENT_PID"
PARENT_START_ENV = "HERMES_GATEWAY_HANDOVER_PARENT_START_TICKS"

READY_TIMEOUT_S = 20.0
# Blue's drain budget already bounds its stop; this is only a safety net for arming the
# deferred PID-record claim, and arming late is safe (blue's own exit path also cleans up).
PREDECESSOR_EXIT_TIMEOUT_S = 1800.0
_POLL_INTERVAL_S = 0.25


def enabled(config: Any) -> bool:
    """Blue-green restart gate: config opt-in + a notify socket (MainPID switch) + POSIX."""
    if sys.platform == "win32" or not os.environ.get("NOTIFY_SOCKET", "").strip():
        return False
    agent = getattr(config, "agent", None)
    return bool(getattr(agent, "blue_green_restart", False))


def successor_env() -> dict[str, str]:
    """Env additions that mark a child as blue's handover successor (PID-reuse guarded)."""
    from gateway.status import get_process_start_time
    return {
        HANDOVER_ENV: "1",
        PARENT_PID_ENV: str(os.getpid()),
        PARENT_START_ENV: str(get_process_start_time(os.getpid())),
    }


@dataclass(frozen=True)
class HandoverContext:
    """Green-side view of its predecessor (``parent_start_ticks`` guards PID reuse)."""

    parent_pid: int
    parent_start_ticks: Optional[int]


def handover_from_env() -> Optional[HandoverContext]:
    """Parse the successor env; None when this boot is not a handover successor."""
    if os.environ.get(HANDOVER_ENV, "").strip() != "1":
        return None
    try:
        parent_pid = int(os.environ.get(PARENT_PID_ENV, ""))
    except ValueError:
        logger.error("Handover env %s set but %s is missing/invalid; treating as a normal boot",
                     HANDOVER_ENV, PARENT_PID_ENV)
        return None
    if parent_pid <= 0 or parent_pid == os.getpid():
        logger.error("Handover parent pid %r is not a predecessor; treating as a normal boot", parent_pid)
        return None
    parent_start: Optional[int]
    try:
        parent_start = int(os.environ.get(PARENT_START_ENV, ""))
    except ValueError:
        parent_start = None
    return HandoverContext(parent_pid=parent_pid, parent_start_ticks=parent_start)


def predecessor_alive(ctx: HandoverContext) -> bool:
    """True while the predecessor process still exists (PID-reuse guarded when we have ticks)."""
    try:
        os.kill(ctx.parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else — still alive
    if ctx.parent_start_ticks is not None:
        from gateway.status import get_process_start_time
        return get_process_start_time(ctx.parent_pid) == ctx.parent_start_ticks
    return True


async def spawn_successor() -> Optional[int]:
    """Blue side: spawn the ``gateway run`` successor child; its pid, or None on failure."""
    from gateway.run import _resolve_hermes_bin, logger as run_logger
    argv = _resolve_hermes_bin()
    if not argv:
        run_logger.error("Blue-green handover: could not resolve the hermes argv; skipping")
        return None
    from hermes_cli.gateway import _profile_arg
    full_argv = [*argv, *shlex.split(_profile_arg()), "gateway", "run"]
    try:
        proc = subprocess.Popen(  # noqa: ASYNC220 -- spawn returns at once; the child outlives this task
            full_argv, env={**os.environ, **successor_env()},
        )
    except OSError as exc:
        run_logger.error("Blue-green handover: failed to spawn successor: %s", exc)
        return None
    run_logger.info("Blue-green handover: spawned successor gateway (PID %d)", proc.pid)
    return proc.pid


def _identify_pid(socket_path) -> Optional[int]:
    """One public ``identify`` round-trip on the control socket; the answering pid or None."""
    import socket as _socket
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as sock:
            sock.settimeout(1.5)
            sock.connect(str(socket_path))
            sock.sendall(json.dumps({"id": 1, "protocol": 1, "verb": "identify"}).encode() + b"\n")
            with sock.makefile("rb") as reader:
                line = reader.readline(4096)
        payload = json.loads(line.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    result = payload.get("result")
    pid = result.get("pid") if isinstance(result, dict) else None
    return pid if isinstance(pid, int) and pid > 0 else None


async def wait_successor_ready(successor_pid: int, timeout: float = READY_TIMEOUT_S) -> bool:
    """Blue side: poll the control socket until *successor_pid* is the gateway answering."""
    from gateway.control_socket import resolve_client_socket_path
    from hermes_constants import get_hermes_home
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        path = resolve_client_socket_path(get_hermes_home())
        if path is not None:
            # Blue may still answer identify until green binds; only green's pid counts.
            if _identify_pid(path) == successor_pid:
                return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return False


async def arm_after_predecessor_exit(
    ctx: HandoverContext,
    on_exit: Callable[[], Any],
    timeout_s: float = PREDECESSOR_EXIT_TIMEOUT_S,
) -> None:
    """Green side: run ``on_exit`` once the predecessor is gone (its stop path releases the
    single-owner records). Arms anyway at the timeout — blue's exit path also cleans up."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if not predecessor_alive(ctx):
            break
        await asyncio.sleep(_POLL_INTERVAL_S)
    else:
        logger.warning("Predecessor (PID %d) still alive after %.0fs; claiming deferred records anyway",
                       ctx.parent_pid, timeout_s)
    outcome = on_exit()
    if inspect.isawaitable(outcome):
        await outcome


def _claim_pid_record(ctx: HandoverContext) -> None:
    """Claim the PID record once the predecessor is gone (the body of the deferred claim)."""
    import atexit
    from gateway.status import get_process_start_time, get_running_pid, remove_pid_file, write_pid_file
    recorded = get_running_pid()
    if recorded is not None and recorded != os.getpid():
        try:
            os.kill(recorded, 0)
            alive = get_process_start_time(recorded) is not None
        except OSError:
            alive = False
        if alive:
            logger.error("Deferred PID claim skipped: a live gateway (PID %d) holds the record", recorded)
            return
        logger.warning("Replacing stale predecessor PID record (PID %d)", recorded)
        remove_pid_file()
    try:
        write_pid_file()
    except FileExistsError as exc:
        logger.error("Deferred PID claim lost the PID-file race: %s", exc)
        return
    atexit.register(remove_pid_file)
    from gateway.run import _claim_host_gateway_role
    try:
        # The predecessor held the host role; with its exit the role is free, but claim it
        # explicitly (force = the handover contract) so host-attach resolution points at us.
        _claim_host_gateway_role(force=True)
    except SystemExit as exc:  # the role claim refuses via SystemExit(75) — degrade, not die
        logger.error("Deferred host-role claim refused after handover: %s", exc)
    except Exception as exc:  # noqa: BLE001 -- role claim failure must not undo the PID claim
        logger.error("Deferred host-role claim failed after handover: %s", exc)
    from hermes_cli.gateway_multiplex_mode import publish_pending_multiplex_decision
    publish_pending_multiplex_decision()
    logger.info("Blue-green handover complete: PID record claimed (predecessor PID %d exited)",
                ctx.parent_pid)


def deferred_pid_claim(ctx: HandoverContext) -> None:
    """Green side: schedule ``_claim_pid_record`` to run once blue exits."""

    async def _run() -> None:
        await arm_after_predecessor_exit(ctx, lambda: _claim_pid_record(ctx))

    asyncio.get_running_loop().create_task(_run())
