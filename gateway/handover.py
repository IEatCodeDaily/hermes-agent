"""Blue-green gateway restart handover.

Successor-first restart: blue (the running gateway) spawns green (``gateway run``) as a
child in the same supervisor cgroup. Green serves sessions once its runtime reaches the
ready state; blue keeps serving its established connections until their turns finish,
then exits through the existing stop path. The MainPID switch is sent by BLUE (not green:
``NotifyAccess=main`` drops notify messages from any other pid) and only after green is
ready and feeding its watchdog, so ``Restart=always`` never fires behind green's back.

Ownership rules that keep the overlap safe:

- The PID record (``gateway.pid``, O_EXCL) stays with blue until blue exits; green claims
  it from a deferred task that polls for the predecessor's exit. A lost claim is fatal for
  green (SIGTERM self-stop): an unrecorded gateway must not keep serving.
- Platform locks: green carries NO takeover authority. Its adapters fail to lock while
  blue holds them and pick up through the existing reconnect watcher once blue's stop
  releases them; blue is never signalled by green.
- Blue sends ``MAINPID=`` only after green's control socket answers ``identify`` with the
  successor pid AND descriptor state ``ready`` (adapters, cron, housekeeping running).

Known protocol remainder (kept out of v1, flag stays default-off): the profile
reservation / session-store handoff, control-socket session affinity for turns parked on
approval/clarify, and fixed-port listeners (api_server) that cannot rebind until blue's
stop closes them.

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
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

HANDOVER_ENV = "HERMES_GATEWAY_HANDOVER"
PARENT_PID_ENV = "HERMES_GATEWAY_HANDOVER_PARENT_PID"
PARENT_START_ENV = "HERMES_GATEWAY_HANDOVER_PARENT_START_TICKS"

READY_TIMEOUT_S = 20.0
FALLBACK_TERMINATE_S = 10.0
# Blue's drain budget already bounds its stop; this is only a safety net for arming the
# deferred PID-record claim, and arming late is safe (blue's own exit path also cleans up).
PREDECESSOR_EXIT_TIMEOUT_S = 1800.0
_POLL_INTERVAL_S = 0.25

_parsed_context: Optional["HandoverContext"] = None


def enabled(flag: bool) -> bool:
    """Blue-green restart gate: pre-loaded config flag + notify socket (MainPID switch)."""
    if sys.platform == "win32" or not os.environ.get("NOTIFY_SOCKET", "").strip():
        return False
    return bool(flag)


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
    """Parse the successor env; None when this boot is not a handover successor.

    The variables are scrubbed from the environment on first parse and the result is
    cached: a handover successor must not leak its authority to ``gateway run`` children
    of its own, and bootstrap reads this twice.
    """
    global _parsed_context
    if _parsed_context is not None:
        return _parsed_context or None
    if os.environ.get(HANDOVER_ENV, "").strip() != "1":
        _parsed_context = False
        return None
    try:
        parent_pid = int(os.environ.get(PARENT_PID_ENV, ""))
    except ValueError:
        logger.error("Handover env %s set but %s is missing/invalid; treating as a normal boot",
                     HANDOVER_ENV, PARENT_PID_ENV)
        _parsed_context = False
        return None
    parent_start: Optional[int]
    try:
        parent_start = int(os.environ.get(PARENT_START_ENV, ""))
    except ValueError:
        parent_start = None
    if parent_pid <= 0 or parent_pid == os.getpid():
        logger.error("Handover parent pid %r is not a predecessor; treating as a normal boot", parent_pid)
        _parsed_context = False
        return None
    for key in (HANDOVER_ENV, PARENT_PID_ENV, PARENT_START_ENV):
        os.environ.pop(key, None)
    _parsed_context = HandoverContext(parent_pid=parent_pid, parent_start_ticks=parent_start)
    return _parsed_context


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


def spawn_successor() -> Optional[subprocess.Popen]:
    """Blue side: spawn the ``gateway run`` successor child; its handle, or None."""
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
    return proc


def _identify_pid_and_state(socket_path) -> Optional[tuple[int, str]]:
    """One public ``identify`` round-trip; (answering pid, descriptor state) or None."""
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
    if not isinstance(result, dict):
        return None
    pid = result.get("pid")
    state = result.get("state")
    if not isinstance(pid, int) or pid <= 0 or not isinstance(state, str):
        return None
    return pid, state


async def wait_successor_ready(proc: subprocess.Popen, timeout: float = READY_TIMEOUT_S) -> bool:
    """Blue side: poll until the successor ITSELF answers identify in the ready state.

    Blue may still answer until green binds the socket name, and a bound-but-starting
    green answers ``state != 'ready'`` — neither counts. Blocking socket I/O runs in a
    thread so the event loop (streaming, adapters, in-flight turns) never stalls.
    """
    from gateway.control_socket import resolve_client_socket_path
    from hermes_constants import get_hermes_home
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if proc.poll() is not None:
            logger.warning("Blue-green handover: successor (PID %d) exited during startup", proc.pid)
            return False
        path = resolve_client_socket_path(get_hermes_home())
        if path is not None:
            answer = await asyncio.to_thread(_identify_pid_and_state, path)
            if answer is not None and answer[0] == proc.pid and answer[1] == "ready":
                return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return False


def terminate_successor(proc: subprocess.Popen) -> None:
    """Blue side: a successor that was never promoted must not outlive the fallback."""
    if proc.poll() is not None:
        return
    logger.warning("Blue-green handover: terminating unpromoted successor (PID %d)", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=FALLBACK_TERMINATE_S)
    except subprocess.TimeoutExpired:
        logger.error("Successor (PID %d) ignored SIGTERM; SIGKILL", proc.pid)
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


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


def _self_stop() -> None:
    """Stop this gateway through its own signal path (clean drain, supervisor revives)."""
    os.kill(os.getpid(), signal.SIGTERM)


def _claim_pid_record(ctx: HandoverContext) -> None:
    """Claim the PID record once the predecessor is gone (the body of the deferred claim).

    A lost claim leaves this process as an unrecorded gateway — fatal by self-stop so the
    supervisor's next boot takes over cleanly instead of two gateways racing the record.
    """
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
            logger.error("Deferred PID claim skipped: a live gateway (PID %d) holds the record; "
                         "this successor stops itself", recorded)
            _self_stop()
            return
        logger.warning("Replacing stale predecessor PID record (PID %d)", recorded)
        remove_pid_file()
    try:
        write_pid_file()
    except FileExistsError as exc:
        logger.error("Deferred PID claim lost the PID-file race (%s); this successor stops itself", exc)
        _self_stop()
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
