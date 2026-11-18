"""Blue-green gateway restart handover.

Successor-first restart: blue (the running gateway) spawns green (``gateway run``) as a
child in the same supervisor cgroup. Green stands by (:func:`stand_by`) BEFORE any ownership
guard — host attach, duplicate-process, PID record, profile reservation — writes a standby
marker, feeds systemd's watchdog and blocks until blue exits. Blue sees the marker, sends
``MAINPID=<green>`` (only the current main may: ``NotifyAccess=main``), drains and exits;
green then boots through the normal guards, which all see a dead owner. Nothing is
bypassed and two processes never hold the stores.

Green carries NO platform-lock takeover authority (takeover SIGTERMs the holder).

Not covered: serving NEW sessions while blue drains (store custody, approval affinity,
fixed-port listeners) — blue's drain still refuses them.

Requires a ``Type=notify`` unit (``NOTIFY_SOCKET`` present) so the MainPID switch works;
without it the handover is refused and restarts keep the classic drain-exit shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HANDOVER_ENV = "HERMES_GATEWAY_HANDOVER"
PARENT_PID_ENV = "HERMES_GATEWAY_HANDOVER_PARENT_PID"
PARENT_START_ENV = "HERMES_GATEWAY_HANDOVER_PARENT_START_TICKS"

READY_TIMEOUT_S = 20.0
FALLBACK_TERMINATE_S = 10.0
# Green is MainPID from standby-confirm on, but its own SystemdWatchdog starts only at READY;
# keep feeding across the post-takeover boot (~30s observed vs WatchdogSec=30s).
BOOT_FEED_S = 120.0
_POLL_INTERVAL_S = 0.25

_parsed_context: Optional[HandoverContext] = None


def standby_marker_path(parent_pid: int) -> str:
    """Green writes this after interpreter/config pre-warm; blue polls for it before the
    MAINPID switch. One file per predecessor pid, in HERMES_HOME (same trust domain as the
    PID record), so a stale marker can never vouch for a recycled pid."""
    from hermes_constants import get_hermes_home
    return str(Path(get_hermes_home()) / f"gateway.handover-standby.{os.getpid()}.{parent_pid}.tmp")


def write_standby_marker(parent_pid: int) -> bool:
    """Green side: pre-warm done, predecessor-wait armed. Consumed by green after takeover."""
    try:
        path = standby_marker_path(parent_pid)
        with open(path, "w") as f:
            f.write(json.dumps({"successor_pid": os.getpid(), "parent_pid": parent_pid}))
        os.chmod(path, 0o600)
        return True
    except OSError:
        return False


def clear_standby_marker(parent_pid: int) -> None:
    from hermes_constants import get_hermes_home
    with contextlib.suppress(OSError):
        os.unlink(standby_marker_path(parent_pid))


def poll_standby_marker(parent_pid: int) -> Optional[int]:
    """Blue side: successor pid if its standby marker is present, else None."""
    from hermes_constants import get_hermes_home
    path = Path(get_hermes_home()) / f"gateway.handover-standby.*.{parent_pid}.tmp"
    matches = sorted(Path(get_hermes_home()).glob(path.name))
    for match in matches:
        try:
            payload = json.loads(match.read_text())
            pid = int(payload.get("successor_pid", 0))
        except (OSError, ValueError):
            continue
        if pid > 0:
            return pid
    return None


async def wait_successor_standby(proc: subprocess.Popen, timeout: float = READY_TIMEOUT_S) -> bool:
    """Blue side: poll for green's standby marker (pre-warm done, watching the predecessor).

    Green runs its ownership bootstrap only AFTER blue exits, so standby — not a bound
    control socket — is the earliest trustworthy signal. Once green confirms standby it is
    also feeding the watchdog, which is what makes an early MAINPID switch safe.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if proc.poll() is not None:
            logger.warning("Blue-green handover: successor (PID %d) exited during startup", proc.pid)
            return False
        if poll_standby_marker(os.getpid()) == proc.pid:  # green keys its marker by OUR pid
            return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return False


def stand_by(ctx: HandoverContext) -> None:
    """Green side, before ANY ownership guard: mark standby, keep systemd's watchdog fed and
    block until the predecessor exits. Sync on purpose — it runs before the event loop.

    No timeout: blue's own restart-handoff watchdog hard-exits it by its deadline, and giving
    up early would boot green into a live owner's guards (exit -> unit restart -> SIGKILL of
    blue mid-drain)."""
    import threading
    with contextlib.suppress(Exception):
        # A long drain is not a wedged boot; run_gateway re-arms it after the guards.
        from hermes_startup_watchdog import disarm_startup_watchdog
        disarm_startup_watchdog()
    gone = threading.Event()
    threading.Thread(target=_feed_watchdog, args=(gone,), daemon=True,
                     name="gateway-handover-watchdog-feed").start()
    write_standby_marker(ctx.parent_pid)
    print(f"⏳ Blue-green handover: standing by until gateway PID {ctx.parent_pid} exits", flush=True)
    try:
        while predecessor_alive(ctx):
            time.sleep(_POLL_INTERVAL_S)
    finally:
        clear_standby_marker(ctx.parent_pid)
        gone.set()


def _feed_watchdog(predecessor_gone) -> None:
    # ponytail: blind feed for BOOT_FEED_S after takeover — systemd cannot see a wedged boot in
    # that window (the startup watchdog still can); stop on the runner's READY if it matters.
    from gateway.systemd_notify import notify
    while not predecessor_gone.wait(1.0):
        notify("WATCHDOG=1")
    end = time.monotonic() + BOOT_FEED_S
    while time.monotonic() < end:
        notify("WATCHDOG=1")
        time.sleep(1.0)


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
        proc = subprocess.Popen(
            full_argv, env={**os.environ, **successor_env()},
        )
    except OSError as exc:
        run_logger.error("Blue-green handover: failed to spawn successor: %s", exc)
        return None
    run_logger.info("Blue-green handover: spawned successor gateway (PID %d)", proc.pid)
    return proc


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
