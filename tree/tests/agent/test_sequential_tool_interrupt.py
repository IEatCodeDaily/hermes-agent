"""Sequential tool execution must abandon the wait when the user interrupts.

Regression tests for the "interrupt doesn't end a running tool" class:
the sequential executor path previously ran the tool inline (when the
deadline was disabled) or waited in 5s slices without checking
``agent._interrupt_requested`` — a non-cooperative tool (e.g. a blocking
FAL ``handler.get()``) held the whole turn hostage until it returned.

Now the wait loop polls the interrupt flag every
``_SEQUENTIAL_INTERRUPT_POLL_SECONDS`` and, after a 3s cooperative grace,
synthesizes a cancelled tool result and abandons the worker.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from tools.interrupt import is_interrupted, is_thread_interrupted, set_interrupt
from tests.waiting import wait_until

from agent import tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolCancelledResult,
    _run_sequential_tool_execution_middleware,
)

class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False
        self.activity = []

    def _touch_activity(self, msg):
        self.activity.append(msg)

@pytest.fixture()
def fake_agent(monkeypatch):
    # This executor-only fixture needs real tool interrupt bits, not AIAgent bootstrap.
    interrupt_api = SimpleNamespace(_set_interrupt=set_interrupt)
    monkeypatch.setattr(tool_executor, "_ra", lambda: interrupt_api)
    return _FakeAgent()

@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    # Keep the test fast: short poll slice, no config lookups.
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    emitted = []
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda agent, **kw: emitted.append(kw),
    )
    yield emitted

def test_interrupt_abandons_noncooperative_tool(monkeypatch, fake_agent, _fast_polls):
    """A blocking tool is abandoned within ~poll+grace once interrupted."""

    started = threading.Event()
    release = threading.Event()
    workers = []

    def _fake_middleware(agent_arg, **kwargs):
        workers.append(threading.current_thread())
        started.set()
        release.wait(30)  # non-cooperative: ignores the real interrupt bit
        return _ManagedToolResult(
            result="late result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )

    def _interrupt_soon():
        if started.wait(5):
            fake_agent._interrupt_requested = True

    interrupter = threading.Thread(target=_interrupt_soon, daemon=True)
    interrupter.start()

    try:
        t0 = time.monotonic()
        managed = _run_sequential_tool_execution_middleware(
            fake_agent,
            function_name="image_generate",
            function_args={"prompt": "x"},
            effective_task_id="t",
            tool_call_id="call_1",
            execute=lambda a: "unused",
        )
        elapsed = time.monotonic() - t0

        assert started.is_set()
        assert isinstance(managed.result, _ToolCancelledResult)
        assert "cancelled" in str(managed.result)
        # Real poll + real 3s grace + slack, not the blocked tool's 30s wait.
        assert elapsed < 10.0
        assert elapsed >= 3.0
        assert workers[0].is_alive()  # result returned before the tool was released
        assert is_thread_interrupted(workers[0].ident)
        assert len(_fast_polls) == 1
        assert _fast_polls[0]["status"] == "cancelled"
    finally:
        release.set()
        interrupter.join(5)
        for worker in workers:
            worker.join(5)
    assert not interrupter.is_alive()
    assert not workers[0].is_alive()
    assert not fake_agent._tool_worker_threads
    assert not is_thread_interrupted(workers[0].ident)
    assert len(_fast_polls) == 1

def test_interrupt_prefers_real_result_from_cooperative_tool(
    monkeypatch, fake_agent, _fast_polls
):
    """A tool that finishes within the grace window returns its real result."""

    def _fake_middleware(agent_arg, **kwargs):
        # Finish only after the executor fans out the real per-thread interrupt.
        wait_until(is_interrupted, timeout=5, what="worker interrupt fan-out")
        return _ManagedToolResult(
            result="real result", args={}, middleware_trace=[],
            blocked=False, dispatched=True,
        )

    monkeypatch.setattr(
        tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware
    )
    monkeypatch.setattr(
        tool_executor, "_resolve_sequential_tool_timeout", lambda: None
    )
    fake_agent._interrupt_requested = True  # interrupted before first poll

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="web_search",
        function_args={},
        effective_task_id="t",
        tool_call_id="call_2",
        execute=lambda a: "unused",
    )

    assert managed.result == "real result"
    assert not isinstance(managed.result, _ToolCancelledResult)
