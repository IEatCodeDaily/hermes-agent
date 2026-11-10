"""Shutdown verdict precedence is identical for startup aborts and normal teardown."""

import logging
from types import SimpleNamespace

import pytest

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.run_shutdown import _resolve_gateway_exit_verdict


@pytest.mark.parametrize(
    "failure,reason,code,signal,restart,service,expected",
    [
        (True, "failed", 78, True, False, True, False),
        (True, None, 78, False, False, False, False),
        (False, None, 78, True, False, True, 78),
        (False, None, None, True, False, True, False),
        (False, None, None, True, True, False, True),
        (False, None, None, False, True, True, GATEWAY_SERVICE_RESTART_EXIT_CODE),
        (False, None, None, False, False, False, True),
    ],
)
def test_exit_verdict_precedence(caplog, failure, reason, code, signal, restart, service, expected):
    runner = SimpleNamespace(should_exit_with_failure=failure, exit_reason=reason, exit_code=code,
                             _restart_requested=restart, _restart_via_service=service)
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        if type(expected) is int:
            with pytest.raises(SystemExit) as exc:
                _resolve_gateway_exit_verdict(runner, signal)
            assert exc.value.code == expected
        else:
            assert _resolve_gateway_exit_verdict(runner, signal) is expected
    assert [r.getMessage() for r in caplog.records] == (
        ["Gateway exiting with failure: failed"] if failure and reason else []
    )
