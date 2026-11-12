"""Declared cooldowns park an API attempt without spending its retry budget."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.retry_utils import reset_delay_from_message
from agent.turn_retry_state import TurnRetryState


MESSAGE = '[claude/claude-opus-5-5] [429]: {"error":{"type":"rate_limit_error"}} (reset after 15m 29s)'


def test_relay_503_is_rate_limit_with_reset():
    error = RuntimeError(MESSAGE)
    error.status_code = 503
    verdict = classify_api_error(error, provider='custom')
    assert verdict.reason == FailoverReason.rate_limit
    assert verdict.error_context['reset_at'] > 0
    assert reset_delay_from_message(MESSAGE) == 929


@pytest.mark.parametrize('message, seconds', [('reset after 15m 29s', 929),
    ('resets in 2h 3m 4s', 7384), ('reset after 1 hour 2 minutes', 3720)])
def test_reset_grammar(message, seconds):
    assert reset_delay_from_message(message) == seconds


def test_declared_wait_precedes_recovery_and_preserves_eight_attempt_budget(monkeypatch):
    from agent import turn_api_error as handler
    from agent.agent_runtime_helpers import extract_api_error_context
    error = RuntimeError(MESSAGE)
    error.status_code = 503
    agent = Mock(provider='custom', model='opus', base_url='http://127.0.0.1:20128',
                 api_key=None, thinking_callback=None, context_compressor=None)
    agent._extract_api_error_context = extract_api_error_context
    monkeypatch.setattr(handler, 'recover_before_classification', lambda *a, **k: (False, 'system'))
    sleeps = []
    monkeypatch.setattr('agent.turn_provider_reset.interruptible_backoff_sleep',
                        lambda *a, **k: sleeps.append(a[1]))
    retry = TurnRetryState()
    result = handler.handle_api_error(agent, api_error=error, _retry=retry, thinking_spinner=None,
        messages=[], api_messages=[], api_kwargs={}, system_message='system', active_system_prompt='system',
        conversation_history=[], approx_tokens=0, retry_count=7, max_retries=8,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1,
        api_request_id='request', api_start_time=0, effective_task_id='task', turn_id='turn')
    assert result.action == 'continue'
    assert result.retry_count == 7 and result.max_retries == 8
    assert 300 <= sleeps[0] <= 331
    agent._try_activate_fallback.assert_not_called()
    agent._emit_diagnostic_wait.assert_called_once()


def test_bounded_wait_and_interrupt(monkeypatch):
    from agent import turn_provider_reset as waiting
    now = 1000.0
    monkeypatch.setattr(waiting.time, 'monotonic', lambda: now)
    monkeypatch.setattr(waiting.time, 'time', lambda: now)
    agent = Mock()
    retry = TurnRetryState()
    classified = SimpleNamespace(retryable=True)
    error = RuntimeError('limited')
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    args = dict(messages=[], conversation_history=[], api_call_count=1)
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 1800}, retry, **args) == ('continue', None)
    assert 300 <= sleeps[0] <= 331
    now += waiting.PROVIDER_RESET_WAIT_CAP_S
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 10}, retry, **args) is None
    assert len(sleeps) == 1
    retry = TurnRetryState(restart_with_redirected_messages=True)
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 10}, retry, **args) == ('break', None)
    retry = TurnRetryState()
    interrupted = {'interrupted': True}
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: interrupted)
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 10}, retry, **args) == ('return', interrupted)


@pytest.mark.parametrize('source', ['header', 'absolute'])
def test_header_and_absolute_reset_are_not_capped_at_ten_minutes(monkeypatch, source):
    from agent import turn_provider_reset as waiting
    error = RuntimeError('busy')
    error.response = SimpleNamespace(headers={'Retry-After': '1800'} if source == 'header' else {})
    context = {} if source == 'header' else {'reset_at': waiting.time.time() + 1800}
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    assert waiting.wait_for_provider_reset(Mock(), error, SimpleNamespace(retryable=True), context,
        TurnRetryState(), messages=[], conversation_history=[], api_call_count=1) == ('continue', None)
    assert 300 <= sleeps[0] <= 331


def test_generic_overload_and_billing_are_not_reclassified():
    for status, message, expected in [(503, 'server overloaded', FailoverReason.overloaded),
                                      (429, 'insufficient_quota', FailoverReason.billing)]:
        error = RuntimeError(message)
        error.status_code = status
        assert classify_api_error(error, provider='custom').reason == expected


@pytest.mark.parametrize('reset', [None, 'garbage', float('inf'), float('nan'), -1])
def test_invalid_reset_never_parks(reset):
    from agent.turn_provider_reset import wait_for_provider_reset
    assert wait_for_provider_reset(Mock(), RuntimeError('bad'), SimpleNamespace(retryable=True),
        {'reset_at': reset}, TurnRetryState(), messages=[], conversation_history=[], api_call_count=1) is None

def test_five_hour_reset_parks_in_probe_intervals(monkeypatch):
    """5h declared reset: first park <= probe interval, budget untouched,
    a second error re-parks, and the total deadline still ends the park."""
    from agent import turn_provider_reset as waiting
    now = 1000.0
    monkeypatch.setattr(waiting.time, "monotonic", lambda: now)
    monkeypatch.setattr(waiting.time, "time", lambda: now)
    agent = Mock()
    sleeps = []
    monkeypatch.setattr(waiting, "interruptible_backoff_sleep", lambda *a, **k: sleeps.append(a[1]))
    retry = TurnRetryState()
    classified = SimpleNamespace(retryable=True)
    error = RuntimeError("limited")
    args = dict(messages=[], conversation_history=[], api_call_count=1)
    context = {"reset_at": now + 5 * 3600}
    assert waiting.wait_for_provider_reset(agent, error, classified, context, retry, **args) == ("continue", None)
    assert waiting.PROVIDER_RESET_PROBE_INTERVAL_S <= sleeps[0] <= waiting.PROVIDER_RESET_PROBE_INTERVAL_S + 31
    assert retry.provider_reset_deadline is not None
    assert waiting.wait_for_provider_reset(agent, error, classified, context, retry, **args) == ("continue", None)
    assert len(sleeps) == 2
    now += waiting.PROVIDER_RESET_WAIT_CAP_S + sleeps[0]
    assert waiting.wait_for_provider_reset(agent, error, classified, {"reset_at": now + 3600}, retry, **args) is None
