"""Declared cooldowns park an API attempt without spending its retry budget."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.error_classifier import FailoverReason, classify_api_error, _structured_limit_context
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


def test_structured_limit_fields_pass_through():
    body = {'code': 'rate_limit_error', 'scope': 'combo', 'reset_at': 1794400000,
            'error': {'code': 'inner', 'limit_kind': 'usage'}}
    assert _structured_limit_context(body) == {
        'limit_code': 'rate_limit_error', 'scope': 'combo', 'limit_kind': 'usage'}
    assert _structured_limit_context({'error': {'limit_kind': 'usage'}}) == {'limit_kind': 'usage'}
    assert _structured_limit_context(None) == {}


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
    agent = Mock(provider='custom', model='opus')
    retry = TurnRetryState()
    classified = SimpleNamespace(retryable=True)
    error = RuntimeError('limited')
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    args = dict(messages=[], conversation_history=[], api_call_count=1)
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 1800}, retry, **args) == ('continue', None)
    assert 300 <= sleeps[0] <= 331
    now += waiting.PROVIDER_RESET_WAIT_CAP_S
    action, payload = waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 10}, retry, **args)
    assert action == 'return' and payload['failed'] is True
    assert payload['failure_reason'] == 'provider_limit_wait_cap'
    assert len(sleeps) == 1
    retry = TurnRetryState(restart_with_redirected_messages=True)
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 10}, retry, **args) == ('break', None)
    retry = TurnRetryState()
    interrupted = {'interrupted': True}
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: interrupted)
    assert waiting.wait_for_provider_reset(agent, error, classified, {'reset_at': now + 10}, retry, **args) == ('return', interrupted)


def test_park_notice_exactly_once_and_resume_notice_once(monkeypatch):
    from agent import turn_provider_reset as waiting
    from agent.turn_provider_reset import close_provider_limit_episode
    now = 1000.0
    monkeypatch.setattr(waiting.time, 'monotonic', lambda: now)
    monkeypatch.setattr(waiting.time, 'time', lambda: now)
    agent = Mock(provider='custom', model='opus')
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    retry = TurnRetryState()
    classified = SimpleNamespace(retryable=True)
    error = RuntimeError('limited')
    args = dict(messages=[], conversation_history=[], api_call_count=1)
    context = {'reset_at': now + 3600}
    assert waiting.wait_for_provider_reset(agent, error, classified, context, retry, **args) == ('continue', None)
    assert waiting.wait_for_provider_reset(agent, error, classified, context, retry, **args) == ('continue', None)
    warnings = [c.args[0] for c in agent._emit_warning.call_args_list]
    assert len(warnings) == 1
    assert 'Stuck on provider limit: custom/opus' in warnings[0]
    assert 'resets ~' in warnings[0] and 'auto-retrying every 5 min' in warnings[0]
    close_provider_limit_episode(agent, retry)
    close_provider_limit_episode(agent, retry)
    resumes = [c.args[0] for c in agent._emit_warning.call_args_list][1:]
    assert len(resumes) == 1 and 'Resumed at' in resumes[0]


def test_delegate_child_fails_fast_structured(monkeypatch):
    from agent import turn_provider_reset as waiting
    monkeypatch.setattr('agent.delegation_context.is_delegated_child_process_context', lambda: True)
    agent = Mock(provider='custom', model='opus')
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    classified = SimpleNamespace(retryable=True, reason=SimpleNamespace(value='rate_limit'))
    action, payload = waiting.wait_for_provider_reset(
        agent, RuntimeError('limited'), classified,
        {'reset_at': waiting.time.time() + 600, 'scope': 'combo'},
        TurnRetryState(), messages=[], conversation_history=[], api_call_count=1)
    assert action == 'return' and payload['failed'] is True
    assert payload['failure_reason'] == 'provider_limit'
    assert 'provider=custom' in payload['error'] and 'model=opus' in payload['error']
    assert 'kind=rate_limit' in payload['error'] and 'reset_at=' in payload['error']
    assert 'scope=combo' in payload['error']
    assert sleeps == []


def test_gateway_stamped_t3_child_fails_fast(monkeypatch):
    from agent import turn_provider_reset as waiting
    agent = Mock(provider='custom', model='opus')
    agent._t3_delegated = True
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    action, payload = waiting.wait_for_provider_reset(
        agent, RuntimeError('limited'), SimpleNamespace(retryable=True),
        {'reset_at': waiting.time.time() + 600}, TurnRetryState(),
        messages=[], conversation_history=[], api_call_count=1)
    assert action == 'return' and payload['failure_reason'] == 'provider_limit'
    assert sleeps == []


def test_no_declared_reset_is_an_ordinary_error():
    from agent.turn_provider_reset import wait_for_provider_reset
    assert wait_for_provider_reset(Mock(provider='custom', model='opus'), RuntimeError('bad'),
        SimpleNamespace(retryable=True), {}, TurnRetryState(),
        messages=[], conversation_history=[], api_call_count=1) is None


def test_header_and_absolute_reset_are_not_capped_at_ten_minutes(monkeypatch, source):
    from agent import turn_provider_reset as waiting
    error = RuntimeError('busy')
    error.response = SimpleNamespace(headers={'Retry-After': '1800'} if source == 'header' else {})
    context = {} if source == 'header' else {'reset_at': waiting.time.time() + 1800}
    sleeps = []
    monkeypatch.setattr(waiting, 'interruptible_backoff_sleep', lambda *a, **k: sleeps.append(a[1]))
    assert waiting.wait_for_provider_reset(Mock(provider='custom', model='opus'), error, SimpleNamespace(retryable=True), context,
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
    assert wait_for_provider_reset(Mock(provider='custom', model='opus'), RuntimeError('bad'), SimpleNamespace(retryable=True),
        {'reset_at': reset}, TurnRetryState(), messages=[], conversation_history=[], api_call_count=1) is None

def test_five_hour_reset_parks_in_probe_intervals(monkeypatch):
    """5h declared reset: first park <= probe interval, budget untouched,
    a second error re-parks, and the total deadline still ends the park."""
    from agent import turn_provider_reset as waiting
    now = 1000.0
    monkeypatch.setattr(waiting.time, "monotonic", lambda: now)
    monkeypatch.setattr(waiting.time, "time", lambda: now)
    agent = Mock(provider='custom', model='opus')
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
    action, payload = waiting.wait_for_provider_reset(agent, error, classified, {"reset_at": now + 3600}, retry, **args)
    assert action == "return" and payload["failed"] is True
    assert payload["failure_reason"] == "provider_limit_wait_cap"
