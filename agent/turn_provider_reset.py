"""Park a retry on a declared provider reset, not on guessed backoff."""
import math
import random
import time

from agent.turn_recovery import interruptible_backoff_sleep

# Total declared-reset waiting per API-call block, not per response. A moving
# reset must not park a turn forever. Ordinary errors still use api_max_retries.
PROVIDER_RESET_WAIT_CAP_S = 6 * 60 * 60
# One park sleeps at most a probe interval: a combo aggregator whose other
# account already recovered must be picked up in minutes, not after the whole
# declared reset. The total deadline below still bounds the entire park.
PROVIDER_RESET_PROBE_INTERVAL_S = 5 * 60


def _hhmm(epoch: float) -> str:
    return time.strftime('%H:%M', time.localtime(epoch))


def _delegated_session(agent) -> bool:
    """Delegate children never park on a limit: fail the run so T3 surfaces the limit
    to the parent thread, whose agent decides retry later / switch model / take over.
    True for in-process delegate_task children (and descendants carrying the env
    marker) and for T3 ACP sessions the gateway stamped ``t3_delegated`` at launch."""
    try:
        from agent.delegation_context import is_delegated_child_process_context
        if is_delegated_child_process_context():
            return True
    except Exception:
        pass
    return getattr(agent, '_t3_delegated', False) is True


def _open_episode(agent, retry, reset_at: float) -> None:
    """Open the park episode with its ONE park notice — never re-announced while the
    episode stays open, no matter how many 5-minute probe parks follow."""
    if getattr(retry, 'provider_limit_episode', None):
        return
    provider = getattr(agent, 'provider', '') or 'unknown'
    model = getattr(agent, 'model', '') or 'unknown'
    retry.provider_limit_episode = {'started': time.time(), 'provider': provider, 'model': model}
    agent._emit_warning(
        f"Stuck on provider limit: {provider}/{model}, resets ~{_hhmm(reset_at)} WIB; "
        "auto-retrying every 5 min")


def close_provider_limit_episode(agent, retry) -> None:
    """Its ONE resume notice, when a parked episode's next API call succeeds; called from
    the turn loop's success path. No-op without an open episode and after the first."""
    episode = getattr(retry, 'provider_limit_episode', None)
    if not episode or episode.get('resumed'):
        return
    episode['resumed'] = True
    minutes = max(1, round((time.time() - episode['started']) / 60))
    agent._emit_warning(f"Resumed at {_hhmm(time.time())} after {minutes} min parked")


def _terminal_failure(messages, api_call_count, reason: str, detail: str):
    return ('return', {
        'final_response': detail, 'messages': messages, 'api_calls': api_call_count,
        'completed': False, 'failed': True, 'error': detail,
        'failure_reason': reason, 'failure_retryable': False,
    })


def wait_for_provider_reset(agent, error, classified, context, retry, *,
                            messages, conversation_history, api_call_count):
    if not classified.retryable:
        return None
    from agent.credential_pool import _parse_absolute_timestamp
    from agent.retry_utils import provider_retry_after_seconds
    seconds = provider_retry_after_seconds(error)
    if seconds is None:
        reset = _parse_absolute_timestamp(context.get('reset_at'))
        seconds = None if reset is None else reset - time.time()
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return None  # no declared reset: an ordinary error — no park, no special casing
    if _delegated_session(agent):
        # Structured, immediate failure: provider, model, limit kind, reset_at (+ any
        # structured scope/code the provider body carried) for the parent to act on.
        reset_at = time.time() + seconds
        extra = ' '.join(f"{k}={context[k]}" for k in
                         ('limit_code', 'limit_kind', 'limit_scope', 'scope') if k in context)
        reason = getattr(classified, 'reason', None)
        detail = (f"provider_limit: provider={getattr(agent, 'provider', '') or '?'} "
                  f"model={getattr(agent, 'model', '') or '?'} "
                  f"kind={getattr(reason, 'value', reason) or '?'} "
                  f"reset_at={time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(reset_at))}"
                  + (f" {extra}" if extra else ""))
        return _terminal_failure(messages, api_call_count, 'provider_limit', detail)
    now = time.monotonic()
    if retry.provider_reset_deadline is None:
        retry.provider_reset_deadline = now + PROVIDER_RESET_WAIT_CAP_S
        _open_episode(agent, retry, time.time() + seconds)
    wait = min(seconds + random.uniform(1, 3),
               PROVIDER_RESET_PROBE_INTERVAL_S + random.uniform(0, 30))
    if wait > retry.provider_reset_deadline - now:
        # Cap expired: FAIL the run with a clear limit error instead of looping forever.
        return _terminal_failure(
            messages, api_call_count, 'provider_limit_wait_cap',
            f"Provider limit did not clear within {PROVIDER_RESET_WAIT_CAP_S // 3600}h: "
            f"provider={getattr(agent, 'provider', '') or '?'} "
            f"model={getattr(agent, 'model', '') or '?'} "
            f"next reset in ~{seconds / 60:.0f} min. Giving up per limit policy; "
            "resend later or switch provider/model.")
    agent._emit_diagnostic_status(f'⏳ Provider cooldown: waiting {wait:.0f}s until reset; retry budget unchanged.')
    agent._emit_diagnostic_wait(f'⏳ Waiting for provider reset in {wait:.0f}s (retry budget unchanged)')
    result = interruptible_backoff_sleep(agent, wait, retry, messages=messages,
        conversation_history=conversation_history, api_call_count=api_call_count,
        abort_message='Interrupted while waiting for provider reset.',
        interrupt_text='Operation interrupted while waiting for provider reset.',
        activity_label='provider reset wait')
    if result is not None:
        return 'return', result
    return ('break' if retry.restart_with_redirected_messages else 'continue'), None
