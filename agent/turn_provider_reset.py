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
        return None
    now = time.monotonic()
    if retry.provider_reset_deadline is None:
        retry.provider_reset_deadline = now + PROVIDER_RESET_WAIT_CAP_S
    wait = min(seconds + random.uniform(1, 3),
                  PROVIDER_RESET_PROBE_INTERVAL_S + random.uniform(0, 30))
    if wait > retry.provider_reset_deadline - now:
        # Do not retry BEFORE a declared reset beyond our total attended bound.
        # Let the existing bounded retry/terminal policy handle that response.
        return None
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
