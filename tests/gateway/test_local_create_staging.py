"""Creation binding stays serial even if its client disconnects mid-discovery."""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_cancelled_create_holds_binding_lock_until_physical_staging_finishes():
    from gateway.session_controls import AuthorityConnection
    from gateway.session_runtime_workers import mutation_tasks
    from tests.waiting import async_wait_until

    entered, release = asyncio.Event(), asyncio.Event()
    events = []

    class Connection:
        authority = SimpleNamespace()

        async def _create_serial(self, ref, params):
            events.append(params)
            if params == 'first':
                entered.set()
                await release.wait()
            return params

    connection = Connection()
    first = asyncio.create_task(AuthorityConnection.create(connection, None, 'first'))
    await asyncio.wait_for(entered.wait(), 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert connection.authority._local_create_lock.locked()
    second = asyncio.create_task(AuthorityConnection.create(connection, None, 'second'))
    await async_wait_until(lambda: len(mutation_tasks(connection.authority)) == 2,
                           timeout=5, what='second create waits for the first staging task')
    assert events == ['first']
    release.set()
    assert await asyncio.wait_for(second, 5) == 'second'
    assert events == ['first', 'second']
