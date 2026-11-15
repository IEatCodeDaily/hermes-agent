"""MCP over a hosted editor's ACP connection, without a stdio helper.

The connection-oriented ACP proposal is still unstable. Keep its wire/SDK
compatibility here; the newer stateless proposal is a different protocol.
"""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import threading
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_LOCK = threading.Lock()
_PROJECTIONS = {}
_MAX_MESSAGE = 8 * 1024 * 1024


class McpServerAcp(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    type: Literal['acp'] = 'acp'
    name: str = Field(min_length=1)
    server_id: str = Field(alias='serverId', min_length=1)


def install_routes(connection, agent):
    """Extend this connection's SDK router, never monkeypatch installed schemas."""
    from acp import schema
    from acp.utils import normalize_result
    # SDK 0.9.0 cannot parse ACP servers. Subclass only the four incoming
    # request models; keep every other field and handler on the SDK's path.
    server_type = McpServerAcp | schema.McpServerHttp | schema.McpServerSse | schema.McpServerStdio
    for method, model, handler in (
        ('session/new', schema.NewSessionRequest, 'new_session'),
        ('session/load', schema.LoadSessionRequest, 'load_session'),
        ('session/resume', schema.ResumeSessionRequest, 'resume_session'),
        ('session/fork', schema.ForkSessionRequest, 'fork_session'),
    ):
        from pydantic import create_model
        extended = create_model(model.__name__ + 'WithAcp', __base__=model,
                                mcp_servers=(list[server_type], Field(default_factory=list, alias='mcpServers')))
        connection._conn._handler.route_request(method, extended, agent, handler, adapt_result=normalize_result)


class ProjectionMCP:
    """An unguessable in-memory reference bound to one authenticated ACP peer."""

    def __init__(self, connection):
        self.connection = connection
        self.loop = asyncio.get_running_loop()
        self.ref = uuid.uuid4().hex
        self.servers = set()
        self.pending = set()
        self.closed = False
        self.closed_event = asyncio.Event()
        with _LOCK:
            _PROJECTIONS[self.ref] = self

    def bind(self, servers):
        result = []
        for server in servers:
            spec = server.model_dump(by_alias=True) if isinstance(server, BaseModel) else dict(server)
            if spec.get('type') == 'acp':
                # The editor may name a server, but never another peer's reference.
                spec = McpServerAcp.model_validate(spec).model_dump(by_alias=True)
                self.servers.add(spec['serverId'])
                spec['_acp_ref'] = self.ref
            result.append(spec)
        return result

    async def _send(self, method, params, notification):
        if self.closed:
            raise ConnectionError('Editor ACP projection closed')
        task = asyncio.current_task()
        self.pending.add(task)
        try:
            wire = self.connection._conn
            send = wire.send_notification if notification else wire.send_request
            return await send(method, params)
        finally:
            self.pending.discard(task)

    async def send(self, method, params, *, notification=False):
        if self.closed or self.loop.is_closed():
            raise ConnectionError('Editor ACP projection closed')
        try:
            if asyncio.get_running_loop() is self.loop:
                return await self._send(method, params, notification)
            # Native MCP runs on its own loop/thread. No blocking future.result():
            # discovery must keep the editor's event loop reading.
            future = asyncio.run_coroutine_threadsafe(self._send(method, params, notification), self.loop)
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            if self.closed:
                raise ConnectionError('Editor ACP projection closed; outcome is unknown') from None
            raise

    async def wait_closed(self):
        if asyncio.get_running_loop() is self.loop:
            await self.closed_event.wait()
        else:
            future = asyncio.run_coroutine_threadsafe(self.closed_event.wait(), self.loop)
            await asyncio.wrap_future(future)

    async def aclose(self):
        self.closed = True
        self.closed_event.set()
        with _LOCK:
            _PROJECTIONS.pop(self.ref, None)
        tasks = list(self.pending)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def resolve_projection(ref, server_id):
    with _LOCK:
        owner = _PROJECTIONS.get(ref)
    if owner is None or owner.closed or server_id not in owner.servers:
        raise ConnectionError('Editor ACP authority unavailable')
    return owner


@asynccontextmanager
async def acp_transport(ref, server_id):
    """Adapt ACP results to the native MCP ClientSession's bounded streams."""
    import anyio
    from mcp import types
    from mcp.shared.message import SessionMessage

    rpc_message = TypeAdapter(types.JSONRPCMessage)
    owner = resolve_projection(ref, server_id)
    reply = await asyncio.wait_for(owner.send('mcp/connect', {'serverId': server_id}), 30)
    connection_id = reply.get('connectionId') if isinstance(reply, dict) else None
    if not isinstance(connection_id, str) or not connection_id:
        raise ConnectionError('Editor returned an invalid MCP connection')
    incoming, read = anyio.create_memory_object_stream(16)
    write, outgoing = anyio.create_memory_object_stream(16)

    async def forward(message):
        payload = message.message.model_dump(by_alias=True, exclude_none=True)
        ident = payload.get('id')
        params = {'connectionId': connection_id, 'method': payload['method']}
        if 'params' in payload:
            params['params'] = payload['params']
        try:
            if len(json.dumps(params).encode()) > _MAX_MESSAGE:
                raise ValueError('MCP message exceeds 8 MiB')
            result = await asyncio.wait_for(owner.send('mcp/message', params, notification=ident is None), 30)
            if ident is not None:
                response = types.JSONRPCResponse(jsonrpc='2.0', id=ident, result=result)
                await incoming.send(SessionMessage(rpc_message.validate_python(
                    response.model_dump(by_alias=True, exclude_none=True))))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # health: allow BLE001 -- isolate arbitrary peer RPC/serialization failures without logging credentials
            from acp.exceptions import RequestError
            if isinstance(exc, RequestError) and ident is not None:
                # Preserve definite outer protocol rejections. This version of
                # T3 itself flattens inner MCP errors; we cannot reconstruct them.
                response = types.JSONRPCError(jsonrpc='2.0', id=ident, error=types.ErrorData(
                    code=exc.code, message=exc.message))
                await incoming.send(SessionMessage(rpc_message.validate_python(
                    response.model_dump(by_alias=True, exclude_none=True))))
                return
            # No recovery ladder may replay a write after an ambiguous dispatch.
            # Do not include peer error text (it can contain launch credentials).
            if ident is None:
                raise ConnectionError('Editor MCP notification failed') from None
            response = types.JSONRPCError(jsonrpc='2.0', id=ident, error=types.ErrorData(
                code=-32000, message='Editor MCP request failed; outcome is unknown. '
                'Hermes did not replay it. Do not retry automatically; inspect external state first.'))
            await incoming.send(SessionMessage(rpc_message.validate_python(
                    response.model_dump(by_alias=True, exclude_none=True))))

    async def pump():
        # ponytail: at most 16 concurrent requests; increase only with a peer
        # contract allowing more, never unbounded task-per-message buffering.
        limiter = anyio.Semaphore(16)

        async def limited(message):
            try:
                await forward(message)
            finally:
                limiter.release()

        async with anyio.create_task_group() as requests:
            async for message in outgoing:
                if 'id' not in message.message.model_dump(exclude_none=True):
                    await forward(message)
                else:
                    await limiter.acquire()
                    requests.start_soon(limited, message)

    async def owner_closed():
        await owner.wait_closed()
        raise ConnectionError('Editor ACP projection closed; outcome is unknown')

    try:
        async with incoming, read, write, outgoing:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(pump)
                tasks.start_soon(owner_closed)
                try:
                    yield read, write
                finally:
                    tasks.cancel_scope.cancel()
    finally:
        # Cleanup must not hold gateway shutdown hostage; peer disposal also
        # closes all its connections if the editor transport has already gone.
        with anyio.CancelScope(shield=True):
            with suppress(Exception):
                await asyncio.wait_for(owner.send('mcp/disconnect', {'connectionId': connection_id}), 5)
