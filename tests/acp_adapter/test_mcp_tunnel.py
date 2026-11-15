"""Connection-oriented MCP-over-ACP: SDK wire compatibility and session ownership."""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_tunnel_reuses_mcp_client_without_process_and_closes():
    from acp_adapter.mcp_tunnel import ProjectionMCP, acp_transport
    from mcp import ClientSession

    messages = []

    class Peer:
        async def send_request(self, method, params):
            messages.append((method, params))
            if method == 'mcp/connect':
                return {'connectionId': 'owned-connection'}
            if method == 'mcp/disconnect':
                return {}
            assert method == 'mcp/message'
            assert params['connectionId'] == 'owned-connection'
            if params['method'] == 'initialize':
                return {'protocolVersion': params['params']['protocolVersion'], 'capabilities': {'tools': {}},
                        'serverInfo': {'name': 'editor', 'version': '1'}}
            if params['method'] == 'tools/list':
                return {'tools': [{'name': 'echo', 'inputSchema': {'type': 'object'}}]}
            if params['method'] == 'tools/call':
                return {'content': [{'type': 'text', 'text': params['params']['arguments']['text']}]}
            return {}

        async def send_notification(self, method, params):
            messages.append((method, params))

    owner = ProjectionMCP(SimpleNamespace(_conn=Peer()))
    spec = owner.bind([{'type': 'acp', 'name': 'editor', 'serverId': 'editor'}])[0]
    async with acp_transport(spec['_acp_ref'], spec['serverId']) as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            assert (await client.list_tools()).tools[0].name == 'echo'
            assert (await client.call_tool('echo', {'text': 'cheap'})).content[0].text == 'cheap'
    assert [m for m, _ in messages].count('mcp/connect') == 1
    assert [m for m, _ in messages].count('mcp/disconnect') == 1
    await owner.aclose()
    with pytest.raises(ConnectionError):
        async with acp_transport(spec['_acp_ref'], 'editor'):
            pytest.fail('closed projection was adopted')


@pytest.mark.asyncio
async def test_server_authority_and_fingerprint_do_not_depend_on_projection_token():
    from acp_adapter.mcp_tunnel import ProjectionMCP, resolve_projection
    from gateway.session_local_mcp import fingerprint, validate_servers

    first = ProjectionMCP(SimpleNamespace())
    second = ProjectionMCP(SimpleNamespace())
    wire = [{'type': 'acp', 'name': 'editor', 'serverId': 'editor'}]
    a, b = first.bind(wire), second.bind(wire)
    assert a != b
    assert fingerprint(a) == fingerprint(b) == fingerprint(wire)
    validate_servers(a)
    with pytest.raises((ValueError, ConnectionError)):
        resolve_projection(a[0]['_acp_ref'], 'foreign')
    from gateway.session_local_mcp import attach_editor_mcp
    from hermes_state_runtime import RuntimeStoreError
    import json
    authority = SimpleNamespace()
    policy = SimpleNamespace(editor_mcp_json=json.dumps({'scope': 'session-one', 'fingerprint': fingerprint(wire)}))
    attach_editor_mcp(authority, policy, a)
    with pytest.raises(RuntimeStoreError, match='acp_mcp_owner_conflict'):
        attach_editor_mcp(authority, policy, b)
    assert authority._local_editor_mcp['session-one'] == a
    await first.aclose()
    attach_editor_mcp(authority, policy, b)
    assert authority._local_editor_mcp['session-one'] == b
    await second.aclose()


@pytest.mark.asyncio
async def test_sent_tool_failure_is_reported_once_not_replayed():
    from acp_adapter.mcp_tunnel import ProjectionMCP, acp_transport
    from mcp import ClientSession
    from mcp.shared import exceptions
    McpError = getattr(exceptions, "MCPError", None) or exceptions.McpError

    calls = []

    class Peer:
        async def send_request(self, method, params):
            if method == 'mcp/connect':
                return {'connectionId': 'one'}
            if method == 'mcp/disconnect':
                return {}
            if params['method'] == 'initialize':
                return {'protocolVersion': params['params']['protocolVersion'], 'capabilities': {},
                        'serverInfo': {'name': 'editor', 'version': '1'}}
            calls.append(params['method'])
            raise ConnectionError('lost after dispatch')

        async def send_notification(self, *args):
            pass

    owner = ProjectionMCP(SimpleNamespace(_conn=Peer()))
    spec = owner.bind([{'type': 'acp', 'name': 'editor', 'serverId': 'editor'}])[0]
    async with acp_transport(spec['_acp_ref'], 'editor') as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            with pytest.raises(McpError, match='outcome is unknown'):
                await client.call_tool('write', {})
    assert calls == ['tools/call']
    await owner.aclose()


def test_native_handler_does_not_retry_acp_calls(monkeypatch):
    from tools import mcp_tool_handlers as handlers
    calls, recovered = [], []

    def fail(*args, **kwargs):
        calls.append('sent')
        raise RuntimeError('invalid or expired session')

    monkeypatch.setattr(handlers._loop, '_run_on_mcp_loop', fail)
    result = handlers._dispatch('editor', SimpleNamespace(_config={'transport': 'acp'}),
        'write', None, 30, [lambda *args: recovered.append(True)], lambda exc: None)
    assert calls == ['sent'] and recovered == []
    assert 'invalid or expired session' in result


@pytest.mark.asyncio
async def test_owner_exit_fails_inflight_call_without_replaying():
    from acp_adapter.mcp_tunnel import ProjectionMCP
    entered = asyncio.Event()

    class Peer:
        async def send_request(self, method, params):
            entered.set()
            await asyncio.Event().wait()

    owner = ProjectionMCP(SimpleNamespace(_conn=Peer()))
    call = asyncio.create_task(owner.send('mcp/message', {}))
    await asyncio.wait_for(entered.wait(), 5)
    await owner.aclose()
    with pytest.raises(ConnectionError, match='outcome is unknown'):
        await call


@pytest.mark.asyncio
async def test_sdk_router_accepts_acp_specs_without_global_schema_patch():
    from acp.schema import NewSessionRequest
    from acp_adapter.mcp_tunnel import install_routes
    from pydantic import ValidationError

    seen = []

    class Agent:
        async def new_session(self, cwd, mcp_servers, **kwargs):
            seen.extend(mcp_servers)
            return {'sessionId': 'created'}

    # Exercise the real SDK router, not a replacement parser.
    from acp.agent.router import build_agent_router
    router = build_agent_router(Agent(), use_unstable_protocol=True)
    connection = SimpleNamespace(_conn=SimpleNamespace(_handler=router))
    install_routes(connection, Agent())
    payload = {'cwd': '/workspace', 'mcpServers': [{'type': 'acp', 'name': 'editor', 'serverId': 'editor'}]}
    assert await router('session/new', payload, False) == {'sessionId': 'created'}
    assert seen[0].server_id == 'editor'
    with pytest.raises(ValidationError):
        NewSessionRequest.model_validate(payload)
