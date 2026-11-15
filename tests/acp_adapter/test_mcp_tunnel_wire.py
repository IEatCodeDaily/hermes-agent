"""Hosted ACP wire -> native MCP client, with two isolated editor authorities."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest

from acp_adapter.hosted import HostedACPServer
from acp_adapter.relay import HEADER_VERSION



@pytest.fixture
def socket_home():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory(prefix='acp-') as root:
        yield Path(root)


@pytest.fixture
def tmp_path(socket_home):
    # The real daemon's Unix socket must fit sun_path as well as the unit listener.
    return socket_home


@pytest.mark.platforms('posix')
@pytest.mark.asyncio
async def test_daemon_create_prompt_resume_fork_keep_native_mcp_scope(daemon, tmp_path, model_peer):
    from tests.acp_adapter.test_gateway_editor_mcp import MCPModelPeer
    from tests.acp_adapter.test_gateway_sessions import viewer
    import sqlite3
    import psutil

    model_peer.RequestHandlerClass = MCPModelPeer
    home = daemon[0]
    spec = [{'type': 'acp', 'name': 'editor', 'serverId': 'editor'}]
    async with peer(home / 'acp.sock', tmp_path, 'OWNED_RESULT_FIRST') as (rpc, calls):
        assert 'result' in await rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await rpc('session/new', cwd=str(tmp_path), mcpServers=spec)
        assert 'result' in created, created
        sid = created['result']['sessionId']
        prompt = await rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'Use owned echo'}])
        assert 'result' in prompt, prompt
        assert any(p.get('method') == 'tools/call' for m, p in calls if m == 'mcp/message'), json.dumps({
            'calls': calls, 'tool_results': [m for r in model_peer.requests for m in r.get('messages', []) if m['role'] == 'tool']})
        print('FULL_DAEMON_TUNNEL_RECEIPT', sid, len(calls))
        async with viewer(daemon) as ws:
            snapshot = await ws.rpc('session.resume', session_id=sid)
            assert 'OWNED_RESULT_FIRST' in json.dumps(snapshot['messages'])
        fork = await rpc('session/fork', cwd=str(tmp_path), sessionId=sid, mcpServers=spec)
        assert 'result' in fork, fork
        assert fork['result']['sessionId'] != sid
        async with peer(home / 'acp.sock', tmp_path, 'FOREIGN') as (other, _):
            await other('initialize', protocolVersion=1, clientCapabilities={})
            refused = await other('session/load', cwd=str(tmp_path), sessionId=sid, mcpServers=spec)
            assert 'error' in refused, refused
        assert not psutil.Process(daemon[1]['pid']).children(), 'MCP tunnel spawned a helper'
    async with peer(home / 'acp.sock', tmp_path, 'OWNED_RESULT_SECOND') as (rpc, calls):
        await rpc('initialize', protocolVersion=1, clientCapabilities={})
        resumed = await rpc('session/load', cwd=str(tmp_path), sessionId=sid, mcpServers=spec)
        assert 'result' in resumed, resumed
        assert 'result' in await rpc('session/prompt', sessionId=sid,
                                    prompt=[{'type': 'text', 'text': 'Use owned echo again'}])
        assert any(p.get('method') == 'tools/call' for m, p in calls if m == 'mcp/message')
    with sqlite3.connect(home / 'state.db') as db:
        assert '_acp_ref' not in '\n'.join(db.iterdump())



@asynccontextmanager
async def peer(path, cwd, label):
    reader, writer = await asyncio.open_unix_connection(str(path))
    pending, calls = {}, []
    connections = set()
    serial = 0
    writer.write((json.dumps({'hermes_acp_relay': HEADER_VERSION, 'env': {}, 'cwd': str(cwd)}) + '\n').encode())

    async def listen():
        while line := await reader.readline():
            frame = json.loads(line)
            if 'method' not in frame:
                pending.pop(frame['id']).set_result(frame)
                continue
            method, params = frame['method'], frame.get('params', {})
            calls.append((method, params))
            if 'id' not in frame:
                continue
            if method == 'session/request_permission':
                result = {'outcome': {'outcome': 'selected', 'optionId': 'allow_once'}}
            elif method == 'mcp/connect':
                result = {'connectionId': label + '-connection-' + str(len(calls))}
                connections.add(result['connectionId'])
            elif method == 'mcp/disconnect':
                result = {}
            else:
                assert method == 'mcp/message'
                assert params['connectionId'] in connections
                inner = params['method']
                if inner == 'initialize':
                    result = {'protocolVersion': params['params']['protocolVersion'], 'capabilities': {'tools': {}},
                              'serverInfo': {'name': label, 'version': '1'}}
                elif inner == 'tools/list':
                    result = {'tools': [{'name': 'echo', 'description': 'owned echo', 'inputSchema': {'type': 'object'}}]}
                elif inner == 'tools/call':
                    result = {'content': [{'type': 'text', 'text': label}]}
                else:
                    result = {}
            writer.write((json.dumps({'jsonrpc': '2.0', 'id': frame['id'], 'result': result}) + '\n').encode())
            await writer.drain()

    task = asyncio.create_task(listen())

    async def rpc(method, **params):
        nonlocal serial
        serial += 1
        # String IDs exercise bidirectional numeric request-ID collision safety.
        ident = 'editor-' + str(serial)
        future = pending[ident] = asyncio.get_running_loop().create_future()
        writer.write((json.dumps({'jsonrpc': '2.0', 'id': ident, 'method': method, 'params': params}) + '\n').encode())
        await writer.drain()
        return await asyncio.wait_for(future, 40)

    try:
        yield rpc, calls
    finally:
        writer.close()
        await writer.wait_closed()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms('posix')
@pytest.mark.asyncio
async def test_hosted_wire_discovers_and_calls_without_a_bridge(socket_home, monkeypatch):
    from acp_adapter import auth
    from acp_adapter.gateway_server import GatewayACPAgent
    from gateway.session_local_mcp import _configs, fingerprint, validate_servers
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_loop import _run_on_mcp_loop, _ensure_mcp_loop
    _ensure_mcp_loop()
    from tools.registry import session_tool_scope

    monkeypatch.setattr(auth, 'build_auth_methods', list)
    observations = []

    async def new_session(agent, cwd, mcp_servers, **kwargs):
        specs = agent._mcp_specs(mcp_servers)
        validate_servers(specs)
        config = _configs(specs, {0: 'wire-editor'}, cwd)['wire-editor']
        server = MCPServerTask('wire-editor')
        scope = 'wire-' + specs[0]['_acp_ref']
        try:
            # The same separate-loop native MCP task discovery and tool call as
            # gateway policy binding, not an alternate registry/client path.
            def exercise():
                with session_tool_scope(scope):
                    async def run():
                        await server.start(config)
                        assert server._tools[0].name == 'echo'
                        return await server.session.call_tool('echo', {})
                    return _run_on_mcp_loop(run, timeout=30)
            result = await asyncio.to_thread(exercise)
            observations.append((fingerprint(specs), result.content[0].text))
            return {'sessionId': result.content[0].text}
        finally:
            await asyncio.to_thread(lambda: _run_on_mcp_loop(server.shutdown, timeout=10))

    monkeypatch.setattr(GatewayACPAgent, 'new_session', new_session)
    server = HostedACPServer(SimpleNamespace(), socket_home)
    assert await server.start()
    try:
        async with peer(server.path, socket_home, 'first') as (one, first_calls):
            async with peer(server.path, socket_home, 'second') as (two, second_calls):
                for rpc in (one, two):
                    init = await rpc('initialize', protocolVersion=1, clientCapabilities={})
                    assert init['result']['agentCapabilities']['mcpCapabilities']['acp'] is True
                spec = [{'type': 'acp', 'name': 'editor', 'serverId': 'editor'}]
                results = await asyncio.gather(*(rpc('session/new', cwd=str(socket_home), mcpServers=spec)
                                                 for rpc in (one, two)))
                assert all('result' in r for r in results), results
                assert [r['result']['sessionId'] for r in results] == ['first', 'second'], results
                assert len(first_calls) and len(second_calls)  # each peer served its own native client
                assert observations[0][0] == observations[1][0]
                for calls in (first_calls, second_calls):
                    assert [m for m, _ in calls].count('mcp/connect') == 1
                    assert [m for m, _ in calls].count('mcp/disconnect') == 1
    finally:
        await server.stop()
