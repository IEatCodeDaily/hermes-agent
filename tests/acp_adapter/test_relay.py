"""Restart safety: never replay an admitted prompt or an old permission answer."""
import json
import os
import socket

from acp_adapter.relay import Relay, _frame


def test_reconnect_keeps_unsent_work_but_fails_sent_work():
    read, write = os.pipe()
    left, right = socket.socketpair()
    relay = Relay('unused', left, stdin_fd=read)
    shown = []
    relay._to_editor = lambda data: shown.append(json.loads(data))
    replacement = peer = None
    try:
        relay.attach(left)
        right.recv(65536)
        relay.on_editor_line(_frame({'id': 1, 'method': 'initialize', 'params': {'protocolVersion': 1}}))
        right.recv(65536)
        relay.on_gateway_line(_frame({'id': 1, 'result': {}}))
        relay._track('session/new', {'cwd': '/work'}, {'sessionId': 's'})
        relay.on_editor_line(_frame({'id': 2, 'method': 'session/prompt', 'params': {'sessionId': 's'}}))
        right.recv(65536)
        relay.on_gateway_line(_frame({'id': 1, 'method': 'session/request_permission', 'params': {}}))
        old_card = shown[-1]['id']
        relay.lost()
        assert shown[-1]['id'] == 2 and 'outcome is unknown' in shown[-1]['error']['message']
        relay.on_editor_line(_frame({'id': 3, 'method': 'session/prompt', 'params': {'sessionId': 's'}}))
        replacement, peer = socket.socketpair()
        peer.settimeout(5)
        relay.attach(replacement)
        replay = [json.loads(line) for line in peer.recv(65536).splitlines()]
        assert [x.get('method') for x in replay] == [None, 'initialize']
        relay.on_editor_line(_frame({'id': old_card, 'result': {'outcome': 'selected'}}))
        relay.on_gateway_line(_frame({'id': 'hermes-relay:init', 'result': {}}))
        resume = json.loads(peer.recv(65536))
        assert resume['method'] == 'session/resume'
        assert resume['params']['_meta']['hermesRelayReattach'] is True
        relay.on_gateway_line(_frame({'id': resume['id'], 'result': {}}))
        queued = [json.loads(line) for line in peer.recv(65536).splitlines()]
        assert [x['id'] for x in queued] == [3]
    finally:
        relay.selector.close()
        for sock in [left, right, replacement, peer]:
            if sock is not None:
                sock.close()
        os.close(read)
        os.close(write)
