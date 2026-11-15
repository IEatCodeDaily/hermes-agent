"""Tests for the monitoring emitter: hot-path invariant + subscriber fan-out."""

from __future__ import annotations

from agent.monitoring.emitter import MonitoringEmitter

def test_emit_never_raises_when_disabled():
    em = MonitoringEmitter(enabled=False)
    em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    assert em.stats()["queued"] == 0
    em.close()

def test_process_singleton_stays_dormant_until_subscribed():
    from agent.monitoring import emitter

    emitter.reset_emitter_for_tests()
    try:
        emitter.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
        singleton = emitter.get_emitter()
        assert singleton.stats()["queued"] == 0
        assert singleton._started is False

        subscriber = lambda _batch: None
        singleton.subscribe(subscriber)
        emitter.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
        assert singleton._started is True
        singleton.unsubscribe(subscriber)
    finally:
        emitter.reset_emitter_for_tests()

def test_unsubscribe_stops_delivery():
    em = MonitoringEmitter()
    seen: list = []
    cb = lambda batch: seen.extend(batch)
    em.subscribe(cb)
    em.emit({"event": "gateway_health", "name": "a"})
    em.flush()
    em.unsubscribe(cb)
    em.emit({"event": "gateway_health", "name": "b"})
    em.flush()
    em.close()
    assert [ev["name"] for ev in seen] == ["a"]


def test_replacement_subscription_never_receives_old_queued_or_captured_work():
    import threading

    em = MonitoringEmitter()
    entered, release = threading.Event(), threading.Event()
    old, new, health = [], [], []

    def stalled(batch):
        entered.set()
        assert release.wait(5)

    previous = lambda batch: old.extend(batch)
    replacement = lambda batch: new.extend(batch)
    em.subscribe(stalled)
    em.subscribe(previous)
    try:
        em.emit({"name": "captured"})
        assert entered.wait(2)
        em.emit({"name": "queued"})
        em.unsubscribe(previous)
        em.subscribe(replacement)
        em.subscribe(lambda batch: health.extend(batch))
        release.set()
        em.flush()
        assert not old and not new and not health
        em.emit({"name": "fresh"})
        em.flush()
        assert [item["name"] for item in new] == ["fresh"]
        assert [item["name"] for item in health] == ["fresh"]
    finally:
        release.set()
        em.close()


def test_last_detach_reuses_bus_without_replaying_old_generation():
    import threading

    em = MonitoringEmitter()
    entered, release = threading.Event(), threading.Event()
    seen = []

    def previous(batch):
        entered.set()
        assert release.wait(5)

    em.subscribe(previous)
    try:
        em.emit({"name": "inflight"})
        assert entered.wait(2)
        em.emit({"name": "old"})
        em.unsubscribe(previous)
        em.subscribe(lambda batch: seen.extend(batch))
        release.set()
        em.emit({"name": "new"})
        em.flush()
        assert [item["name"] for item in seen] == ["new"]
        assert em._thread.is_alive()
    finally:
        release.set()
        em.close()
