"""OTLP exporter tests: config resolution, span mapping, streaming subscriber.

No SQLite involved — monitoring is an egress path, so the exporter consumes
emitter batches directly. Uses the in-memory OTel span exporter; skipped when
the optional otlp extra is not installed.
"""

from __future__ import annotations

import pytest

otel = pytest.importorskip("opentelemetry.sdk.trace", reason="otlp extra not installed")

import agent.monitoring.otlp_exporter as OE
from agent.monitoring.emitter import MonitoringEmitter


def _mem_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_gateway_health_event_maps_to_span_with_attrs():
    provider, mem = _mem_provider()
    n = OE.export_batch(provider, [{
        "event": "gateway_health", "name": "gateway.lifecycle",
        "old_state": "starting", "new_state": "running",
        "active_agents": 2, "pid": 4242,
    }])
    assert n == 1
    spans = mem.get_finished_spans()
    assert spans[0].name == "hermes.gateway_health"
    attrs = dict(spans[0].attributes or {})
    assert attrs["hermes.old_state"] == "starting"
    assert attrs["hermes.new_state"] == "running"
    assert attrs["hermes.active_agents"] == 2






def test_headers_resolve_from_env_not_value(monkeypatch):
    monkeypatch.setenv("DD_KEY_ENV", "secret-value")
    resolved = OE._resolve_headers({"DD-API-KEY": "DD_KEY_ENV", "X-Missing": "NOPE_ENV"})
    assert resolved == {"DD-API-KEY": "secret-value"}




def test_trace_resource_includes_stable_hashed_instance():
    attrs = OE._resource_attributes(
        {"monitoring": {"install_id": "private-install-id"}}
    )

    assert attrs["service.name"] == "hermes-gateway"
    assert attrs["service.instance.id"].startswith("sha256:")
    assert len(attrs["service.instance.id"]) == len("sha256:") + 24
    assert "private-install-id" not in str(attrs)
    assert attrs["telemetry.scope"] == "gateway_monitoring"


def test_trace_resource_includes_configured_deployment_environment():
    attrs = OE._resource_attributes({
        "monitoring": {
            "install_id": "private-install-id",
            "gateway_health_export": {
                "resource_attributes": {"deployment.environment.name": "production"},
            },
        },
    })

    assert attrs["deployment.environment.name"] == "production"
    assert attrs["service.name"] == "hermes-gateway"




def test_streamer_receives_events_and_respects_filter(monkeypatch):
    provider, mem = _mem_provider()
    monkeypatch.setattr(OE, "_make_provider", lambda cfg: (provider, None))
    streamer = OE.OTLPStreamer(
        {}, event_filter=lambda ev: ev.get("event") == "gateway_health")

    em = MonitoringEmitter()
    em.subscribe(streamer)
    em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    em.emit({"event": "model_call", "provider": "anthropic"})  # filtered out
    em.flush()
    em.close()

    spans = mem.get_finished_spans()
    assert [s.name for s in spans] == ["hermes.gateway_health"]
    assert streamer.exported == 1


@pytest.mark.parametrize("fails", [False, True])
def test_shutdown_overlapping_roots_retires_exporter_after_processor_noop(fails):
    from types import SimpleNamespace

    calls = []
    stopped = False

    def exporter_shutdown():
        calls.append("exporter")

    exporter = SimpleNamespace(shutdown=exporter_shutdown)

    def processor_shutdown():
        nonlocal stopped
        if stopped:
            return
        stopped = True
        calls.append("processor")
        if fails:
            raise RuntimeError("stopped before exporter shutdown")
        exporter.shutdown()

    processor = SimpleNamespace(shutdown=processor_shutdown)

    def provider_shutdown():
        calls.append("provider")
        processor.shutdown()

    provider = SimpleNamespace(shutdown=provider_shutdown)
    assert OE._shutdown_resources([[provider], [processor, exporter]]) is not fails
    assert calls == ["provider", "processor", "exporter"]


def test_factory_success_preserves_health_provider_processor_tuple(monkeypatch):
    from types import SimpleNamespace

    exporter = object()
    processor = object()
    attached = []
    provider = SimpleNamespace(add_span_processor=attached.append)

    def make_processor(sink):
        assert sink is exporter
        return processor

    sdk = {"TracerProvider": lambda **kwargs: provider,
           "BatchSpanProcessor": make_processor,
           "Resource": SimpleNamespace(create=lambda attrs: attrs)}
    monkeypatch.setattr(OE, "_require_sdk", lambda: sdk)
    monkeypatch.setattr(OE, "_resource_attributes", lambda cfg: {})
    monkeypatch.setattr(OE, "build_exporter", lambda cfg: exporter)
    assert OE._make_provider({}) == (provider, processor)
    assert attached == [processor]


@pytest.mark.parametrize("stage", ["exporter", "processor", "attach", "attached"])
def test_factory_failure_keeps_legacy_signature_and_cleans_up_off_loop(monkeypatch, stage):
    import threading
    from types import SimpleNamespace

    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    made, closed, threads = [], [], []

    class Exporter:
        def shutdown(self):
            closed.append("exporter")
            threads.append(threading.get_ident())

    class Processor:
        def __init__(self, exporter):
            if stage == "processor":
                raise RuntimeError("factory failed")
            self.exporter = exporter
            self.stopped = False
            made.append("processor")

        def shutdown(self):
            if not self.stopped:
                self.stopped = True
                closed.append("processor")
                self.exporter.shutdown()

    class Provider:
        def __init__(self, **kwargs):
            self.processors = []
            made.append("provider")

        def add_span_processor(self, processor):
            if stage == "attached":
                self.processors.append(processor)
            raise RuntimeError("factory failed")

        def shutdown(self):
            closed.append("provider")
            threads.append(threading.get_ident())
            entered.set()
            assert release.wait(10)
            for processor in self.processors:
                processor.shutdown()

    def exporter(config):
        if stage == "exporter":
            raise RuntimeError("factory failed")
        made.append("exporter")
        return Exporter()

    sdk = {"TracerProvider": Provider, "BatchSpanProcessor": Processor,
           "Resource": SimpleNamespace(create=lambda attrs: attrs)}
    monkeypatch.setattr(OE, "_require_sdk", lambda: sdk)
    monkeypatch.setattr(OE, "_resource_attributes", lambda cfg: {})
    monkeypatch.setattr(OE, "build_exporter", exporter)
    real_cleanup = OE._shutdown_resources
    finished = threading.Event()

    def cleanup(resources):
        try:
            return real_cleanup(resources)
        finally:
            finished.set()

    monkeypatch.setattr(OE, "_shutdown_resources", cleanup)
    try:
        # The health-streamer caller still passes only config and receives
        # the original construction exception, without waiting on SDK shutdown.
        with pytest.raises(RuntimeError, match="factory failed"):
            OE.OTLPStreamer({})
        assert entered.wait(2)
        assert not finished.is_set()
        release.set()
        assert finished.wait(2)
        assert set(closed) == set(made)
        assert threads and all(t != owner for t in threads)
    finally:
        release.set()
        assert finished.wait(2)


def test_failing_streamer_never_breaks_emitter(monkeypatch):
    def boom(cfg):
        raise RuntimeError("no provider")

    em = MonitoringEmitter()

    def bad_subscriber(batch):
        raise RuntimeError("export down")

    seen: list = []
    em.subscribe(bad_subscriber)
    em.subscribe(lambda batch: seen.extend(batch))
    em.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
    em.flush()
    em.close()
    assert len(seen) == 1
