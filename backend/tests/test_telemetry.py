"""Tracing is disabled by default; enabled traces use the local SDK/export path."""

from unittest.mock import Mock

import pytest
from langchain_core.tools import StructuredTool
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import ValidationError

from app import telemetry
from app.config import Settings


def test_disabled_tracing_has_no_provider():
    assert telemetry.setup_tracing(Settings(_env_file=None)) is None
    with telemetry.trace_span("disabled") as span:
        assert span is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://app.phoenix.arize.com/v1/traces",
        "http://example.com:6006/v1/traces",
        "http://localhost:6006/v1/traces?redirect=elsewhere",
    ],
)
def test_only_loopback_phoenix_endpoints(endpoint):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, phoenix_endpoint=endpoint)


def test_setup_and_shutdown(monkeypatch):
    import phoenix.otel
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider = Mock()
    register = Mock(return_value=provider)
    instrument = Mock()
    uninstrument = Mock()
    monkeypatch.setattr(phoenix.otel, "register", register)
    monkeypatch.setattr(LangChainInstrumentor, "instrument", instrument)
    monkeypatch.setattr(LangChainInstrumentor, "uninstrument", uninstrument)
    settings = Settings(_env_file=None, phoenix_enabled=True)
    try:
        assert telemetry.setup_tracing(settings) is provider
        assert telemetry.setup_tracing(settings) is provider
        register.assert_called_once()
        assert register.call_args.kwargs["endpoint"] == settings.phoenix_endpoint
        assert register.call_args.kwargs["headers"] == {}
        instrument.assert_called_once_with(tracer_provider=provider)
    finally:
        telemetry.shutdown_tracing()
    provider.force_flush.assert_called_once()
    provider.shutdown.assert_called_once()
    uninstrument.assert_called_once()
    assert telemetry._provider is None


def test_real_sdk_tool_parenting_failures_and_cleanup(monkeypatch):
    from openinference.instrumentation.langchain import LangChainInstrumentor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_provider", provider)
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    def lookup(query: str) -> str:
        """Return test evidence."""
        return query

    @telemetry.traced("graph.test")
    def failed():
        return {"status": "failed"}

    @telemetry.traced("raising")
    def raising():
        raise RuntimeError("Test failure")

    try:
        with telemetry.trace_span("research.test"):
            StructuredTool.from_function(lookup).invoke({"query": "risk"})
            failed()
            with pytest.raises(RuntimeError):
                raising()
        spans = exporter.get_finished_spans()
        root = next(span for span in spans if span.name == "research.test")
        tool = next(span for span in spans if span.name == "lookup")
        assert tool.parent.span_id == root.context.span_id
        assert tool.attributes["openinference.span.kind"] == "TOOL"
        for name in ("graph.test", "raising"):
            span = next(span for span in spans if span.name == name)
            assert span.status.status_code == StatusCode.ERROR
    finally:
        instrumentor.uninstrument()
        provider.shutdown()


def test_score_annotations_keep_human_reviews_distinct():
    row = {
        "span_id": "1" * 16,
        "case_id": "case",
        "workflow": "graph",
        "execution_success": True,
        "metrics": {
            "answer_terms_match": False,
            "semantic_correctness": True,
            "unsupported_claim_rate": None,
        },
    }
    annotations = telemetry.score_annotations([row])
    assert {a["name"] for a in annotations} == {
        "execution_success",
        "answer_terms_match",
    }
    assert all(a["annotator_kind"] == "CODE" for a in annotations)
    row["review"] = {"reviewer": "human tester", "notes": "Checked source passage"}
    annotations = telemetry.score_annotations([row])
    human = next(a for a in annotations if a["name"] == "semantic_correctness")
    assert human["annotator_kind"] == "HUMAN"
    assert human["result"]["explanation"] == "Checked source passage"


def test_annotation_export_handles_ingestion_delay_and_failure(monkeypatch):
    import httpx

    real_client = httpx.Client
    requests = []
    statuses = iter([404, 200])

    def handler(request):
        requests.append(request)
        return httpx.Response(next(statuses))

    monkeypatch.setattr(
        telemetry.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(telemetry.time, "sleep", lambda seconds: None)
    row = {
        "span_id": "1" * 16,
        "case_id": "case",
        "workflow": "baseline",
        "execution_success": False,
        "metrics": {"citation_validity": None},
    }
    settings = Settings(_env_file=None, phoenix_enabled=True)
    assert telemetry.export_scores([row], settings) == {"exported": 1, "error": None}
    assert len(requests) == 2
    assert requests[0].url.path == "/v1/span_annotations"
    monkeypatch.setattr(
        telemetry.httpx,
        "Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503))
        ),
    )
    result = telemetry.export_scores([row], settings)
    assert result["exported"] == 0 and "failed" in result["error"]
