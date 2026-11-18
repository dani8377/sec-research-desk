"""Opt-in local Phoenix tracing; no exporter or instrumentation when disabled."""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urlsplit

import httpx

from app.config import Settings

P = ParamSpec("P")
R = TypeVar("R")

_provider: Any = None


def setup_tracing(settings: Settings) -> Any:
    global _provider
    if not settings.phoenix_enabled:
        return None
    if _provider is not None:
        return _provider
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    _provider = register(
        endpoint=settings.phoenix_endpoint,
        project_name="sec-research-desk",
        protocol="http/protobuf",
        batch=True,
        headers={},
        set_global_tracer_provider=False,
        auto_instrument=False,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    return _provider


def shutdown_tracing() -> None:
    global _provider
    if _provider is not None:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().uninstrument()
        _provider.force_flush()
        _provider.shutdown()
        _provider = None


@contextmanager
def trace_span(name: str, **attributes: str) -> Iterator[Any]:
    if _provider is None:
        yield None
    else:
        attributes = {
            "openinference.span.kind": (
                "EVALUATOR" if name == "evaluation.case" else "CHAIN"
            ),
            **attributes,
        }
        with _provider.get_tracer("sec-research-desk").start_as_current_span(
            name, attributes=attributes
        ) as span:
            yield span


def traced(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with trace_span(name) as span:
                result = function(*args, **kwargs)
                status = (
                    result.get("status")
                    if isinstance(result, dict)
                    else getattr(result, "status", None)
                )
                if span is not None and status == "failed":
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(
                        Status(StatusCode.ERROR, "Research workflow failed")
                    )
                return result

        return wrapped

    return decorate


METRIC_EXPLANATIONS = {
    "execution_success": "A validated result was returned; not answer correctness.",
    "status_match": "Expected answered/insufficient-evidence status only.",
    "answer_terms_match": "Reference-term presence only; not semantic correctness.",
    "citation_validity": "Evidence identity and source metadata, not claim entailment.",
    "required_tools_present": "Required tool attempts, including failed attempts.",
    "cited_reference_overlap": "Cited-reference substring overlap; lexical proxy.",
    "retrieval_reference_hit": "Retrieved-reference substring overlap; lexical proxy.",
    "numerical_accuracy": "Expected numeric token only; not calculation accuracy.",
}


def score_annotations(rows: list[dict]) -> list[dict]:
    annotations = []
    for row in rows:
        if not row.get("span_id"):
            continue
        for name, value in {
            "execution_success": row["execution_success"],
            **row["metrics"],
        }.items():
            if value is None:
                continue
            human = name in {"semantic_correctness", "unsupported_claim_rate"}
            review = row.get("review", {})
            if human and not review.get("reviewer"):
                continue
            annotations.append(
                {
                    "span_id": row["span_id"],
                    "name": name,
                    "annotator_kind": "HUMAN" if human else "CODE",
                    "identifier": review["reviewer"] if human else "sec-research-desk",
                    "result": {
                        "score": float(value),
                        "label": ("pass" if value else "fail")
                        if isinstance(value, bool)
                        else str(value),
                        "explanation": review.get("notes")
                        if human
                        else METRIC_EXPLANATIONS.get(name, "Deterministic check."),
                    },
                    "metadata": {
                        "case_id": row["case_id"],
                        "workflow": row["workflow"],
                    },
                }
            )
    return annotations


def export_scores(rows: list[dict], settings: Settings) -> dict:
    """Export code checks and explicit human reviews through the local Phoenix API."""
    data = score_annotations(rows)
    if not settings.phoenix_enabled or not data:
        return {"exported": 0, "error": None}
    endpoint = urlsplit(settings.phoenix_endpoint)
    url = f"{endpoint.scheme}://{endpoint.netloc}/v1/span_annotations"
    exported = 0
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for start in range(0, len(data), 100):
                batch = data[start : start + 100]
                for attempt in range(5):
                    response = client.post(url, json={"data": batch})
                    # Exported spans may still be in Phoenix's ingestion queue.
                    if response.status_code == 404 and attempt < 4:
                        time.sleep(0.25 * 2**attempt)
                        continue
                    response.raise_for_status()
                    break
                exported += len(batch)
    except httpx.HTTPError as exc:
        # Keep the result file useful even if the optional collector is unavailable.
        error = f"Phoenix annotation export failed ({type(exc).__name__})"
        logging.getLogger(__name__).warning(error)
        return {"exported": exported, "error": error}
    return {"exported": exported, "error": None}
