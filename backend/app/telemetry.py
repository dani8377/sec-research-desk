"""Opt-in local Phoenix tracing; no exporter or instrumentation when disabled."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

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
