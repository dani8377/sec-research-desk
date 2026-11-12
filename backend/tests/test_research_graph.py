"""Graph behavior and checkpoint recovery using scripted models and services."""

import json
import os
from unittest.mock import Mock
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from psycopg import sql
from psycopg.conninfo import make_conninfo
from test_research import call, draft
from test_research import evidence as evidence

from app.config import Settings
from app.main import create_app
from app.rag_embeddings import RAGError
from app.rag_models import Evidence, SearchResult
from app.research_graph import GraphResearchService
from app.research_models import GraphResearchRequest, ResearchPlan
from app.research_routes import get_graph_service
from app.research_service import ResearchError


@pytest.fixture
def graph_setup(monkeypatch: pytest.MonkeyPatch, evidence: Evidence) -> tuple:
    sec, rag, model = Mock(), Mock(), Mock()
    rag.search.return_value = SearchResult(embedding_model="test", evidence=[evidence])
    planner, synthesizer = Mock(), Mock()
    planner.invoke.return_value = ResearchPlan(
        scope="filing_research", objectives=["Find reported risks"], limitation=None
    )
    synthesizer.invoke.return_value = draft()
    model.with_structured_output.side_effect = lambda schema, **kw: (
        planner if schema is ResearchPlan else synthesizer
    )
    model.bind_tools.return_value.invoke.side_effect = [
        call(),
        AIMessage(content="done"),
    ]
    monkeypatch.setattr("app.research_graph.chat_model", lambda settings: model)
    service = GraphResearchService(
        sec, rag, Settings(_env_file=None), checkpointer=InMemorySaver()
    )
    return service, sec, rag, model, planner, synthesizer


def request() -> GraphResearchRequest:
    return GraphResearchRequest(question="What risks does company 1 report?")


def test_graph_citations_and_checkpointed_state(
    graph_setup: tuple, evidence: Evidence
) -> None:
    service, _, rag, _, _, synthesizer = graph_setup
    query = request()
    result = service.run(query)
    assert result.status == "completed" and result.errors == []
    assert result.result.evidence == [evidence]
    assert result.model_calls == 4
    assert result.tools_called == ["search_filing_sections"]
    assert result.next_nodes == []
    assert rag.search.call_args.args[0].cik == "0000000001"
    synthesis = json.loads(synthesizer.invoke.call_args.args[0][1].content)
    assert synthesis["question"] == query.question
    assert synthesis["evidence"][0]["text"] == evidence.chunk.text
    graph = service.build_graph(service.checkpointer)
    history = list(graph.get_state_history(service._config(query.run_id)))
    assert len(history) >= 7
    for checkpoint in history:
        json.dumps(checkpoint.values)  # no opaque model objects in graph state
    # A new service object can inspect the saved run without another model call.
    restored = GraphResearchService(
        service.sec, service.rag, service.settings, checkpointer=service.checkpointer
    )
    assert restored.get_run(query.run_id) == result
    assert restored.get_run(query.run_id, resume=True) == result
    rag.search.assert_called_once()
    with pytest.raises(ResearchError, match="already exists"):
        service.run(query)


def test_unsupported_route(graph_setup: tuple) -> None:
    service, _, rag, model, planner, synthesizer = graph_setup
    planner.invoke.return_value = ResearchPlan(
        scope="unsupported",
        objectives=["Calculate revenue growth"],
        limitation="Financial calculation tools are unavailable.",
    )
    result = service.run(GraphResearchRequest(question="Calculate revenue growth"))
    assert result.status == "completed"
    assert result.result.status == "insufficient_evidence"
    assert result.model_calls == 1
    assert result.result.claims == []
    assert result.tools_called == []
    rag.search.assert_not_called()
    model.bind_tools.assert_not_called()
    synthesizer.invoke.assert_not_called()


def test_empty_evidence_route(graph_setup: tuple) -> None:
    service, _, rag, _, _, synthesizer = graph_setup
    rag.search.return_value = SearchResult(embedding_model="test", evidence=[])
    result = service.run(request())
    assert result.result.status == "insufficient_evidence"
    assert result.result.evidence == []
    assert result.model_calls == 3
    synthesizer.invoke.assert_not_called()


def test_fabricated_citation_is_saved_failure(graph_setup: tuple) -> None:
    service, _, _, _, _, synthesizer = graph_setup
    synthesizer.invoke.return_value = draft("invented")
    result = service.run(request())
    assert result.status == "failed" and result.result is None
    assert result.errors[0].stage == "validate"
    assert "not retrieved" in result.errors[0].message
    assert service.get_run(result.run_id) == result
    assert service.get_run(result.run_id, resume=True) == result


@pytest.mark.parametrize("stage", ["plan", "select_tools", "synthesize"])
def test_model_failures_are_explicit(graph_setup: tuple, stage: str) -> None:
    service, _, _, model, planner, synthesizer = graph_setup
    runnable = {
        "plan": planner,
        "select_tools": model.bind_tools.return_value,
        "synthesize": synthesizer,
    }[stage]
    runnable.invoke.side_effect = ValueError("model unavailable")
    result = service.run(request())
    assert result.status == "failed" and result.result is None
    assert result.errors[0].stage == stage
    assert result.errors[0].status_code == 503


def test_tool_failure_is_never_empty_evidence(graph_setup: tuple) -> None:
    service, _, rag, _, _, synthesizer = graph_setup
    rag.search.side_effect = RAGError("Database unavailable", 503)
    result = service.run(request())
    assert result.status == "failed" and result.result is None
    assert result.tools_called == ["search_filing_sections"]
    assert result.errors[0].stage == "execute_tool"
    assert result.errors[0].message == "Database unavailable"
    synthesizer.invoke.assert_not_called()


def test_invalid_input_can_be_corrected_once(graph_setup: tuple) -> None:
    service, _, rag, model, _, _ = graph_setup
    model.bind_tools.return_value.invoke.side_effect = [
        call(cik="1", section="Item 1A"),
        call(),
        AIMessage(content="done"),
    ]
    result = service.run(request())
    assert result.status == "completed"
    assert result.tools_called == ["search_filing_sections"] * 2
    assert len(result.result.tool_errors) == 1
    assert "query" in result.result.tool_errors[0]
    rag.search.assert_called_once()


def test_repeated_invalid_inputs_fail(graph_setup: tuple) -> None:
    service, _, rag, model, _, _ = graph_setup
    model.bind_tools.return_value.invoke.side_effect = [
        call(cik="1", section="Item 1A"),
        call(cik="1", section="Item 1A"),
    ]
    result = service.run(request())
    assert result.status == "failed"
    assert len(result.tools_called) == 2
    assert "repeatedly" in result.errors[0].message
    rag.search.assert_not_called()


@pytest.mark.parametrize(
    "reply",
    [
        call("unknown"),
        AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "find_company",
                    "args": "{",
                    "id": "broken",
                    "error": "invalid",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "find_company", "args": {"query": "test"}, "id": ""}],
        ),
    ],
)
def test_invalid_tool_selection_fails(graph_setup: tuple, reply: AIMessage) -> None:
    service, _, rag, model, _, _ = graph_setup
    model.bind_tools.return_value.invoke.side_effect = [reply]
    result = service.run(request())
    assert result.status == "failed"
    assert result.errors[0].stage == "select_tools"
    rag.search.assert_not_called()


def test_tool_budget_is_bounded(graph_setup: tuple) -> None:
    service, _, rag, model, _, _ = graph_setup
    service.settings.research_max_tool_calls = 2
    model.bind_tools.return_value.invoke.side_effect = [call(), call(), call()]
    result = service.run(request())
    assert result.status == "failed" and "tool-call limit" in result.errors[0].message
    assert rag.search.call_count == 2


def test_context_budget_is_bounded(graph_setup: tuple, evidence: Evidence) -> None:
    service, _, rag, _, _, synthesizer = graph_setup
    service.settings.research_max_context_chars = 4000
    chunk = evidence.chunk.model_copy(update={"text": "x" * 5000, "char_end": 5000})
    rag.search.return_value = SearchResult(
        embedding_model="test", evidence=[evidence.model_copy(update={"chunk": chunk})]
    )
    result = service.run(request())
    assert result.status == "failed" and "context limit" in result.errors[0].message
    assert result.model_calls == 2  # the rejected next request never reaches Ollama
    rag.search.assert_called_once()
    synthesizer.invoke.assert_not_called()


def test_crash_resumes_only_pending_tool(
    graph_setup: tuple, evidence: Evidence
) -> None:
    service, _, rag, model, _, _ = graph_setup
    first = call().tool_calls[0]
    second = {**first, "id": "call-2", "args": {"cik": "1", "query": "facilities"}}
    model.bind_tools.return_value.invoke.side_effect = [
        AIMessage(content="", tool_calls=[first, second]),
        AIMessage(content="done"),
    ]
    # An unexpected interruption escapes; prior completed nodes remain durable.
    rag.search.side_effect = [
        SearchResult(embedding_model="test", evidence=[evidence]),
        RuntimeError("simulated process interruption"),
        SearchResult(embedding_model="test", evidence=[evidence]),
    ]
    query = request()
    with pytest.raises(RuntimeError, match="interruption"):
        service.run(query)
    saved = service.get_run(query.run_id)
    assert saved.status == "running"
    assert saved.next_nodes == ["execute_tool"]
    assert saved.tools_called == ["search_filing_sections"]
    restored = GraphResearchService(
        service.sec, service.rag, service.settings, checkpointer=service.checkpointer
    )
    result = restored.get_run(query.run_id, resume=True)
    assert result.status == "completed"
    assert result.tools_called == ["search_filing_sections"] * 2
    assert rag.search.call_count == 3
    assert rag.search.call_args_list[0].args[0].query == "risks"
    assert all(c.args[0].query == "facilities" for c in rag.search.call_args_list[1:])


def test_missing_run_and_isolation(graph_setup: tuple) -> None:
    service, *_ = graph_setup
    with pytest.raises(ResearchError) as error:
        service.get_run(uuid4())
    assert error.value.status_code == 404
    result = service.run(request())
    with pytest.raises(ResearchError):
        service.get_run(uuid4(), resume=True)
    assert service.get_run(result.run_id) == result


def test_api(graph_setup: tuple) -> None:
    service, *_ = graph_setup
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_graph_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/research/graph", json={"question": "risks"})
        assert response.status_code == 200
        run = response.json()
        assert run["status"] == "completed"
        path = f"/research/graph/{run['run_id']}"
        assert client.get(path).json() == run
        assert client.post(path + "/resume").json() == run
        assert client.get("/research/graph/invalid").status_code == 422
        assert client.get(f"/research/graph/{uuid4()}").status_code == 404
        assert client.post("/research/graph", json={"question": " "}).status_code == 422
        assert (
            client.post(
                "/research/graph", json={"question": "risks", "run_id": run["run_id"]}
            ).status_code
            == 409
        )


def test_checkpoint_database_failure(
    graph_setup: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, *_ = graph_setup
    service.checkpointer = None
    monkeypatch.setattr(
        "app.research_graph.psycopg.connect",
        Mock(side_effect=psycopg.OperationalError("private connection details")),
    )
    with pytest.raises(ResearchError) as error:
        service.run(request())
    assert error.value.status_code == 503
    assert "private" not in str(error.value)


@pytest.mark.parametrize("interrupted", [False, True])
def test_postgres_checkpoint_survives_connection_restart(
    graph_setup: tuple, interrupted: bool
) -> None:
    dsn = os.environ.get("SEC_RESEARCH_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set SEC_RESEARCH_TEST_DATABASE_URL for PostgreSQL checkpoints")
    service, _, rag, model, _, _ = graph_setup
    schema = "graph_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        service.settings.database_url = Settings(
            _env_file=None,
            database_url=make_conninfo(dsn, options=f"-c search_path={schema},public"),
        ).database_url
        service.checkpointer = None
        query = request()
        if interrupted:
            rag.search.side_effect = [
                RuntimeError("interruption"),
                rag.search.return_value,
            ]
            with pytest.raises(RuntimeError, match="interruption"):
                service.run(query)
        else:
            result = service.run(query)
            assert result.status == "completed"
        restored = GraphResearchService(service.sec, service.rag, service.settings)
        if interrupted:
            assert restored.get_run(query.run_id).next_nodes == ["execute_tool"]
            result = restored.get_run(query.run_id, resume=True)
            assert result.status == "completed"
            assert model.bind_tools.return_value.invoke.call_count == 2
        else:
            assert restored.get_run(query.run_id) == result
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


@pytest.mark.skipif(
    os.environ.get("SEC_RESEARCH_TEST_CHAT") != "1", reason="Opt-in local Ollama test"
)
def test_real_ollama_graph(evidence: Evidence) -> None:
    # Real planning, tools and synthesis, with synthetic evidence and no SEC traffic.
    sec, rag = Mock(), Mock()
    rag.search.return_value = SearchResult(embedding_model="test", evidence=[evidence])
    service = GraphResearchService(sec, rag, Settings(), checkpointer=InMemorySaver())
    result = service.run(
        GraphResearchRequest(
            question=(
                "Use search_filing_sections for CIK 1. What does the retrieved filing "
                "explicitly say may affect operations? Cite the passage."
            )
        )
    )
    assert result.status == "completed", result.errors
    assert result.result.status == "answered"
    assert result.result.evidence == [evidence]
    assert "search_filing_sections" in result.tools_called
