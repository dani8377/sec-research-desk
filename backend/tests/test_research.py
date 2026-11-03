"""Offline baseline tests use synthetic evidence and scripted model responses."""

import json
import os
from datetime import UTC, date, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.rag_embeddings import RAGError
from app.rag_models import Evidence, FilingChunk, SearchResult
from app.research_models import ResearchDraft, ResearchRequest
from app.research_routes import get_research_service
from app.research_service import ResearchError, ResearchService, chat_model
from app.research_tools import research_tools
from app.sec_models import Filing, FilingHistory


@pytest.fixture
def evidence() -> Evidence:
    text = "Supply chain disruption may affect operations."
    return Evidence(
        similarity=0.8,
        chunk=FilingChunk(
            chunk_id="synthetic-evidence",
            filing=Filing(
                cik="0000000001",
                accession_number="0000000001-24-000001",
                filing_date=date(2024, 2, 1),
                report_date=date(2023, 12, 31),
                form="10-K",
                primary_document="synthetic.htm",
                filing_url="https://www.sec.gov/Archives/edgar/data/1/synthetic.htm",
                source_url="https://data.sec.gov/submissions/CIK0000000001.json",
            ),
            retrieved_at=datetime(2024, 2, 1, tzinfo=UTC),
            section="Item 1A. Risk Factors",
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            text=text,
        ),
    )


def draft(key: str = "synthetic-evidence") -> ResearchDraft:
    return ResearchDraft(
        status="answered",
        claims=[
            {
                "text": "Supply chain disruption is a reported risk.",
                "kind": "reported_fact",
                "evidence_ids": [key],
            }
        ],
        limitations=["Synthetic test evidence only."],
    )


def call(name: str = "search_filing_sections", **args: object) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {"cik": "1", "query": "risks"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, evidence: Evidence) -> tuple:
    sec, rag, model = Mock(), Mock(), Mock()
    rag.search.return_value = SearchResult(embedding_model="test", evidence=[evidence])
    model.bind_tools.return_value.invoke.side_effect = [
        call(),
        AIMessage(content="done"),
    ]
    model.with_structured_output.return_value.invoke.return_value = draft()
    monkeypatch.setattr("app.research_service.chat_model", lambda settings: model)
    return ResearchService(sec, rag, Settings(_env_file=None)), sec, rag, model


def test_baseline_citations(setup: tuple, evidence: Evidence) -> None:
    service, _, rag, model = setup
    result = service.research(ResearchRequest(question="What are company 1 risks?"))
    assert result.evidence == [evidence]
    assert result.tools_called == ["search_filing_sections"]
    assert result.model_calls == 3
    assert rag.search.call_args.args[0].cik == "0000000001"
    messages = model.bind_tools.return_value.invoke.call_args.args[0]
    tool_result = next(m for m in messages if isinstance(m, ToolMessage))
    assert "synthetic-evidence" in tool_result.content
    synthesis = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "What are company 1 risks?" in synthesis[1].content
    assert "synthetic-evidence" in synthesis[1].content
    assert len(synthesis) == 2
    source = json.loads(synthesis[1].content)["evidence"][0]
    assert source["evidence_id"] == evidence.chunk.chunk_id
    assert source["text"] == evidence.chunk.text
    assert source["filing"] == evidence.chunk.filing.model_dump(mode="json")
    model.with_structured_output.assert_called_once_with(
        ResearchDraft, method="json_schema"
    )


@pytest.mark.parametrize("key", ["invented", "https://invented.example"])
def test_fabricated_citations_fail(setup: tuple, key: str) -> None:
    service, _, _, model = setup
    model.with_structured_output.return_value.invoke.return_value = draft(key)
    with pytest.raises(ResearchError, match="not retrieved"):
        service.research(ResearchRequest(question="risks"))


def test_empty_retrieval(setup: tuple) -> None:
    service, _, rag, model = setup
    rag.search.return_value = SearchResult(embedding_model="test", evidence=[])
    model.with_structured_output.return_value.invoke.return_value = ResearchDraft(
        status="insufficient_evidence", claims=[], limitations=["No indexed evidence."]
    )
    result = service.research(ResearchRequest(question="risks"))
    assert result.status == "insufficient_evidence" and result.evidence == []


@pytest.mark.parametrize(
    "reply",
    [
        call("unknown_tool"),
        call(cik="invalid", query="risks"),
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
    ],
)
def test_invalid_tools(setup: tuple, reply: AIMessage) -> None:
    service, _, rag, model = setup
    model.bind_tools.return_value.invoke.side_effect = [reply]
    with pytest.raises(ResearchError):
        service.research(ResearchRequest(question="risks"))
    rag.search.assert_not_called()


def test_bounded_calls(setup: tuple) -> None:
    service, _, rag, model = setup
    service.settings.research_max_tool_calls = 2
    model.bind_tools.return_value.invoke.side_effect = [call(), call(), call()]
    with pytest.raises(ResearchError, match="tool-call limit"):
        service.research(ResearchRequest(question="risks"))
    assert rag.search.call_count == 2
    model.with_structured_output.assert_not_called()


def test_tool_failure(setup: tuple) -> None:
    service, _, rag, model = setup
    rag.search.side_effect = RAGError("Database unavailable", 503)
    with pytest.raises(ResearchError, match="Database unavailable") as error:
        service.research(ResearchRequest(question="risks"))
    assert error.value.status_code == 503
    model.with_structured_output.assert_not_called()


@pytest.mark.parametrize("stage", ["tools", "structured"])
def test_model_failure(setup: tuple, stage: str) -> None:
    service, _, _, model = setup
    runnable = (
        model.bind_tools.return_value
        if stage == "tools"
        else model.with_structured_output.return_value
    )
    runnable.invoke.side_effect = ValueError("bad output or connection error")
    with pytest.raises(ResearchError, match="Local chat request"):
        service.research(ResearchRequest(question="risks"))


def test_context_bound(setup: tuple, evidence: Evidence) -> None:
    service, _, rag, model = setup
    service.settings.research_max_context_chars = 4000
    chunk = evidence.chunk.model_copy(update={"text": "x" * 5000, "char_end": 5000})
    rag.search.return_value = SearchResult(
        embedding_model="test", evidence=[evidence.model_copy(update={"chunk": chunk})]
    )
    with pytest.raises(ResearchError, match="context limit"):
        service.research(ResearchRequest(question="risks"))
    assert model.bind_tools.return_value.invoke.call_count == 1


def test_tools_delegate(evidence: Evidence) -> None:
    sec, rag = Mock(), Mock()
    sec.find_company.return_value = []
    sec.get_company_filings.return_value = FilingHistory(
        cik="0000000001",
        company_name="Synthetic",
        includes_historical=False,
        filings=[evidence.chunk.filing] * 3,
    )
    tools = {tool.name: tool for tool in research_tools(sec, rag, {})}
    assert tools["find_company"].invoke({"query": " TEST "}) == {"companies": []}
    sec.find_company.assert_called_once_with("TEST", 5)
    result = tools["get_company_filings"].invoke({"cik": "1", "limit": 1})
    assert len(result["filings"]) == 1
    sec.get_company_filings.assert_called_once_with("0000000001", form="10-K")
    with pytest.raises(ValidationError):
        tools["find_company"].invoke({"query": "TEST", "arbitrary": "value"})
    with pytest.raises(ValidationError):
        tools["search_filing_sections"].invoke(
            {"cik": "1", "query": "risk", "limit": 6}
        )


@pytest.mark.parametrize("question", ["", " ", "x" * 2001])
def test_request_validation(question: str) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(question=question)


def test_schema_requirements() -> None:
    for payload in (
        {"status": "answered", "claims": [], "limitations": []},
        {"status": "insufficient_evidence", "claims": [], "limitations": []},
    ):
        with pytest.raises(ValidationError):
            ResearchDraft.model_validate(payload)


def test_api(setup: tuple) -> None:
    service, _, _, _ = setup
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_research_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/research/baseline", json={"question": "risks"})
        assert response.status_code == 200
        assert (
            response.json()["evidence"][0]["chunk"]["chunk_id"] == "synthetic-evidence"
        )
        assert (
            client.post("/research/baseline", json={"question": " "}).status_code == 422
        )
        service.research = Mock(side_effect=ResearchError("Unavailable", 503))
        response = client.post("/research/baseline", json={"question": "risks"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Unavailable"}


@pytest.mark.skipif(
    os.environ.get("SEC_RESEARCH_TEST_CHAT") != "1", reason="Opt-in local Ollama test"
)
def test_real_ollama(evidence: Evidence) -> None:
    # Actual inference with synthetic services; no live SEC or database calls.
    settings = Settings()
    sec, rag = Mock(), Mock()
    rag.search.return_value = SearchResult(embedding_model="test", evidence=[evidence])
    model = chat_model(settings)
    tools = research_tools(sec, rag, {})
    reply = model.bind_tools(tools).invoke(
        "Call search_filing_sections with cik 1 and query supply chain risks. "
        "You must use that tool now, without answering in prose."
    )
    assert isinstance(reply, AIMessage) and not reply.invalid_tool_calls
    assert reply.tool_calls and reply.tool_calls[0]["name"] == "search_filing_sections"
    tool = next(tool for tool in tools if tool.name == "search_filing_sections")
    tool.invoke(reply.tool_calls[0]["args"])
    result = model.with_structured_output(ResearchDraft, method="json_schema").invoke(
        "No evidence exists. Return status insufficient_evidence, claims [], "
        'limitations ["No indexed evidence."].'
    )
    assert result.status == "insufficient_evidence" and not result.claims
    result = ResearchService(sec, rag, settings).research(
        ResearchRequest(
            question=(
                "Use search_filing_sections for CIK 1. What does the retrieved filing "
                "explicitly say may affect operations? Cite the passage."
            )
        )
    )
    assert result.status == "answered"
    assert "search_filing_sections" in result.tools_called
    assert result.evidence == [evidence]


def test_invalid_arguments_can_be_corrected_once(setup: tuple) -> None:
    service, _, rag, model = setup
    model.bind_tools.return_value.invoke.side_effect = [
        call(cik="1", section="Item 1A"),
        call(),
        AIMessage(content="done"),
    ]
    result = service.research(ResearchRequest(question="risks"))
    assert result.status == "answered"
    assert result.model_calls == 4
    assert len(result.tool_errors) == 1
    assert "query" in result.tool_errors[0]
    assert result.tools_called == ["search_filing_sections"] * 2
    rag.search.assert_called_once()
    messages = model.bind_tools.return_value.invoke.call_args.args[0]
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in messages)


def test_repeated_invalid_arguments_fail(setup: tuple) -> None:
    service, _, rag, model = setup
    model.bind_tools.return_value.invoke.side_effect = [
        call(cik="1", section="Item 1A"),
        call(cik="1", section="Item 1A"),
    ]
    with pytest.raises(ResearchError, match="repeatedly supplied invalid"):
        service.research(ResearchRequest(question="risks"))
    rag.search.assert_not_called()
    model.with_structured_output.assert_not_called()
