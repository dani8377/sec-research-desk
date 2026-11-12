"""MCP discovery, tools, errors and graph execution over real stdio, offline."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock

import anyio
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError
from test_research import evidence as evidence

from app.config import Settings
from app.mcp_client import MCPResearchClient, mcp_research_tools
from app.mcp_server import create_server
from app.rag_embeddings import RAGError
from app.rag_models import Evidence, SearchResult
from app.research_graph import GraphResearchService
from app.research_models import GraphResearchRequest
from app.research_service import ResearchError
from app.sec_models import Company, FilingHistory


@pytest.fixture
def transport(tmp_path: Path, evidence: Evidence, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "evidence.json"
    source.write_text(evidence.model_dump_json())
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-c",
            "import sys, runpy; sys.path.insert(0, 'tests'); "
            "runpy.run_path('tests/test_mcp.py', run_name='__main__')",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "MCP_TEST_EVIDENCE": str(source)},
    )
    monkeypatch.setattr("app.mcp_client.stdio_client", lambda _: stdio_client(params))
    return params


def test_discovery_and_interfaces(transport: StdioServerParameters) -> None:
    async def check() -> None:
        async with stdio_client(transport) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert {tool.name for tool in tools} == {
                    "find_company",
                    "get_company_filings",
                    "search_filing_sections",
                }
                for tool in tools:
                    assert tool.inputSchema["additionalProperties"] is False
                    assert tool.outputSchema
                    assert tool.annotations.readOnlyHint
                company = await session.call_tool("find_company", {"query": "TEST"})
                assert not company.isError
                assert company.structuredContent["companies"][0]["cik"] == "0000000001"
                filings = await session.call_tool("get_company_filings", {"cik": "1"})
                assert not filings.isError
                filing = filings.structuredContent["filings"][0]
                assert filing["accession_number"] == "0000000001-24-000001"
                for name, arguments in [
                    ("unknown", {}),
                    ("find_company", {"query": ""}),
                    ("find_company", {"query": "TEST", "extra": True}),
                    ("get_company_filings", {"cik": "0"}),
                    (
                        "search_filing_sections",
                        {"cik": "1", "query": "risk", "limit": 6},
                    ),
                    (
                        "search_filing_sections",
                        {
                            "cik": "1",
                            "query": "risk",
                            "report_date_from": "2025-01-01",
                            "report_date_to": "2024-01-01",
                        },
                    ),
                ]:
                    result = await session.call_tool(name, arguments)
                    assert result.isError, (name, arguments)

    anyio.run(check)


def test_adapter_evidence_and_validation(transport, evidence: Evidence) -> None:
    registry = {}
    tools = {t.name: t for t in mcp_research_tools(Settings(_env_file=None), registry)}
    with pytest.raises(ValidationError):
        tools["search_filing_sections"].invoke({"cik": "1"})
    result = tools["search_filing_sections"].invoke({"cik": "1", "query": "risks"})
    assert SearchResult.model_validate(result).evidence == [evidence]
    assert registry == {evidence.chunk.chunk_id: evidence}
    tools["search_filing_sections"].invoke({"cik": "1", "query": "empty"})
    assert len(registry) == 1
    with pytest.raises(ResearchError, match="mismatched"):
        tools["search_filing_sections"].invoke({"cik": "2", "query": "risks"})


@pytest.mark.parametrize("query,status", [("unavailable", 503), ("missing", 404)])
def test_service_errors(transport, query: str, status: int) -> None:
    client = MCPResearchClient(Settings(_env_file=None))
    with pytest.raises(ResearchError) as error:
        anyio.run(client.call, "search_filing_sections", {"cik": "1", "query": query})
    assert error.value.status_code == status


def test_graph_uses_mcp(transport, evidence: Evidence, monkeypatch) -> None:
    from langchain_core.messages import AIMessage
    from test_research import call, draft

    from app.research_models import ResearchPlan

    model = Mock()
    planner, synthesizer = Mock(), Mock()
    planner.invoke.return_value = ResearchPlan(
        scope="filing_research", objectives=["Find risks"], limitation=None
    )
    synthesizer.invoke.return_value = draft()
    model.with_structured_output.side_effect = lambda schema, **kw: (
        planner if schema is ResearchPlan else synthesizer
    )
    model.bind_tools.return_value.invoke.side_effect = [
        call(),
        AIMessage(content="done"),
    ]
    monkeypatch.setattr("app.research_graph.chat_model", lambda _: model)
    # No direct service is available: tools must traverse MCP.
    service = GraphResearchService(
        None, None, Settings(_env_file=None), checkpointer=InMemorySaver()
    )
    result = service.run(GraphResearchRequest(question="What are the risks?"))
    assert result.status == "completed", result.errors
    assert result.result.evidence == [evidence]
    assert result.tools_called == ["search_filing_sections"]
    model.bind_tools.return_value.invoke.side_effect = [
        call(query="unavailable", cik="1")
    ]
    failed = service.run(GraphResearchRequest(question="What are the risks?"))
    assert failed.status == "failed"
    assert failed.errors[0].status_code == 503
    assert failed.tools_called == ["search_filing_sections"]


def test_production_server_starts_without_services(tmp_path: Path) -> None:
    # Real entry point and forwarded Settings; no live SEC/database/Ollama requests.
    settings = Settings(_env_file=None, sec_user_agent=None, sec_cache_dir=tmp_path)
    with pytest.raises(ResearchError) as error:
        anyio.run(MCPResearchClient(settings).call, "find_company", {"query": "TEST"})
    assert error.value.status_code == 503
    assert "SEC_RESEARCH_SEC_USER_AGENT" in str(error.value)


@pytest.mark.parametrize("mode", ["timeout", "startup"])
def test_transport_failures_are_explicit(monkeypatch, mode: str) -> None:
    @asynccontextmanager
    async def unavailable(params):
        if mode == "startup":
            raise OSError("Fixture process startup failure")
        await anyio.sleep_forever()
        yield

    monkeypatch.setattr("app.mcp_client.stdio_client", unavailable)
    settings = Settings(_env_file=None, mcp_timeout_seconds=0.05)
    with pytest.raises(ResearchError, match="Local MCP request failed") as error:
        anyio.run(MCPResearchClient(settings).call, "find_company", {"query": "TEST"})
    assert error.value.status_code == 503


if __name__ == "__main__":
    from mcp.server.stdio import stdio_server

    async def serve_fixture() -> None:
        item = Evidence.model_validate_json(
            Path(os.environ["MCP_TEST_EVIDENCE"]).read_text()
        )
        sec, rag = Mock(), Mock()
        sec.find_company.return_value = [
            Company(
                cik="0000000001",
                ticker="TEST",
                name="Synthetic company",
                source_url="https://www.sec.gov/files/company_tickers.json",
            )
        ]
        sec.get_company_filings.return_value = FilingHistory(
            cik="0000000001",
            company_name="Synthetic company",
            includes_historical=False,
            filings=[item.chunk.filing],
        )

        def search(request):
            if request.query in {"unavailable", "missing"}:
                raise RAGError(
                    "Fixture retrieval failed",
                    503 if request.query == "unavailable" else 404,
                )
            return SearchResult(
                embedding_model="test",
                evidence=[] if request.query == "empty" else [item],
            )

        rag.search.side_effect = search
        server = create_server(sec, rag)
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    anyio.run(serve_fixture)
