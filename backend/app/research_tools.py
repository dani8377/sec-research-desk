"""Typed, read-only LangChain tools reusing SEC and RAG services."""

from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.rag_models import CompanyCIK, Evidence, SearchRequest, SearchResult
from app.rag_service import RAGService
from app.sec_models import Company, FilingHistory
from app.sec_service import SECService


class CompanyResult(BaseModel):
    companies: list[Company]


class CompanyQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]
    limit: int = Field(default=5, ge=1, le=10)


class FilingQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cik: CompanyCIK
    limit: int = Field(default=5, ge=1, le=10)


class EvidenceQuery(SearchRequest):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=3, ge=1, le=5)


def research_tools(
    sec: SECService, rag: RAGService, evidence: dict[str, Evidence]
) -> list[BaseTool]:
    # The evidence registry belongs to one request, never shared across users/runs.
    def find_company(**kwargs: object) -> dict:
        query = CompanyQuery.model_validate(kwargs)
        return {
            "companies": [
                company.model_dump(mode="json")
                for company in sec.find_company(query.query, query.limit)
            ]
        }

    def get_company_filings(**kwargs: object) -> dict:
        query = FilingQuery.model_validate(kwargs)
        history = sec.get_company_filings(query.cik, form="10-K")
        return history.model_copy(
            update={"filings": history.filings[: query.limit]}
        ).model_dump(mode="json")

    def search_filing_sections(**kwargs: object) -> dict:
        query = EvidenceQuery.model_validate(kwargs)
        result = rag.search(query)
        for item in result.evidence:
            evidence[item.chunk.chunk_id] = item
        return result.model_dump(mode="json")

    functions = {
        "find_company": find_company,
        "get_company_filings": get_company_filings,
        "search_filing_sections": search_filing_sections,
    }
    return [
        StructuredTool.from_function(
            functions[name], name=name, description=description, args_schema=schema
        )
        for name, description, schema in TOOL_DEFINITIONS
    ]


TOOL_DEFINITIONS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("find_company", "Find SEC companies by ticker, CIK or name.", CompanyQuery),
    (
        "get_company_filings",
        "List recent 10-K filings; older history is not included.",
        FilingQuery,
    ),
    (
        "search_filing_sections",
        "Search already indexed 10-K sections by CIK and optional filing/"
        "period filters. Use limit=3 (maximum 5). section accepts an exact "
        "title or Item identifier such as Item 7. "
        "Empty results mean no matching indexed evidence. "
        "Cite chunk.chunk_id. Does not download or ingest filings.",
        EvidenceQuery,
    ),
)

OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "find_company": CompanyResult,
    "get_company_filings": FilingHistory,
    "search_filing_sections": SearchResult,
}
