"""Synchronous LangChain adapters consuming the local MCP stdio server."""

import json
import os
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import anyio
from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings
from app.rag_models import Evidence, SearchResult
from app.research_service import ResearchError
from app.research_tools import OUTPUT_MODELS, TOOL_DEFINITIONS


class MCPResearchClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def call(self, name: str, arguments: dict) -> dict:
        # Forward validated application settings, including API-injected settings.
        env = dict(os.environ)
        config = self.settings.model_dump(mode="json")
        config["database_url"] = self.settings.database_url.get_secret_value()
        env["SEC_RESEARCH_MCP_SETTINGS"] = json.dumps(config)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
        )
        try:
            with anyio.fail_after(self.settings.mcp_timeout_seconds):
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(
                        reader,
                        writer,
                        read_timeout_seconds=timedelta(
                            seconds=self.settings.mcp_timeout_seconds
                        ),
                    ) as session:
                        await session.initialize()
                        available = await session.list_tools()
                        if name not in {tool.name for tool in available.tools}:
                            raise ResearchError("MCP research tool is unavailable")
                        result = await session.call_tool(name, arguments)
            if result.isError:
                error = result.structuredContent or {}
                raise ResearchError(
                    str(error.get("error", "MCP research tool failed")),
                    int(error.get("status_code", 502)),
                )
            return (
                OUTPUT_MODELS[name]
                .model_validate(result.structuredContent)
                .model_dump(mode="json")
            )
        except ResearchError:
            raise
        except Exception as exc:
            raise ResearchError(
                "Local MCP request failed; check the MCP server and its output", 503
            ) from exc


def mcp_research_tools(
    settings: Settings, evidence: dict[str, Evidence]
) -> list[BaseTool]:
    client = MCPResearchClient(settings)

    def adapter(name: str) -> Callable[..., dict]:
        def invoke(**arguments: object) -> dict:
            result = anyio.run(client.call, name, arguments)
            if name == "search_filing_sections":
                parsed = SearchResult.model_validate(result)
                # Accept only evidence matching the requested company and filing.
                for item in parsed.evidence:
                    if item.chunk.filing.cik != arguments["cik"] or (
                        arguments.get("accession_number")
                        and item.chunk.filing.accession_number
                        != arguments["accession_number"]
                    ):
                        raise ResearchError("MCP returned mismatched filing evidence")
                for item in parsed.evidence:
                    evidence[item.chunk.chunk_id] = item
            return result

        return invoke

    # Reuse the baseline's explicit schemas/descriptions, never its service calls.
    return [
        StructuredTool.from_function(
            adapter(name),
            name=name,
            description=description,
            args_schema=schema,
        )
        for name, description, schema in TOOL_DEFINITIONS
    ]
