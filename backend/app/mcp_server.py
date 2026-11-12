"""Local stdio MCP interface over the existing read-only research tools."""

import json
import os

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from app.config import Settings
from app.rag_embeddings import LocalEmbeddings, RAGError
from app.rag_service import RAGService
from app.rag_store import VectorStore
from app.research_tools import OUTPUT_MODELS, research_tools
from app.sec_client import SECClient, SECError
from app.sec_service import SECService


def create_server(sec: SECService, rag: RAGService) -> Server:
    server = Server("sec-research-desk")
    tools = {tool.name: tool for tool in research_tools(sec, rag, {})}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.args_schema.model_json_schema(),
                outputSchema=OUTPUT_MODELS[tool.name].model_json_schema(),
                annotations=types.ToolAnnotations(
                    readOnlyHint=True, destructiveHint=False, openWorldHint=True
                ),
            )
            for tool in tools.values()
        ]

    # Pydantic enforces semantic constraints as well as the published JSON schema.
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> dict | types.CallToolResult:
        if name not in tools:
            return error_result("Unknown research tool", 422)
        try:
            inputs = tools[name].args_schema.model_validate(arguments)
        except ValidationError:
            return error_result("Invalid research tool inputs", 422)
        try:
            # No evidence registry survives between MCP requests.
            tool = next(
                item for item in research_tools(sec, rag, {}) if item.name == name
            )
            result = await anyio.to_thread.run_sync(
                lambda: tool.invoke(inputs.model_dump())
            )
            return OUTPUT_MODELS[name].model_validate(result).model_dump(mode="json")
        except ValidationError:
            return error_result("Invalid research service output", 502)
        except (SECError, RAGError) as exc:
            return error_result(str(exc), exc.status_code)
        except Exception:
            return error_result("Research tool execution failed", 502)

    return server


def error_result(message: str, status_code: int) -> types.CallToolResult:
    return types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text=message)],
        structuredContent={"error": message, "status_code": status_code},
    )


async def serve() -> None:
    config = os.environ.get("SEC_RESEARCH_MCP_SETTINGS")
    settings = Settings(_env_file=None, **json.loads(config)) if config else Settings()
    client = SECClient(settings)
    sec = SECService(client)
    rag = RAGService(sec, LocalEmbeddings(settings), VectorStore(settings), settings)
    server = create_server(sec, rag)
    try:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        client.close()


if __name__ == "__main__":
    anyio.run(serve)
