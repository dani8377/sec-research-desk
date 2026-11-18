"""Shared deterministic input repair and bounded model context."""

import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    messages_to_dict,
)

from app.research_tools import TOOL_DEFINITIONS

SCHEMAS = {name: schema for name, _, schema in TOOL_DEFINITIONS}


def prepare_arguments(
    name: str, arguments: dict, question: str
) -> tuple[dict, list[str]]:
    arguments = dict(arguments)
    warnings = []
    if name == "search_filing_sections" and "query" not in arguments:
        # Only reuse known user input; never infer a CIK, accession or financial fact.
        arguments["query"] = question
        warnings.append(
            "search_filing_sections: missing query used the original question."
        )
    return SCHEMAS[name].model_validate(arguments).model_dump(mode="json"), warnings


def tool_key(name: str, arguments: dict) -> str:
    return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)


def selection_reply(reply: AIMessage) -> AIMessage:
    # Discard intermediate prose; retain tool calls for protocol pairing.
    return reply.model_copy(update={"content": ""})


def tool_message_content(name: str, result: dict, known_ids: set[str]) -> str:
    if name != "search_filing_sections":
        return json.dumps(result, ensure_ascii=False)
    items = result.get("evidence", [])
    previews = []
    for item in items:
        chunk = item["chunk"]
        if chunk["chunk_id"] in known_ids:
            continue
        previews.append(
            {
                "evidence_id": chunk["chunk_id"],
                "section": chunk["section"],
                "cik": chunk["filing"]["cik"],
                "accession_number": chunk["filing"]["accession_number"],
                "preview": chunk["text"][:350],
                "preview_truncated": len(chunk["text"]) > 350,
            }
        )
    return json.dumps(
        {
            "new_evidence": previews,
            "returned_evidence_ids": [item["chunk"]["chunk_id"] for item in items],
            "warnings": result.get("warnings", []),
            "note": "Full passages and original metadata are retained for synthesis. "
            "Repeated evidence is not repeated here. Stop when enough evidence "
            "is collected; do not repeat identical searches.",
        },
        ensure_ascii=False,
    )


def fit_messages(messages: list[BaseMessage], budget: int) -> list[BaseMessage]:
    """Trim complete old tool rounds, preserving the request and latest round."""
    messages = list(messages)
    while len(json.dumps(messages_to_dict(messages), ensure_ascii=False)) > budget:
        # Each round begins with an AI tool request and ends with its tool responses.
        starts = [
            i for i in range(2, len(messages)) if isinstance(messages[i], AIMessage)
        ]
        if len(starts) < 2:
            raise ValueError("Research context limit exceeded; narrow the question")
        start, end = starts[0], starts[1]
        ai = messages[start]
        expected = {call["id"] for call in ai.tool_calls}
        received = {
            m.tool_call_id
            for m in messages[start + 1 : end]
            if isinstance(m, ToolMessage)
        }
        if expected != received:
            raise ValueError("Cannot trim an incomplete research tool round")
        del messages[start:end]
    return messages
