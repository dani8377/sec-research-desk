"""Shared context behavior uses complete tool rounds and original evidence."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from app.research_context import fit_messages, prepare_arguments, tool_message_content


def test_missing_query_uses_question_but_never_guesses_identifiers():
    args, warnings = prepare_arguments(
        "search_filing_sections", {"cik": "1"}, "What operating system is reported?"
    )
    assert args["query"] == "What operating system is reported?"
    assert args["cik"] == "0000000001" and warnings
    with pytest.raises(ValidationError):
        prepare_arguments("search_filing_sections", {}, "Company 1 risks")
    with pytest.raises(ValidationError):
        prepare_arguments("search_filing_sections", {"cik": "1", "query": ""}, "risks")


def test_trim_preserves_request_and_whole_tool_pairs():
    initial = [SystemMessage("system"), HumanMessage("question")]
    first = AIMessage(
        content="",
        tool_calls=[{"name": "search_filing_sections", "args": {}, "id": "one"}],
    )
    last = AIMessage(
        content="",
        tool_calls=[{"name": "search_filing_sections", "args": {}, "id": "two"}],
    )
    messages = initial + [
        first,
        ToolMessage("x" * 3000, tool_call_id="one"),
        last,
        ToolMessage("result", tool_call_id="two"),
    ]
    trimmed = fit_messages(messages, 2200)
    assert trimmed == initial + messages[-2:]
    assert len(messages) == 6  # No mutation of previously checkpointed input.
    with pytest.raises(ValueError, match="incomplete"):
        fit_messages(
            initial + [first, last, ToolMessage("x" * 3000, tool_call_id="two")], 2200
        )


def test_previews_are_bounded_and_known_evidence_is_not_duplicated():
    chunk = {
        "chunk_id": "one",
        "section": "Item 1",
        "text": "x" * 4000,
        "filing": {"cik": "1", "accession_number": "test"},
    }
    result = {"evidence": [{"chunk": chunk}], "warnings": ["Fallback disclosed"]}
    preview = json.loads(tool_message_content("search_filing_sections", result, set()))
    assert len(preview["new_evidence"][0]["preview"]) == 350
    assert preview["new_evidence"][0]["preview_truncated"] is True
    assert result["evidence"][0]["chunk"]["text"] == "x" * 4000
    repeated = json.loads(
        tool_message_content("search_filing_sections", result, {"one"})
    )
    assert repeated["new_evidence"] == []
    assert repeated["returned_evidence_ids"] == ["one"]
    assert repeated["warnings"] == ["Fallback disclosed"]
