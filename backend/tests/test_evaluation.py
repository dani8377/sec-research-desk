"""Deterministic evaluator tests: no local model, database or SEC requests."""

import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.evaluation import (
    DATASET,
    EvaluationDataset,
    evaluate_case,
    run_statistics,
    score,
    summarize,
)
from app.research_models import GraphResearchRun, ResearchResult
from app.research_service import ResearchError


def dataset():
    return EvaluationDataset.model_validate_json(DATASET.read_bytes())


def answer(case):
    text = case.reference_quote
    return ResearchResult(
        status="answered",
        claims=[{"text": text, "kind": "reported_fact", "evidence_ids": ["known"]}],
        limitations=[],
        model="test",
        tools_called=case.required_tools,
        model_calls=3,
        evidence=[
            {
                "similarity": 1.0,
                "chunk": {
                    "chunk_id": "known",
                    "section": f"{case.section}. Heading",
                    "text": text,
                    "char_start": 0,
                    "char_end": len(text),
                    "chunk_index": 0,
                    "retrieved_at": "2026-11-03T00:00:00Z",
                    "filing": {
                        "cik": case.cik,
                        "accession_number": case.accession_number,
                        "filing_url": case.source_url,
                        "source_url": "https://data.sec.gov/submissions/CIK0000320193.json",
                        "filing_date": "2025-10-31",
                        "report_date": "2025-09-27",
                        "primary_document": "aapl-20250927.htm",
                        "form": "10-K",
                    },
                },
            }
        ],
    )


def test_dataset_contract_and_reference_terms():
    data = dataset()
    assert len(data.cases) == 30
    assert len({case.category for case in data.cases}) >= 7
    for case in data.cases:
        if case.expected_status == "answered":
            # All deterministic expectations must be present in the saved SEC quote.
            assert all(
                term.casefold() in case.reference_quote.casefold()
                for term in case.expected_terms
            )
            assert score(case, answer(case))["answer_terms_match"]
    payload = json.loads(DATASET.read_text())
    payload["cases"].append(payload["cases"][0])
    with pytest.raises(ValidationError, match="unique"):
        EvaluationDataset.model_validate(payload)
    payload["cases"][0]["source_url"] = None
    with pytest.raises(ValidationError, match="source metadata"):
        EvaluationDataset.model_validate(payload)


def test_scoring_citations_and_missing_terms():
    case = dataset().cases[1]
    result = answer(case)
    metrics = score(case, result)
    assert metrics["citation_validity"] is True
    assert metrics["cited_reference_overlap"] is True
    assert metrics["semantic_correctness"] is None
    assert metrics["unsupported_claim_rate"] is None
    result.claims[0].evidence_ids = ["invented"]
    assert score(case, result)["citation_validity"] is False
    result.claims[0].evidence_ids = ["known"]
    result.evidence[0].chunk.filing.accession_number = "wrong"
    assert score(case, result)["citation_validity"] is False
    result.claims[0].text = "Unsupported answer"
    assert score(case, result)["answer_terms_match"] is False


def test_numeric_token_boundaries():
    case = dataset().cases[19]
    result = answer(case)
    assert score(case, result)["numerical_accuracy"] is True
    result.claims[0].text = "It spans 152 weeks."
    assert score(case, result)["numerical_accuracy"] is False


def test_abstention_has_no_vacuous_citation_score():
    case = dataset().cases[-1]
    result = ResearchResult(
        status="insufficient_evidence",
        claims=[],
        limitations=["Absent period"],
        evidence=[],
        model="test",
        tools_called=[],
        model_calls=1,
    )
    metrics = score(case, result)
    assert metrics["status_match"] and metrics["answer_terms_match"]
    assert metrics["citation_validity"] is None
    result = answer(dataset().cases[0])
    result.status = "insufficient_evidence"
    assert score(case, result)["answer_terms_match"] is False


@pytest.mark.parametrize("workflow", ["baseline", "graph"])
def test_runner_records_failures_and_continues(workflow):
    baseline, graph = Mock(), Mock()
    baseline.research.side_effect = ResearchError("Local inference unavailable", 503)
    graph.run.return_value = GraphResearchRun(
        run_id=uuid4(),
        status="failed",
        question="test",
        plan=None,
        result=None,
        errors=[{"stage": "plan", "message": "failure", "status_code": 503}],
        tools_called=[],
        model_calls=1,
        next_nodes=[],
    )
    row = evaluate_case(dataset().cases[0], workflow, baseline, graph)
    assert not row["execution_success"] and row["errors"]
    assert not row["metrics"]["status_match"]
    assert row["latency_seconds"] >= 0
    assert row["statistics"]["total_tokens"] is None
    summary = summarize([row])[workflow]
    assert summary["execution_success_rate"] == 0
    assert summary["metrics"]["citation_validity"]["scored"] == 0
    assert summary["failures"][0]["case_id"] == dataset().cases[0].id


def test_collect_actual_local_tool_runs():
    from langchain_core.tools import StructuredTool
    from langchain_core.tracers.context import collect_runs

    def search_filing_sections(query: str) -> dict:
        """Return deterministic retrieval for evaluating trace capture."""
        return {"evidence": [], "query": query}

    tool = StructuredTool.from_function(search_filing_sections)
    with collect_runs() as collected:
        tool.invoke({"query": "risk"})
    stats = run_statistics(collected.traced_runs)
    assert stats["tools_called"] == ["search_filing_sections"]
    assert len(stats["retrievals"]) == 1
    assert "risk" in json.dumps(stats["retrievals"])


def test_run_statistics_counts_nested_calls_and_partial_usage():
    def run(id, kind, outputs=None, children=None):
        return SimpleNamespace(
            id=id,
            name="test",
            run_type=kind,
            outputs=outputs,
            child_runs=children or [],
        )

    llm = run(
        "llm",
        "llm",
        {
            "generations": [
                [{"message": {"kwargs": {"usage_metadata": {"total_tokens": 17}}}}]
            ]
        },
    )
    failed = run("failed", "llm")
    root = run("root", "chain", children=[llm, failed])
    stats = run_statistics([root, llm])
    assert stats["model_calls"] == 2 and stats["total_tokens"] == 17
    assert stats["token_usage_complete"] is False


def test_reference_verification_detects_changed_source_and_quotes():
    import hashlib

    from app.evaluation import verify_source

    data = dataset()
    case = data.cases[0]
    html = (
        f"<html><body><h2>{case.section}. Business</h2>"
        f"<p>{case.reference_quote}</p></body></html>"
    )
    data = data.model_copy(
        update={
            "cases": [case],
            "source_html_sha256": hashlib.sha256(html.encode()).hexdigest(),
        }
    )
    verify_source(data, html)
    with pytest.raises(ValueError, match="hash differs"):
        verify_source(data, html + "changed")
    case.reference_quote = "An invented statement."
    with pytest.raises(ValueError, match="not found"):
        verify_source(data, html)
    case.source_url = "https://wrong.example"
    with pytest.raises(ValueError, match="Wrong source metadata"):
        verify_source(data, html)


def test_retrieval_reference_hit_includes_uncited_evidence_and_rejects_wrong_source():
    from app.evaluation import retrieval_reference_hit

    case = dataset().cases[0]
    result = answer(case)
    stats = {
        "retrievals": [
            {
                "outputs": {
                    "output": {
                        "evidence": [
                            item.model_dump(mode="json") for item in result.evidence
                        ]
                    }
                }
            }
        ]
    }
    assert retrieval_reference_hit(case, stats) is True
    stats["retrievals"][0]["outputs"]["output"]["evidence"][0]["chunk"]["filing"][
        "cik"
    ] = "wrong"
    assert retrieval_reference_hit(case, stats) is False
    assert retrieval_reference_hit(case, {"retrievals": []}) is False
    assert retrieval_reference_hit(dataset().cases[-1], stats) is None


def test_failed_workflow_still_scores_recorded_tool_attempts():
    from langchain_core.tools import StructuredTool

    def search_filing_sections(query: str) -> dict:
        """Return empty test evidence."""
        return {"evidence": []}

    tool = StructuredTool.from_function(search_filing_sections)

    def fail_after_retrieval(request):
        tool.invoke({"query": "business"})
        raise ResearchError("Synthesis failed")

    baseline = Mock()
    baseline.research.side_effect = fail_after_retrieval
    row = evaluate_case(dataset().cases[0], "baseline", baseline, Mock())
    assert not row["execution_success"]
    assert row["metrics"]["required_tools_present"] is True
    assert row["metrics"]["retrieval_reference_hit"] is False
