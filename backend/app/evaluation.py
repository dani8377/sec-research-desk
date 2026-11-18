"""Local paired evaluation. Automated checks are proxies, not semantic judges."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal

import httpx
from langchain_core.tracers.context import collect_runs
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import Settings
from app.filing_parser import parse_sections
from app.rag_embeddings import LocalEmbeddings
from app.rag_service import RAGService
from app.rag_store import VectorStore
from app.research_graph import GraphResearchService
from app.research_models import GraphResearchRequest, ResearchRequest, ResearchResult
from app.research_service import ResearchService
from app.sec_client import SECClient
from app.sec_models import Filing
from app.sec_service import SECService
from app.telemetry import export_scores, setup_tracing, shutdown_tracing, trace_span

DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    category: str
    question: str = Field(min_length=1, max_length=2000)
    expected_status: Literal["answered", "insufficient_evidence"]
    required_tools: list[str]
    expected_terms: list[str]
    reference_quote: str = Field(min_length=1)
    section: str | None
    source_url: str | None
    cik: str | None
    accession_number: str | None
    expected_number: str | None

    @model_validator(mode="after")
    def grounded_answer(self) -> "EvaluationCase":
        if self.expected_status == "answered" and not all(
            (
                self.expected_terms,
                self.section,
                self.source_url,
                self.cik,
                self.accession_number,
            )
        ):
            raise ValueError("Answered cases require source metadata and answer terms")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    source_html_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_filing: Filing
    source_retrieved_at: datetime
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "EvaluationDataset":
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique")
        return self


def verify_source(dataset: EvaluationDataset, html: str) -> None:
    if hashlib.sha256(html.encode()).hexdigest() != dataset.source_html_sha256:
        raise ValueError("Source HTML hash differs from the verified filing snapshot")
    sections, _ = parse_sections(html)
    for case in dataset.cases:
        if case.expected_status != "answered":
            continue
        source = dataset.source_filing
        if (case.cik, case.accession_number, case.source_url) != (
            source.cik,
            source.accession_number,
            source.filing_url,
        ):
            raise ValueError(f"Wrong source metadata for {case.id}")
        if not any(
            title.split(".")[0] == case.section
            and normalized(case.reference_quote) in normalized(text)
            for title, text in sections
        ):
            raise ValueError(f"Reference quote not found in source section: {case.id}")


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def score(case: EvaluationCase, result: ResearchResult | None) -> dict:
    """Identity/lexical checks only; leave semantic measures explicitly unscored."""
    empty = {
        "status_match": False,
        "answer_terms_match": False,
        "citation_validity": None,
        "required_tools_present": False,
        "cited_reference_overlap": None,
        "numerical_accuracy": None,
        "semantic_correctness": None,
        "unsupported_claim_rate": None,
    }
    if result is None:
        return empty
    text = normalized(" ".join(claim.text for claim in result.claims))
    evidence = {item.chunk.chunk_id: item.chunk for item in result.evidence}
    cited = [key for claim in result.claims for key in claim.evidence_ids]
    # Checking known source metadata is stronger than just accepting any returned ID.
    valid = (
        all(
            key in evidence
            and evidence[key].filing.cik == case.cik
            and evidence[key].filing.accession_number == case.accession_number
            and evidence[key].filing.filing_url == case.source_url
            for key in cited
        )
        if cited
        else None
    )
    quote = normalized(case.reference_quote)
    # A reference may straddle chunk boundaries; evaluate term overlap separately.
    overlap = (
        any(
            normalized(chunk.text) in quote or quote in normalized(chunk.text)
            for chunk in evidence.values()
        )
        if evidence and case.expected_status == "answered"
        else None
    )
    number = None
    if case.expected_number is not None and result.status == "answered":
        number = case.expected_number in re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return {
        **empty,
        "status_match": result.status == case.expected_status,
        "answer_terms_match": (
            result.status == case.expected_status
            and (
                all(normalized(term) in text for term in case.expected_terms)
                if case.expected_status == "answered"
                else not result.claims
            )
        ),
        "citation_validity": valid,
        "required_tools_present": set(case.required_tools) <= set(result.tools_called),
        "cited_reference_overlap": overlap,
        "numerical_accuracy": number,
    }


def run_statistics(roots: list) -> dict:
    """Read local LangChain run records, including failed calls and graph children."""
    runs = {}

    def visit(run: object) -> None:
        if run.id in runs:
            return
        runs[run.id] = run
        for child in run.child_runs:
            visit(child)

    for root in roots:
        visit(root)
    llms = [run for run in runs.values() if run.run_type == "llm"]
    tools = [run for run in runs.values() if run.run_type == "tool"]
    tokens = []
    for run in llms:
        for batch in (run.outputs or {}).get("generations", []):
            for generation in batch:
                usage = (
                    generation.get("message", {})
                    .get("kwargs", {})
                    .get("usage_metadata")
                )
                if usage and "total_tokens" in usage:
                    tokens.append(usage["total_tokens"])
    return {
        "model_calls": len(llms),
        "tools_called": [run.name for run in tools],
        "total_tokens": sum(tokens) if tokens else None,
        "token_usage_complete": bool(llms) and len(tokens) == len(llms),
        # Keep full retrieval outputs, before synthesis prunes uncited evidence.
        "retrievals": [
            {"inputs": run.inputs, "outputs": run.outputs, "error": run.error}
            for run in tools
            if run.name == "search_filing_sections"
        ],
    }


def retrieval_reference_hit(case: EvaluationCase, statistics: dict) -> bool | None:
    """Reference overlap in all retrieved chunks, including uncited evidence."""
    if case.expected_status != "answered":
        return None
    quote = normalized(case.reference_quote)
    for retrieval in statistics["retrievals"]:
        output = (retrieval["outputs"] or {}).get("output")
        if not isinstance(output, dict):
            continue
        for item in output.get("evidence", []):
            chunk = item["chunk"]
            text = normalized(chunk["text"])
            if (
                chunk["filing"]["cik"] == case.cik
                and chunk["filing"]["accession_number"] == case.accession_number
                and (quote in text or text in quote)
            ):
                return True
    return False


def evaluate_case(
    case: EvaluationCase,
    workflow: Literal["baseline", "graph"],
    baseline: ResearchService,
    graph: GraphResearchService,
) -> dict:
    started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    result = None
    errors = []
    graph_run = None
    trace_id = span_id = None
    with (
        collect_runs() as collected,
        trace_span("evaluation.case", case_id=case.id, workflow=workflow) as span,
    ):
        if span is not None:
            context = span.get_span_context()
            trace_id, span_id = f"{context.trace_id:032x}", f"{context.span_id:016x}"
            span.set_attribute("input.value", case.question)
            span.set_attribute("input.mime_type", "text/plain")
        try:
            if workflow == "baseline":
                result = baseline.research(ResearchRequest(question=case.question))
            else:
                run = graph.run(GraphResearchRequest(question=case.question))
                graph_run = run.model_dump(mode="json")
                result = run.result if run.status == "completed" else None
                errors = [error.model_dump() for error in run.errors]
        except Exception as exc:
            # A benchmark records unexpected failures too; never turn them into answers.
            errors = [{"type": type(exc).__name__, "message": str(exc)}]
        statistics = run_statistics(collected.traced_runs)
        metrics = score(case, result)
        metrics["required_tools_present"] = set(case.required_tools) <= set(
            statistics["tools_called"]
        )
        metrics["retrieval_reference_hit"] = retrieval_reference_hit(case, statistics)
        if span is not None:
            span.set_attribute("evaluation.execution_success", result is not None)
            span.set_attribute("evaluation.status_match", metrics["status_match"])
            span.set_attribute("output.mime_type", "application/json")
            span.set_attribute(
                "output.value",
                json.dumps(
                    {
                        "status": result.status if result else "failed",
                        "claims": [claim.model_dump() for claim in result.claims]
                        if result
                        else [],
                        "limitations": result.limitations if result else [],
                        "errors": errors,
                    },
                    ensure_ascii=False,
                ),
            )
            if result is None:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, "Research execution failed"))
    return {
        "case_id": case.id,
        "trace_id": trace_id,
        "span_id": span_id,
        "category": case.category,
        "workflow": workflow,
        "started_at": started_at,
        "latency_seconds": perf_counter() - started,
        "execution_success": result is not None,
        "metrics": metrics,
        "statistics": statistics,
        "result": result.model_dump(mode="json") if result else None,
        "graph_run": graph_run,
        "errors": errors,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for workflow in ("baseline", "graph"):
        selected = [row for row in rows if row["workflow"] == workflow]
        if not selected:
            continue
        metrics = {}
        for name in selected[0]["metrics"]:
            values = [
                row["metrics"][name]
                for row in selected
                if row["metrics"][name] is not None
            ]
            metrics[name] = {
                "mean": sum(values) / len(values) if values else None,
                "scored": len(values),
            }
        summary[workflow] = {
            "cases": len(selected),
            "execution_success_rate": sum(row["execution_success"] for row in selected)
            / len(selected),
            "mean_latency_seconds": sum(row["latency_seconds"] for row in selected)
            / len(selected),
            "model_calls": sum(row["statistics"]["model_calls"] for row in selected),
            "metrics": metrics,
            "failures": [
                {"case_id": row["case_id"], "errors": row["errors"]}
                for row in selected
                if not row["execution_success"]
            ],
        }
    return summary


def runtime_manifest(settings: Settings) -> dict:
    """Record model weights, implementation and package versions before a live run."""
    response = httpx.get(settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=5)
    response.raise_for_status()
    installed = {m["name"]: m["digest"] for m in response.json()["models"]}
    models = {}
    for name in (settings.chat_model, settings.embedding_model):
        tag = name if ":" in name else name + ":latest"
        if tag not in installed:
            raise ValueError(f"Required local model is not installed: {tag}")
        models[name] = installed[tag]
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    return {
        "model_digests": models,
        "implementation_sha256": digest.hexdigest(),
        "package_versions": {
            name: version(name)
            for name in (
                "langchain-core",
                "langchain-ollama",
                "langgraph",
                "mcp",
                "arize-phoenix-otel",
                "openinference-instrumentation-langchain",
            )
        },
        "embedding_dimensions": settings.embedding_dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-source-html", type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = EvaluationDataset.model_validate_json(raw)
    if args.verify_source_html:
        verify_source(dataset, args.verify_source_html.read_text())
        print("All answerable references match the pinned SEC filing and sections.")
        return
    if args.output is None:
        parser.error("--output is required for model evaluation")
    cases = [
        case for case in dataset.cases if args.cases is None or case.id in args.cases
    ]
    if not cases or (args.cases and set(args.cases) - {c.id for c in cases}):
        parser.error("Unknown or empty case selection")
    settings = Settings()
    runtime = runtime_manifest(settings)
    store = VectorStore(settings)
    inventory = store.inventory(settings.embedding_model)
    provider = setup_tracing(settings)
    client = SECClient(settings)
    sec = SECService(client)
    rag = RAGService(sec, LocalEmbeddings(settings), store, settings)
    baseline = ResearchService(sec, rag, settings)
    graph = GraphResearchService(sec, rag, settings)
    # Explicit allowlist: never serialize settings containing credentials/contact data.
    manifest = {
        **runtime,
        "corpus": inventory,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_version": dataset.version,
        "evaluator_version": "reliability-v2",
        "case_ids": [case.id for case in cases],
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "chat_context_size": settings.chat_context_size,
        "chat_max_tokens": settings.chat_max_tokens,
        "tool_budget": settings.research_max_tool_calls,
        "context_chars": settings.research_max_context_chars,
        "tracing_enabled": settings.phoenix_enabled,
        "transport": {"baseline": "direct", "graph": "MCP stdio"},
    }
    rows = []
    try:
        with args.output.open("x") as output:
            output.write(json.dumps({"manifest": manifest}) + "\n")
            output.flush()
            for index, case in enumerate(cases):
                # Alternate order to reduce systematic cache/warmup advantage.
                order = (
                    ("baseline", "graph") if index % 2 == 0 else ("graph", "baseline")
                )
                for workflow in order:
                    row = evaluate_case(case, workflow, baseline, graph)
                    rows.append(row)
                    output.write(json.dumps(row, default=str) + "\n")
                    output.flush()
                    print(
                        f"{case.id} {workflow}: success={row['execution_success']} "
                        f"status_match={row['metrics']['status_match']}",
                        flush=True,
                    )
            if provider:
                provider.force_flush()
            exported = export_scores(rows, settings)
            output.write(
                json.dumps(
                    {
                        "summary": summarize(rows),
                        "phoenix_annotations": exported,
                    }
                )
                + "\n"
            )
    finally:
        client.close()
        if provider:
            shutdown_tracing()


if __name__ == "__main__":
    main()
