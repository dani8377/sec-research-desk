"""Offline behavior tests; annual content and identifiers are synthetic."""

import os
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.filing_parser import chunk_filing, parse_sections
from app.main import create_app
from app.rag_embeddings import LocalEmbeddings, RAGError
from app.rag_models import Evidence, IngestRequest, SearchRequest
from app.rag_routes import get_rag_service
from app.rag_service import RAGService
from app.rag_store import VectorStore
from app.sec_client import SECClient, SECError
from app.sec_models import Filing, FilingHistory
from app.sec_service import SECService

HTML = (Path(__file__).parent / "fixtures" / "annual_synthetic.html").read_text()
NOW = datetime(2024, 2, 1, tzinfo=UTC)


@pytest.fixture
def filing() -> Filing:
    return Filing(
        cik="0000000001",
        accession_number="0000000001-24-000001",
        filing_date=date(2024, 2, 1),
        report_date=date(2023, 12, 31),
        form="10-K",
        primary_document="synthetic.htm",
        filing_url="https://www.sec.gov/Archives/edgar/data/1/000000000124000001/synthetic.htm",
        source_url="https://data.sec.gov/submissions/CIK0000000001.json",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        sec_cache_dir=tmp_path,
        sec_user_agent="Offline tests test@example.com",
    )


def test_parser_sections_remove_navigation_and_hidden_content() -> None:
    sections, warnings = parse_sections(HTML)
    assert warnings == []
    assert [title for title, _ in sections] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management Discussion",
    ]
    text = " ".join(body for _, body in sections)
    assert "hidden" not in text and "code" not in text
    assert "Illustrative row qualitative value" in text
    assert "Supply chain disruptions" in sections[1][1]
    assert "Management discusses" in sections[2][1]


def test_original_sec_html_excerpt() -> None:
    html = (Path(__file__).parent / "fixtures" / "annual_excerpt.html").read_text()
    sections, warnings = parse_sections(html)
    assert not warnings
    assert sections[0][0] == "Item 1A. Risk Factors"
    assert sections[1][0].startswith("Item 7. Management’s Discussion")
    assert "consolidated financial statements" in sections[0][1]
    assert "fiscal year 2025" in sections[1][1]


def test_parser_unknown_sections_and_unsupported_content() -> None:
    sections, warnings = parse_sections(
        "<html><body><p>Unlabelled content</p></body></html>"
    )
    assert sections == [("Unidentified section", "Unlabelled content")]
    assert warnings
    for html in ("%PDF annual filing", "plain text", "<html><body></body></html>"):
        with pytest.raises(ValueError):
            parse_sections(html)


def test_chunks_preserve_evidence_offsets_and_stable_identity(filing: Filing) -> None:
    sections, _ = parse_sections(HTML)
    chunks = chunk_filing(filing, NOW, sections, chunk_size=100, overlap=20)
    assert len(chunks) > len(sections)
    assert chunks == chunk_filing(filing, NOW, sections, chunk_size=100, overlap=20)
    bodies = dict(sections)
    for chunk in chunks:
        assert chunk.text == bodies[chunk.section][chunk.char_start : chunk.char_end]
        assert len(chunk.text) <= 100
        assert chunk.filing == filing
        assert chunk.retrieved_at == NOW
    for title, body in sections:
        selected = [chunk for chunk in chunks if chunk.section == title]
        assert selected[0].char_start == 0
        assert selected[-1].char_end == len(body)
        for first, second in zip(selected, selected[1:], strict=False):
            assert first.char_start < second.char_start <= first.char_end
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    with pytest.raises(ValueError):
        chunk_filing(filing, NOW, sections, chunk_size=100, overlap=100)


def make_service(settings: Settings, filing: Filing) -> RAGService:
    sec = Mock(spec=SECService)
    sec.client = Mock(spec=SECClient)
    sec.get_company_filings.return_value = FilingHistory(
        cik=filing.cik,
        company_name="Synthetic test company",
        includes_historical=False,
        filings=[filing],
    )
    sec.client.get_filing_html.return_value = (HTML, NOW)
    embeddings = Mock(spec=LocalEmbeddings)
    embeddings.model = "test-embeddings"
    embeddings.documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    embeddings.query.return_value = [1.0, 0.0]
    store = Mock(spec=VectorStore)
    return RAGService(sec, embeddings, store, settings)


def test_ingestion_and_search_keep_metadata(settings: Settings, filing: Filing) -> None:
    service = make_service(settings, filing)
    request = IngestRequest(cik="1", accession_number=filing.accession_number)
    result = service.ingest(request)
    chunks, vectors, model = service.store.replace.call_args.args
    assert result.chunk_count == len(chunks) == len(vectors)
    assert all(chunk.filing == filing for chunk in chunks)
    assert all(chunk.retrieved_at == NOW for chunk in chunks)
    service.sec.get_company_filings.assert_called_once_with(
        filing.cik, form="10-K", include_historical=False
    )
    service.store.search.return_value = [Evidence(chunk=chunks[1], similarity=0.9)]
    search = SearchRequest(
        cik="1", query="supplier disruptions", section=chunks[1].section
    )
    result = service.search(search)
    assert result.evidence[0].chunk.filing.filing_url == filing.filing_url
    assert result.evidence[0].chunk.filing.report_date == filing.report_date
    service.store.search.assert_called_once_with(search, [1.0, 0.0], model)


def test_ingest_failure_does_not_replace_existing_data(
    settings: Settings, filing: Filing
) -> None:
    service = make_service(settings, filing)
    service.embeddings.documents.side_effect = RAGError("embedding failed")
    with pytest.raises(RAGError, match="embedding failed"):
        service.ingest(IngestRequest(cik="1", accession_number=filing.accession_number))
    service.store.replace.assert_not_called()


def test_missing_filing_does_not_download(settings: Settings, filing: Filing) -> None:
    service = make_service(settings, filing)
    service.sec.get_company_filings.return_value.filings = []
    with pytest.raises(RAGError) as error:
        service.ingest(IngestRequest(cik="1", accession_number=filing.accession_number))
    assert error.value.status_code == 404
    service.sec.client.get_filing_html.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"form": "10-Q"},
        {"primary_document": None},
        {"filing_url": "https://attacker.invalid/annual.htm"},
        {
            "filing_url": "http://www.sec.gov/Archives/edgar/data/1/000000000124000001/synthetic.htm"
        },
        {
            "filing_url": "https://www.sec.gov/Archives/edgar/data/1/000000000124000001/../bad.htm"
        },
    ],
)
def test_unsafe_filing_source_rejected(filing: Filing, change: dict) -> None:
    with pytest.raises(RAGError):
        RAGService._validate_source(filing.model_copy(update=change))


def test_html_uses_shared_sec_cache(settings: Settings, filing: Filing) -> None:
    handler = Mock(return_value=httpx.Response(200, text=HTML))
    with_client = SECClient(settings, httpx.MockTransport(handler))
    try:
        first = with_client.get_filing_html(filing.filing_url)
        second = with_client.get_filing_html(filing.filing_url)
        assert first == second
        assert first[0] == HTML
        assert handler.call_count == 1
        assert handler.call_args.args[0].headers["Accept"] == "text/html"
    finally:
        with_client.close()


def test_non_html_not_cached(settings: Settings, filing: Filing) -> None:
    client = SECClient(
        settings, httpx.MockTransport(lambda _: httpx.Response(200, text="%PDF"))
    )
    try:
        with pytest.raises(SECError, match="HTML"):
            client.get_filing_html(filing.filing_url)
        assert not list(settings.sec_cache_dir.glob("*.json"))
    finally:
        client.close()


def test_corrupt_html_cache_is_refetched(settings: Settings, filing: Filing) -> None:
    import json

    handler = Mock(return_value=httpx.Response(200, text=HTML))
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        client.get_filing_html(filing.filing_url)
        path = next(settings.sec_cache_dir.glob("*.json"))
        entry = json.loads(path.read_text())
        entry["payload"] = json.dumps({"content": "invalid cached HTML"})
        path.write_text(json.dumps(entry))
        assert client.get_filing_html(filing.filing_url)[0] == HTML
        assert handler.call_count == 2
    finally:
        client.close()


def test_embedding_batches_and_query_prefix(settings: Settings) -> None:
    settings.embedding_dimensions = 2
    settings.embedding_batch_size = 1
    embeddings = LocalEmbeddings(settings)
    embeddings.client = Mock()
    embeddings.client.embed_documents.side_effect = lambda texts: [
        [1.0, 0.5] for _ in texts
    ]
    assert embeddings.documents(["one", "two"]) == [[1.0, 0.5], [1.0, 0.5]]
    assert embeddings.query("risks") == [1.0, 0.5]
    assert [
        call.args[0] for call in embeddings.client.embed_documents.call_args_list
    ] == [["search_document: one"], ["search_document: two"], ["search_query: risks"]]


@pytest.mark.parametrize(
    "vectors", [[], [[0.0, 0.0]], [[float("nan"), 1.0]], [[float("inf"), 1.0]], [[1.0]]]
)
def test_invalid_embedding_output(settings: Settings, vectors: list) -> None:
    settings.embedding_dimensions = 2
    embeddings = LocalEmbeddings(settings)
    embeddings.client = Mock()
    embeddings.client.embed_documents.return_value = vectors
    with pytest.raises(RAGError, match="invalid embeddings"):
        embeddings.documents(["test"])


def test_embedding_failure_is_explicit(settings: Settings) -> None:
    embeddings = LocalEmbeddings(settings)
    embeddings.client = Mock()
    embeddings.client.embed_documents.side_effect = ConnectionError("private error")
    with pytest.raises(RAGError, match="check Ollama") as error:
        embeddings.query("risks")
    assert "private error" not in str(error.value)


@pytest.mark.skipif(
    os.environ.get("SEC_RESEARCH_TEST_OLLAMA") != "1",
    reason="Set SEC_RESEARCH_TEST_OLLAMA=1 for a real local embedding smoke test",
)
def test_real_ollama_retrieval() -> None:
    embeddings = LocalEmbeddings(Settings(_env_file=None))
    vectors = embeddings.documents(
        [
            "Supply chain disruptions and reliance on component suppliers are risks.",
            "Management invested in offices, buildings and factory equipment.",
        ]
    )
    query = embeddings.query("What are the risks from supply chain disruptions?")

    def cosine(vector: list[float]) -> float:
        import math

        dot = sum(a * b for a, b in zip(vector, query, strict=True))
        return dot / math.sqrt(sum(a * a for a in vector) * sum(b * b for b in query))

    assert cosine(vectors[0]) > cosine(vectors[1])


def test_database_failure_hides_connection_details(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = Mock(side_effect=psycopg.OperationalError("secret connection details"))
    monkeypatch.setattr("app.rag_store.psycopg.connect", connect)
    with pytest.raises(RAGError, match="Local vector database") as error:
        VectorStore(settings).search(
            SearchRequest(cik="1", query="risks"), [1.0, 0.0], "test"
        )
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        {"cik": "0", "query": "risks"},
        {"cik": "1", "query": " "},
        {"cik": "1", "query": "risks", "limit": 21},
        {"cik": "1", "query": "risks", "form": "10-Q"},
        {
            "cik": "1",
            "query": "risks",
            "report_date_from": "2024-01-01",
            "report_date_to": "2023-01-01",
        },
    ],
)
def test_search_validation(body: dict) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(body)


def test_rag_api(settings: Settings, filing: Filing) -> None:
    service = make_service(settings, filing)
    app = create_app(settings)
    app.dependency_overrides[get_rag_service] = lambda: service
    with TestClient(app) as client:
        result = client.post(
            "/rag/ingest",
            json={"cik": "1", "accession_number": filing.accession_number},
        )
        assert result.status_code == 200
        assert result.json()["chunk_count"] == 3
        service.store.search.return_value = []
        response = client.post("/rag/search", json={"cik": "1", "query": "risks"})
        assert response.status_code == 200
        assert response.json()["evidence"] == []
        assert (
            client.post("/rag/search", json={"cik": "0", "query": "risks"}).status_code
            == 422
        )
        service.embeddings.query.side_effect = RAGError("Ollama unavailable")
        assert (
            client.post("/rag/search", json={"cik": "1", "query": "risks"}).status_code
            == 503
        )
        assert client.get("/health").status_code == 200
