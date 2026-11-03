"""Opt-in pgvector integration; uses an isolated schema in the supplied database."""

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.config import Settings
from app.filing_parser import chunk_filing
from app.rag_embeddings import RAGError
from app.rag_models import FilingChunk, SearchRequest
from app.rag_store import VectorStore
from app.sec_models import Filing


@pytest.fixture
def store() -> Iterator[VectorStore]:
    dsn = os.environ.get("SEC_RESEARCH_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set SEC_RESEARCH_TEST_DATABASE_URL for local pgvector tests")
    schema = "rag_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        test_dsn = make_conninfo(dsn, options=f"-c search_path={schema},public")
        yield VectorStore(Settings(_env_file=None, database_url=test_dsn))
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def chunks(
    cik: str = "0000000001", accession: str = "0000000001-24-000001"
) -> list[FilingChunk]:
    filing = Filing(
        cik=cik,
        accession_number=accession,
        filing_date=date(2024, 2, 1),
        report_date=date(2023, 12, 31),
        form="10-K",
        primary_document="synthetic.htm",
        filing_url=(
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/synthetic.htm"
        ),
        source_url=f"https://data.sec.gov/submissions/CIK{cik}.json",
    )
    return chunk_filing(
        filing,
        datetime(2024, 2, 1, tzinfo=UTC),
        [
            ("Item 1A. Risk Factors", "Synthetic supply chain evidence"),
            ("Item 7. Management Discussion", "Synthetic facilities evidence"),
        ],
    )


def test_cosine_ranking_citations_and_filters(store: VectorStore) -> None:
    source = chunks()
    store.replace(source, [[1.0, 0.0], [0.0, 1.0]], "test-model")
    store.replace(
        chunks("0000000002", "0000000002-24-000001"),
        [[1.0, 0.0], [1.0, 0.0]],
        "test-model",
    )
    store.replace(source, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "other-dimensions")
    request = SearchRequest(cik="1", query="supply chain")
    result = store.search(request, [1.0, 0.0], "test-model")
    assert len(result) == 2
    assert result[0].chunk == source[0]
    assert result[0].similarity == pytest.approx(1.0)
    assert result[1].similarity == pytest.approx(0.0)
    assert all(item.chunk.filing.cik == request.cik for item in result)
    assert store.search(request, [1.0, 0.0], "absent-model") == []
    for filters in (
        {"accession_number": "0000000001-24-000002"},
        {"report_date_from": date(2024, 1, 1)},
        {"report_date_to": date(2022, 12, 31)},
        {"section": "absent section"},
    ):
        assert (
            store.search(request.model_copy(update=filters), [1.0, 0.0], "test-model")
            == []
        )
    filtered = request.model_copy(update={"section": source[1].section, "limit": 1})
    assert store.search(filtered, [1.0, 0.0], "test-model")[0].chunk == source[1]
    for section in ("Item 7", "Item 7."):
        filtered = request.model_copy(update={"section": section})
        result = store.search(filtered, [1.0, 0.0], "test-model")
        assert [item.chunk for item in result] == [source[1]]
    filtered = request.model_copy(update={"section": "Item 1"})
    assert store.search(filtered, [1.0, 0.0], "test-model") == []
    # Same model name with another dimension must never enter the distance operation.
    store.replace(
        chunks(accession="0000000001-24-000003"), [[1.0, 0.0, 0.0]] * 2, "test-model"
    )
    assert len(store.search(request, [1.0, 0.0], "test-model")) == 2


def test_reingestion_replaces_and_failed_transaction_rolls_back(
    store: VectorStore,
) -> None:
    source = chunks()
    request = SearchRequest(cik="1", query="supply chain")
    store.replace(source, [[1.0, 0.0], [0.0, 1.0]], "test-model")
    store.replace(source[:1], [[1.0, 0.0]], "test-model")
    assert len(store.search(request, [1.0, 0.0], "test-model")) == 1
    # pgvector rejects NaN after DELETE; the transaction must preserve the old row.
    with pytest.raises(RAGError):
        store.replace(source, [[1.0, 0.0], [float("nan"), 0.0]], "test-model")
    assert store.search(request, [1.0, 0.0], "test-model")[0].chunk == source[0]
