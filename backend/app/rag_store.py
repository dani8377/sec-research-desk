"""Small exact cosine-search store; filing replacements are transactional."""

import re
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from app.config import Settings
from app.rag_embeddings import RAGError
from app.rag_models import Evidence, FilingChunk, SearchRequest

SCHEMA = """
CREATE TABLE IF NOT EXISTS filing_chunks (
    cik text NOT NULL,
    accession_number text NOT NULL,
    embedding_model text NOT NULL,
    dimensions integer NOT NULL,
    chunk_id text NOT NULL,
    form text NOT NULL,
    report_date date,
    section text NOT NULL,
    evidence jsonb NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (cik, accession_number, embedding_model, chunk_id)
);
CREATE INDEX IF NOT EXISTS filing_chunks_filter
ON filing_chunks (cik, embedding_model, dimensions, form, report_date);
"""


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.dsn = settings.database_url.get_secret_value()
        self._initialized = False

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as conn:
                if not self._initialized:
                    # Serialize initial DDL across application threads/processes.
                    conn.execute("SELECT pg_advisory_xact_lock(731029341)")
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.execute(SCHEMA)
                    conn.commit()
                    self._initialized = True
                register_vector(conn)
                yield conn
        except psycopg.Error as exc:
            # Do not expose connection strings or upstream database details.
            raise RAGError(
                "Local vector database unavailable; check PostgreSQL and pgvector"
            ) from exc

    def replace(
        self, chunks: list[FilingChunk], vectors: list[list[float]], model: str
    ) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Provide one embedding per filing chunk")
        filing = chunks[0].filing
        if any(chunk.filing != filing for chunk in chunks):
            raise ValueError("A replacement must contain exactly one filing")
        with self._connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"{filing.cik}:{filing.accession_number}:{model}",),
            )
            conn.execute(
                "DELETE FROM filing_chunks WHERE cik = %s "
                "AND accession_number = %s AND embedding_model = %s",
                (filing.cik, filing.accession_number, model),
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO filing_chunks
                    (cik, accession_number, embedding_model, dimensions, chunk_id,
                     form, report_date, section, evidence, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (
                            filing.cik,
                            filing.accession_number,
                            model,
                            len(vector),
                            chunk.chunk_id,
                            filing.form,
                            filing.report_date,
                            chunk.section,
                            Jsonb(chunk.model_dump(mode="json")),
                            Vector(vector),
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )

    def search(
        self, request: SearchRequest, vector: list[float], model: str
    ) -> list[Evidence]:
        conditions = [
            "cik = %s",
            "embedding_model = %s",
            "dimensions = %s",
            "form = %s",
        ]
        values: list[object] = [request.cik, model, len(vector), request.form]
        for column, operator, value in (
            ("accession_number", "=", request.accession_number),
            ("report_date", ">=", request.report_date_from),
            ("report_date", "<=", request.report_date_to),
        ):
            if value is not None:
                conditions.append(f"{column} {operator} %s")
                values.append(value)
        if request.section is not None:
            # Match an Item boundary, so Item 7 never includes Item 7A.
            item = re.fullmatch(r"Item ([0-9]{1,2}[A-C]?)\.?", request.section)
            if item:
                conditions.append("section LIKE %s")
                values.append(f"Item {item[1]}. %")
            else:
                conditions.append("section = %s")
                values.append(request.section)
        # Exact search keeps small local corpora simple and applies filters first.
        sql = f"""SELECT evidence, 1 - (embedding <=> %s) AS similarity
                  FROM filing_chunks WHERE {" AND ".join(conditions)}
                  ORDER BY embedding <=> %s, chunk_id LIMIT %s"""
        with self._connection() as conn:
            rows = conn.execute(
                sql, [Vector(vector), *values, Vector(vector), request.limit]
            ).fetchall()
        return [
            Evidence(chunk=FilingChunk.model_validate(row[0]), similarity=row[1])
            for row in rows
        ]
