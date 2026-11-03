"""Separate annual-filing ingestion and cited-evidence retrieval operations."""

from urllib.parse import urlsplit

from app.config import Settings
from app.filing_parser import chunk_filing, parse_sections
from app.rag_embeddings import LocalEmbeddings, RAGError
from app.rag_models import IngestRequest, IngestResult, SearchRequest, SearchResult
from app.rag_store import VectorStore
from app.sec_models import Filing
from app.sec_service import SECService


class RAGService:
    def __init__(
        self,
        sec: SECService,
        embeddings: LocalEmbeddings,
        store: VectorStore,
        settings: Settings,
    ) -> None:
        self.sec = sec
        self.embeddings = embeddings
        self.store = store
        self.settings = settings

    def ingest(self, request: IngestRequest) -> IngestResult:
        history = self.sec.get_company_filings(
            request.cik, form="10-K", include_historical=request.include_historical
        )
        filing = next(
            (
                item
                for item in history.filings
                if item.accession_number == request.accession_number
            ),
            None,
        )
        if filing is None:
            raise RAGError(
                "Requested 10-K not found in selected filing history; "
                "older filings may require include_historical=true",
                404,
            )
        self._validate_source(filing)
        html, retrieved_at = self.sec.client.get_filing_html(filing.filing_url)
        try:
            sections, warnings = parse_sections(html)
        except ValueError as exc:
            raise RAGError(str(exc), 422) from exc
        chunks = chunk_filing(
            filing,
            retrieved_at,
            sections,
            chunk_size=self.settings.rag_chunk_size,
            overlap=self.settings.rag_chunk_overlap,
        )
        # Finish all model work before atomically replacing previously indexed chunks.
        vectors = self.embeddings.documents([chunk.text for chunk in chunks])
        self.store.replace(chunks, vectors, self.embeddings.model)
        return IngestResult(
            filing=filing,
            chunk_count=len(chunks),
            embedding_model=self.embeddings.model,
            warnings=warnings,
        )

    def search(self, request: SearchRequest) -> SearchResult:
        vector = self.embeddings.query(request.query)
        evidence = self.store.search(request, vector, self.embeddings.model)
        return SearchResult(embedding_model=self.embeddings.model, evidence=evidence)

    @staticmethod
    def _validate_source(filing: Filing) -> None:
        url = urlsplit(filing.filing_url)
        accession = filing.accession_number.replace("-", "")
        expected = f"/Archives/edgar/data/{int(filing.cik)}/{accession}/"
        if (
            filing.form != "10-K"
            or not filing.primary_document
            or url.scheme != "https"
            or url.netloc != "www.sec.gov"
            or not url.path.startswith(expected)
            or url.query
            or url.fragment
            or any(part in {".", ".."} for part in url.path.split("/"))
            or "%" in url.path
        ):
            raise RAGError("Unsupported or unsafe SEC primary filing document", 422)
