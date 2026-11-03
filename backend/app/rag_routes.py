from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.rag_models import IngestRequest, IngestResult, SearchRequest, SearchResult
from app.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["Filing evidence"])


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


Service = Annotated[RAGService, Depends(get_rag_service)]


@router.post("/ingest", response_model=IngestResult)
def ingest_filing(body: IngestRequest, service: Service) -> IngestResult:
    return service.ingest(body)


@router.post("/search", response_model=SearchResult)
def search_evidence(body: SearchRequest, service: Service) -> SearchResult:
    return service.search(body)
