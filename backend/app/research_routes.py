from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.research_models import ResearchRequest, ResearchResult
from app.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["Research baseline"])


def get_research_service(request: Request) -> ResearchService:
    return request.app.state.research_service


@router.post("/baseline", response_model=ResearchResult)
def research(
    body: ResearchRequest,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchResult:
    return service.research(body)
