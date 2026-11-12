from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from app.research_graph import GraphResearchService
from app.research_models import (
    GraphResearchRequest,
    GraphResearchRun,
    ResearchRequest,
    ResearchResult,
)
from app.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["Research"])


def get_research_service(request: Request) -> ResearchService:
    return request.app.state.research_service


@router.post("/baseline", response_model=ResearchResult)
def research(
    body: ResearchRequest,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchResult:
    return service.research(body)


def get_graph_service(request: Request) -> GraphResearchService:
    return request.app.state.graph_research_service


@router.post("/graph", response_model=GraphResearchRun)
def graph_research(
    body: GraphResearchRequest,
    service: Annotated[GraphResearchService, Depends(get_graph_service)],
) -> GraphResearchRun:
    return service.run(body)


@router.get("/graph/{run_id}", response_model=GraphResearchRun)
def get_graph_run(
    run_id: UUID,
    service: Annotated[GraphResearchService, Depends(get_graph_service)],
) -> GraphResearchRun:
    return service.get_run(run_id)


@router.post("/graph/{run_id}/resume", response_model=GraphResearchRun)
def resume_graph_run(
    run_id: UUID,
    service: Annotated[GraphResearchService, Depends(get_graph_service)],
) -> GraphResearchRun:
    return service.get_run(run_id, resume=True)
