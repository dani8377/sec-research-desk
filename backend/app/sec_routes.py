from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import AfterValidator, StringConstraints

from app.sec_models import Company, FactsResult, FilingHistory, normalize_cik
from app.sec_service import SECService

router = APIRouter(prefix="/companies", tags=["SEC data"])
CIK = Annotated[str, Path(pattern=r"^[0-9]{1,10}$"), AfterValidator(normalize_cik)]


def get_sec_service(request: Request) -> SECService:
    return request.app.state.sec_service


Service = Annotated[SECService, Depends(get_sec_service)]


@router.get("", response_model=list[Company])
def find_company(
    service: Service,
    q: Annotated[
        str,
        Query(max_length=200),
        StringConstraints(strip_whitespace=True, min_length=1),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[Company]:
    return service.find_company(q, limit)


@router.get("/{cik}/filings", response_model=FilingHistory)
def get_company_filings(
    cik: CIK,
    service: Service,
    form: Annotated[str | None, Query(min_length=1, max_length=30)] = None,
    include_historical: bool = False,
) -> FilingHistory:
    return service.get_company_filings(
        cik, form=form, include_historical=include_historical
    )


@router.get(
    "/{cik}/facts", response_model=FactsResult, response_model_exclude_unset=True
)
def get_company_facts(cik: CIK, service: Service) -> FactsResult:
    return service.get_company_facts(cik)
