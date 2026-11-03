"""Validated ingestion, retrieval and citation contracts."""

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)

from app.sec_models import Filing, normalize_cik

CompanyCIK = Annotated[str, AfterValidator(normalize_cik)]
Accession = Annotated[str, Field(pattern=r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")]


class IngestRequest(BaseModel):
    cik: CompanyCIK
    accession_number: Accession
    include_historical: bool = False


class SearchRequest(BaseModel):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    cik: CompanyCIK
    accession_number: Accession | None = None
    form: Literal["10-K"] = "10-K"
    report_date_from: date | None = None
    report_date_to: date | None = None
    section: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Exact title or Item identifier, for example Item 7 or Item 1A.",
    )
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def ordered_dates(self) -> "SearchRequest":
        if (
            self.report_date_from
            and self.report_date_to
            and self.report_date_from > self.report_date_to
        ):
            raise ValueError("report_date_from must not exceed report_date_to")
        return self


class FilingChunk(BaseModel):
    chunk_id: str
    filing: Filing
    retrieved_at: datetime
    section: str
    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_offsets(self) -> "FilingChunk":
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("Chunk offsets must match the evidence text length")
        return self


class IngestResult(BaseModel):
    filing: Filing
    chunk_count: int
    embedding_model: str
    warnings: list[str]


class Evidence(BaseModel):
    chunk: FilingChunk
    similarity: float = Field(allow_inf_nan=False)


class SearchResult(BaseModel):
    embedding_model: str
    evidence: list[Evidence]
