"""Validated SEC wire formats and public service results."""

import re
from datetime import date, datetime
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictFloat,
    StrictInt,
    model_validator,
)


def normalize_cik(value: str | int) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9]{1,10}", text) or int(text) == 0:
        raise ValueError("CIK must contain 1–10 digits and be greater than zero")
    return text.zfill(10)


class SECModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class TickerEntry(SECModel):
    cik_str: int = Field(gt=0, le=9999999999)
    ticker: str = Field(min_length=1)
    title: str = Field(min_length=1)


class TickerDirectory(RootModel[dict[str, TickerEntry]]):
    pass


class Company(BaseModel):
    cik: str
    ticker: str
    name: str
    source_url: str


class FilingColumns(SECModel):
    accessionNumber: list[
        Annotated[str, Field(pattern=r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")]
    ]
    filingDate: list[date]
    reportDate: list[str]
    form: list[Annotated[str, Field(min_length=1)]]
    primaryDocument: list[str]

    @model_validator(mode="after")
    def aligned_columns(self) -> "FilingColumns":
        count = len(self.accessionNumber)
        for name in ("filingDate", "reportDate", "form", "primaryDocument"):
            if len(getattr(self, name)) != count:
                raise ValueError(f"SEC filing column {name} has inconsistent length")
        for value in self.reportDate:
            if value:
                date.fromisoformat(value)
        return self


class HistoricalFile(SECModel):
    name: str = Field(pattern=r"^CIK[0-9]{10}-submissions-[0-9]+\.json$")
    filingFrom: date
    filingTo: date


class SubmissionsFilings(SECModel):
    recent: FilingColumns
    files: list[HistoricalFile]


class Submissions(SECModel):
    cik: Annotated[str | int, AfterValidator(normalize_cik)]
    name: str
    filings: SubmissionsFilings


class Filing(BaseModel):
    cik: str
    accession_number: str
    filing_date: date
    report_date: date | None
    form: str
    primary_document: str | None
    filing_url: str
    source_url: str


class FilingHistory(BaseModel):
    cik: str
    company_name: str
    includes_historical: bool
    filings: list[Filing]


class Fact(SECModel):
    val: StrictInt | StrictFloat = Field(allow_inf_nan=False)
    end: date
    start: date | None = None
    accn: str = Field(pattern=r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
    filed: date
    form: str
    fy: int | None = None
    fp: str | None = None
    frame: str | None = None


class Concept(SECModel):
    label: str | None
    description: str | None = None
    units: dict[str, list[Fact]]


class CompanyFacts(SECModel):
    cik: int = Field(gt=0, le=9999999999)
    entityName: str
    facts: dict[str, dict[str, Concept]]


class FactsResult(BaseModel):
    cik: str
    source_url: str
    retrieved_at: datetime
    data: CompanyFacts
