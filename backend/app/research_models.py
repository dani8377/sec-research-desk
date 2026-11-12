"""Validated research requests, plans and evidence-backed response contracts."""

from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.rag_models import Evidence

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[Text, Field(max_length=2000)]


class ResearchClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[Text, Field(max_length=2000)]
    kind: Literal["reported_fact", "interpretation"]
    evidence_ids: list[Text] = Field(min_length=1, max_length=10)


class ResearchDraft(BaseModel):
    """Model output; source metadata is attached by Python, never generated."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence"]
    claims: list[ResearchClaim] = Field(max_length=12)
    limitations: list[Annotated[Text, Field(max_length=2000)]] = Field(max_length=12)

    @model_validator(mode="after")
    def meaningful_result(self) -> "ResearchDraft":
        if self.status == "answered" and not self.claims:
            raise ValueError("An answered result must contain cited claims")
        if self.status == "insufficient_evidence" and not self.limitations:
            raise ValueError("Insufficient evidence must include a limitation")
        return self


class ResearchResult(ResearchDraft):
    model: str
    evidence: list[Evidence]
    tools_called: list[str]
    model_calls: int
    tool_errors: list[str] = Field(default_factory=list)


class GraphResearchRequest(ResearchRequest):
    run_id: UUID = Field(default_factory=uuid4)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["filing_research", "unsupported"]
    objectives: list[Annotated[Text, Field(max_length=500)]] = Field(
        min_length=1, max_length=5
    )
    limitation: Annotated[Text, Field(max_length=1000)] | None

    @model_validator(mode="after")
    def unsupported_reason(self) -> "ResearchPlan":
        if self.scope == "unsupported" and not self.limitation:
            raise ValueError("Unsupported research requires an explanation")
        return self


class ResearchFailure(BaseModel):
    stage: str
    message: str
    status_code: int


class GraphResearchRun(BaseModel):
    run_id: UUID
    status: Literal["running", "completed", "failed"]
    question: str
    plan: ResearchPlan | None
    result: ResearchResult | None
    errors: list[ResearchFailure]
    tools_called: list[str]
    model_calls: int
    next_nodes: list[str]
