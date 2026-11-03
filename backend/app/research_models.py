"""Structured baseline requests and evidence-backed response contracts."""

from typing import Annotated, Literal

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
