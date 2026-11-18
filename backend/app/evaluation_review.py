"""Prepare and apply explicit human reviews without rerunning local models."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from app.config import Settings
from app.evaluation import EvaluationDataset, summarize
from app.telemetry import export_scores


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    workflow: Literal["baseline", "graph"]
    semantic_correctness: StrictBool | None = None
    unsupported_claims: int | None = Field(default=None, ge=0, strict=True)
    reviewer: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def attributable_rating(self) -> "HumanReview":
        if self.semantic_correctness is not None or self.unsupported_claims is not None:
            if not self.reviewer.strip() or not self.notes.strip():
                raise ValueError("Rated reviews require a human reviewer and notes")
        return self


class ReviewFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviews: list[HumanReview]

    @model_validator(mode="after")
    def unique_pairs(self) -> "ReviewFile":
        keys = [(r.case_id, r.workflow) for r in self.reviews]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate review case/workflow")
        return self


def load_records(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records or "manifest" not in records[0]:
        raise ValueError("Missing evaluation manifest")
    return raw, records


def apply_reviews(raw: bytes, rows: list[dict], reviews: ReviewFile) -> list[dict]:
    if hashlib.sha256(raw).hexdigest() != reviews.run_sha256:
        raise ValueError("Reviews refer to a different evaluation run")
    keys = [(row["case_id"], row["workflow"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Evaluation run contains duplicate case/workflow pairs")
    output = json.loads(json.dumps(rows))
    by_key = {(row["case_id"], row["workflow"]): row for row in output}
    for review in reviews.reviews:
        key = (review.case_id, review.workflow)
        if key not in by_key:
            raise ValueError(f"Unknown review case/workflow: {key}")
        row = by_key[key]
        if review.semantic_correctness is None and review.unsupported_claims is None:
            continue
        row["review"] = review.model_dump()
        row["metrics"]["semantic_correctness"] = review.semantic_correctness
        row["metrics"]["unsupported_claim_rate"] = None
        if review.unsupported_claims is not None:
            count = len((row["result"] or {}).get("claims", []))
            if not count or review.unsupported_claims > count:
                raise ValueError("Unsupported claim count must fit an existing answer")
            row["metrics"]["unsupported_claim_rate"] = review.unsupported_claims / count
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--template", type=Path)
    group.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    raw, records = load_records(args.run)
    rows = [r for r in records if "case_id" in r]
    if args.template:
        questions = {}
        if args.dataset:
            dataset_raw = args.dataset.read_bytes()
            if (
                hashlib.sha256(dataset_raw).hexdigest()
                != records[0]["manifest"]["dataset_sha256"]
            ):
                parser.error("Dataset does not match the evaluation run")
            dataset = EvaluationDataset.model_validate_json(dataset_raw)
            questions = {
                case.id: case.model_dump(mode="json") for case in dataset.cases
            }
        # Keep evidence context separate from the importable ratings.
        template = ReviewFile(
            run_sha256=hashlib.sha256(raw).hexdigest(),
            reviews=[
                HumanReview(case_id=r["case_id"], workflow=r["workflow"]) for r in rows
            ],
        )
        with args.template.open("x") as output:
            output.write(template.model_dump_json(indent=2) + "\n")
        if questions:
            context = args.template.with_suffix(".context.json")
            with context.open("x") as output:
                json.dump(
                    [
                        {
                            "case_id": row["case_id"],
                            "workflow": row["workflow"],
                            "reference": questions[row["case_id"]],
                            "result": row["result"],
                            "retrievals": row["statistics"]["retrievals"],
                            "errors": row["errors"],
                            "trace_id": row.get("trace_id"),
                        }
                        for row in rows
                    ],
                    output,
                    indent=2,
                )
        return
    if not args.output:
        parser.error("--output is required when applying reviews")
    reviewed = apply_reviews(
        raw, rows, ReviewFile.model_validate_json(args.reviews.read_bytes())
    )
    settings = Settings()
    with args.output.open("x") as output:
        output.write(json.dumps(records[0]) + "\n")
        for row in reviewed:
            output.write(json.dumps(row) + "\n")
        output.flush()
        export = export_scores(reviewed, settings)
        output.write(
            json.dumps(
                {
                    "summary": summarize(reviewed),
                    "phoenix_annotations": export,
                    "reviewed_run_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
