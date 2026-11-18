"""Human reviews must be attributable and tied to the exact saved run."""

import hashlib

import pytest
from pydantic import ValidationError

from app.evaluation_review import HumanReview, ReviewFile, apply_reviews


def run_rows():
    return [
        {
            "case_id": "one",
            "workflow": "baseline",
            "result": {"claims": [{"text": "claim"}, {"text": "claim two"}]},
            "metrics": {"semantic_correctness": None, "unsupported_claim_rate": None},
        }
    ]


def test_human_reviews_apply_without_changing_original():
    raw = b"original evaluation run"
    reviews = ReviewFile(
        run_sha256=hashlib.sha256(raw).hexdigest(),
        reviews=[
            HumanReview(
                case_id="one",
                workflow="baseline",
                semantic_correctness=False,
                unsupported_claims=1,
                reviewer="human tester",
                notes="Second claim lacks support",
            )
        ],
    )
    rows = run_rows()
    reviewed = apply_reviews(raw, rows, reviews)
    assert reviewed[0]["metrics"] == {
        "semantic_correctness": False,
        "unsupported_claim_rate": 0.5,
    }
    assert rows[0]["metrics"]["semantic_correctness"] is None
    with pytest.raises(ValueError, match="different evaluation"):
        apply_reviews(b"different run", rows, reviews)
    reviews.reviews[0].unsupported_claims = 3
    with pytest.raises(ValueError, match="claim count"):
        apply_reviews(raw, rows, reviews)


def test_empty_reviews_stay_unscored_and_ratings_require_attribution():
    raw = b"run"
    pending = HumanReview(case_id="one", workflow="baseline")
    reviews = ReviewFile(run_sha256=hashlib.sha256(raw).hexdigest(), reviews=[pending])
    assert apply_reviews(raw, run_rows(), reviews) == run_rows()
    with pytest.raises(ValidationError, match="human reviewer"):
        HumanReview(case_id="one", workflow="baseline", semantic_correctness=True)
    with pytest.raises(ValidationError):
        HumanReview(
            case_id="one",
            workflow="baseline",
            semantic_correctness="yes",
            reviewer="reviewer",
            notes="notes",
        )
    with pytest.raises(ValidationError, match="Duplicate"):
        ReviewFile(run_sha256=reviews.run_sha256, reviews=[pending, pending])


def test_replacing_review_cannot_attribute_an_old_rating_to_a_new_reviewer():
    raw = b"previously reviewed run"
    rows = run_rows()
    rows[0]["review"] = {"reviewer": "first person", "notes": "Original review"}
    rows[0]["metrics"]["unsupported_claim_rate"] = 0.5
    reviews = ReviewFile(
        run_sha256=hashlib.sha256(raw).hexdigest(),
        reviews=[
            HumanReview(
                case_id="one",
                workflow="baseline",
                semantic_correctness=True,
                reviewer="second person",
                notes="Reviewed answer correctness only",
            )
        ],
    )
    reviewed = apply_reviews(raw, rows, reviews)
    assert reviewed[0]["review"]["reviewer"] == "second person"
    assert reviewed[0]["metrics"]["semantic_correctness"] is True
    assert reviewed[0]["metrics"]["unsupported_claim_rate"] is None
    assert rows[0]["metrics"]["unsupported_claim_rate"] == 0.5
