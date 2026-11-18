# Reliability follow-up to milestone 7

These changes address measured backend failures in milestones 3–7. The remaining
milestone 8 concerns the frontend, visualizations, documentation and demo; it does
not schedule these reliability fixes.

## Workflow changes

- Section names are resolved against the actual indexed section labels for the
  requested company, filing, form, dates, embedding model and dimensions. Short
  names such as `Business` and `Risk Factors`, full titles and case-insensitive
  Item identifiers are accepted. Unknown/ambiguous names return a corrective
  `422` response listing the available sections. A missing corpus stays an empty
  result rather than an invented section.
- HTTP search keeps strict section filtering by default. Research tools enable
  `allow_section_fallback` by default: an empty valid section search retries once
  without that section, preserving all other filters and reusing the embedding.
  The response and final `tool_warnings` disclose this wider search. Set the flag
  false when the user requires a particular section. Unknown names are reported
  as invalid filters rather than silently broadened.
- Only the research workflows repair an omitted search `query`, by reusing the
  original user question. The repair is recorded in `tool_warnings`. Empty or
  invalid queries still fail validation; missing CIKs/accessions are not guessed.
  Standalone MCP input schemas still require a query.
- Both workflows share compact tool-selection previews. Complete passages,
  source metadata and evidence IDs stay in the evidence registry. Already seen
  chunks are not repeated in previews. Intermediate model prose is discarded;
  the model is instructed to respond briefly when tool selection is finished.
- Identical successful tool requests are fingerprinted after input validation.
  A repeat consumes the attempt budget but does not perform another tool/MCP
  request. The workflow completes any already queued tools and then synthesizes
  using its retained evidence. Graph fingerprints and warnings are checkpointed.
- If selection context still exceeds its character budget, only complete older
  tool-call/result rounds are removed. System instructions, the original request
  and the latest round are retained. Incomplete tool pairs cannot be trimmed.
  Full evidence remains available independently of this conversation.
- Synthesis receives complete passages within the configured character budget;
  oversized/excess passages are omitted with an explicit limitation. Excluded
  evidence IDs cannot support claims. If no passage fits, the result explicitly
  reports insufficient evidence. This is a character guard, not an exact tokenizer.
- The baseline now returns a deterministic insufficient-evidence result when
  nothing was retrieved, matching the graph's existing protection. Empty evidence
  is never sent to model synthesis. Service/transport failures remain failures.

No new services, dependencies, financial calculation tools or model-generated
financial arithmetic were introduced.

## Reading the improved Phoenix interface

Open `http://127.0.0.1:6006`, select `sec-research-desk`, and include the run date
in the time filter. Each `evaluation.case` is now an `EVALUATOR` span with the
question in Input and a concise answer/status/errors object in Output. Failed
executions mark the root span as an error; successful abstention is distinct
from answering an answerable question.

Every newly recorded result includes `trace_id` and `span_id`. The two attributes
`evaluation.execution_success` and `evaluation.status_match` remain available.
After a run, the evaluator flushes traces and exports non-null checks as Phoenix
span annotations with `annotator_kind=CODE`. The annotation explanations identify
lexical checks and citation identity as proxies. Full scores remain in JSONL.

The export uses the running collector's supported `/v1/span_annotations` API,
batches at most 100 annotations, and retries an ingestion-queue `404` at most
four times after the initial attempt. Other failures are logged and saved in the
final `phoenix_annotations` record. Optional collector failure does not erase
research results or turn a successful research execution into a model failure.
Old recorded results without span IDs cannot be automatically linked to traces.

Human correctness and unsupported-claim ratings are never automatically invented.
Genuine manual reviews are exported as `HUMAN` annotations, separately from code
checks. `unsupported_claim_rate` is a fraction where **lower is better**.

## Manual review

Generate an unfilled review file and a companion file containing references,
answers, retrieved evidence, failures and trace IDs. From `backend/`:

```bash
uv run python -m app.evaluation_review \
  --run evaluation/results-reliability.jsonl \
  --dataset evaluation/dataset.json \
  --template /tmp/sec-research-reviews.json
```

A person should assess whether the answer addresses the question, whether each
claim is supported by its cited evidence, and whether an abstention was justified.
Use `semantic_correctness` for the answer-level judgment and `unsupported_claims`
for a count of unsupported claims. Enter `reviewer` and explanatory `notes` for
any rating. Leave undecided fields null. Do not label an LLM-generated judgment
as a human review.

Apply the completed file to a new results file without rerunning inference:

```bash
SEC_RESEARCH_PHOENIX_ENABLED=true uv run python -m app.evaluation_review \
  --run evaluation/results-reliability.jsonl \
  --reviews /tmp/sec-research-reviews.json \
  --output /tmp/sec-research-reviewed-results.jsonl
```

The review file is tied to the exact original run SHA-256. Unknown/duplicate
case/workflow pairs, unattributed ratings and impossible unsupported-claim counts
are rejected. Existing files are never overwritten. Unrated answers stay
unscored. Replacing a rated review clears its unspecified human scores, so an
older person's rating cannot be attributed to the new reviewer. Summaries show
the number of reviewed answers; the mean unsupported
claim fraction is a per-answer average, not a pooled claim count.

## Separate evaluation set

`heldout.json` contains eight new cases using Microsoft's official 2025 Form 10-K:
CIK `0000789019`, accession `0000950170-25-100235`, filed 2025-07-30,
report date 2025-06-30. Source:
https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm.

Six answerable cases cover business facts, a reported founding year, product
categories, competition, management commentary and a reported percentage. Two
cases require abstention. Quotes were checked against the original filing and
sections using its recorded HTML SHA-256. Expected numerical tokens were copied
from source paragraphs; no arithmetic was used to generate expected values.

The questions were set aside before the live follow-up experiments, and no
prompts were tuned against their observed results. This provides another company
and filing format, but is still a small, source-checked dataset without independent
human semantic ratings. It is not a permanently secret or blinded test set.

Both filings must be indexed before running these experiments. Microsoft's
filing was ingested locally into 247 chunks; Apple's original 142 chunks were
preserved. The runner records the corpus inventory, chunk-ID fingerprints, model
digests, package versions and implementation hash in each manifest. The original
30-case dataset and November 12 results are unchanged. Preserve new independent
output files for repeated trials; one trial does not measure run-to-run variance.

```bash
SEC_RESEARCH_PHOENIX_ENABLED=true uv run python -m app.evaluation \
  --output evaluation/results-reliability-new.jsonl

SEC_RESEARCH_PHOENIX_ENABLED=true uv run python -m app.evaluation \
  --dataset evaluation/heldout.json \
  --output evaluation/heldout-results-new.jsonl
```

The local inference server reported Ollama `0.40.14` during the follow-up,
with Qwen loaded at a 16,384-token context and nonzero reported VRAM usage.
The original experiment did not record an Ollama server version. Before/after
latencies are observations across separate dates, not a controlled measurement
of a code-only speedup. Baseline and graph within each new run share the same
local server and model configuration.

## Recorded results

The original Apple comparison is saved in `results-reliability.jsonl`. All
60 case/workflow pairs were attempted. The original dataset and initial results
were byte-compared with Git and remain unchanged. The saved summary was
recomputed from individual records and verified.

| Measure | Baseline initial | Baseline follow-up | Graph initial | Graph follow-up |
| --- | ---: | ---: | ---: | ---: |
| Completed with a valid result | 22/30 | 30/30 | 27/30 | 29/30 |
| Expected answer/abstention status | 22/30 | 30/30 | 24/30 | 29/30 |
| Reference-term proxy matched | 20/30 | 30/30 | 22/30 | 28/30 |
| Answered answerable cases | 18/20 | 20/20 | 14/20 | 19/20 |
| Correctly abstained on unsupported cases | 4/10 | 10/10 | 10/10 | 10/10 |
| Reference hit in retrieved evidence | 14/20 | 18/20 | 14/20 | 18/20 |
| Citation identity on scored answers | 18/18 | 20/20 | 14/14 | 19/19 |
| Reported-number token check | 0 scored | 1/1 | 0 scored | 1/1 |
| Mean latency, seconds | 32.8 | 50.2 | 51.9 | 59.2 |
| Median latency, seconds | 32.0 | 49.2 | 51.9 | 68.6 |
| Model calls | 83 | 102 | 95 | 123 |
| Reported input + output tokens | 152,533 | 271,623 | 170,293 | 297,154 |

Completion and the reference proxies improved on these known development cases;
this is not a measured semantic-accuracy score. Mean latency increased and more
model calls/tokens were used. The graph now answers more cases instead of
terminating early, but a code-only explanation for timing differences is not
established. It remains slower than the direct baseline in the aggregate.

The remaining graph failure is `apple-12`: tool selection exceeded the six-attempt
budget after searches with different query arguments. Exact request deduplication
does not detect every semantically redundant search. The workflow still reports
that limit as an explicit failure. The graph's `apple-10` answer remains a lexical
false negative under the unchanged rubric. No context-limit failures occurred in
this follow-up, but these runs do not prove context handling is correct for all
questions or corpus sizes.

Phoenix stored 360 annotations for the 60 new Apple spans, all tagged `CODE`.
No human semantic correctness or unsupported-claim scores were generated.

The Apple run began before a redundant duplicate definition of the identical
`_selection_messages` helper was removed. Reconstructing that duplicate exactly
reproduces the recorded implementation hash; the effective method and workflow
behavior were unchanged. The Microsoft run uses the cleaned source hash.
Neither prompts nor model settings were tuned during either live run.

### Separate Microsoft results

The separate Microsoft run is saved in `heldout-results.jsonl`: eight cases,
16 unique case/workflow executions. It completed without execution failures.
The reference and case definitions were not edited after viewing these results.

| Measure | Baseline | Graph |
| --- | ---: | ---: |
| Completed with a valid result | 8/8 | 8/8 |
| Expected answer/abstention status | 8/8 | 8/8 |
| Reference-term proxy matched | 7/8 | 7/8 |
| Citation identity | 6/6 | 6/6 |
| Reference hit in retrieved evidence | 3/6 | 4/6 |
| Reported-number token checks | 2/2 | 2/2 |
| Mean latency, seconds | 31.6 | 47.8 |
| Model calls | 22 | 32 |
| Reported input + output tokens | 38,876 | 59,105 |

Both reference-term misses were `msft-heldout-05`, which asks for significant
expense categories. The rubric requires the substring `employees`, while both
answers use `employee compensation`. This is a lexical mismatch; it is not proof
of a wrong answer. The full answer and cited evidence are retained for review.
Reference-hit checks also miss cases where the reference paragraph spans chunk
boundaries or another passage supports the answer. These proxies should not be
reported as semantic accuracy or complete retrieval relevance.

The new set improves coverage beyond Apple, but still consists of only two
issuers across the combined experiments, a single trial per question and no
independent human semantic ratings. The reported-number checks concern copied
facts, not financial calculations. No claim is made that all material claims are
supported simply because citation identity passes.

### Audit and implementation notes

All 76 new case/workflow pairs, every deterministic metric, both dataset hashes,
and both saved summaries were verified. Phoenix persisted all 76 linked
`EVALUATOR` spans with question/answer summaries and all 464 exported checks
(360 Apple, 104 Microsoft), each labelled `CODE`. No human ratings were generated.

Each manifest fingerprints the source loaded when its process started. An
identical duplicate `_selection_messages` definition was removed after the Apple
process started and before the Microsoft process. Reconstructing that redundant
definition reproduces the Apple source fingerprint exactly; the effective method
body and all research behavior are unchanged. The Microsoft fingerprint matches
the research source before final manual-review replacement safety was tightened.
That later change only clears stale human ratings when replacing a review; it
does not affect model execution or automatic checks. No prompts or model settings
were tuned during either recorded run.

The final ordinary suite passes with 177 tests passed and 7 optional integration
tests skipped. Four PostgreSQL integration tests passed in isolated schemas,
covering section resolution, corpus inventory and durable graph recovery. Ruff
lint/format and Git whitespace checks passed. The manual-review CLI was exercised
end to end with 60 unfilled reviews: scores stayed unscored and the original run
was unchanged. Source-reference verification passed against Microsoft's original
captured SEC HTML.

Remaining work includes evidence-coverage stopping for differently worded
repeated searches, lower inference cost/latency, repeated trials, broader data
and independent human review. The tool budget remains bounded and explicit:
`apple-12` is still an execution failure rather than an invented successful answer.
