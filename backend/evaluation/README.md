# Milestone 7 evaluation

`dataset.json` contains 30 cases scoped to the tools actually implemented:
20 answerable questions and 10 requests requiring abstention. The answerable
cases cover Apple business facts, risk factors, management commentary, and one
reported numerical fact. Negative cases cover unavailable financial calculations,
multi-company calculations, historical comparisons without the historical filing,
and absent future-period evidence.

The reference paragraphs are verbatim normalized text from Apple's
[2025 Form 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm),
CIK `0000320193`, accession `0000320193-25-000079`, filed 2025-10-31,
report date 2025-09-27. The complete original HTML was downloaded through the
existing SEC client on 2026-11-03. All 20 reference quotes were checked against
their original Item sections during implementation. The dataset retains the
source HTML SHA-256, retrieval timestamp, filing metadata, quotes, answer terms,
and explicit reasons for unsupported cases. No financial figures were generated
or calculated by a model to establish expected answers.

This is an initial source-checked dataset, not an independently human-reviewed
benchmark. It is concentrated on one issuer and one filing. The negative cases
are expected abstentions for the current product scope, not evidence that the
requested information never exists at the SEC.

## Running

Start PostgreSQL and native Ollama, install `qwen3:8b` and `nomic-embed-text`,
configure the SEC contact header, and ingest this exact annual filing through
`POST /rag/ingest` (see the root README). Both workflows use the same local
index. Do not run other research requests or change that index during evaluation.
The runner does not ingest filings automatically. Navigation tools may fetch
SEC data through the normal cache and fair-access limiter; ordinary tests remain
offline.

From `backend/`, run the full paired benchmark:

```bash
uv run python -m app.evaluation --output evaluation/my-results.jsonl
```

The output must be a new file; existing results are never overwritten. Select
individual cases by repeating `--case`, for example:

```bash
uv run python -m app.evaluation --output evaluation/sample.jsonl \
  --case apple-02 --case scope-27
```

To recheck references against a saved copy of the exact source HTML without
starting models or connecting to the database:

```bash
uv run python -m app.evaluation --verify-source-html /path/to/aapl-20250927.htm
```

A hash mismatch fails explicitly: use the original captured HTML snapshot, or
review and version the references before accepting a different source snapshot.

## Methodology

For each case, run both production services sequentially, alternating which goes
first. The baseline uses direct service tools; the graph uses the same tool
contracts over MCP stdio and real PostgreSQL checkpoints. Both share the chat
model, temperature zero, context/output limits, tool budget, synthesis prompt,
citation validator and local index. Graph planning and MCP/checkpoint overhead
remain part of its measured cost. This measures the implemented systems, not
an isolated experiment on graph topology alone.

The first JSONL record is an allowlisted configuration and dataset hash manifest.
Each following record contains the answer, evidence, errors, graph run reference,
wall-clock latency, actual model/tool calls and reported tokens. Retrieval tool
inputs and outputs are captured before uncited evidence is discarded. Failed
model/tool calls remain counted. Missing token metadata is `null`, and partial
usage is explicitly marked. Each case is flushed immediately; interruptions leave
partial records available. A final record summarizes completed runs. Interrupted
runs have no final summary and must not be presented as complete experiments.

Automated metrics:

- `status_match`: expected answer/abstention status; failures score false.
- `answer_terms_match`: expected status and all reference terms in claims;
  abstentions must have no claims. This is a lexical proxy, not correctness.
- `citation_validity`: cited IDs exist in returned evidence and have the expected
  CIK, accession and official filing URL. It does not establish entailment.
- `required_tools_present`: the expected tool subset appears in recorded tool
  attempts, including failed attempts. It does not penalize unnecessary calls or
  prove correct arguments. Cases with no required tools satisfy this trivially.
- `cited_reference_overlap`: a cited chunk contains the reference paragraph or
  is contained in it. Section/chunk boundaries can produce false negatives.
- `retrieval_reference_hit`: at least one retrieved chunk overlaps the reference
  paragraph and matches its CIK/accession, including evidence not cited in the
  answer. Failed or missing retrieval scores false on answerable cases;
  unsupported cases are unscored. This is a reference-hit proxy for retrieval
  relevance, subject to the same paragraph/chunk boundary limitations.
- `numerical_accuracy`: exact numeric-token match for the one reported-number
  case. This is not an evaluation of financial calculation capability or units.

Citation and numerical checks without applicable evidence remain unscored
(`null`), with scored denominators in summaries. Semantic correctness and
unsupported-claim rate also remain `null`; no LLM judge score is substituted.
For semantic review, inspect each claim against the saved reference and actual
cited passage, record supported/unsupported claims, assess whether the answer
addresses the question, and review all retrieved passages for relevance. Keep
interpretation judgments separate from deterministic identity/number checks.

## Phoenix

From the repository root:

```bash
docker compose --profile observability up -d phoenix
```

Set `SEC_RESEARCH_PHOENIX_ENABLED=true` for the backend or evaluation process.
The collector/UI is at `http://127.0.0.1:6006`; traces use `/v1/traces`.
Only loopback HTTP collector endpoints are accepted. Tracing is disabled by
default, and Phoenix is an optional Compose profile with local SQLite storage
in a named volume. No extra PostgreSQL server, account or cloud service is used.
Authentication is explicitly disabled for this loopback-only development service.

Request spans contain LangChain model and tool spans; graph spans identify each
node, and evaluation spans record case/workflow IDs and execution/status checks.
Model spans include reported token usage. Errors mark spans as errors, including
graph failures returned as data. MCP invocation is traced in the parent process;
there is no cross-process trace propagation or instrumentation inside the stdio
child. SEC retries remain inside their enclosing tool span. Traces include
research questions and retrieved public filing text; settings and credentials
are not added as span attributes.

The integration uses the official
[Phoenix OTEL SDK](https://arize-phoenix.readthedocs.io/projects/otel/)
and OpenInference LangChain instrumentation. Export is asynchronous; collector
failure is visible in exporter diagnostics and does not change an answer into an
inference failure. To stop the optional collector without removing traces:

```bash
docker compose --profile observability stop phoenix
```

## Recorded environment

The recorded run uses `qwen3:8b` digest
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`
and `nomic-embed-text:latest` digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
The index contains only the target Apple filing: 142 chunks, 768 dimensions,
model key `nomic-embed-text`. The SHA-256 of its sorted chunk IDs joined by
newlines is
`bc38dbd0ac6c1ab1168e883f4b3befa1fb2d625ad4ceb3ffdab0d206b7d182eb`.

Installed versions are pinned in `backend/uv.lock`: LangChain Core 1.7.4,
LangChain Ollama 1.1.0, LangGraph 1.2.20, MCP 1.30.0, Phoenix OTEL 0.17.3,
and OpenInference LangChain 0.1.86. The Compose collector is Phoenix 21.0.0,
pinned to the tested container digest. The experiment uses the default chat
context of 16,384 tokens, output limit of 2,048 tokens, six tool attempts and
24,000 context characters. Model weights are released between calls, as in
normal application use.

## Results

The full run on 2026-11-12 is saved in [results.jsonl](results.jsonl):
all 30 cases were attempted by both workflows (60 unique executions). Its
manifest matches the dataset SHA-256, and its final summary was recomputed from
the individual records and checked. Retrieval-reference hits and tool-subset
checks were finalized from saved tool outputs/attempts after execution; no model
outputs were changed and no model calls were rerun to adjust these scores.

| Measure | Baseline | Graph |
| --- | ---: | ---: |
| Completed with a valid result | 22/30 (73.3%) | 27/30 (90.0%) |
| Expected answer/abstention status | 22/30 (73.3%) | 24/30 (80.0%) |
| Reference-term proxy matched | 20/30 (66.7%) | 22/30 (73.3%) |
| Answered answerable cases | 18/20 | 14/20 |
| Correctly abstained on unsupported cases | 4/10 | 10/10 |
| Citation identity on scored answers | 18/18 | 14/14 |
| Cited-reference overlap on scored answers | 14/18 | 11/14 |
| Reference hit in all retrieved evidence | 14/20 | 14/20 |
| Mean latency, seconds | 32.8 | 51.9 |
| Median latency, seconds | 32.0 | 51.9 |
| Model calls | 83 | 95 |
| Tool attempts | 29 | 31 |
| Reported input + output tokens | 152,533 | 170,293 |

Token usage was available for every recorded model call. Neither workflow
successfully answered the reported-number case, so numerical accuracy has zero
scored live answers; it cannot be estimated from this run. Semantic correctness
and unsupported-claim rate were not scored. Citation identity passes show the
source validator working, not that every cited claim is supported.

The graph improved completion and unsupported-request handling on this dataset,
but answered fewer supported questions and took longer. Its overall advantage
in expected status comes from abstention cases. These results do not establish
a general quality improvement from orchestration.

### Failure cases and metric limitations

- Baseline `apple-03` and `apple-20`: repeated invalid tool arguments.
- Baseline `scope-21`–`scope-25`: synthesis cited IDs not retrieved; the validator
  rejected the answers. These are explicit failed executions, not accepted
  hallucinated financial results.
- Baseline `scope-26`: cumulative context budget exceeded.
- Graph `apple-09`, `apple-12`, `apple-18`: context budget exceeded at tool
  selection. The saved retrievals for these cases contained overlapping
  reference evidence; context accumulation prevented successful completion.
- Graph `apple-02`, `apple-03`, `apple-20`: returned insufficient evidence despite
  the answer existing in the indexed filing. Their retrievals missed the
  reference passages.
- Graph `apple-01`: answered with general statements about hardware and
  competition instead of identifying the requested product types. Its real
  citations pass the identity check while the reference-term check fails.
- Both `apple-10` answers name overseas outsourcing locations without using
  the expected word “outside.” This is a lexical false negative, showing why
  the reference-term proxy is not semantic correctness.
- Baseline `apple-12` discusses foreign-currency effects but explicitly lacks
  evidence about pricing. Its answered status does not prove it fully addressed
  the question.
- Graph `apple-05` took 408 seconds, versus 30 seconds for the baseline. The
  final record shows four model calls and one search attempt; trace inspection
  identified lengthy model generation. This is an inference latency outlier,
  not evidence of slow vector search.

The reference-hit proxy misses some successfully answered questions because a
reference paragraph can span chunks, or another passage can support an answer.
It also does not assess ranking or the relevance of every returned chunk.
One run per question, one company/filing, a small lexical rubric and unblinded
source review limit these findings. There are no confidence intervals, repeated
seed trials or independent human semantic ratings. The five-case development
pilot is excluded from the reported full-run totals. No prompts or model settings
were tuned during the full run.

The measured follow-up priorities are context-budget management, more reliable
tool arguments, retrieval of known answers and a broader independently reviewed
rubric. Those workflow changes are outside this evaluation milestone.

### Validation

The ordinary pytest suite passes with 161 tests passed and 7 optional integration
tests skipped. Ruff lint, Ruff format, Compose configuration and Git whitespace
checks also pass. Tests use saved or synthetic data and do not require live SEC
calls. New coverage exercises
source verification, metric failures, real LangChain run collection, numerical
token boundaries, SDK span parenting/error status, local-only endpoints, and
tracing cleanup. A real local export smoke check confirmed that Phoenix stored
a parent span and a LangChain tool span; the full experiment also exported
model, tool, graph and evaluation spans to that collector. A final API query
confirmed 60 stored evaluation spans and 178 model-call spans.
