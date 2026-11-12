# SEC Research Desk

A local-first research application for US company SEC filings. Milestones 1–5
provide a FastAPI backend, SEC data access, annual-filing evidence retrieval,
and LangChain baseline and LangGraph research workflows using local Ollama models and
PostgreSQL/pgvector.

## Install on macOS

Install [uv](https://docs.astral.sh/uv/) using Homebrew:

```bash
brew install uv
```

From the repository root, create the optional local configuration and install
Python 3.12 and the backend dependencies (including development tools):

```bash
cp .env.example .env
uv python install 3.12
cd backend
uv sync --python 3.12 --locked
```

uv creates `backend/.venv`; activation is unnecessary when using `uv run`.
`backend/uv.lock` records the resolved dependency versions.

## Run

From `backend/`:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response: `{"status":"ok"}`. API documentation is available at
<http://127.0.0.1:8000/docs>. Stop the server with Ctrl-C.

## Configuration

Settings load from the repository root `.env` file regardless of the working
directory. Environment variables override `.env` values; defaults apply when
neither is provided. Invalid settings fail application startup.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEC_RESEARCH_APP_NAME` | `SEC Research Desk` | FastAPI application title |
| `SEC_RESEARCH_DEBUG` | `false` | FastAPI debug mode; accepts `true` or `false` |
| `SEC_RESEARCH_SEC_USER_AGENT` | unset | Application name followed by a real contact email; required for SEC requests |
| `SEC_RESEARCH_SEC_CACHE_DIR` | repository `.cache/sec` | Local disk cache directory |
| `SEC_RESEARCH_SEC_CACHE_TTL_SECONDS` | `3600` | Cache lifetime; `0` disables caching |
| `SEC_RESEARCH_SEC_REQUEST_INTERVAL_SECONDS` | `0.2` | Minimum interval between request starts; at least `0.1` |
| `SEC_RESEARCH_SEC_TIMEOUT_SECONDS` | `30` | HTTP timeout in seconds |
| `SEC_RESEARCH_SEC_MAX_RETRIES` | `2` | Retries after the initial request; maximum `5` |

Before using SEC endpoints, set `SEC_RESEARCH_SEC_USER_AGENT` in `.env` to your
application name and real contact email, following the
[SEC fair-access guidance](https://www.sec.gov/about/webmaster-frequently-asked-questions).
Health checks work without this setting; SEC requests return `503` until it is set.

## SEC data API

```bash
curl 'http://127.0.0.1:8000/companies?q=AAPL'
curl 'http://127.0.0.1:8000/companies/320193/filings?form=10-K'
curl 'http://127.0.0.1:8000/companies/320193/filings?include_historical=true'
curl 'http://127.0.0.1:8000/companies/320193/facts'
curl 'http://127.0.0.1:8000/companies/320193/facts?taxonomy=us-gaap&concept=Assets'
```

Company lookup matches an exact ticker or CIK, or part of a company name,
without case sensitivity. Exact ticker matches come first. `limit` defaults to
20 (maximum 100); no matches returns an empty list. The SEC ticker directory
covers listed companies and does not guarantee every EDGAR filer; filing and
facts endpoints accept a known CIK directly, with or without leading zeros.

Filing history returns recent filings by default. `include_historical=true`
downloads all additional history files listed by the SEC; this can take longer
on a cold cache. `form` matches an exact form, so `10-K` excludes `10-K/A`.
Each filing retains its CIK, accession number, filing/report dates and source URLs.

Company Facts retains the SEC taxonomy/concept/unit structure and original
observation fields, including periods and filing references. The response adds
the source URL and retrieval time (the original fetch time on cache hits).
Full responses can contain several megabytes of JSON, making browser display
slow. For interactive testing in `/docs`, set `taxonomy=us-gaap` and
`concept=Assets` to view a smaller result. Filters are exact and case-sensitive;
`concept` requires `taxonomy` (`422` otherwise), and missing taxonomies or
concepts return `404`. Filtering preserves all observations and source metadata
for the selected concepts; the complete upstream response remains cached.
The first request still downloads the complete SEC Company Facts response.
Syntax highlighting is disabled in `/docs` to reduce display overhead.
No metrics, period selection or cross-company normalization are performed.
The [SEC Company Facts API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
covers standard taxonomy facts applying to the whole entity; custom concepts
and other absent data are not inferred.

Responses are validated before caching. Expired or corrupt cache entries are
refetched; failed requests do not fall back to stale data. Cache I/O failures
are logged and leave retrieval available. Timeouts, HTTP 429 and transient server
errors have bounded retries respecting `Retry-After` (waits above 60 seconds
return `503`). HTTP 403 is reported without retrying, missing SEC data returns
`404`, and malformed upstream data returns `502`.

Rate limiting is shared by threads within one application process. Run one
backend worker locally and account for other SEC clients on the same network;
multiple processes do not share a rate limiter.

## Annual-filing retrieval (milestone 3)

Install and start Docker Desktop. From the repository root, start PostgreSQL:

```bash
docker compose up -d --wait postgres
```

Compose binds PostgreSQL to `127.0.0.1:5432` and persists data in a named volume.
The included credentials are for local development. `docker compose stop`
stops the database without removing its data. The backend creates the vector
extension and chunk table on first use; the database role needs permission to
create them. No database or Ollama connection is required for health/SEC APIs.

Install [Ollama for macOS](https://ollama.com/download/mac), launch it, and pull
the embedding model:

```bash
ollama pull nomic-embed-text
```

Ollama runs natively on macOS. Filing ingestion and search only need the
embedding model; the research baseline also uses Qwen (see below). Embeddings use LangChain's
[Ollama integration](https://docs.langchain.com/oss/python/integrations/embeddings/ollama).
The Nomic model uses the documented
[document/query prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes).

Additional settings (all use the `SEC_RESEARCH_` prefix):

| Variable suffix | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://sec_research:sec_research@127.0.0.1:5432/sec_research` | Local PostgreSQL connection |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama server |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Installed embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Expected output size; mismatches fail explicitly |
| `EMBEDDING_BATCH_SIZE` | `16` | Maximum chunks per embedding request |
| `EMBEDDING_TIMEOUT_SECONDS` | `120` | Timeout per local model request |
| `RAG_CHUNK_SIZE` | `1800` | Maximum characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Character overlap within a section; must be smaller than chunk size |

With the backend running and the SEC contact header configured, select an
accession from `/companies/320193/filings?form=10-K` and ingest it. This example
uses Apple's 2025 annual filing:

```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{"cik":"320193","accession_number":"0000320193-25-000079"}'

curl -X POST http://127.0.0.1:8000/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"cik":"320193","query":"supply chain risks","limit":5}'
```

Ingestion resolves the primary document through SEC submissions, downloads it
using the existing cache/rate limiter, removes hidden HTML/XBRL and navigation,
splits at recognized Item headings and creates overlapping chunks within each
section. Content without recognized headings has an explicit unidentified
section label. The response includes the filing, chunk count, model and parser
warnings. Only HTML `10-K` primary documents are supported; amendments, quarterly
reports, PDFs and plain-text filings are outside this milestone. Set
`include_historical=true` in the ingestion body for older submission histories.

Ingestion runs synchronously and may take time on a cold cache. Re-ingesting
replaces that filing's chunks for the selected model in one transaction, after
all embeddings succeed; a failure preserves the previous stored evidence.
Search never downloads or ingests documents automatically.

Search requires a company CIK and accepts optional `accession_number`, `section`
(exact returned section label or Item identifier such as `Item 7`),
`report_date_from`, `report_date_to` (inclusive report dates), and `limit` (1–20). The supported `form` is `10-K`. It uses exact
cosine search in pgvector, filtered by company, model and dimensions. Each result
contains similarity and the verbatim normalized chunk text, original filing URL,
CIK, accession, filing/report dates, form, section and retrieval timestamp.
`char_start`/`char_end` are section-relative offsets into normalized extracted
text, not HTML byte positions. They and the stable `chunk_id` identify evidence;
no invented URL fragments or page numbers are added. An empty result means no
indexed evidence matches the filters. Similarity alone does not establish that
evidence supports a claim; answer synthesis and validation come later.

Parser section detection is heuristic, and tables are linearized text rather
than validated financial data. Retrieved text remains untrusted source content.
The parser has been smoke-tested on Apple's full 2025 10-K; broader filing-format
coverage and retrieval-quality evaluation remain future work. Model names and
dimensions separate vector spaces. Pin a model tag and re-ingest when its weights
or chunk settings change; changing weights under the same tag is not detected.
Other embedding models may require their own input-prefix configuration in code.
Ollama/database failures return `503`; invalid embedding outputs return `502`.

## Research baseline (milestone 4)

With Ollama running, install the chat model:

```bash
ollama pull qwen3:8b
```

Ingest the relevant annual filings using `/rag/ingest` first. Then:

```bash
curl -X POST http://127.0.0.1:8000/research/baseline \
  -H 'Content-Type: application/json' \
  -d '{"question":"What supply chain risks does Apple report? Identify the filing period."}'
```

The baseline uses `ChatOllama` and typed LangChain tools for company lookup,
recent annual filing history and metadata-filtered search of indexed sections.
It executes a simple bounded tool loop, followed by a separate native Ollama
JSON-schema output call validated with Pydantic. This is the baseline for later
LangGraph comparison; the baseline runs independently of the graph below.
No new framework is needed: `langchain-core` is declared directly alongside the
existing `langchain-ollama` integration.

The response contains `status` (`answered` or `insufficient_evidence`), `claims`,
`limitations`, `evidence`, `model`, `tools_called`, `model_calls` and `tool_errors`.
Each claim is marked `reported_fact` or `interpretation` and cites retrieved chunk IDs.
Synthesis receives the original question and retrieved evidence directly, without
intermediate model prose. Python rejects unknown IDs and attaches the original
text and filing metadata.
Company lookup and filing history are navigation tools, not claim evidence.
Missing indexed data should produce an explicit insufficient-evidence result.
SEC/RAG failures abort the request rather than becoming successful empty results.
An invalid tool input gets one correction opportunity via an explicit error tool
message; a second invalid attempt fails. Invalid attempts consume the tool budget
and remain visible in `tools_called` and `tool_errors` if correction succeeds.
Malformed model outputs, invalid tool calls, invented citations and exceeded
budgets fail explicitly (`502`); model/connection/parsing failures return `503`.
Underlying SEC/RAG failures retain their HTTP status. There are no model-request
retries or silent format fallbacks. Health and SEC APIs do not require a chat model.

Additional settings use the `SEC_RESEARCH_` prefix:

| Variable suffix | Default | Purpose |
| --- | --- | --- |
| `CHAT_MODEL` | `qwen3:8b` | Installed model supporting tools and JSON-schema output |
| `CHAT_TIMEOUT_SECONDS` | `180` | Timeout per model request |
| `CHAT_CONTEXT_SIZE` | `16384` | Ollama context tokens |
| `CHAT_MAX_TOKENS` | `2048` | Maximum generated tokens per call |
| `RESEARCH_MAX_TOOL_CALLS` | `6` | Total tool attempts, including invalid inputs and multi-call responses |
| `RESEARCH_MAX_CONTEXT_CHARS` | `24000` | Character budget for accumulated messages |

At most the tool budget plus two model calls run per request. The character
budget is a conservative guard, not an exact tokenizer count; reduce it for
smaller model contexts. Chat weights are released after each call (`keep_alive=0`)
to limit persistent memory usage, at the cost of additional loading latency.
Requests are synchronous and can take several minutes. Use one local request at
a time. Model configuration must support tool calling and native structured
outputs; switching models requires rerunning the opt-in integration test.

This milestone supports qualitative annual-filing research. It has no Company
Facts or financial calculation tool and must report those questions as unsupported
when numerical analysis is required. It checks citation identity, not semantic
entailment: a valid citation can still support an incorrect interpretation.
Prompt instructions treat filing text as untrusted, but are not a guarantee
against prompt injection. Answer quality, unsupported claims and retrieval
relevance need the milestone 7 evaluation dataset. Search only covers ingested
filings; it does not establish that they are the latest filings available.

The normal tests use scripted model outputs and synthetic evidence. To test
actual tool calling, structured outputs and a complete baseline execution
against the configured local model (synthetic services, no live SEC calls):

```bash
cd backend
SEC_RESEARCH_TEST_CHAT=1 uv run pytest tests/test_research.py -k real_ollama
```

## LangGraph research workflow (milestone 5)

The graph uses the same model, research tools, synthesis prompt and citation
validator as the baseline. It adds a structured planning call and explicit nodes
for tool selection, individual tool execution, synthesis and validation.
Planning routes unsupported requests directly to an insufficient-evidence result.
Empty retrieval also skips model synthesis. Tool selection can return to retrieval
within the existing tool budget. Milestone 6 routes its tools through MCP (see below);
financial calculations remain unsupported.

With PostgreSQL and Ollama running and the relevant annual filings indexed:

```bash
curl -X POST http://127.0.0.1:8000/research/graph \
  -H 'Content-Type: application/json' \
  -d '{"question":"What supply chain risks does Apple report? Identify the filing period."}'
```

The response includes `run_id`, `status` (`running`, `completed` or `failed`),
`plan`, `result`, `errors`, `tools_called`, `model_calls` and `next_nodes`.
`completed` means the workflow finished; inspect `result.status` to distinguish
an answer from insufficient evidence. `result` has the baseline response schema.
Failed workflows have no result and include the failing stage, message and
underlying status code. These saved failures return HTTP `200` as run records;
request validation and checkpoint/access failures use HTTP error responses.

Checkpoints store the question, plan, tool messages, retrieved evidence, call
counters, draft, validated result and errors in the local PostgreSQL database.
Tables are created on first graph use with the official
[PostgreSQL checkpointer](https://docs.langchain.com/oss/python/langgraph/add-memory).
The database role needs table/index creation permissions. The baseline, SEC and
health endpoints do not require checkpoint storage.

Use the returned UUID to inspect or resume a run:

```text
GET  /research/graph/{run_id}
POST /research/graph/{run_id}/resume
```

Supply an optional UUID `run_id` in the initial JSON body to retain a known ID
if the HTTP request is interrupted. Duplicate IDs return `409`; unknown runs
return `404`. Each successful node is checkpointed synchronously, including each
individual tool in a batch. Resume continues pending nodes after a process
interruption; a tool interrupted before its checkpoint may execute again.
Completed and terminal failed runs are returned unchanged by resume. Start a new
run after correcting a terminal failure. There are no automatic model retries;
invalid tool arguments receive one correction opportunity, and existing SEC
retries remain bounded. Keep the same configuration when resuming; a changed
chat model is rejected. Checkpoint storage has no automatic retention policy.

Use one backend worker and one research request at a time. Concurrent graph
requests return `409`, and a PostgreSQL advisory lock prevents two processes from
advancing the same run. Requests remain synchronous. The graph makes at most the
tool budget plus three model calls, including planning. Citation validation checks
retrieved IDs and attaches original metadata; semantic support, numerical accuracy
and planning quality still require evaluation. This workflow retains the baseline's
annual-filing scope and prompt-injection limitations.

Normal graph tests use scripted models and in-memory checkpoints. Opt-in tests
exercise real local inference and PostgreSQL persistence with synthetic evidence:

```bash
cd backend
SEC_RESEARCH_TEST_CHAT=1 uv run pytest tests/test_research_graph.py -k real_ollama
SEC_RESEARCH_TEST_DATABASE_URL=postgresql://sec_research:sec_research@127.0.0.1:5432/sec_research \
  uv run pytest tests/test_research_graph.py -k postgres
```

The PostgreSQL test creates and removes a unique test schema. Neither integration
test sends SEC requests. No comparative quality or reliability improvement is
claimed before the milestone 7 evaluation.

## MCP research tools (milestone 6)

The official Python MCP SDK exposes the same three read-only research tools:

| Tool | Inputs | Output |
| --- | --- | --- |
| `find_company` | `query`, optional `limit` (1–10) | SEC company matches and source URLs |
| `get_company_filings` | `cik`, optional `limit` (1–10) | Recent 10-K metadata with original source references |
| `search_filing_sections` | `cik`, `query`, optional filing/section/date filters and `limit` (1–5) | Indexed evidence with chunk IDs, text and filing metadata |

Run the standalone stdio server from `backend/`:

```bash
uv run python -m app.mcp_server
```

For a local MCP client, use the absolute path to `backend/.venv/bin/python` as
its command, arguments `["-m", "app.mcp_server"]`, and `backend/` as its working
directory. The server loads the existing root `.env`. Protocol messages use
stdout; diagnostics use stderr. No listening port or additional service is needed.
Discovery publishes explicit input/output JSON schemas and read-only annotations.
Pydantic also checks semantic constraints such as CIKs and ordered date filters.
Tool failures set MCP `isError`; service errors retain their message and status
in structured error data. Successful empty searches remain successful empty results.

`/research/graph` now consumes these tools through the official MCP stdio client.
Each tool call launches and closes a local server process, initializes a session,
discovers the tool, invokes it and validates its structured output. This simple
lifecycle adds process startup overhead but leaves no persistent MCP worker.
The child receives the API's validated settings; it does not replace explicit
settings with values from `.env`. `SEC_RESEARCH_MCP_TIMEOUT_SECONDS` defaults to
180 seconds and bounds startup, discovery and invocation together. Transport
failures/timeouts become explicit failed graph runs; there is no direct-call
fallback. Invalid model arguments still get the graph's existing one correction
opportunity before any MCP request. Retrieved evidence remains in graph checkpoints
and passes through the existing citation validator. The baseline keeps direct
service calls with the same schemas and descriptions for later comparison.

Scope remains qualitative research over already indexed annual reports. The
planned `get_company_facts`, `get_filing_content` and
`compare_financial_metrics` MCP tools are not exposed in this milestone: bounded
fact selection, direct filing-content contracts and deterministic metric analysis
remain future work. Existing HTTP Company Facts and RAG ingestion are unchanged.
One local research request at a time is still the supported operating mode.

Offline interface tests use real subprocess stdio sessions with synthetic services,
including a graph run through MCP, error propagation and citation preservation:

```bash
cd backend
uv run pytest tests/test_mcp.py
```

These tests do not evaluate real-model answer quality. The existing opt-in Ollama
graph test isolates model behavior with direct synthetic tools; MCP transport is
covered separately by the interface tests.

## Test and lint

From `backend/`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

To apply formatting, run `uv run ruff format .`.
Tests run locally without external services or live SEC requests.

Optional checks against a running local database and an installed Ollama model,
from `backend/`:

```bash
SEC_RESEARCH_TEST_DATABASE_URL=postgresql://sec_research:sec_research@127.0.0.1:5432/sec_research \
  uv run pytest tests/test_rag_postgres.py
SEC_RESEARCH_TEST_OLLAMA=1 uv run pytest tests/test_rag.py -k real_ollama
```

Database tests create and remove a unique test schema; use a local test database
with schema/extension creation permissions. Optional checks are skipped unless
enabled. See [fixture provenance and test limitations](backend/tests/fixtures/README.md).
