# SEC Research Desk

A local-first research application for US company SEC filings. Milestones 1–2
provide a FastAPI backend, company lookup, filing histories and Company Facts.

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

## Test and lint

From `backend/`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

To apply formatting, run `uv run ruff format .`.
Tests run locally without external services or live SEC requests.
