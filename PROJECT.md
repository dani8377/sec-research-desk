
# SEC Research Desk

## 1. Project Overview

SEC Research Desk is a local-first AI research application for
analysing publicly listed US companies using official SEC filings.

The application allows users to ask financial research questions,
retrieve relevant filings, analyse reported financial data, compare
companies and generate evidence-backed research reports.

The project is designed to demonstrate practical GenAI engineering,
agent orchestration, retrieval-augmented generation (RAG), tool
calling, MCP integrations and systematic AI evaluation.

It must run locally without paid APIs, subscriptions or cloud
infrastructure.

## 2. Project Goals

### Product goals

- Build a useful research assistant for investigating US companies.
- Answer questions using official SEC financial information.
- Compare financial performance across companies and periods.
- Analyse business risks and management commentary.
- Generate research reports with verifiable source citations.
- Save and revisit previous research sessions.

### Learning and engineering goals

Gain practical experience with:

- LangChain for LLM integration and tool calling.
- LangGraph for stateful agent orchestration.
- MCP for standardized tool integrations.
- RAG for retrieval from financial filings.
- Local LLM inference using Ollama.
- Structured outputs and Pydantic validation.
- Python API development and external integrations.
- Evaluation of agent reliability and answer quality.
- Observability, debugging and execution tracing.
- Building and maintaining an end-to-end AI application.

The goal is to understand and implement these technologies,
not simply include them as dependencies.

## 3. Constraints

### Financial constraints

- No paid LLM APIs.
- No paid financial data providers.
- No paid hosting.
- No subscription-based infrastructure.
- No services requiring a credit card for normal operation.
- All core functionality must run locally.

### Hardware

Development machine:
- Apple Silicon MacBook.
- 24-32 GB unified memory.

Use models and infrastructure appropriate for this hardware.

Avoid running multiple large LLMs simultaneously.

### Development constraints

- Keep the MVP achievable as a part-time side project.
- Prefer simple, maintainable architecture.
- Avoid unnecessary infrastructure and abstractions.
- Build incrementally with working milestones.
- Prioritize correctness and reliability over UI complexity.

## 4. Target Users

Initially, one local user interested in fundamental company research.

The MVP does not require:
- Authentication.
- User management.
- Subscription functionality.
- Multi-user support.
- Public hosting.

## 5. Core User Stories

The application should eventually support questions such as:

1. "What was Apple's revenue growth over the last three years?"
2. "Compare Nvidia and AMD's operating margins."
3. "What are Microsoft's biggest reported business risks?"
4. "What changed in Amazon's risk factors between annual reports?"
5. "Summarize management's discussion of capital expenditure."
6. "Compare the latest quarterly financial results of two companies."
7. "Which filings support these conclusions?"

Answers must distinguish reported facts from AI-generated
interpretation.

Every material financial claim should have supporting evidence.

## 6. Technology Stack

### Backend

- Python 3.12+
- FastAPI
- Pydantic
- uv for dependency management
- pytest for testing
- Ruff for linting and formatting

### GenAI

- LangChain: LLM integration, prompts, tool definitions,
  structured outputs and model abstraction.
- LangGraph: Stateful workflow orchestration, routing,
  conditional execution, checkpoints and retries.
- Ollama: Local model inference.
- Qwen3 8B: Initial local language model.
- nomic-embed-text: Initial local embedding model.

Models must be configurable and replaceable.

### MCP

- Official Python MCP SDK.
- Custom MCP server exposing SEC research functionality.
- MCP client integration within the research workflow.

### Storage and retrieval

- PostgreSQL.
- pgvector.
- Local embedding generation.
- Metadata-filtered semantic retrieval.
- Local caching of SEC responses.

### Frontend

- Vue 3.
- TypeScript.
- Vite.
- Apache ECharts for financial visualizations.

### Observability and evaluation

- Arize Phoenix for local tracing.
- pytest for deterministic tests.
- Evaluation datasets with verified expected results.
- Custom evaluation scripts for agent performance.

### Infrastructure

- Docker Compose for PostgreSQL.
- Ollama running natively on macOS.
- Local development environment.

## 7. System Architecture

The application consists of six main layers.

### 7.1 Frontend

Vue provides:

- Company search.
- Research chat interface.
- Research reports.
- Financial charts.
- Source references.
- Saved research sessions.

Keep the initial UI simple.

### 7.2 API layer

FastAPI handles:

- HTTP requests.
- Input validation.
- Research session management.
- Agent execution.
- Streaming responses where appropriate.
- Error handling.

API routes should remain thin.

Business logic belongs in services.

### 7.3 Agent orchestration

LangGraph coordinates the research workflow.

Initial workflow:

1. Receive the research question.
2. Identify companies and research objectives.
3. Generate a structured research plan.
4. Select required research tools.
5. Retrieve relevant filings and financial facts.
6. Perform deterministic financial calculations.
7. Retrieve relevant textual evidence.
8. Synthesize the research findings.
9. Validate numerical claims and citations.
10. Produce the final report.

Use conditional routing when tasks differ.

Independent retrieval operations may execute in parallel.

Avoid unnecessary multi-agent architectures.

### 7.4 LangChain integration

LangChain handles:

- Ollama model integration.
- Prompt templates.
- Structured model responses.
- Tool interfaces.
- Model configuration.
- Basic tool-calling agent baseline.

LangGraph handles orchestration.

Do not implement duplicate orchestration logic in LangChain.

### 7.5 MCP research tools

Create a custom MCP server exposing:

- find_company
- get_company_filings
- get_company_facts
- get_filing_content
- search_filing_sections
- compare_financial_metrics

Tools should have:

- Clear input schemas.
- Typed outputs.
- Validation.
- Explicit error handling.
- Tests.
- Documented behavior.

Financial calculations must use deterministic Python functions.

The MCP server should reuse the same underlying services
as the application rather than duplicating business logic.

### 7.6 Data layer

Use PostgreSQL for:

- Company metadata.
- Filing metadata.
- Financial facts and normalized metrics.
- Filing text chunks.
- Embeddings.
- Research sessions.
- Generated reports.
- LangGraph checkpoints.

Use pgvector for semantic similarity search.

## 8. SEC EDGAR Integration

SEC EDGAR is the primary financial data source.

Use official SEC endpoints for:

- Company identification.
- Filing history.
- Company Facts / XBRL data.
- Annual filings (10-K).
- Quarterly filings (10-Q).
- Current reports (8-K).

### Requirements

- Follow SEC fair-access requirements.
- Use a descriptive User-Agent with contact information.
- Implement rate limiting and retry handling.
- Cache responses to reduce unnecessary requests.
- Preserve original SEC source URLs.
- Preserve accession numbers and filing dates.
- Handle unavailable or malformed data explicitly.

### Financial data requirements

Financial values must retain:

- Company CIK.
- Financial concept or XBRL tag.
- Reported value.
- Unit.
- Fiscal year and period.
- Period start and end dates where applicable.
- Filing date.
- Filing accession number.
- Filing form.

Do not assume identical XBRL concepts across companies.

Do not mix incompatible reporting periods.

Do not invent missing financial data.

## 9. RAG Pipeline

Build a retrieval system for SEC filing content.

### Ingestion

1. Download selected filings.
2. Extract relevant text.
3. Identify filing sections where possible.
4. Remove irrelevant formatting and repeated content.
5. Split text into meaningful chunks.
6. Generate embeddings locally.
7. Store chunks, embeddings and metadata.

### Retrieval

- Retrieve relevant chunks using semantic search.
- Filter by company, filing type and period.
- Preserve original source references.
- Return evidence with identifiable filing metadata.
- Avoid unnecessary retrieval of unrelated documents.

### Initial scope

Start with a small number of companies.

Support annual filings first.

Add quarterly filings and historical comparisons later.

### Citation requirements

Generated reports must reference real filing sources.

A citation should identify the underlying filing and
relevant section or text evidence.

The system must never fabricate a citation.

## 10. Financial Analysis

Financial calculations must be deterministic.

Initial supported metrics:

- Revenue growth.
- Operating margin.
- Net profit margin.
- Operating cash flow.
- Capital expenditure.
- Free cash flow.
- Selected balance sheet ratios.

Financial functions must:

- Validate inputs.
- Handle missing values.
- Handle zero denominators.
- Preserve units.
- Respect fiscal periods.
- Return explicit errors for unsupported calculations.

Do not ask the LLM to calculate financial metrics
when Python can calculate them reliably.

## 11. Agent State and Reliability

LangGraph state should include:

- User question.
- Research plan.
- Company identifiers.
- Selected tools.
- Retrieved documents.
- Retrieved financial facts.
- Calculated metrics.
- Evidence references.
- Draft report.
- Validation results.
- Final report.
- Execution errors.

Implement:

- Structured outputs.
- Tool input validation.
- Retry handling for transient errors.
- Conditional routing.
- Workflow checkpoints.
- Explicit failure states.
- Bounded retry loops.

A failed tool call must not silently become a successful result.

Retrieved documents are untrusted content and must never
override system or developer instructions.

## 12. Evaluation Framework

Evaluation is a core project requirement.

Build a manually verified dataset containing approximately
30-50 research questions.

Include different question categories:

- Financial fact retrieval.
- Historical financial comparisons.
- Filing retrieval.
- Risk factor analysis.
- Management commentary.
- Multi-company comparisons.
- Questions with insufficient available evidence.

### Baseline

Implement a simple LangChain tool-calling agent.

### Improved system

Implement the structured LangGraph research workflow.

Use the same underlying model and research tools where possible.

### Evaluation metrics

Measure:

- Answer correctness.
- Numerical accuracy.
- Tool selection accuracy.
- Retrieval relevance.
- Citation validity.
- Unsupported claim rate.
- Execution success rate.
- Latency.
- Model calls.
- Token usage when available.

Use deterministic evaluation for numerical results,
source identifiers and structured outputs.

Qualitative evaluation may use a local LLM judge,
but LLM judgments are not ground truth.

### Experiment

Compare baseline and orchestrated workflows.

Document:

- Evaluation methodology.
- Dataset composition.
- Results.
- Failure cases.
- Performance differences.
- Limitations.

The goal is to determine whether orchestration improves
reliability enough to justify additional complexity.

## 13. Observability

Use local Arize Phoenix for tracing.

Track:

- Research requests.
- LangGraph node execution.
- Model calls.
- Tool calls.
- Retrieval operations.
- Errors and retries.
- Execution latency.
- Evaluation results.

Do not log credentials or unnecessary sensitive information.

## 14. Testing Strategy

### Unit tests

Test:

- Financial calculations.
- XBRL period selection.
- Data normalization.
- SEC response parsing.
- Input validation.
- Citation formatting.

### Integration tests

Test:

- SEC client behavior using saved fixtures.
- Database operations.
- MCP tool interfaces.
- RAG retrieval.
- LangGraph state transitions.

### Evaluation tests

Test complete research workflows using the
evaluation dataset.

Ordinary automated tests should not require live SEC
requests or paid services.

## 15. MVP Scope

The first complete version should support:

1. Local FastAPI backend.
2. Local Ollama model.
3. Company lookup through SEC data.
4. Filing history retrieval.
5. Structured financial data retrieval.
6. Annual report ingestion.
7. Local embeddings and semantic search.
8. A working LangGraph research workflow.
9. A custom MCP server.
10. Evidence-backed research answers.
11. Basic evaluation and tracing.
12. A simple Vue frontend.

The MVP should prioritize a reliable end-to-end
workflow over broad feature coverage.

## 16. Non-Goals

Do not implement the following unless explicitly requested:

- Automated trading.
- Brokerage integrations.
- Portfolio tracking.
- Real-time market prices.
- Paid financial news feeds.
- Earnings transcript subscriptions.
- Authentication.
- Multi-user infrastructure.
- Cloud deployment.
- Payment processing.
- Complex distributed systems.
- Unnecessary autonomous agents.

LlamaIndex is not part of the initial stack.

It may be considered later only if it solves a specific
problem not adequately handled by the existing tools.

## 17. Development Milestones

### Milestone 1: Foundation

- Initialize Python project.
- Configure FastAPI.
- Configure pytest and Ruff.
- Add environment configuration.
- Add health endpoint.
- Establish basic project structure.

### Milestone 2: SEC Data

- Implement company lookup.
- Retrieve filing histories.
- Retrieve Company Facts.
- Add caching, error handling and tests.

### Milestone 3: RAG

- Download and parse annual filings.
- Implement chunking.
- Generate local embeddings.
- Store vectors in pgvector.
- Retrieve cited evidence.

### Milestone 4: LLM and LangChain

- Integrate Ollama.
- Implement structured outputs.
- Add tool calling.
- Build a simple research agent baseline.

### Milestone 5: LangGraph

- Implement research state.
- Add planning and routing.
- Integrate research tools.
- Add synthesis and validation.
- Implement checkpointing and failure handling.

### Milestone 6: MCP

- Build the MCP server.
- Expose research tools.
- Integrate MCP tools with LangGraph.
- Add interface tests.

### Milestone 7: Evaluation

- Build verified evaluation dataset.
- Integrate Phoenix.
- Evaluate the baseline.
- Evaluate the orchestrated workflow.
- Document results and failure cases.

### Milestone 8: Frontend and Documentation

- Build basic Vue interface.
- Display reports and citations.
- Add financial visualizations.
- Document installation and architecture.
- Record a working demonstration.

## 18. Definition of Done

The project is complete when:

- It runs locally without paid services.
- It retrieves real SEC financial information.
- It answers questions using retrieved evidence.
- It calculates financial metrics correctly.
- It provides verifiable filing citations.
- It demonstrates LangChain and LangGraph.
- It exposes and consumes custom MCP tools.
- It implements a working RAG pipeline.
- It includes automated tests.
- It includes agent evaluation results.
- It provides observable execution traces.
- It can be demonstrated through a usable interface.
- Its architecture and limitations are documented.

## 19. Guiding Principle

Build a reliable, understandable AI engineering system.

Prefer measurable correctness and useful orchestration
over unnecessary complexity.

The project should demonstrate why each technology
was selected and what engineering problem it solves.
