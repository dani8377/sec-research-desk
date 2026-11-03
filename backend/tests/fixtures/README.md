SEC fixtures for offline tests:

- `submissions.json`: selected top-level fields, first three recent filing rows
  and historical-file references from
  https://data.sec.gov/submissions/CIK0000320193.json.
- `history.json`: first two rows from
  https://data.sec.gov/submissions/CIK0000320193-submissions-001.json.
- `facts.json`: original top-level fields, selected concepts
  (`EntityCommonStockSharesOutstanding`, `Assets`, `Revenues`,
  `EffectiveIncomeTaxRateReconciliationFdiiAmount`) and their first
  two observations per unit from
  https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json.

These three responses were downloaded on 2026-11-03. Values and filing
identifiers were retained without modification; the files are reduced samples,
not complete histories. Assertions compare parsing and metadata with these
samples, not financial metric selection or calculation.

`tickers.json` is a manually curated format fixture using known Apple and
Microsoft identifiers and the `cik_str`, `ticker`, `title` schema documented at
https://www.sec.gov/about/webmaster-frequently-asked-questions and exposed by
https://www.sec.gov/files/company_tickers.json. The live ticker download returned
403 in the development environment; this fixture is not a captured response.

RAG fixtures:

- `annual_synthetic.html`: hand-written parser test content with inline XBRL,
  navigation, hidden elements, split inline headings and tabular text. It contains
  no actual company statements or financial figures. The CIK and accession used
  with it in `test_rag.py` and `test_rag_postgres.py` are synthetic test identifiers.
- `annual_excerpt.html`: four original HTML elements extracted from Apple's
  2025 10-K, accession `0000320193-25-000079`, downloaded on 2026-11-03 from
  https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm.
  Includes Item 1A and Item 7 headings and short paragraphs, wrapped in a minimal
  HTML body. This is a parser fixture, not a complete filing or retrieval corpus.

Offline tests verify parsing, section-bounded chunking, metadata and failure
behavior. Database tests use synthetic vectors to verify cosine ranking and
filtering; they do not measure semantic answer quality. The opt-in Ollama test
checks actual embedding dimensions and relative relevance for two known texts;
this small smoke test is not a retrieval benchmark.
