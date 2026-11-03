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
