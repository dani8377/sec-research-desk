
---
name: sec-edgar
description: Retrieve and parse SEC company filings and XBRL data. Use when implementing or modifying SEC data integrations.
---

# SEC EDGAR development

## Before implementation

1. Inspect the existing SEC client and related tests.
2. Check the current official SEC documentation.
3. Verify API endpoints, response schemas and field names.
4. Do not assume that all companies use identical XBRL tags.

## Data handling

- Identify companies using their SEC CIK.
- Preserve filing accession numbers and filing dates.
- Preserve the original SEC filing URL.
- Respect SEC fair-access requirements.
- Use a descriptive User-Agent.
- Cache external responses where appropriate.
- Handle missing and custom XBRL facts explicitly.

## Evidence requirements

Every retrieved financial fact should retain:

- Company CIK
- Filing accession number
- Filing form
- Filing date
- Fiscal period
- XBRL concept or tag
- Value and unit

Never substitute an estimated value for a missing reported value.

## Testing

- Use saved SEC response fixtures.
- Do not rely on live SEC requests for ordinary unit tests.
- Test missing, malformed and custom financial facts.
- Verify period selection and unit conversion.

## Output

Report the SEC endpoints used, files changed,
tests executed and any limitations discovered.
