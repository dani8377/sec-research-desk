"""SEC research operations reused by HTTP and future MCP tools."""

from datetime import date
from urllib.parse import quote

from app.sec_client import SECClient, SECError
from app.sec_models import (
    Company,
    CompanyFacts,
    FactsResult,
    Filing,
    FilingColumns,
    FilingHistory,
    Submissions,
    TickerDirectory,
    normalize_cik,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"


class SECService:
    def __init__(self, client: SECClient) -> None:
        self.client = client

    def find_company(self, query: str, limit: int = 20) -> list[Company]:
        query = query.strip()
        if not query or not 1 <= limit <= 100:
            raise ValueError("Provide a nonempty query and a limit between 1 and 100")
        directory, _ = self.client.get(TICKERS_URL, TickerDirectory)
        needle = query.casefold()
        cik = (
            normalize_cik(query)
            if query.isascii()
            and query.isdigit()
            and len(query) <= 10
            and int(query) > 0
            else None
        )
        matches = [
            row
            for row in directory.root.values()
            if needle == row.ticker.casefold()
            or needle in row.title.casefold()
            or (cik is not None and normalize_cik(row.cik_str) == cik)
        ]
        matches.sort(
            key=lambda row: (row.ticker.casefold() != needle, row.title, row.ticker)
        )
        return [
            Company(
                cik=normalize_cik(row.cik_str),
                ticker=row.ticker,
                name=row.title,
                source_url=TICKERS_URL,
            )
            for row in matches[:limit]
        ]

    def get_company_filings(
        self, cik: str, *, form: str | None = None, include_historical: bool = False
    ) -> FilingHistory:
        cik = normalize_cik(cik)
        source = f"{SUBMISSIONS_URL}CIK{cik}.json"
        data, _ = self.client.get(source, Submissions)
        if normalize_cik(data.cik) != cik:
            raise SECError("SEC submissions CIK does not match the requested company")
        filings = self._filings(cik, data.filings.recent, source)
        if include_historical:
            for file in data.filings.files:
                if not file.name.startswith(f"CIK{cik}-"):
                    raise SECError("SEC historical file belongs to a different company")
                url = SUBMISSIONS_URL + file.name
                columns, _ = self.client.get(url, FilingColumns)
                filings.extend(self._filings(cik, columns, url))
        unique = {filing.accession_number: filing for filing in reversed(filings)}
        selected = [
            item for item in unique.values() if form is None or item.form == form
        ]
        selected.sort(
            key=lambda item: (item.filing_date, item.accession_number), reverse=True
        )
        return FilingHistory(
            cik=cik,
            company_name=data.name,
            includes_historical=include_historical,
            filings=selected,
        )

    @staticmethod
    def _filings(cik: str, columns: FilingColumns, source: str) -> list[Filing]:
        results = []
        for index, accession in enumerate(columns.accessionNumber):
            document = columns.primaryDocument[index]
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            url = (
                f"{base}{accession.replace('-', '')}/{quote(document, safe='/')}"
                if document
                else f"{base}{accession}-index.htm"
            )
            results.append(
                Filing(
                    cik=cik,
                    accession_number=accession,
                    filing_date=columns.filingDate[index],
                    report_date=date.fromisoformat(columns.reportDate[index])
                    if columns.reportDate[index]
                    else None,
                    form=columns.form[index],
                    primary_document=document or None,
                    filing_url=url,
                    source_url=source,
                )
            )
        return results

    def get_company_facts(
        self, cik: str, *, taxonomy: str | None = None, concept: str | None = None
    ) -> FactsResult:
        cik = normalize_cik(cik)
        if concept is not None and taxonomy is None:
            raise SECError("Provide a taxonomy when filtering by concept", 422)
        source = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        data, retrieved_at = self.client.get(source, CompanyFacts)
        if normalize_cik(data.cik) != cik:
            raise SECError("SEC Company Facts CIK does not match the requested company")
        if taxonomy is not None:
            if taxonomy not in data.facts:
                raise SECError(
                    "Requested taxonomy is absent from SEC Company Facts", 404
                )
            concepts = data.facts[taxonomy]
            if concept is not None:
                if concept not in concepts:
                    raise SECError(
                        "Requested concept is absent from SEC Company Facts", 404
                    )
                concepts = {concept: concepts[concept]}
            # Keep the complete response in the cache; filter only this result.
            data = data.model_copy(update={"facts": {taxonomy: concepts}})
        return FactsResult(
            cik=cik, source_url=source, retrieved_at=retrieved_at, data=data
        )
