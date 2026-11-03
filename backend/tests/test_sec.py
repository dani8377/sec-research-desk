import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.sec_client import SECClient, SECError
from app.sec_models import CompanyFacts, TickerDirectory, normalize_cik
from app.sec_routes import get_sec_service
from app.sec_service import SECService

FIXTURES = Path(__file__).parent / "fixtures"
TICKERS = "https://www.sec.gov/files/company_tickers.json"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        sec_user_agent="Research Tests test@example.com",
        sec_cache_dir=tmp_path,
        sec_max_retries=2,
    )


@pytest.fixture(autouse=True)
def no_wait(monkeypatch: pytest.MonkeyPatch) -> Mock:
    sleep = Mock()
    monkeypatch.setattr("app.sec_client.time.sleep", sleep)
    return sleep


@pytest.fixture
def service(settings: Settings) -> Iterator[SECService]:
    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("company_tickers.json"):
            name = "tickers.json"
        elif "/companyfacts/" in path:
            name = "facts.json"
        elif "-submissions-" in path:
            name = "history.json"
        else:
            name = "submissions.json"
        return httpx.Response(200, json=fixture(name))

    client = SECClient(settings, httpx.MockTransport(respond))
    try:
        yield SECService(client)
    finally:
        client.close()


@pytest.mark.parametrize("query", ["aapl", "Apple", "320193", "0000320193", " Apple "])
def test_find_company(service: SECService, query: str) -> None:
    companies = service.find_company(query)
    assert len(companies) == 1
    assert companies[0].cik == "0000320193"
    assert companies[0].ticker == "AAPL"
    assert companies[0].source_url == TICKERS


def test_company_search_missing_and_limit(service: SECService) -> None:
    assert service.find_company("No such company") == []
    assert service.find_company("000000000000000") == []
    assert len(service.find_company("o", limit=1)) == 1
    with pytest.raises(ValueError):
        service.find_company(" ")


@pytest.mark.parametrize(
    "value", ["0", "0000000000", "-1", "aapl", "12345678901", "１２"]
)
def test_invalid_cik(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_cik(value)


def test_filing_history_and_sources(service: SECService) -> None:
    history = service.get_company_filings("320193")
    assert history.includes_historical is False
    assert len(history.filings) == 3
    first = history.filings[0]
    raw = fixture("submissions.json")["filings"]["recent"]
    assert first.cik == "0000320193"
    assert first.accession_number == raw["accessionNumber"][0]
    assert first.filing_date.isoformat() == raw["filingDate"][0]
    assert first.source_url.endswith("/CIK0000320193.json")
    assert first.filing_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        + raw["accessionNumber"][0].replace("-", "")
        + "/"
        + raw["primaryDocument"][0]
    )
    full = service.get_company_filings("320193", include_historical=True)
    assert full.includes_historical
    assert len(full.filings) == 5
    assert full.filings[-1].source_url.endswith("-submissions-001.json")
    filtered = service.get_company_filings("320193", form="4", include_historical=True)
    assert filtered.filings
    assert all(item.form == "4" for item in filtered.filings)
    assert service.get_company_filings("320193", form="nonexistent").filings == []


def test_facts_preserve_values_units_periods_and_accessions(
    service: SECService,
) -> None:
    result = service.get_company_facts("320193")
    assert result.cik == "0000320193"
    assert result.source_url.endswith("/companyfacts/CIK0000320193.json")
    raw = fixture("facts.json")
    data = result.data.model_dump(mode="json", exclude_unset=True)
    assert data == raw
    assert (
        data["facts"]["us-gaap"]["EffectiveIncomeTaxRateReconciliationFdiiAmount"][
            "label"
        ]
        is None
    )
    assert result.retrieved_at.tzinfo is not None
    assert "custom" not in result.data.facts  # No invented concepts or normalization.


def test_filtered_facts_preserve_observations_and_full_cache(
    service: SECService,
) -> None:
    result = service.get_company_facts("320193", taxonomy="us-gaap", concept="Assets")
    expected = fixture("facts.json")["facts"]["us-gaap"]["Assets"]
    assert result.data.model_dump(mode="json", exclude_unset=True)["facts"] == {
        "us-gaap": {"Assets": expected}
    }
    taxonomy = service.get_company_facts("320193", taxonomy="us-gaap")
    assert set(taxonomy.data.facts) == {"us-gaap"}
    assert (
        taxonomy.data.facts["us-gaap"].keys()
        == fixture("facts.json")["facts"]["us-gaap"].keys()
    )
    full = service.get_company_facts("320193")
    assert full.data.model_dump(mode="json", exclude_unset=True) == fixture(
        "facts.json"
    )
    assert result.retrieved_at == taxonomy.retrieved_at == full.retrieved_at
    assert result.source_url == full.source_url


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"concept": "Assets"}, 422),
        ({"taxonomy": "missing"}, 404),
        ({"taxonomy": "us-gaap", "concept": "missing"}, 404),
        ({"taxonomy": "us-gaap", "concept": "assets"}, 404),
    ],
)
def test_facts_filter_errors(service: SECService, filters: dict, expected: int) -> None:
    with pytest.raises(SECError) as error:
        service.get_company_facts("320193", **filters)
    assert error.value.status_code == expected


def test_cache_persists_between_clients(settings: Settings) -> None:
    handler = Mock(return_value=httpx.Response(200, json=fixture("tickers.json")))
    first = SECClient(settings, httpx.MockTransport(handler))
    _, retrieved_at = first.get(TICKERS, TickerDirectory)
    first.close()
    second = SECClient(settings, httpx.MockTransport(handler))
    try:
        _, cached_at = second.get(TICKERS, TickerDirectory)
        assert cached_at == retrieved_at
        assert handler.call_count == 1
        assert list(settings.sec_cache_dir.glob("*.json"))
    finally:
        second.close()


@pytest.mark.parametrize("cache_state", ["expired", "corrupt", "invalid_payload"])
def test_invalid_cache_is_refetched(settings: Settings, cache_state: str) -> None:
    handler = Mock(return_value=httpx.Response(200, json=fixture("tickers.json")))
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        client.get(TICKERS, TickerDirectory)
        path = next(settings.sec_cache_dir.glob("*.json"))
        entry = json.loads(path.read_text())
        if cache_state == "expired":
            entry["retrieved_at"] = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        elif cache_state == "invalid_payload":
            entry["payload"] = "{}not-json"
        path.write_text("not-json" if cache_state == "corrupt" else json.dumps(entry))
        client.get(TICKERS, TickerDirectory)
        assert handler.call_count == 2
    finally:
        client.close()


def test_disabled_cache_and_rate_limiting(settings: Settings, no_wait: Mock) -> None:
    settings.sec_cache_ttl_seconds = 0
    client = SECClient(
        settings,
        httpx.MockTransport(
            lambda _: httpx.Response(200, json=fixture("tickers.json"))
        ),
    )
    try:
        client.get(TICKERS, TickerDirectory)
        client.get(TICKERS, TickerDirectory)
        assert no_wait.call_count == 1
        assert 0 < no_wait.call_args.args[0] <= 0.2
        assert not list(settings.sec_cache_dir.glob("*.json"))
    finally:
        client.close()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_retry(settings: Settings, no_wait: Mock, status: int) -> None:
    handler = Mock(
        side_effect=[
            httpx.Response(status, headers={"Retry-After": "2"}),
            httpx.Response(200, json=fixture("tickers.json")),
        ]
    )
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        client.get(TICKERS, TickerDirectory)
        assert handler.call_count == 2
        no_wait.assert_any_call(2)
    finally:
        client.close()


@pytest.mark.parametrize(
    ("status", "expected", "calls"),
    [
        (403, 503, 1),
        (404, 404, 1),
        (429, 503, 3),
        (500, 503, 3),
        (400, 502, 1),
    ],
)
def test_http_failure_is_not_cached(
    settings: Settings, status: int, expected: int, calls: int
) -> None:
    handler = Mock(return_value=httpx.Response(status))
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        with pytest.raises(SECError) as error:
            client.get(TICKERS, TickerDirectory)
        assert error.value.status_code == expected
        assert handler.call_count == calls
        assert not list(settings.sec_cache_dir.glob("*.json"))
    finally:
        client.close()


def test_long_retry_after_stops(settings: Settings) -> None:
    handler = Mock(return_value=httpx.Response(429, headers={"Retry-After": "120"}))
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        with pytest.raises(SECError, match="longer wait"):
            client.get(TICKERS, TickerDirectory)
        assert handler.call_count == 1
    finally:
        client.close()


def test_timeout_is_bounded(settings: Settings) -> None:
    handler = Mock(side_effect=httpx.ReadTimeout("timeout"))
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        with pytest.raises(SECError, match="timed out"):
            client.get(TICKERS, TickerDirectory)
        assert handler.call_count == 3
    finally:
        client.close()


@pytest.mark.parametrize(
    "payload", ["<html>blocked</html>", '{"0": {"ticker": "AAPL"}}']
)
def test_malformed_response(settings: Settings, payload: str) -> None:
    client = SECClient(
        settings, httpx.MockTransport(lambda _: httpx.Response(200, text=payload))
    )
    try:
        with pytest.raises(SECError, match="malformed"):
            client.get(TICKERS, TickerDirectory)
        assert not list(settings.sec_cache_dir.glob("*.json"))
    finally:
        client.close()


def test_misaligned_filing_columns(settings: Settings) -> None:
    data = fixture("submissions.json")
    data["filings"]["recent"]["form"].pop()
    client = SECClient(
        settings, httpx.MockTransport(lambda _: httpx.Response(200, json=data))
    )
    try:
        with pytest.raises(SECError, match="malformed"):
            SECService(client).get_company_filings("320193")
    finally:
        client.close()


def test_missing_user_agent_makes_no_request(settings: Settings) -> None:
    settings.sec_user_agent = None
    handler = Mock()
    client = SECClient(settings, httpx.MockTransport(handler))
    try:
        with pytest.raises(SECError, match="SEC_RESEARCH_SEC_USER_AGENT"):
            client.get(TICKERS, TickerDirectory)
        handler.assert_not_called()
    finally:
        client.close()


@pytest.mark.parametrize("operation", ["get_company_facts", "get_company_filings"])
def test_mismatched_company_rejected(service: SECService, operation: str) -> None:
    with pytest.raises(SECError, match="does not match"):
        getattr(service, operation)("789019")


def test_api(settings: Settings, service: SECService) -> None:
    app = create_app(settings)
    app.dependency_overrides[get_sec_service] = lambda: service
    with TestClient(app) as client:
        assert (
            client.get("/companies", params={"q": "aapl"}).json()[0]["cik"]
            == "0000320193"
        )
        assert client.get("/companies/320193/filings").status_code == 200
        response = client.get("/companies/320193/facts")
        assert response.status_code == 200
        assert response.json()["data"]["facts"] == fixture("facts.json")["facts"]
        filtered = client.get(
            "/companies/320193/facts",
            params={"taxonomy": "us-gaap", "concept": "Assets"},
        )
        assert filtered.status_code == 200
        assert filtered.json()["data"]["facts"] == {
            "us-gaap": {"Assets": fixture("facts.json")["facts"]["us-gaap"]["Assets"]}
        }
        for params, status in [
            ({"concept": "Assets"}, 422),
            ({"taxonomy": ""}, 422),
            ({"taxonomy": "missing"}, 404),
            ({"taxonomy": "us-gaap", "concept": "missing"}, 404),
        ]:
            assert (
                client.get("/companies/320193/facts", params=params).status_code
                == status
            )
        assert client.get("/companies/789019/facts").status_code == 502
        for path in [
            "/companies?q=%20%20",
            "/companies?q=a&limit=0",
            "/companies?q=a&limit=101",
            "/companies",
            "/companies/0/facts",
            "/companies/abc/filings",
            "/companies/12345678901/facts",
        ]:
            assert client.get(path).status_code == 422, path


def test_api_unconfigured_sec_still_has_health(settings: Settings) -> None:
    settings.sec_user_agent = None
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/companies?q=aapl").status_code == 503


def test_fact_numeric_validation() -> None:
    data = fixture("facts.json")
    data["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["val"] = True
    with pytest.raises(ValidationError):
        CompanyFacts.model_validate(data)
