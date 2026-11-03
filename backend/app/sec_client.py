"""Shared SEC transport with validated disk caching and bounded requests."""

import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)


class SECError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class SECClient:
    def __init__(
        self, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.http = httpx.Client(
            headers={
                "User-Agent": settings.sec_user_agent or "",
                "Accept": "application/json",
            },
            timeout=settings.sec_timeout_seconds,
            transport=transport,
        )
        self._lock = threading.Lock()
        self._last_request = float("-inf")

    def close(self) -> None:
        self.http.close()

    def get[Model: BaseModel](
        self, url: str, schema: type[Model]
    ) -> tuple[Model, datetime]:
        if not self.settings.sec_user_agent:
            raise SECError(
                "Set SEC_RESEARCH_SEC_USER_AGENT to your application name "
                "and contact email",
                503,
            )
        # One client is shared across API threads, including retries and cache writes.
        with self._lock:
            path = (
                self.settings.sec_cache_dir
                / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
            )
            cached = self._read_cache(path, schema)
            if cached is not None:
                return cached
            payload = self._download(url)
            try:
                result = schema.model_validate_json(payload)
            except ValidationError as exc:
                raise SECError(
                    "SEC returned malformed or unexpected JSON data"
                ) from exc
            retrieved_at = datetime.now(UTC)
            self._write_cache(path, payload, retrieved_at)
            return result, retrieved_at

    def _read_cache[Model: BaseModel](
        self, path: Path, schema: type[Model]
    ) -> tuple[Model, datetime] | None:
        if self.settings.sec_cache_ttl_seconds == 0:
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            retrieved_at = datetime.fromisoformat(entry["retrieved_at"])
            age = (datetime.now(UTC) - retrieved_at).total_seconds()
            if 0 <= age < self.settings.sec_cache_ttl_seconds:
                return schema.model_validate_json(entry["payload"]), retrieved_at
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError):
            logger.warning("Could not read SEC cache entry %s; fetching again", path)
        return None

    def _write_cache(self, path: Path, payload: str, retrieved_at: datetime) -> None:
        if self.settings.sec_cache_ttl_seconds == 0:
            return
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {"retrieved_at": retrieved_at.isoformat(), "payload": payload},
                    stream,
                )
            temporary.replace(path)
        except OSError:
            logger.warning("Could not write SEC cache entry %s", path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove temporary SEC cache file %s", temporary
                    )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        value = response.headers.get("Retry-After")
        if value:
            try:
                delay = float(value)
            except ValueError:
                try:
                    delay = (
                        parsedate_to_datetime(value) - datetime.now(UTC)
                    ).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    delay = 2**attempt
            # Do not retry earlier than requested or block a worker indefinitely.
            if delay > 60:
                raise SECError("SEC requested a longer wait; try again later", 503)
            return max(0, delay)
        return float(2**attempt)

    def _download(self, url: str) -> str:
        for attempt in range(self.settings.sec_max_retries + 1):
            wait = self.settings.sec_request_interval_seconds - (
                time.monotonic() - self._last_request
            )
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self.http.get(url)
            except httpx.RequestError as exc:
                if attempt == self.settings.sec_max_retries:
                    raise SECError("SEC request failed or timed out", 503) from exc
                time.sleep(2**attempt)
                continue
            if response.status_code == 404:
                raise SECError("Requested SEC data was not found", 404)
            if response.status_code == 403:
                raise SECError(
                    "SEC denied access; check User-Agent and fair-access limits", 503
                )
            if response.status_code == 429 or response.status_code in {
                500,
                502,
                503,
                504,
            }:
                if attempt == self.settings.sec_max_retries:
                    raise SECError(
                        "SEC is temporarily unavailable or rate limiting requests", 503
                    )
                time.sleep(self._retry_delay(response, attempt))
                continue
            if response.status_code != 200:
                raise SECError(
                    f"SEC returned unexpected HTTP status {response.status_code}"
                )
            return response.text
        raise SECError("SEC retry limit reached", 503)
