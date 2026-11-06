"""Tavily search client. Everything it returns is UNTRUSTED web content.

Results must pass through the quarantine layer before any model sees them. This client only
makes the request and parses the reply defensively: urls are validated, text is length-capped,
bad items are dropped and counted, and errors are fixed categories that never contain response
content. The api key goes only in the Authorization header and is never logged or shown.
"""

from __future__ import annotations

import http.client
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tokenshield.bifrost_client import Transport, _urllib_transport

DEFAULT_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT = 30.0
MAX_QUERY_CHARS = 400
MAX_TITLE_CHARS = 200
MAX_CONTENT_CHARS = 4000
MAX_DATE_CHARS = 64
MAX_DOMAINS = 20

SEARCH_DEPTHS = frozenset({"basic", "advanced"})
TOPICS = frozenset({"general", "news"})
TIME_RANGES = frozenset({"day", "week", "month", "year"})

_URL_RE = re.compile(r"https://[A-Za-z0-9.\-]+(:\d{1,5})?(/[A-Za-z0-9._~/\-]{0,100})?")
_KEY_RE = re.compile(r"[A-Za-z0-9._\-]{8,200}")
_DOMAIN_RE = re.compile(r"[a-z0-9.\-]{1,100}")
_RESULT_URL_RE = re.compile(r"https?://[^\s<>\"'\\]{1,500}")


class TavilyError(Exception):
    """A Tavily call failed. The message is a fixed category such as http_401."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float | None
    published_date: str | None


@dataclass(frozen=True)
class SearchResponse:
    results: tuple[SearchResult, ...]
    dropped: int


def _clean_text(value: object, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _parse_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _parse_result(item: object) -> SearchResult | None:
    if not isinstance(item, Mapping):
        return None
    url = item.get("url")
    if not isinstance(url, str) or not _RESULT_URL_RE.fullmatch(url):
        return None
    return SearchResult(
        title=_clean_text(item.get("title"), MAX_TITLE_CHARS),
        url=url,
        content=_clean_text(item.get("content"), MAX_CONTENT_CHARS),
        score=_parse_score(item.get("score")),
        published_date=_clean_text(item.get("published_date"), MAX_DATE_CHARS) or None,
    )


def _parse_response(body: str, max_results: int) -> SearchResponse:
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise TavilyError("bad_response") from None
    if not isinstance(data, Mapping):
        raise TavilyError("bad_response")
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise TavilyError("bad_response")
    parsed = [_parse_result(item) for item in raw_results[:max_results]]
    results = tuple(result for result in parsed if result is not None)
    return SearchResponse(results=results, dropped=len(parsed) - len(results))


def _check_domains(domains: Sequence[str]) -> list[str]:
    if isinstance(domains, (str, bytes)) or not isinstance(domains, Sequence):
        raise ValueError("invalid include_domains")  # noqa: TRY004 - single error type
    if not 1 <= len(domains) <= MAX_DOMAINS:
        raise ValueError("invalid include_domains")
    for domain in domains:
        if not isinstance(domain, str) or not _DOMAIN_RE.fullmatch(domain):
            raise ValueError("invalid include_domains")
    return list(domains)


class TavilyClient:
    def __init__(
        self,
        api_key: str,
        *,
        url: str = DEFAULT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not _KEY_RE.fullmatch(api_key):
            raise ValueError("invalid api key")
        if not _URL_RE.fullmatch(url):
            raise ValueError("invalid url")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._api_key = api_key
        self._url = url
        self.timeout = timeout
        self._transport = transport or _urllib_transport

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, transport: Transport | None = None
    ) -> TavilyClient:
        return cls(env.get("TAVILY_API_KEY", ""), transport=transport)

    def __repr__(self) -> str:
        return f"TavilyClient(url={self._url!r})"

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str | None = None,
        include_domains: Sequence[str] | None = None,
    ) -> SearchResponse:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ValueError("invalid query")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValueError("invalid max_results")  # noqa: TRY004 - single error type
        if not 1 <= max_results <= 10:
            raise ValueError("invalid max_results")
        if search_depth not in SEARCH_DEPTHS:
            raise ValueError("invalid search_depth")
        if topic not in TOPICS:
            raise ValueError("invalid topic")
        if time_range is not None and time_range not in TIME_RANGES:
            raise ValueError("invalid time_range")

        payload: dict[str, object] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "topic": topic,
            "include_answer": False,
            "include_raw_content": False,
        }
        if time_range is not None:
            payload["time_range"] = time_range
        if include_domains is not None:
            payload["include_domains"] = _check_domains(include_domains)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            status, text = self._transport(self._url, headers, body, self.timeout)
        except (OSError, http.client.HTTPException):
            raise TavilyError("network_error") from None
        if not 200 <= status < 300:
            raise TavilyError(f"http_{status}")
        return _parse_response(text, max_results)