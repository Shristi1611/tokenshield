"""Local HTTP API for the frontend. Standard library only. For local demos, not deployment.

Security defaults
- Binds to loopback only, and rejects requests whose Host header is not that loopback address
  (blocks DNS rebinding).
- A request carrying an Origin that is not allowed is rejected, so other websites cannot spend
  your tokens. Non-browser clients send no Origin.
- POST bodies must be application/json (this forces a CORS preflight in browsers) and are size
  capped.
- Keys stay in this process. Errors are fixed codes and never contain exception text, request
  content or keys.
- One live model request at a time (busy gives 429), to respect free-tier rate limits.

Endpoints
  GET  /health
  GET  /api/benchmark   saved benchmark summary and a per-run table
  POST /api/shield      {"messages": [...]}  -> shielded messages and a per-message report
  POST /api/compare     {"question": "...", "document": "..."} -> baseline vs shielded
  POST /api/freshness   {"answer": "..."}  -> per-claim verdict and freshness score

Source URLs in /api/freshness come from untrusted search results. They are validated http(s)
URLs, but the frontend must render them as plain text or escaped links.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, urlsplit

from tokenshield.benchmark import (
    DEFAULT_MAIN_MODEL,
    SYSTEM_PROMPT,
    default_policy,
    load_records,
    summarize,
)
from tokenshield.bifrost_client import DEFAULT_BASE_URL, BifrostClient, BifrostError
from tokenshield.chat import BifrostChat, ChatClient, run_baseline, run_shielded
from tokenshield.extractor import LLMClient
from tokenshield.freshness import (
    ClaimReport,
    FreshnessError,
    SearchProvider,
    Verdict,
    check_answer,
)
from tokenshield.gate import GatePolicy
from tokenshield.meter import MAIN, QUARANTINE, TokenMeter
from tokenshield.multiturn import load_conv_records
from tokenshield.reporting import report_data
from tokenshield.shield import KIND_KEY, ItemReport, shield_messages
from tokenshield.tavily_client import TavilyClient

MAX_BODY_BYTES = 300_000
MAX_MESSAGES = 50
MAX_MESSAGES_CHARS = 200_000
MAX_QUESTION_CHARS = 1_000
MAX_DOCUMENT_CHARS = 20_000
MAX_ANSWER_CHARS = 8_000
EXCERPT_CHARS = 1_500
RESULT_SUITES = {
    "readme_long": ("multiturn_pad150_v2.jsonl",),
    "readme_short": ("multiturn_pad0.jsonl",),
    "logs": ("logbench_v2.jsonl", "logbench.jsonl"),
    "flip": ("flipbench.jsonl",),
}
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
DEFAULT_ORIGINS = frozenset(
    {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    }
)

Body = dict[str, object]
Reply = tuple[int, Body]


class ApiError(Exception):
    """A request failed. `code` is a fixed string and `detail` a fixed category, never content."""

    def __init__(self, status: int, code: str, detail: str | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.detail = detail

    def body(self) -> Body:
        result: Body = {"error": self.code}
        if self.detail is not None:
            result["reason"] = self.detail
        return result


def _mapping(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ApiError(400, "invalid_request")
    return payload


def _text(data: Mapping[str, object], key: str, limit: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ApiError(400, "invalid_request")
    return value


def _tokens(meter: TokenMeter, role: str) -> Body:
    totals = meter.totals(role)
    return {"input": totals.input_tokens, "output": totals.output_tokens, "calls": totals.calls}


def _item_json(item: ItemReport) -> Body:
    return {
        "index": item.index,
        "role": item.role,
        "kind": item.kind.value,
        "action": item.action.value,
        "reason": item.reason,
        "source_hash": item.source_hash,
        "chars_in": item.chars_in,
        "chars_out": item.chars_out,
        "truncated": item.truncated,
        "input_tokens": item.input_tokens,
        "output_tokens": item.output_tokens,
    }


def _claim_json(item: ClaimReport) -> Body:
    return {
        "category": item.claim.category.value,
        "subject": item.claim.subject,
        "value": item.claim.value,
        "risk": item.risk,
        "verdict": item.verdict.value,
        "score": item.score,
        "note": item.note,
        "sources": [{"domain": urlparse(url).netloc, "url": url} for url in item.sources],
    }


def _side(text: str, meter: TokenMeter) -> Body:
    totals = meter.totals()
    return {
        "answer": text[:EXCERPT_CHARS],
        "answer_chars": len(text),
        "main_tokens": _tokens(meter, MAIN),
        "quarantine_tokens": _tokens(meter, QUARANTINE),
        "total_tokens": totals.total_tokens,
        "unreported_calls": totals.unreported_calls,
    }


class ApiService:
    def __init__(
        self,
        quarantine: LLMClient,
        chat: ChatClient,
        search: SearchProvider | None,
        policy: GatePolicy,
        results_path: Path,
        *,
        max_checked: int = 2,
        results_per_claim: int = 2,
        results_dir: Path | None = None,
    ) -> None:
        self._quarantine = quarantine
        self._chat = chat
        self._search = search
        self._policy = policy
        self._results_path = results_path
        self._results_dir = results_dir if results_dir is not None else results_path.parent
        self._max_checked = max_checked
        self._results_per_claim = results_per_claim
        self.live_lock = threading.Lock()
        self._routes: dict[str, tuple[str, Callable[[object], Body]]] = {
            "/health": ("GET", self._health),
            "/api/benchmark": ("GET", self._benchmark),
            "/api/results": ("GET", self._results),
            "/api/shield": ("POST", self._shield),
            "/api/compare": ("POST", self._compare),
            "/api/freshness": ("POST", self._freshness),
        }

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], results_path: Path = Path("results/benchmark.jsonl")
    ) -> ApiService:
        quarantine = BifrostClient.from_env(env)
        chat = BifrostChat(
            env.get("TOKENSHIELD_BIFROST_URL", DEFAULT_BASE_URL),
            env.get("TOKENSHIELD_BIFROST_VK", ""),
            env.get("TOKENSHIELD_MAIN_MODEL", DEFAULT_MAIN_MODEL),
            max_tokens=2500,
        )
        try:
            search: SearchProvider | None = TavilyClient.from_env(env)
        except ValueError:
            search = None
        return cls(quarantine, chat, search, default_policy(), results_path)

    @property
    def freshness_enabled(self) -> bool:
        return self._search is not None

    def handle(self, method: str, path: str, payload: object) -> Reply:
        route = self._routes.get(path)
        if route is None:
            return 404, {"error": "not_found"}
        expected_method, action = route
        if method != expected_method:
            return 405, {"error": "method_not_allowed"}
        try:
            return 200, action(payload)
        except ApiError as error:
            return error.status, error.body()
        except Exception:  # noqa: BLE001 - never leak exception text to a client
            return 500, {"error": "internal_error"}

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self.live_lock.acquire(blocking=False):
            raise ApiError(429, "busy")
        try:
            yield
        finally:
            self.live_lock.release()

    def _health(self, _payload: object) -> Body:
        return {"status": "ok"}

    def _benchmark(self, _payload: object) -> Body:
        records = list(load_records(self._results_path).values())
        if not records:
            return {"available": False}
        runs = [
            {
                "task_id": record.task_id,
                "variant": record.variant,
                "mode": record.mode,
                "ok": record.ok,
                "correct": record.correct,
                "followed": record.followed,
                "cost_tokens": record.cost_tokens,
                "action": record.action,
            }
            for record in sorted(records, key=lambda record: record.key)
        ]
        return {"available": True, "summary": summarize(records), "runs": runs}

    def _results(self, _payload: object) -> Body:
        suites: Body = {}
        for label, names in RESULT_SUITES.items():
            for name in names:
                path = self._results_dir / name
                if not path.exists():
                    continue
                records = list(load_conv_records(path).values())
                if records:
                    suites[label] = {"file": name, **report_data(records)}
                break
        return {"available": bool(suites), "suites": suites}

    def _shield(self, payload: object) -> Body:
        data = _mapping(payload)
        messages = data.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
            raise ApiError(400, "invalid_request")
        if not all(isinstance(message, Mapping) for message in messages):
            raise ApiError(400, "invalid_request")
        if len(json.dumps(messages)) > MAX_MESSAGES_CHARS:
            raise ApiError(413, "too_large")
        meter = TokenMeter()
        with self._exclusive():
            outcome = shield_messages(self._quarantine, self._policy, messages, meter)
        return {
            "messages": list(outcome.messages),
            "items": [_item_json(item) for item in outcome.items],
            "quarantine_tokens": _tokens(meter, QUARANTINE),
        }

    def _compare(self, payload: object) -> Body:
        data = _mapping(payload)
        question = _text(data, "question", MAX_QUESTION_CHARS)
        document = _text(data, "document", MAX_DOCUMENT_CHARS)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{question}\nReply with exactly one line: ANSWER: <value>",
            },
            {"role": "user", KIND_KEY: "file", "content": document},
        ]
        baseline_meter = TokenMeter()
        shielded_meter = TokenMeter()
        with self._exclusive():
            try:
                baseline = run_baseline(self._chat, messages, baseline_meter, "api:baseline")
                shielded, outcome = run_shielded(
                    self._quarantine,
                    self._chat,
                    self._policy,
                    messages,
                    shielded_meter,
                    "api:shielded",
                )
            except BifrostError as error:
                raise ApiError(502, "upstream_error", str(error)) from None
        baseline_total = baseline_meter.totals().total_tokens
        shielded_total = shielded_meter.totals().total_tokens
        saved = baseline_total - shielded_total
        shielded_side = _side(shielded.text, shielded_meter)
        shielded_side["action"] = outcome.items[2].action.value
        shielded_side["view"] = str(outcome.messages[2]["content"])
        return {
            "baseline": _side(baseline.text, baseline_meter),
            "shielded": shielded_side,
            "saved_tokens": saved,
            "saved_fraction": saved / baseline_total if baseline_total else None,
        }

    def _freshness(self, payload: object) -> Body:
        data = _mapping(payload)
        answer = _text(data, "answer", MAX_ANSWER_CHARS)
        if self._search is None:
            raise ApiError(503, "not_configured")
        meter = TokenMeter()
        with self._exclusive():
            try:
                report = check_answer(
                    self._quarantine,
                    self._quarantine,
                    self._search,
                    answer,
                    meter,
                    max_checked=self._max_checked,
                    results_per_claim=self._results_per_claim,
                )
            except FreshnessError as error:
                raise ApiError(502, "upstream_error", error.reason) from None
        return {
            "claims": [_claim_json(item) for item in report.claims],
            "searches": report.searches,
            "counts": {verdict.value: report.count(verdict) for verdict in Verdict},
            "tokens": _tokens(meter, QUARANTINE),
        }


def make_server(
    service: ApiService,
    host: str = "127.0.0.1",
    port: int = 8765,
    allowed_origins: Iterable[str] = DEFAULT_ORIGINS,
) -> ThreadingHTTPServer:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("host must be a loopback address")
    origins = frozenset(allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        server_version = "tokenshield"
        sys_version = ""
        timeout = 10

        def log_message(self, *_args: object) -> None:
            return None

        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send(self, status: int, body: Body) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self) -> bool:
            bound_port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}
            if self.headers.get("Host") not in hosts:
                self._send(403, {"error": "forbidden_host"})
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in origins:
                self._send(403, {"error": "forbidden_origin"})
                return False
            return True

        def _read_json(self) -> object:
            length_text = self.headers.get("Content-Length")
            if length_text is None:
                raise ApiError(411, "length_required")
            try:
                length = int(length_text)
            except ValueError:
                raise ApiError(400, "invalid_request") from None
            if length < 0:
                raise ApiError(400, "invalid_request")
            if length > MAX_BODY_BYTES:
                raise ApiError(413, "too_large")
            raw = self.rfile.read(length)
            media_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if media_type != "application/json":
                raise ApiError(415, "unsupported_media_type")
            try:
                return json.loads(raw)
            except (ValueError, RecursionError):
                raise ApiError(400, "invalid_json") from None

        def do_GET(self) -> None:  
            if self._allowed():
                status, body = service.handle("GET", urlsplit(self.path).path, None)
                self._send(status, body)

        def do_POST(self) -> None:  
            if not self._allowed():
                return
            try:
                payload = self._read_json()
            except ApiError as error:
                self._send(error.status, error.body())
                return
            status, body = service.handle("POST", urlsplit(self.path).path, payload)
            self._send(status, body)

        def do_OPTIONS(self) -> None:  
            if not self._allowed():
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

    return ThreadingHTTPServer((host, port), Handler)