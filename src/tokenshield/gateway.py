"""TokenShield gateway: an OpenAI-compatible chat endpoint that shields requests, then forwards
them to Bifrost.

Point any OpenAI-style client at this server instead of Bifrost. For every request it
  1. checks the optional session's running token total against the budget (a circuit breaker),
  2. reduces untrusted messages to typed facts or fixed placeholders (the Layer 1 shield),
  3. forwards the shielded request to Bifrost with the server's own virtual key,
  4. returns Bifrost's reply with a `tokenshield` report added.

Trust: system, user and assistant messages pass unchanged. Tool messages are treated as build or
test logs and reduced with LOG_FACTS. A message may carry `ts_kind` ("file" for a README-like
file; "web" or "skill" for content that has no schema and is therefore blocked) to LOWER its
trust. Nothing can raise trust. Content without a schema is blocked, never passed.

Security defaults: loopback only, Host header checked, any request carrying an Origin header is
rejected (browsers cannot use this server), JSON bodies only, size caps, fixed error codes that
never contain request content or keys, streaming rejected. The budget stops the next request after
a session exceeds it, so one request can overshoot.

Limits: tool results are assumed to be test or build logs; other tools need their own schema (see
catalog.py). Unsupported request parameters are dropped. There is no streaming.
"""

from __future__ import annotations

import http.client
import json
import math
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from tokenshield.api import ApiError, Body, Reply
from tokenshield.bifrost_client import (
    _KEY_RE,
    _MODEL_RE,
    _URL_RE,
    DEFAULT_BASE_URL,
    EFFORTS,
    BifrostClient,
    Transport,
    _token_count,
    _urllib_transport,
)
from tokenshield.catalog import gateway_policy
from tokenshield.extractor import LLMClient
from tokenshield.gate import GatePolicy
from tokenshield.meter import QUARANTINE, TokenMeter
from tokenshield.sessions import SessionLedger, valid_session
from tokenshield.shield import ItemReport, shield_messages

MAX_BODY_BYTES = 300_000
MAX_MESSAGES = 50
MAX_TOOLS = 32
MAX_OUTPUT_TOKENS = 8192
DEFAULT_PORT = 8766
SESSION_HEADER = "X-Tokenshield-Session"
CHAT_PATH = "/v1/chat/completions"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _bounded(value: object, low: float, high: float) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, "invalid_request")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or not low <= value <= high:
        raise ApiError(400, "invalid_request")
    return value


def _params(data: Mapping[str, object]) -> dict[str, object]:
    """The few request parameters that are forwarded. Anything else is dropped."""
    params: dict[str, object] = {}
    if "temperature" in data:
        params["temperature"] = _bounded(data["temperature"], 0, 2)
    if "top_p" in data:
        params["top_p"] = _bounded(data["top_p"], 0, 1)
    if "max_tokens" in data:
        value = data["max_tokens"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ApiError(400, "invalid_request")
        if not 1 <= value <= MAX_OUTPUT_TOKENS:
            raise ApiError(400, "invalid_request")
        params["max_tokens"] = value
    if "reasoning_effort" in data:
        effort = data["reasoning_effort"]
        if not isinstance(effort, str) or effort not in EFFORTS:
            raise ApiError(400, "invalid_request")
        params["reasoning_effort"] = effort
    if "tools" in data:
        tools = data["tools"]
        if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
            raise ApiError(400, "invalid_request")
        if not all(isinstance(tool, Mapping) for tool in tools):
            raise ApiError(400, "invalid_request")
        params["tools"] = tools
    if "tool_choice" in data:
        choice = data["tool_choice"]
        if not isinstance(choice, (str, Mapping)):
            raise ApiError(400, "invalid_request")
        params["tool_choice"] = choice
    return params


def _json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _item_json(item: ItemReport) -> Body:
    return {
        "index": item.index,
        "role": item.role,
        "kind": item.kind.value,
        "action": item.action.value,
        "reason": item.reason,
    }


class GatewayService:
    def __init__(
        self,
        quarantine: LLMClient,
        policy: GatePolicy,
        upstream_url: str,
        virtual_key: str,
        ledger: SessionLedger,
        *,
        timeout: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        base = upstream_url.rstrip("/")
        if not _URL_RE.fullmatch(base):
            raise ValueError("invalid upstream_url")
        if not _KEY_RE.fullmatch(virtual_key):
            raise ValueError("invalid virtual key")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._quarantine = quarantine
        self._policy = policy
        self._url = f"{base}{CHAT_PATH}"
        self._virtual_key = virtual_key
        self.ledger = ledger
        self._timeout = timeout
        self._transport = transport or _urllib_transport

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, transport: Transport | None = None
    ) -> GatewayService:
        raw_budget = env.get("TOKENSHIELD_SESSION_BUDGET", "").strip()
        try:
            budget = float(raw_budget) if raw_budget else None
        except ValueError:
            raise ValueError("invalid session budget") from None
        return cls(
            BifrostClient.from_env(env),
            gateway_policy(),
            env.get("TOKENSHIELD_BIFROST_URL", DEFAULT_BASE_URL),
            env.get("TOKENSHIELD_BIFROST_VK", ""),
            SessionLedger(budget),
            transport=transport,
        )

    def handle(
        self, method: str, path: str, payload: object, session: str | None = None
    ) -> Reply:
        if path == "/health":
            if method != "GET":
                return 405, {"error": "method_not_allowed"}
            return 200, {"status": "ok"}
        if path != CHAT_PATH:
            return 404, {"error": "not_found"}
        if method != "POST":
            return 405, {"error": "method_not_allowed"}
        try:
            return 200, self._chat(payload, session)
        except ApiError as error:
            return error.status, error.body()
        except Exception:  # noqa: BLE001 - never leak exception text to a client
            return 500, {"error": "internal_error"}

    def _chat(self, payload: object, session: str | None) -> Body:
        if not isinstance(payload, Mapping):
            raise ApiError(400, "invalid_request")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
            raise ApiError(400, "invalid_request")
        if not all(isinstance(message, Mapping) for message in messages):
            raise ApiError(400, "invalid_request")
        model = payload.get("model")
        if not isinstance(model, str) or _MODEL_RE.fullmatch(model) is None:
            raise ApiError(400, "invalid_request")
        if payload.get("stream") not in (None, False):
            raise ApiError(400, "streaming_not_supported")
        params = _params(payload)
        if session is not None:
            if not valid_session(session):
                raise ApiError(400, "invalid_session")
            if self.ledger.over_budget(session):
                raise ApiError(429, "budget_exceeded", "session")

        meter = TokenMeter()
        outcome = shield_messages(self._quarantine, self._policy, messages, meter)
        upstream_payload = {"model": model, "messages": list(outcome.messages), **params}
        headers = {"Content-Type": "application/json", "x-bf-vk": self._virtual_key}
        body = json.dumps(upstream_payload).encode("utf-8")
        try:
            status, text = self._transport(self._url, headers, body, self._timeout)
        except (OSError, http.client.HTTPException):
            raise ApiError(502, "upstream_error", "network_error") from None
        if not 200 <= status < 300:
            raise ApiError(502, "upstream_error", f"http_{status}")
        try:
            reply = json.loads(text)
        except (ValueError, RecursionError):
            raise ApiError(502, "upstream_error", "bad_response") from None
        if not isinstance(reply, dict) or not isinstance(reply.get("choices"), list):
            raise ApiError(502, "upstream_error", "bad_response")

        usage = reply.get("usage")
        main_in = _token_count(usage, "prompt_tokens")
        main_out = _token_count(usage, "completion_tokens")
        main_tokens = None if main_in is None and main_out is None else (main_in or 0) + (main_out or 0)
        quarantine = meter.totals(QUARANTINE)
        used = None
        if session is not None:
            used = self.ledger.add(session, (main_tokens or 0) + quarantine.total_tokens)
        reply["tokenshield"] = {
            "items": [_item_json(item) for item in outcome.items],
            "quarantine_tokens": {
                "input": quarantine.input_tokens,
                "output": quarantine.output_tokens,
                "calls": quarantine.calls,
            },
            "main_tokens": main_tokens,
            "session": (
                None
                if used is None
                else {"used": used, "budget": _json_number(self.ledger.budget)}
            ),
        }
        return reply


def make_gateway_server(
    service: GatewayService, host: str = "127.0.0.1", port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("host must be a loopback address")

    class Handler(BaseHTTPRequestHandler):
        server_version = "tokenshield-gateway"
        sys_version = ""
        timeout = 10

        def log_message(self, *_args: object) -> None:
            return None

        def _send(self, status: int, body: Body) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self) -> bool:
            bound_port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}
            if self.headers.get("Host") not in hosts:
                self._send(403, {"error": "forbidden_host"})
                return False
            if self.headers.get("Origin") is not None:
                self._send(403, {"error": "browser_not_allowed"})
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
            session = self.headers.get(SESSION_HEADER)
            status, body = service.handle("POST", urlsplit(self.path).path, payload, session)
            self._send(status, body)

        def do_OPTIONS(self) -> None:
            if self._allowed():
                self._send(405, {"error": "method_not_allowed"})

    return ThreadingHTTPServer((host, port), Handler)