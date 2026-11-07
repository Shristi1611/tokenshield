"""Real LLM client: calls the local Bifrost gateway (OpenAI-compatible chat completions).

It implements the LLMClient protocol used by the extractor. The virtual key is sent only in
the x-bf-vk header and is never logged, printed or placed in an error message. Errors are
short fixed categories and never contain response content, which may echo untrusted text.

Limits: one request, no retries, no streaming. Reasoning models spend part of max_tokens on
reasoning, so a low max_tokens can produce an empty answer (the extractor then fails closed).
reasoning_effort is optional and passed through as-is when set.
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from tokenshield.extractor import Completion

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "groq/openai/gpt-oss-20b"
DEFAULT_QUARANTINE_EFFORT = "low"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 1024
MAX_RESPONSE_BYTES = 1_000_000
EFFORTS = frozenset({"low", "medium", "high"})

_URL_RE = re.compile(r"https?://[A-Za-z0-9._\-]+(:\d{1,5})?")
_KEY_RE = re.compile(r"[A-Za-z0-9._\-]{8,200}")
_MODEL_RE = re.compile(r"[A-Za-z0-9._:/\-]{1,100}")

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, str]]


class BifrostError(Exception):
    """A gateway call failed. The message is a fixed category such as http_401."""


def load_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. Blank lines, comments and lines without '=' are skipped."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(MAX_RESPONSE_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read(MAX_RESPONSE_BYTES).decode("utf-8", "replace")


def _token_count(usage: object, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _parse_completion(body: str) -> Completion:
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
    except (ValueError, RecursionError, KeyError, IndexError, TypeError):
        raise BifrostError("bad_response") from None
    text = content if isinstance(content, str) else ""
    usage = data.get("usage")
    return Completion(
        text,
        _token_count(usage, "prompt_tokens"),
        _token_count(usage, "completion_tokens"),
    )


class BifrostClient:
    def __init__(
        self,
        base_url: str,
        virtual_key: str,
        model: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        if not _URL_RE.fullmatch(base):
            raise ValueError("invalid base_url")
        if not _KEY_RE.fullmatch(virtual_key):
            raise ValueError("invalid virtual key")
        if not _MODEL_RE.fullmatch(model):
            raise ValueError("invalid model")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if reasoning_effort is not None and reasoning_effort not in EFFORTS:
            raise ValueError("invalid reasoning_effort")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._url = f"{base}/v1/chat/completions"
        self._virtual_key = virtual_key
        self._transport = transport or _urllib_transport

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, transport: Transport | None = None
    ) -> BifrostClient:
        effort = env.get("TOKENSHIELD_QUARANTINE_EFFORT", DEFAULT_QUARANTINE_EFFORT)
        return cls(
            base_url=env.get("TOKENSHIELD_BIFROST_URL", DEFAULT_BASE_URL),
            virtual_key=env.get("TOKENSHIELD_BIFROST_VK", ""),
            model=env.get("TOKENSHIELD_EXTRACTOR_MODEL", DEFAULT_MODEL),
            reasoning_effort=None if effort in ("", "default") else effort,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"BifrostClient(url={self._url!r}, model={self.model!r})"

    def complete(self, system: str, user: str) -> Completion:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json", "x-bf-vk": self._virtual_key}
        body = json.dumps(payload).encode("utf-8")
        try:
            status, text = self._transport(self._url, headers, body, self.timeout)
        except (OSError, http.client.HTTPException):
            raise BifrostError("network_error") from None
        if not 200 <= status < 300:
            raise BifrostError(f"http_{status}")
        return _parse_completion(text)