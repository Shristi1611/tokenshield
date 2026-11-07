"""Chat-style access to the Bifrost gateway, plus the two agent paths the benchmark compares.

run_baseline is the naive agent: raw untrusted content goes straight to the main model.
run_shielded passes every message through the shield first. Both meter every call.
"""

from __future__ import annotations

import http.client
import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from tokenshield.bifrost_client import _MODEL_RE, BifrostClient, BifrostError, _parse_completion
from tokenshield.extractor import Completion, LLMClient
from tokenshield.gate import GatePolicy
from tokenshield.meter import MAIN, TokenMeter
from tokenshield.shield import KIND_KEY, ShieldOutcome, shield_messages


class ChatClient(Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...


class BifrostChat(BifrostClient):
    def chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        chosen_model = self.model if model is None else model
        if not _MODEL_RE.fullmatch(chosen_model):
            raise ValueError("invalid model")
        limit = self.max_tokens if max_tokens is None else max_tokens
        if limit < 1:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, object] = {
            "model": chosen_model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": limit,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json", "x-bf-vk": self._virtual_key}
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise BifrostError("bad_request") from None
        try:
            status, text = self._transport(self._url, headers, body, self.timeout)
        except (OSError, http.client.HTTPException):
            raise BifrostError("network_error") from None
        if not 200 <= status < 300:
            raise BifrostError(f"http_{status}")
        return _parse_completion(text)


def _call_main(
    main: ChatClient,
    messages: Sequence[Mapping[str, object]],
    meter: TokenMeter,
    label: str,
) -> Completion:
    try:
        completion = main.chat(messages)
    except BifrostError:
        meter.record(label, MAIN, False)
        raise
    meter.record(label, MAIN, True, completion.input_tokens, completion.output_tokens)
    return completion


def run_baseline(
    main: ChatClient,
    messages: Sequence[Mapping[str, object]],
    meter: TokenMeter,
    label: str,
) -> Completion:
    """The naive agent: untrusted content reaches the main model as raw text."""
    plain = [{key: value for key, value in m.items() if key != KIND_KEY} for m in messages]
    return _call_main(main, plain, meter, label)


def run_shielded(
    quarantine: LLMClient,
    main: ChatClient,
    policy: GatePolicy,
    messages: Sequence[Mapping[str, object]],
    meter: TokenMeter,
    label: str,
) -> tuple[Completion, ShieldOutcome]:
    """The shielded agent: untrusted content reaches the main model only as typed facts."""
    outcome = shield_messages(quarantine, policy, messages, meter)
    return _call_main(main, outcome.messages, meter, label), outcome