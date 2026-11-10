"""Tests for the quarantine model's output budget."""

import json

import pytest

from tokenshield.bifrost_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_QUARANTINE_MAX_TOKENS,
    BifrostClient,
)

KEY = "sk-bf-0123456789abcdef"


class FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(json.loads(body))
        reply = {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
        return 200, json.dumps(reply)


def payload(env):
    transport = FakeTransport()
    client = BifrostClient.from_env({"TOKENSHIELD_BIFROST_VK": KEY, **env}, transport=transport)
    client.complete("s", "u")
    return transport.calls[0]


def test_the_quarantine_default_leaves_room_for_reasoning():
    assert DEFAULT_QUARANTINE_MAX_TOKENS == 2048
    assert DEFAULT_QUARANTINE_MAX_TOKENS > DEFAULT_MAX_TOKENS
    assert payload({})["max_tokens"] == 2048


@pytest.mark.parametrize(("value", "expected"), [("3000", 3000), (" 512 ", 512)])
def test_from_env_can_override_the_budget(value, expected):
    assert payload({"TOKENSHIELD_QUARANTINE_MAX_TOKENS": value})["max_tokens"] == expected


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_setting_uses_the_default(value):
    assert payload({"TOKENSHIELD_QUARANTINE_MAX_TOKENS": value})["max_tokens"] == 2048


@pytest.mark.parametrize("value", ["lots", "1.5", "0x10"])
def test_non_numbers_are_rejected(value):
    with pytest.raises(ValueError, match="invalid max_tokens"):
        BifrostClient.from_env(
            {"TOKENSHIELD_BIFROST_VK": KEY, "TOKENSHIELD_QUARANTINE_MAX_TOKENS": value}
        )


@pytest.mark.parametrize("value", ["0", "-5"])
def test_non_positive_budgets_are_rejected(value):
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        BifrostClient.from_env(
            {"TOKENSHIELD_BIFROST_VK": KEY, "TOKENSHIELD_QUARANTINE_MAX_TOKENS": value}
        )


def test_direct_construction_keeps_the_old_default():
    assert BifrostClient("http://localhost:8080", KEY, "groq/m").max_tokens == DEFAULT_MAX_TOKENS