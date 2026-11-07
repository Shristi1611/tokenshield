"""Tests for the optional reasoning-effort setting on the Bifrost clients."""

import json

import pytest

from tokenshield.bifrost_client import BifrostClient
from tokenshield.chat import BifrostChat

KEY = "sk-bf-0123456789abcdef"


class FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(json.loads(body))
        reply = {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
        return 200, json.dumps(reply)


def payload_of(client_class, **kwargs):
    transport = FakeTransport()
    client = client_class("http://localhost:8080", KEY, "groq/m", transport=transport, **kwargs)
    if client_class is BifrostChat:
        client.chat([{"role": "user", "content": "q"}])
    else:
        client.complete("s", "u")
    return transport.calls[0]


@pytest.mark.parametrize("client_class", [BifrostClient, BifrostChat])
def test_no_effort_is_sent_by_default(client_class):
    assert "reasoning_effort" not in payload_of(client_class)


@pytest.mark.parametrize("client_class", [BifrostClient, BifrostChat])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_the_effort_is_sent_when_set(client_class, effort):
    assert payload_of(client_class, reasoning_effort=effort)["reasoning_effort"] == effort


@pytest.mark.parametrize("bad", ["", "none", "LOW", "xhigh", "minimal", 5])
def test_invalid_efforts_are_rejected(bad):
    with pytest.raises(ValueError, match="invalid reasoning_effort"):
        BifrostClient("http://localhost:8080", KEY, "groq/m", reasoning_effort=bad)


def from_env_payload(env):
    transport = FakeTransport()
    BifrostClient.from_env(
        {"TOKENSHIELD_BIFROST_VK": KEY, **env}, transport=transport
    ).complete("s", "u")
    return transport.calls[0]


def test_from_env_defaults_the_quarantine_effort_to_low():
    assert from_env_payload({})["reasoning_effort"] == "low"


def test_from_env_can_override_the_effort():
    payload = from_env_payload({"TOKENSHIELD_QUARANTINE_EFFORT": "high"})
    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("value", ["default", ""])
def test_from_env_can_switch_the_effort_off(value):
    assert "reasoning_effort" not in from_env_payload({"TOKENSHIELD_QUARANTINE_EFFORT": value})


def test_from_env_rejects_an_unknown_effort():
    with pytest.raises(ValueError, match="invalid reasoning_effort"):
        BifrostClient.from_env(
            {"TOKENSHIELD_BIFROST_VK": KEY, "TOKENSHIELD_QUARANTINE_EFFORT": "bogus"}
        )