"""Tests for the optional JSON mode on the Bifrost clients."""

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


def complete_payload(**kwargs):
    transport = FakeTransport()
    client = BifrostClient("http://localhost:8080", KEY, "groq/m", transport=transport, **kwargs)
    client.complete("s", "u")
    return transport.calls[0]


def test_json_mode_is_off_by_default():
    assert "response_format" not in complete_payload()


def test_json_mode_asks_for_a_json_object():
    assert complete_payload(json_mode=True)["response_format"] == {"type": "json_object"}


def test_json_mode_can_be_toggled_after_construction():
    transport = FakeTransport()
    client = BifrostClient("http://localhost:8080", KEY, "groq/m", transport=transport)
    client.json_mode = True
    client.complete("s", "u")
    client.json_mode = False
    client.complete("s", "u")
    assert "response_format" in transport.calls[0]
    assert "response_format" not in transport.calls[1]


def env_payload(value):
    transport = FakeTransport()
    env = {"TOKENSHIELD_BIFROST_VK": KEY}
    if value is not None:
        env["TOKENSHIELD_QUARANTINE_JSON"] = value
    BifrostClient.from_env(env, transport=transport).complete("s", "u")
    return transport.calls[0]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_from_env_switches_json_mode_on(value):
    assert env_payload(value)["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "maybe"])
def test_from_env_leaves_json_mode_off(value):
    assert "response_format" not in env_payload(value)


def test_the_main_model_client_never_sends_a_response_format():
    transport = FakeTransport()
    chat = BifrostChat(
        "http://localhost:8080", KEY, "groq/m", json_mode=True, transport=transport
    )
    chat.chat([{"role": "user", "content": "q"}])
    assert "response_format" not in transport.calls[0]