"""Tests for the optional trusted context passed to the extractor."""

import json

import pytest

from tokenshield.extractor import Completion, build_system_prompt, extract
from tokenshield.schema import README_FACTS


class FakeClient:
    def __init__(self):
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return Completion(
            json.dumps(
                {
                    "has_install_steps": True,
                    "has_test_instructions": False,
                    "license": "mit",
                    "primary_language": "python",
                    "num_code_blocks": 4,
                }
            )
        )


def test_no_context_leaves_the_prompt_unchanged():
    client = FakeClient()
    extract(client, README_FACTS, "text")
    assert client.calls[0][0] == build_system_prompt(README_FACTS)
    assert "Trusted task context" not in client.calls[0][0]


def test_context_is_appended_after_the_field_list():
    client = FakeClient()
    context = "Subject: Python latest stable version."
    extract(client, README_FACTS, "text", context=context)
    system = client.calls[0][0]
    assert system == build_system_prompt(README_FACTS, context)
    assert system.endswith(f"\n\nTrusted task context: {context}")
    assert system.startswith(build_system_prompt(README_FACTS))


def test_context_never_enters_the_untrusted_block():
    client = FakeClient()
    extract(client, README_FACTS, "page text", context="Subject: abc")
    assert "Subject: abc" not in client.calls[0][1]


@pytest.mark.parametrize(
    "context",
    [
        "line\nbreak",
        'has "quotes"',
        "<tag>",
        "back`tick",
        "{brace}",
        "tab\there",
        "unicode \u00e9",
        "x" * 801,
    ],
)
def test_unsafe_contexts_are_rejected_before_any_model_call(context):
    client = FakeClient()
    with pytest.raises(ValueError, match="invalid context"):
        extract(client, README_FACTS, "text", context=context)
    assert client.calls == []


@pytest.mark.parametrize("context", ["", "Subject: Python (latest) 3.13", "x" * 800])
def test_safe_contexts_are_accepted(context):
    client = FakeClient()
    extract(client, README_FACTS, "text", context=context)
    assert len(client.calls) == 1