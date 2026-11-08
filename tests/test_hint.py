"""Tests for field hints and the hardened extractor prompt."""

import pytest

from tokenshield.extractor import build_system_prompt
from tokenshield.schema import README_FACTS, FactSchema, FieldSpec, FieldType


def test_hints_default_to_empty():
    assert FieldSpec("a", FieldType.BOOL).hint == ""


def test_a_hint_follows_its_field_description():
    schema = FactSchema("s", (FieldSpec("a", FieldType.BOOL, hint="Check the header."),))
    assert '- "a": true or false Check the header.' in build_system_prompt(schema)


def test_a_hint_follows_the_optional_marker():
    spec = FieldSpec("a", FieldType.TOKEN, required=False, hint="Be careful.")
    assert "(optional) Be careful." in build_system_prompt(FactSchema("s", (spec,)))


def test_fields_without_hints_are_described_as_before():
    schema = FactSchema("s", (FieldSpec("a", FieldType.BOOL),))
    assert '- "a": true or false\n\nIf a required value' in build_system_prompt(schema)


@pytest.mark.parametrize(
    "hint", ["line\nbreak", 'has "quotes"', "<tag>", "{brace}", "x" * 201, "caf\u00e9"]
)
def test_unsafe_hints_are_rejected(hint):
    with pytest.raises(ValueError, match="unsafe hint"):
        FieldSpec("a", FieldType.BOOL, hint=hint)


def test_the_longest_safe_hint_is_accepted():
    assert len(FieldSpec("a", FieldType.BOOL, hint="x" * 200).hint) == 200


def test_the_readme_version_field_asks_for_the_version_when_present():
    spec = next(field for field in README_FACTS.fields if field.name == "declared_version")
    assert spec.required is False
    assert "Always include it" in spec.hint
    assert "Always include it" in build_system_prompt(README_FACTS)


def test_the_prompt_warns_that_the_data_may_try_to_instruct():
    prompt = build_system_prompt(README_FACTS)
    assert "tries to instruct you" in prompt
    assert "only the JSON object" in prompt


def test_only_one_field_is_marked_optional():
    assert build_system_prompt(README_FACTS).count("(optional)") == 1