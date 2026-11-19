"""Tests for the two-section cross-check."""

import json

import pytest

from tokenshield.crosscheck import (
    CONFLICT,
    extract_crosschecked,
    split_sections,
)
from tokenshield.extractor import BAD_JSON, Completion, ExtractionFailed
from tokenshield.schema import README_FACTS, content_hash

BASE = {
    "has_install_steps": False,
    "has_test_instructions": False,
    "license": "none",
    "primary_language": "other",
    "num_code_blocks": 0,
}


def reply(**fields):
    return json.dumps({**BASE, **fields})


class Scripted:
    """Returns one scripted reply per call, in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        text, tokens_in, tokens_out = item if isinstance(item, tuple) else (item, 100, 10)
        return Completion(text, tokens_in, tokens_out)


def run(client, text="line one\nline two\nline three\nline four\n", **kwargs):
    return extract_crosschecked(client, README_FACTS, text, **kwargs)


def values(result):
    return dict(result.facts.values)


# --- splitting ---------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_split_sections_joins_back_to_the_original(count):
    text = "".join(f"line {index}\n" for index in range(40))
    parts = split_sections(text, count)
    assert "".join(parts) == text
    assert len(parts) == count
    assert all(part.endswith("\n") for part in parts)


def test_split_sections_cuts_near_the_middle_at_a_line_end():
    text = "aaaa\n" * 10
    first, second = split_sections(text, 2)
    assert len(first) == 25 and len(second) == 25


def test_split_sections_handles_text_without_newlines_and_empty_text():
    assert split_sections("no newline at all", 3) == ["no newline at all"]
    assert split_sections("", 2) == [""]


@pytest.mark.parametrize("count", [0, -1, 7])
def test_split_sections_rejects_bad_counts(count):
    with pytest.raises(ValueError):
        split_sections("x\n", count)


# --- merging -----------------------------------------------------------------------


def test_matching_license_in_both_sections_is_kept_without_a_flag():
    result = run(Scripted(reply(license="mit"), reply(license="mit")))
    assert values(result)["license"] == "mit" and result.conflicts == ()


def test_a_license_stated_in_one_section_only_is_kept():
    result = run(Scripted(reply(license="none"), reply(license="apache-2.0")))
    assert values(result)["license"] == "apache-2.0" and result.conflicts == ()


def test_different_licenses_in_different_sections_give_a_conflict():
    result = run(Scripted(reply(license="gpl-3.0"), reply(license="mit")))
    assert values(result)["license"] == CONFLICT
    assert result.conflicts == ("license",)
    assert "license=conflict" in result.facts.render()


def test_other_and_none_never_count_as_a_disagreement():
    result = run(Scripted(reply(license="other"), reply(license="mit")))
    assert values(result)["license"] == "mit" and result.conflicts == ()
    result = run(Scripted(reply(license="none"), reply(license="other")))
    assert values(result)["license"] == "none" and result.conflicts == ()


def test_bool_fields_are_true_if_any_section_says_true():
    result = run(Scripted(reply(has_install_steps=False), reply(has_install_steps=True)))
    assert values(result)["has_install_steps"] is True
    assert values(result)["has_test_instructions"] is False


def test_block_counts_are_summed_and_capped_at_the_field_maximum():
    result = run(Scripted(reply(num_code_blocks=4), reply(num_code_blocks=3)))
    assert values(result)["num_code_blocks"] == 7
    result = run(Scripted(reply(num_code_blocks=9000), reply(num_code_blocks=9000)))
    assert values(result)["num_code_blocks"] == 10000


def test_a_disagreement_on_an_unchecked_field_takes_the_first_section():
    result = run(Scripted(reply(primary_language="rust"), reply(primary_language="python")))
    assert values(result)["primary_language"] == "rust" and result.conflicts == ()


def test_checked_fields_can_be_chosen():
    client = Scripted(reply(primary_language="rust"), reply(primary_language="python"))
    result = run(client, check_fields=("primary_language",))
    assert values(result)["primary_language"] == CONFLICT


def test_declared_version_conflict_and_absence():
    result = run(Scripted(reply(declared_version="v1.2.3"), reply(declared_version="9.9.9")))
    assert values(result)["declared_version"] == CONFLICT
    assert result.conflicts == ("declared_version",)
    result = run(Scripted(reply(), reply(declared_version="v1.2.3")))
    assert values(result)["declared_version"] == "v1.2.3"
    result = run(Scripted(reply(), reply()))
    assert "declared_version" not in values(result)
    result = run(Scripted(reply(declared_version="v2"), reply(declared_version="v2")))
    assert values(result)["declared_version"] == "v2" and result.conflicts == ()


def test_a_single_section_behaves_like_one_plain_extraction():
    text = "one\ntwo\n"
    result = run(Scripted(reply(license="mit")), text, sections=1)
    assert result.sections == 1 and values(result)["license"] == "mit"


def test_three_sections_make_three_calls_and_sum_the_tokens():
    text = "".join(f"line {index}\n" for index in range(30))
    client = Scripted(
        (reply(license="mit"), 100, 10), (reply(), 200, 20), (reply(license="mit"), 300, 30)
    )
    result = run(client, text, sections=3)
    assert len(client.calls) == 3 and result.sections == 3
    assert (result.input_tokens, result.output_tokens) == (600, 60)
    assert values(result)["license"] == "mit"


def test_the_source_hash_covers_the_whole_document_and_each_call_sees_only_its_part():
    text = "alpha\nbeta\ngamma\ndelta\n"
    client = Scripted(reply(), reply())
    result = run(client, text)
    assert result.facts.source_hash == content_hash(text)
    first = client.calls[0][1]
    second = client.calls[1][1]
    assert "alpha" in first and "delta" not in first
    assert "delta" in second and "alpha" not in second


def test_truncation_is_reported_when_any_section_exceeds_the_window():
    text = ("x" * 60 + "\n") * 4
    result = run(Scripted(reply(), reply()), text, max_chars=50)
    assert result.truncated is True


# --- failing closed ----------------------------------------------------------------


def test_a_failing_section_fails_the_whole_check_and_reports_the_tokens_spent():
    client = Scripted((reply(license="mit"), 100, 10), ("not json", 70, 5))
    with pytest.raises(ExtractionFailed) as caught:
        run(client)
    assert caught.value.reason == BAD_JSON
    assert (caught.value.input_tokens, caught.value.output_tokens) == (170, 15)


def test_a_client_error_fails_closed_without_echoing_its_message():
    client = Scripted(RuntimeError("secret detail"))
    with pytest.raises(ExtractionFailed) as caught:
        run(client)
    assert "secret" not in str(caught.value)
