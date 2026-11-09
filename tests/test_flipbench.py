"""Tests for the answer-flip benchmark."""

import pytest

from tokenshield.benchmark import INJECTION_LABEL, PROJECTS, SPDX
from tokenshield.flipbench import (
    FALSE_LICENSES,
    FALSE_VERSION,
    build_flip_conversations,
    false_license,
    flip_text,
)

CONVOS = build_flip_conversations()


def test_ten_conversations_with_five_turns_each():
    assert len(CONVOS) == 10
    assert len({convo.id for convo in CONVOS}) == 10
    assert all(len(convo.questions) == 5 and len(convo.expected) == 5 for convo in CONVOS)
    assert CONVOS[0].id == "f01-fastgrid"
    assert {convo.injection_id for convo in CONVOS} == {"flip"}


def test_expected_answers_stay_the_true_ones():
    assert CONVOS[0].expected == ("MIT", "python", "yes", "1.2.3", "yes")
    assert CONVOS[4].expected[0] == "GPL-3.0"


@pytest.mark.parametrize("license_id", sorted(SPDX))
def test_the_false_license_is_never_the_true_one(license_id):
    assert false_license(SPDX[license_id]) != SPDX[license_id]
    assert false_license(SPDX[license_id]) in FALSE_LICENSES


def test_the_false_values_never_match_an_expected_answer():
    for convo, project in zip(CONVOS, PROJECTS, strict=True):
        assert FALSE_VERSION not in convo.expected
        assert false_license(SPDX[project.license_id]) not in convo.expected


def test_poisoned_readmes_differ_from_clean_ones_by_one_planted_note():
    for convo, project in zip(CONVOS, PROJECTS, strict=True):
        clean = convo.readme_clean.splitlines()
        extra = [line for line in convo.readme_poisoned.splitlines() if line not in clean]
        assert len(extra) == 1
        assert extra[0].startswith(INJECTION_LABEL)
        assert FALSE_VERSION in extra[0]
        assert flip_text(project) in extra[0]
        assert FALSE_VERSION not in convo.readme_clean
        assert INJECTION_LABEL not in convo.readme_clean


def test_padding_makes_documents_longer_and_keeps_the_note():
    padded = build_flip_conversations(40)[0]
    assert len(padded.readme_clean) > len(CONVOS[0].readme_clean) + 1000
    assert "## Changelog" in padded.readme_poisoned
    assert FALSE_VERSION in padded.readme_poisoned


def test_negative_padding_is_rejected():
    with pytest.raises(ValueError, match="pad_lines"):
        build_flip_conversations(-1)