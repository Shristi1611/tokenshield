"""Tests for seeding a new results file with finished baseline conversations."""

import pytest

from tokenshield.multiturn import ConvRecord, append_conv_record, load_conv_records
from tokenshield.seeding import seed_baseline


def crec(**overrides):
    values = {
        "conv_id": "c01",
        "variant": "clean",
        "mode": "baseline",
        "ok": True,
        "error": None,
        "turns": 5,
        "correct": 5,
        "followed": 0,
        "main_in": 400,
        "main_out": 100,
        "quarantine_in": 0,
        "quarantine_out": 0,
        "quarantine_calls": 0,
        "unreported": 0,
        "retry_tokens": 0,
        "turn_tokens": (80, 90, 100, 110, 120),
        "action": None,
        "seconds": 1.0,
    }
    values.update(overrides)
    return ConvRecord(**values)


def test_only_finished_baseline_runs_are_copied(tmp_path):
    source = tmp_path / "old.jsonl"
    dest = tmp_path / "new.jsonl"
    append_conv_record(source, crec())
    append_conv_record(source, crec(variant="poisoned"))
    append_conv_record(source, crec(mode="shielded"))
    append_conv_record(source, crec(conv_id="c02", ok=False, error="http_429"))
    assert seed_baseline(source, dest) == 2
    assert sorted(load_conv_records(dest)) == [
        ("c01", "clean", "baseline"),
        ("c01", "poisoned", "baseline"),
    ]


def test_runs_the_destination_already_finished_are_not_copied_again(tmp_path):
    source = tmp_path / "old.jsonl"
    dest = tmp_path / "new.jsonl"
    append_conv_record(source, crec())
    assert seed_baseline(source, dest) == 1
    assert seed_baseline(source, dest) == 0
    assert len(dest.read_text(encoding="utf-8").splitlines()) == 1


def test_a_failed_run_in_the_destination_is_replaced(tmp_path):
    source = tmp_path / "old.jsonl"
    dest = tmp_path / "new.jsonl"
    append_conv_record(dest, crec(ok=False, error="http_429"))
    append_conv_record(source, crec())
    assert seed_baseline(source, dest) == 1
    assert load_conv_records(dest)[("c01", "clean", "baseline")].ok is True


def test_the_source_file_is_left_unchanged(tmp_path):
    source = tmp_path / "old.jsonl"
    append_conv_record(source, crec())
    before = source.read_text(encoding="utf-8")
    seed_baseline(source, tmp_path / "new.jsonl")
    assert source.read_text(encoding="utf-8") == before


def test_a_missing_source_copies_nothing(tmp_path):
    assert seed_baseline(tmp_path / "missing.jsonl", tmp_path / "new.jsonl") == 0


def test_source_and_destination_must_differ(tmp_path):
    path = tmp_path / "same.jsonl"
    append_conv_record(path, crec())
    with pytest.raises(ValueError, match="differ"):
        seed_baseline(path, path)