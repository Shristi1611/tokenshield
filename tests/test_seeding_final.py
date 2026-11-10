"""Tests that deterministic baseline failures are seeded too."""

from tokenshield.multiturn import ConvRecord, append_conv_record, load_conv_records
from tokenshield.seeding import seed_baseline


def crec(ok=True, error=None, mode="baseline"):
    return ConvRecord(
        conv_id="c01", variant="poisoned", mode=mode, ok=ok, error=error, turns=4, correct=4,
        followed=4, main_in=24000, main_out=252, quarantine_in=0, quarantine_out=0,
        quarantine_calls=0, unreported=0, retry_tokens=0, turn_tokens=(), action=None,
        seconds=1.0,
    )


def test_a_request_too_large_baseline_is_copied_once(tmp_path):
    source = tmp_path / "old.jsonl"
    dest = tmp_path / "new.jsonl"
    append_conv_record(source, crec(ok=False, error="http_413"))
    assert seed_baseline(source, dest) == 1
    assert seed_baseline(source, dest) == 0
    assert load_conv_records(dest)[("c01", "poisoned", "baseline")].error == "http_413"


def test_rate_limit_failures_are_still_not_copied(tmp_path):
    source = tmp_path / "old.jsonl"
    append_conv_record(source, crec(ok=False, error="http_429"))
    assert seed_baseline(source, tmp_path / "new.jsonl") == 0


def test_shielded_failures_are_never_copied(tmp_path):
    source = tmp_path / "old.jsonl"
    append_conv_record(source, crec(ok=False, error="http_413", mode="shielded"))
    assert seed_baseline(source, tmp_path / "new.jsonl") == 0