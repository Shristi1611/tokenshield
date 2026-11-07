"""Tests for the token meter."""

import threading

import pytest

from tokenshield.meter import MAIN, QUARANTINE, TokenMeter


def test_empty_meter_has_zero_totals():
    totals = TokenMeter().totals()
    assert (totals.calls, totals.failed_calls, totals.unreported_calls) == (0, 0, 0)
    assert totals.total_tokens == 0


def test_records_are_summed_by_role():
    meter = TokenMeter()
    meter.record("a", QUARANTINE, True, 100, 20)
    meter.record("b", MAIN, True, 50, 10)
    meter.record("c", MAIN, True, 5, 1)
    assert meter.totals().total_tokens == 186
    quarantine = meter.totals(QUARANTINE)
    assert (quarantine.calls, quarantine.input_tokens, quarantine.output_tokens) == (1, 100, 20)
    main = meter.totals(MAIN)
    assert (main.calls, main.input_tokens, main.output_tokens) == (2, 55, 11)


def test_failed_and_unreported_calls_are_counted_not_hidden():
    meter = TokenMeter()
    meter.record("a", QUARANTINE, False, 30, 5)
    meter.record("b", QUARANTINE, True)
    meter.record("c", QUARANTINE, True, None, 7)
    totals = meter.totals()
    assert totals.calls == 3
    assert totals.failed_calls == 1
    assert totals.unreported_calls == 2
    assert (totals.input_tokens, totals.output_tokens) == (30, 12)


@pytest.mark.parametrize("bad", [True, False, -1, 1.5, "7", [], object()])
def test_invalid_token_counts_are_treated_as_unreported(bad):
    meter = TokenMeter()
    meter.record("a", MAIN, True, bad, bad)
    record = meter.records()[0]
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert meter.totals().unreported_calls == 1


def test_unknown_roles_are_rejected():
    with pytest.raises(ValueError, match="role"):
        TokenMeter().record("a", "other", True)
    with pytest.raises(ValueError, match="role"):
        TokenMeter().totals("other")


def test_reset_clears_everything():
    meter = TokenMeter()
    meter.record("a", MAIN, True, 1, 1)
    meter.reset()
    assert meter.records() == ()
    assert meter.totals().calls == 0


def test_records_snapshot_is_not_live():
    meter = TokenMeter()
    snapshot = meter.records()
    meter.record("a", MAIN, True, 1, 1)
    assert snapshot == ()
    assert len(meter.records()) == 1


def test_concurrent_recording_loses_nothing():
    meter = TokenMeter()

    def work():
        for _ in range(200):
            meter.record("x", MAIN, True, 1, 1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert meter.totals().calls == 1600