"""Tests for the honest reporting module."""

import pytest

from tokenshield.multiturn import ConvRecord
from tokenshield.reporting import (
    PairedCost,
    bootstrap_interval,
    failed_runs,
    format_report,
    paired_costs,
    saving,
)


def crec(conv_id="c01", variant="clean", mode="baseline", ok=True, error=None, turns=5,
         correct=5, followed=0, main_in=400, main_out=100, quarantine_in=0, quarantine_out=0):
    return ConvRecord(
        conv_id=conv_id, variant=variant, mode=mode, ok=ok, error=error, turns=turns,
        correct=correct, followed=followed, main_in=main_in, main_out=main_out,
        quarantine_in=quarantine_in, quarantine_out=quarantine_out, quarantine_calls=0,
        unreported=0, retry_tokens=0, turn_tokens=(), action=None, seconds=1.0,
    )


def base(conv_id, variant, cost):
    return crec(conv_id, variant, "baseline", main_in=cost - 100, main_out=100)


def shield(conv_id, variant, cost):
    return crec(
        conv_id, variant, "shielded", main_in=cost - 60, main_out=10,
        quarantine_in=40, quarantine_out=10,
    )


def dataset():
    items = []
    for conv in ("c01", "c02", "c03"):
        items += [
            base(conv, "clean", 500), shield(conv, "clean", 200),
            base(conv, "poisoned", 1000), shield(conv, "poisoned", 200),
        ]
    items.append(
        crec("c04", "poisoned", "baseline", ok=False, error="http_413", turns=4,
             main_in=1900, main_out=100)
    )
    return items


def pairs(*values):
    return [PairedCost(f"c{index}", b, s) for index, (b, s) in enumerate(values)]


def test_cost_helpers_match_their_names():
    assert base("c", "clean", 500).cost_tokens == 500
    assert shield("c", "clean", 200).cost_tokens == 200


def test_only_conversations_where_both_agents_finished_are_paired():
    items = dataset()
    assert [pair.conv_id for pair in paired_costs(items, "poisoned")] == ["c01", "c02", "c03"]
    items.append(shield("c04", "poisoned", 300))
    assert len(paired_costs(items, "poisoned")) == 3
    items.append(base("c05", "clean", 500))
    assert len(paired_costs(items, "clean")) == 3


def test_saving_values():
    assert saving(pairs((100, 40), (200, 80))) == pytest.approx(0.6)
    assert saving(pairs((100, 150))) == pytest.approx(-0.5)
    assert saving([]) is None
    assert saving(pairs((0, 5))) is None


def test_the_bootstrap_of_a_constant_saving_is_that_saving():
    low, high = bootstrap_interval(pairs((100, 40), (200, 80), (300, 120)), resamples=200)
    assert low == pytest.approx(0.6)
    assert high == pytest.approx(0.6)


def test_the_bootstrap_is_deterministic_for_a_seed_and_ordered():
    data = pairs((100, 40), (100, 60), (100, 30), (100, 50), (100, 70))
    first = bootstrap_interval(data, resamples=300, seed=5)
    assert first == bootstrap_interval(data, resamples=300, seed=5)
    low, high = first
    assert low <= saving(data) <= high


def test_the_bootstrap_needs_at_least_two_pairs():
    assert bootstrap_interval([]) is None
    assert bootstrap_interval(pairs((100, 40))) is None


def test_the_bootstrap_without_baseline_tokens_gives_none():
    assert bootstrap_interval(pairs((0, 5), (0, 6)), resamples=50) is None


@pytest.mark.parametrize("kwargs", [{"resamples": 0}, {"level": 0}, {"level": 1}, {"level": 1.5}])
def test_the_bootstrap_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_interval(pairs((100, 40), (100, 50)), **kwargs)


def test_failed_runs_are_listed_with_what_they_spent():
    failures = failed_runs(dataset())
    assert len(failures) == 1
    failure = failures[0]
    assert (failure.conv_id, failure.variant, failure.mode) == ("c04", "poisoned", "baseline")
    assert (failure.error, failure.turns_done, failure.tokens_spent) == ("http_413", 4, 2000)


def test_a_failed_run_without_an_error_is_reported_as_unknown():
    assert failed_runs([crec(ok=False, error=None)])[0].error == "unknown"


def test_the_report_has_every_section_and_the_honest_numbers():
    text = format_report("demo", dataset(), resamples=200)
    assert "== demo ==" in text
    assert "clean: +60.0% over 3 conversations" in text
    assert "poisoned: +80.0% over 3 conversations" in text
    assert "95% bootstrap interval +60.0% to +60.0%" in text
    assert "baseline/clean: 15/15 correct, 0/15 hijacked" in text
    assert "failed runs: 1" in text
    assert "c04 poisoned baseline: http_413 after 4 turns, 2000 tokens spent" in text
    assert "baseline/poisoned: 1250" in text
    assert "shielded/poisoned: 200" in text
    assert "poisoned: +84.0%" in text


def test_a_single_pair_has_no_interval():
    text = format_report("one", [base("c01", "clean", 500), shield("c01", "clean", 200)])
    assert "over 1 conversations" in text
    assert "bootstrap" not in text


def test_an_empty_report_does_not_crash():
    text = format_report("empty", [])
    assert "n/a" in text
    assert "failed runs: 0" in text