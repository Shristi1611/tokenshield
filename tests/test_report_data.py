"""Tests for the structured report data used by the API and the frontend."""

import json

import pytest

from tokenshield.multiturn import ConvRecord
from tokenshield.reporting import report_data


def crec(conv_id, variant, mode, cost, ok=True, error=None, turns=5, correct=5, followed=0,
         turn_tokens=()):
    return ConvRecord(
        conv_id=conv_id, variant=variant, mode=mode, ok=ok, error=error, turns=turns,
        correct=correct, followed=followed, main_in=cost - 100, main_out=100,
        quarantine_in=0, quarantine_out=0, quarantine_calls=0, unreported=0,
        retry_tokens=0, turn_tokens=tuple(turn_tokens), action=None, seconds=1.0,
    )


def dataset():
    items = []
    for conv in ("c01", "c02", "c03"):
        items += [
            crec(conv, "clean", "baseline", 500, turn_tokens=(80, 90, 100, 110, 120)),
            crec(conv, "clean", "shielded", 200),
            crec(conv, "poisoned", "baseline", 1000, correct=4, followed=2),
            crec(conv, "poisoned", "shielded", 200),
        ]
    items.append(crec("c04", "poisoned", "baseline", 2000, ok=False, error="http_413", turns=4))
    return items


def test_savings_and_intervals():
    data = report_data(dataset(), resamples=200)
    clean = data["savings"]["clean"]
    assert clean["conversations"] == 3
    assert clean["saving"] == pytest.approx(0.6)
    assert clean["interval"] == pytest.approx([0.6, 0.6])
    assert data["savings"]["poisoned"]["saving"] == pytest.approx(0.8)


def test_group_counts_and_costs():
    groups = report_data(dataset(), resamples=50)["groups"]
    base = groups["baseline/clean"]
    assert (base["runs"], base["finished"], base["turns"]) == (3, 3, 15)
    assert (base["correct"], base["hijacked"], base["mean_cost"]) == (15, 0, 500)
    assert base["per_turn"] == [80, 90, 100, 110, 120]
    poisoned = groups["baseline/poisoned"]
    assert (poisoned["runs"], poisoned["finished"], poisoned["turns"]) == (4, 3, 15)
    assert (poisoned["correct"], poisoned["hijacked"]) == (12, 6)
    assert poisoned["mean_cost"] == 1250
    assert groups["shielded/clean"]["per_turn"] == []


def test_failed_runs_are_listed_with_their_spend():
    failures = report_data(dataset(), resamples=50)["failed_runs"]
    assert failures == [
        {
            "conv_id": "c04",
            "variant": "poisoned",
            "mode": "baseline",
            "error": "http_413",
            "turns_done": 4,
            "tokens_spent": 2000,
        }
    ]


def test_the_data_is_json_safe():
    json.dumps(report_data(dataset(), resamples=50))


def test_empty_data():
    data = report_data([])
    assert data["savings"]["clean"] == {"conversations": 0, "saving": None, "interval": None}
    assert data["failed_runs"] == []
    assert data["groups"]["baseline/clean"]["mean_cost"] is None
    assert data["groups"]["baseline/clean"]["per_turn"] == []