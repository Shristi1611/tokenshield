"""Tests for the conformal token budgets."""

import math
import random

import pytest

from tokenshield.budget import (
    Budget,
    calibrate,
    detection_rate,
    leave_one_out_false_alarms,
    min_samples,
)


@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.5, 1), (0.2, 4), (0.1, 9), (0.05, 19), (0.01, 99)]
)
def test_min_samples(alpha, expected):
    assert min_samples(alpha) == expected


def test_threshold_is_the_ranked_sample():
    budget = calibrate(list(range(1, 20)), alpha=0.1)
    assert budget.threshold == 18.0
    assert budget.samples == 19
    assert budget.finite


def test_threshold_is_the_maximum_at_the_minimum_sample_size():
    samples = [5, 1, 9, 3, 7, 2, 8, 4, 6]
    assert calibrate(samples, alpha=0.1).threshold == 9.0


def test_too_few_samples_give_an_infinite_budget_that_never_flags():
    budget = calibrate([1, 2, 3, 4, 5, 6, 7, 8], alpha=0.1)
    assert budget.threshold == math.inf
    assert not budget.finite
    assert budget.exceeded(10**18) is False


def test_empty_calibration_gives_an_infinite_budget():
    assert calibrate([], alpha=0.1).threshold == math.inf


def test_exceeded_is_strict():
    budget = calibrate(list(range(1, 10)), alpha=0.1)
    assert budget.exceeded(9) is False
    assert budget.exceeded(9.01) is True


def test_budget_fields_are_exposed():
    assert Budget(5.0, 0.1, 9).threshold == 5.0


@pytest.mark.parametrize("alpha", [0, 0.0005, 0.6, 1, -1, True])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        calibrate([1, 2, 3], alpha)
    with pytest.raises(ValueError, match="alpha"):
        min_samples(alpha)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_samples_are_rejected(bad):
    with pytest.raises(ValueError, match="samples"):
        calibrate([1, 2, bad], alpha=0.1)


def test_false_alarm_rate_matches_alpha_on_exchangeable_data():
    rng = random.Random(7)
    trials = 3000
    flagged = 0
    for _ in range(trials):
        budget = calibrate([rng.random() for _ in range(19)], alpha=0.1)
        if budget.exceeded(rng.random()):
            flagged += 1
    assert 0.07 <= flagged / trials <= 0.13


def test_leave_one_out_flags_only_the_largest_run():
    assert leave_one_out_false_alarms(range(1, 11), alpha=0.1) == (1, 10)


def test_leave_one_out_skips_when_the_others_are_too_few():
    assert leave_one_out_false_alarms([1, 2, 3, 4, 5], alpha=0.1) == (0, 0)


def test_leave_one_out_with_identical_samples_flags_nothing():
    assert leave_one_out_false_alarms([300] * 10, alpha=0.1) == (0, 10)


def test_detection_rate():
    budget = Budget(100.0, 0.1, 9)
    assert detection_rate(budget, [50, 150, 200]) == pytest.approx(2 / 3)


def test_detection_rate_is_none_without_a_finite_budget_or_attacks():
    assert detection_rate(Budget(math.inf, 0.1, 3), [1000]) is None
    assert detection_rate(Budget(100.0, 0.1, 9), []) is None