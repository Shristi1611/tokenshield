"""Layer 2: conformal token budgets.

Given the token totals of clean runs, a split-conformal quantile gives a budget. If a new clean
run is exchangeable with the calibration runs, it exceeds the budget with probability at most
alpha. That bounds false alarms only. How many attacks the budget catches is measured, never
promised. With few calibration runs the budget is coarse: at least min_samples(alpha) clean runs
are needed for a finite threshold, and below that the budget never flags anything.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

MIN_ALPHA = 0.001
MAX_ALPHA = 0.5
_EPSILON = 1e-9


@dataclass(frozen=True)
class Budget:
    threshold: float
    alpha: float
    samples: int

    @property
    def finite(self) -> bool:
        return math.isfinite(self.threshold)

    def exceeded(self, tokens: float) -> bool:
        return tokens > self.threshold


def _check_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not MIN_ALPHA <= alpha <= MAX_ALPHA:
        raise ValueError("alpha must be between 0.001 and 0.5")


def _rank(samples: int, alpha: float) -> int:
    return math.ceil((samples + 1) * (1 - alpha) - _EPSILON)


def min_samples(alpha: float = 0.1) -> int:
    """Fewest clean runs that give a finite threshold at this alpha."""
    _check_alpha(alpha)
    count = 1
    while _rank(count, alpha) > count:
        count += 1
    return count


def calibrate(samples: Iterable[float], alpha: float = 0.1) -> Budget:
    _check_alpha(alpha)
    values = sorted(float(value) for value in samples)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("samples must be finite and non-negative")
    rank = _rank(len(values), alpha)
    threshold = values[rank - 1] if 1 <= rank <= len(values) else math.inf
    return Budget(threshold, alpha, len(values))


def leave_one_out_false_alarms(samples: Iterable[float], alpha: float = 0.1) -> tuple[int, int]:
    """Calibrate on all but one clean run and test the held-out one. Returns (flagged, tested)."""
    values = list(samples)
    flagged = 0
    tested = 0
    for index, value in enumerate(values):
        budget = calibrate(values[:index] + values[index + 1 :], alpha)
        if not budget.finite:
            continue
        tested += 1
        if budget.exceeded(value):
            flagged += 1
    return flagged, tested


def detection_rate(budget: Budget, attacked: Iterable[float]) -> float | None:
    """Fraction of attacked runs that exceed the budget, or None if it cannot be computed."""
    values = list(attacked)
    if not budget.finite or not values:
        return None
    return sum(1 for value in values if budget.exceeded(value)) / len(values)