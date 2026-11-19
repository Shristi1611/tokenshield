"""Compare every agent mode on the same conversations: accuracy, hijack rate, tokens, savings.

Intervals are percentile bootstraps over whole conversations. Failed runs are listed, never dropped.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from tokenshield.baselines import ALL_MODES
from tokenshield.benchmark import BASELINE, CLEAN, POISONED, VARIANTS
from tokenshield.multiturn import ConvRecord
from tokenshield.reporting import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    PairedCost,
    bootstrap_interval,
    saving,
)

Interval = tuple[float, float] | None


def _rate(items: Sequence[ConvRecord], top: Callable[[ConvRecord], int]) -> float | None:
    turns = sum(item.turns for item in items)
    return sum(top(item) for item in items) / turns if turns else None


def rate_interval(
    items: Sequence[ConvRecord],
    top: Callable[[ConvRecord], int],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 1,
) -> Interval:
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [items[rng.randrange(len(items))] for _ in range(len(items))]
        value = _rate(sample, top)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    tail = (1.0 - level) / 2.0
    low = values[int(tail * (len(values) - 1))]
    high = values[round((1.0 - tail) * (len(values) - 1))]
    return low, high


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group(items: Sequence[ConvRecord]) -> dict[str, Any]:
    done = [item for item in items if item.ok]
    return {
        "runs": len(items),
        "ok": len(done),
        "turns": sum(item.turns for item in done),
        "accuracy": _rate(done, lambda item: item.correct),
        "accuracy_ci": rate_interval(done, lambda item: item.correct),
        "followed": _rate(done, lambda item: item.followed),
        "followed_ci": rate_interval(done, lambda item: item.followed),
        "mean_cost": _mean([float(item.cost_tokens) for item in done]),
    }


def _paired(
    records: dict[tuple[str, str, str], ConvRecord], mode: str, variant: str
) -> list[PairedCost]:
    pairs: list[PairedCost] = []
    for conv_id in sorted({key[0] for key in records}):
        base = records.get((conv_id, variant, BASELINE))
        other = records.get((conv_id, variant, mode))
        if base is not None and other is not None and base.ok and other.ok:
            pairs.append(PairedCost(conv_id, base.cost_tokens, other.cost_tokens))
    return pairs


def compare_modes(records: Iterable[ConvRecord]) -> dict[str, Any]:
    items = list(records)
    by_key = {item.key: item for item in items}
    modes = [mode for mode in ALL_MODES if any(item.mode == mode for item in items)]
    groups: dict[str, dict[str, Any]] = {}
    for mode in modes:
        for variant in VARIANTS:
            selected = [item for item in items if item.mode == mode and item.variant == variant]
            group = _group(selected)
            if mode != BASELINE:
                pairs = _paired(by_key, mode, variant)
                group["pairs"] = len(pairs)
                group["saving"] = saving(pairs)
                group["saving_ci"] = bootstrap_interval(pairs)
            groups[f"{mode}/{variant}"] = group
    failed = [
        {"conv": item.conv_id, "variant": item.variant, "mode": item.mode, "error": item.error}
        for item in items
        if not item.ok
    ]
    return {"modes": modes, "groups": groups, "failed": failed}


def _ci(value: float | None, interval: Interval, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    head = f"{value * 100:{sign}.0f}%"
    if interval is None:
        return head
    return f"{head} ({interval[0] * 100:{sign}.0f} to {interval[1] * 100:{sign}.0f})"


def format_comparison(summary: dict[str, Any]) -> str:
    lines = ["Mode comparison (95% bootstrap over conversations; small samples, indicative only)", ""]
    for variant in (CLEAN, POISONED):
        lines.append(f"{variant} documents")
        lines.append(
            f"  {'mode':<11}{'ok/runs':>8}{'accuracy':>22}{'hijacked':>22}"
            f"{'mean tokens':>13}{'saved vs plain':>24}"
        )
        for mode in summary["modes"]:
            group = summary["groups"][f"{mode}/{variant}"]
            cost = "n/a" if group["mean_cost"] is None else f"{group['mean_cost']:.0f}"
            saved = (
                "-"
                if mode == BASELINE
                else _ci(group.get("saving"), group.get("saving_ci"), signed=True)
            )
            lines.append(
                f"  {mode:<11}{group['ok']:>4}/{group['runs']:<3}"
                f"{_ci(group['accuracy'], group['accuracy_ci']):>22}"
                f"{_ci(group['followed'], group['followed_ci']):>22}"
                f"{cost:>13}{saved:>24}"
            )
        lines.append("")
    if summary["failed"]:
        lines.append("failed runs (kept, never dropped):")
        lines += [
            f"  {item['conv']} {item['variant']} {item['mode']}: {item['error']}"
            for item in summary["failed"]
        ]
    else:
        lines.append("failed runs: 0")
    return "\n".join(lines)