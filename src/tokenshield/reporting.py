"""Honest reporting over saved multi-turn records.

Token savings come with a percentile-bootstrap interval that resamples whole conversations (the
unit that repeats), not turns. With few conversations generated from one template the interval is
indicative only: it shows run-to-run spread, not how the result would vary across real documents.

Failed runs are never dropped. Each is listed with what it had already spent, and a second
comparison counts that spend. It is a lower bound, because a failed run stops early.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tokenshield.benchmark import BASELINE, MODES, SHIELDED, VARIANTS
from tokenshield.multiturn import ConvRecord

DEFAULT_RESAMPLES = 2000
DEFAULT_LEVEL = 0.95


@dataclass(frozen=True)
class PairedCost:
    conv_id: str
    baseline: int
    shielded: int


@dataclass(frozen=True)
class FailedRun:
    conv_id: str
    variant: str
    mode: str
    error: str
    turns_done: int
    tokens_spent: int


def paired_costs(records: Iterable[ConvRecord], variant: str) -> list[PairedCost]:
    """Conversations whose baseline and shielded runs both finished."""
    by_key = {record.key: record for record in records}
    conv_ids = sorted({key[0] for key in by_key})
    pairs: list[PairedCost] = []
    for conv_id in conv_ids:
        base = by_key.get((conv_id, variant, BASELINE))
        shield = by_key.get((conv_id, variant, SHIELDED))
        if base is not None and shield is not None and base.ok and shield.ok:
            pairs.append(PairedCost(conv_id, base.cost_tokens, shield.cost_tokens))
    return pairs


def saving(pairs: Sequence[PairedCost]) -> float | None:
    total = sum(pair.baseline for pair in pairs)
    if total == 0:
        return None
    return 1.0 - sum(pair.shielded for pair in pairs) / total


def bootstrap_interval(
    pairs: Sequence[PairedCost],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 1,
) -> tuple[float, float] | None:
    if resamples < 1 or not 0.0 < level < 1.0:
        raise ValueError("resamples must be positive and level between 0 and 1")
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    count = len(pairs)
    values: list[float] = []
    for _ in range(resamples):
        value = saving([pairs[rng.randrange(count)] for _ in range(count)])
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    tail = (1.0 - level) / 2.0
    return values[int(tail * (len(values) - 1))], values[int((1.0 - tail) * (len(values) - 1))]


def failed_runs(records: Iterable[ConvRecord]) -> list[FailedRun]:
    failures = [
        FailedRun(
            item.conv_id, item.variant, item.mode, item.error or "unknown", item.turns,
            item.cost_tokens,
        )
        for item in records
        if not item.ok
    ]
    return sorted(failures, key=lambda item: (item.conv_id, item.variant, item.mode))


def _group(items: Sequence[ConvRecord], mode: str, variant: str) -> list[ConvRecord]:
    return [item for item in items if item.mode == mode and item.variant == variant]


def _turn_counts(items: Sequence[ConvRecord]) -> tuple[int, int, int]:
    done = [item for item in items if item.ok]
    return (
        sum(item.correct for item in done),
        sum(item.followed for item in done),
        sum(item.turns for item in done),
    )


def _mean_cost(items: Sequence[ConvRecord]) -> float | None:
    return sum(item.cost_tokens for item in items) / len(items) if items else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def format_report(
    title: str,
    records: Iterable[ConvRecord],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 1,
) -> str:
    items = list(records)
    lines = [f"== {title} ==", ""]
    lines.append("tokens saved by the shield (positive = fewer tokens), finished pairs only:")
    for variant in VARIANTS:
        pairs = paired_costs(items, variant)
        text = f"  {variant}: {_pct(saving(pairs))} over {len(pairs)} conversations"
        interval = bootstrap_interval(pairs, resamples=resamples, seed=seed)
        if interval is not None:
            text += f", 95% bootstrap interval {_pct(interval[0])} to {_pct(interval[1])}"
        lines.append(text)

    lines += ["", "turns answered correctly / hijacked (long or multi-line reply), finished runs:"]
    for mode in MODES:
        for variant in VARIANTS:
            correct, hijacked, turns = _turn_counts(_group(items, mode, variant))
            lines.append(
                f"  {mode}/{variant}: {correct}/{turns} correct, {hijacked}/{turns} hijacked"
            )

    failures = failed_runs(items)
    lines += ["", f"failed runs: {len(failures)} (their spend is a lower bound)"]
    for failure in failures:
        lines.append(
            f"  {failure.conv_id} {failure.variant} {failure.mode}: {failure.error} after "
            f"{failure.turns_done} turns, {failure.tokens_spent} tokens spent"
        )

    lines += ["", "mean tokens per conversation, counting the spend of failed runs:"]
    means: dict[tuple[str, str], float | None] = {}
    for mode in MODES:
        for variant in VARIANTS:
            means[(mode, variant)] = _mean_cost(_group(items, mode, variant))
            lines.append(f"  {mode}/{variant}: {_num(means[(mode, variant)])}")
    lines.append("saving when failed runs are counted:")
    for variant in VARIANTS:
        base = means[(BASELINE, variant)]
        shield = means[(SHIELDED, variant)]
        value = None if not base or shield is None else 1.0 - shield / base
        lines.append(f"  {variant}: {_pct(value)}")
    return "\n".join(lines)