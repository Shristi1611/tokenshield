"""Multi-turn benchmark: the same README is questioned over several turns.

Why: in a single turn the shield cannot save tokens, because the quarantine model reads the whole
document too. In a conversation the baseline agent re-sends the raw document with every turn (the
history grows), while the shielded agent quarantines it once and re-sends only a few typed facts.
This module measures whether that pays off.

Counts are raw tokens as reported by the provider. Prompt caching would lower the baseline's
price per token but not its token count, so a dollar comparison needs its own measurement.

Limits: small and synthetic, written by the project author; five in-schema questions per README
(out-of-schema questions are excluded because the shield cannot answer them); optional padding
adds benign filler text so that documents can be made longer.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from tokenshield.benchmark import (
    BASELINE,
    CLEAN,
    DEFAULT_RETRIES,
    INJECTIONS,
    MODES,
    POISONED,
    PROJECTS,
    QUESTIONS,
    RETRYABLE,
    SHIELDED,
    SYSTEM_PROMPT,
    VARIANTS,
    _expected,
    followed_injection,
    grade,
    render_readme,
)
from tokenshield.bifrost_client import BifrostError
from tokenshield.chat import ChatClient, _call_main
from tokenshield.extractor import LLM_ERROR, Completion, LLMClient
from tokenshield.gate import GatePolicy, SourceKind
from tokenshield.meter import MAIN, QUARANTINE, TokenMeter
from tokenshield.shield import KIND_KEY, ShieldOutcome, shield_messages

TURN_KINDS = ("license", "language", "install", "version", "tests")
FORMAT_SUFFIX = "\nReply with exactly one line: ANSWER: <value>"
QUARANTINE_ERROR = "quarantine_llm_error"

FILLER = (
    "Improved the wording of the getting started guide.",
    "Fixed a typo in one of the examples.",
    "Clarified the description of an option in the reference.",
    "Tidied the layout of the contributing notes.",
    "Added a short note about supported platforms.",
    "Reworded an error message to be easier to read.",
    "Updated the screenshots in the documentation.",
    "Reordered a few sections for readability.",
    "Added a link to the community chat.",
    "Corrected the capitalisation of a heading.",
    "Explained a default setting in more detail.",
    "Removed an outdated remark from the faq.",
)


def pad_readme(text: str, lines: int) -> str:
    """Insert a benign changelog (no numbers, no answer-bearing words) before Contributing."""
    if lines < 0:
        raise ValueError("lines must not be negative")
    if lines == 0:
        return text
    block = ["## Changelog", ""]
    block += [f"- {FILLER[index % len(FILLER)]}" for index in range(lines)]
    block.append("")
    head, marker, tail = text.partition("## Contributing")
    return head + "\n".join(block) + "\n" + marker + tail


@dataclass(frozen=True)
class Conversation:
    id: str
    questions: tuple[str, ...]
    expected: tuple[str, ...]
    readme_clean: str
    readme_poisoned: str
    injection_id: str


def build_conversations(pad_lines: int = 0) -> tuple[Conversation, ...]:
    if pad_lines < 0:
        raise ValueError("pad_lines must not be negative")
    result = []
    for index, project in enumerate(PROJECTS):
        injection_id, injection = INJECTIONS[index % len(INJECTIONS)]
        result.append(
            Conversation(
                id=f"c{index + 1:02d}-{project.name}",
                questions=tuple(QUESTIONS[kind] for kind in TURN_KINDS),
                expected=tuple(_expected(project, kind) for kind in TURN_KINDS),
                readme_clean=pad_readme(render_readme(project), pad_lines),
                readme_poisoned=pad_readme(render_readme(project, injection), pad_lines),
                injection_id=injection_id,
            )
        )
    return tuple(result)


def initial_messages(convo: Conversation, variant: str, mode: str) -> list[dict[str, object]]:
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    if mode not in MODES:
        raise ValueError("unknown mode")
    readme = convo.readme_poisoned if variant == POISONED else convo.readme_clean
    document: dict[str, object] = {"role": "user", "content": readme}
    if mode == SHIELDED:
        document[KIND_KEY] = "file"
    return [{"role": "system", "content": SYSTEM_PROMPT}, document]


# --- records ------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvRecord:
    conv_id: str
    variant: str
    mode: str
    ok: bool
    error: str | None
    turns: int
    correct: int
    followed: int
    main_in: int
    main_out: int
    quarantine_in: int
    quarantine_out: int
    quarantine_calls: int
    unreported: int
    retry_tokens: int
    turn_tokens: tuple[int, ...]
    action: str | None
    seconds: float

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.conv_id, self.variant, self.mode)

    @property
    def total_tokens(self) -> int:
        return self.main_in + self.main_out + self.quarantine_in + self.quarantine_out

    @property
    def cost_tokens(self) -> int:
        """Everything the conversation cost, including quarantine attempts that were retried."""
        return self.total_tokens + self.retry_tokens

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> ConvRecord:
        data = json.loads(line)
        if not isinstance(data, dict):
            raise TypeError("not a record")
        if set(data) != _FIELD_NAMES:
            raise ValueError("unexpected fields")
        turn_tokens = data["turn_tokens"]
        if not isinstance(turn_tokens, list):
            raise TypeError("bad turn_tokens")
        data["turn_tokens"] = tuple(turn_tokens)
        return cls(**data)


_FIELD_NAMES = frozenset(field.name for field in fields(ConvRecord))


def _parse_line(line: str) -> ConvRecord | None:
    try:
        return ConvRecord.from_json(line)
    except (ValueError, TypeError):
        return None


def load_conv_records(path: Path) -> dict[tuple[str, str, str], ConvRecord]:
    """Latest record per conversation. Unreadable lines are ignored."""
    records: dict[tuple[str, str, str], ConvRecord] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        record = _parse_line(line) if line.strip() else None
        if record is not None:
            records[record.key] = record
    return records


def append_conv_record(path: Path, record: ConvRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(record.to_json() + "\n")


# --- running ------------------------------------------------------------------------

Sleep = Callable[[float], None]
Clock = Callable[[], float]


def _tokens(completion: Completion) -> int:
    return (completion.input_tokens or 0) + (completion.output_tokens or 0)


def _shield_with_retries(
    quarantine: LLMClient,
    policy: GatePolicy,
    messages: Sequence[dict[str, object]],
    retries: Sequence[float],
    sleep: Sleep,
) -> tuple[ShieldOutcome | None, TokenMeter, int]:
    """Quarantine the document once, retrying when the cheap model itself fails."""
    wasted = 0
    attempts = 0
    while True:
        attempts += 1
        meter = TokenMeter()
        outcome = shield_messages(quarantine, policy, messages, meter)
        if not any(item.reason == LLM_ERROR for item in outcome.items):
            return outcome, meter, wasted
        wasted += meter.totals().total_tokens
        if attempts > len(retries):
            return None, meter, wasted
        sleep(retries[attempts - 1])


def _ask(
    main: ChatClient,
    messages: Sequence[dict[str, object]],
    meter: TokenMeter,
    label: str,
    retries: Sequence[float],
    sleep: Sleep,
) -> Completion:
    attempts = 0
    while True:
        attempts += 1
        try:
            completion = _call_main(main, messages, meter, label)
        except BifrostError as error:
            if str(error) not in RETRYABLE or attempts > len(retries):
                raise
            sleep(retries[attempts - 1])
        else:
            return completion


def run_conversation(
    convo: Conversation,
    variant: str,
    mode: str,
    quarantine: LLMClient,
    main: ChatClient,
    policy: GatePolicy,
    *,
    retries: Sequence[float] = DEFAULT_RETRIES,
    pace: float = 0.0,
    sleep: Sleep = time.sleep,
    clock: Clock = time.perf_counter,
) -> ConvRecord:
    base = initial_messages(convo, variant, mode)
    label = f"{convo.id}:{variant}:{mode}"
    started = clock()
    error: str | None = None
    action: str | None = None
    wasted = 0
    history: list[dict[str, object]] = []
    quarantine_meter = TokenMeter()

    if mode == SHIELDED:
        outcome, quarantine_meter, wasted = _shield_with_retries(
            quarantine, policy, base, retries, sleep
        )
        if outcome is None:
            error = QUARANTINE_ERROR
        else:
            history = [dict(message) for message in outcome.messages]
            file_items = [item for item in outcome.items if item.kind is SourceKind.FILE]
            action = file_items[0].action.value if file_items else None
            if pace > 0:
                sleep(pace)
    else:
        history = [dict(message) for message in base]

    main_meter = TokenMeter()
    turn_tokens: list[int] = []
    correct = 0
    followed = 0
    if error is None:
        for index, question in enumerate(convo.questions):
            history.append({"role": "user", "content": question + FORMAT_SUFFIX})
            try:
                completion = _ask(
                    main, history, main_meter, f"{label}:t{index + 1}", retries, sleep
                )
            except BifrostError as failure:
                error = str(failure)
                break
            turn_tokens.append(_tokens(completion))
            correct += int(grade(completion.text, convo.expected[index]))
            followed += int(followed_injection(completion.text))
            history.append({"role": "assistant", "content": completion.text})
            if pace > 0:
                sleep(pace)

    main_totals = main_meter.totals(MAIN)
    quarantine_totals = quarantine_meter.totals(QUARANTINE)
    return ConvRecord(
        conv_id=convo.id,
        variant=variant,
        mode=mode,
        ok=error is None,
        error=error,
        turns=len(turn_tokens),
        correct=correct,
        followed=followed,
        main_in=main_totals.input_tokens,
        main_out=main_totals.output_tokens,
        quarantine_in=quarantine_totals.input_tokens,
        quarantine_out=quarantine_totals.output_tokens,
        quarantine_calls=quarantine_totals.calls,
        unreported=main_totals.unreported_calls + quarantine_totals.unreported_calls,
        retry_tokens=wasted,
        turn_tokens=tuple(turn_tokens),
        action=action,
        seconds=round(clock() - started, 2),
    )


def run_multiturn(
    convos: Sequence[Conversation],
    quarantine: LLMClient,
    main: ChatClient,
    policy: GatePolicy,
    out_path: Path,
    *,
    modes: Sequence[str] = MODES,
    pace: float = 6.0,
    retries: Sequence[float] = DEFAULT_RETRIES,
    sleep: Sleep = time.sleep,
    on_record: Callable[[ConvRecord], None] | None = None,
) -> list[ConvRecord]:
    """Run every (conversation, variant, mode). Finished runs are skipped, failed ones retried."""
    if not set(modes) <= set(MODES):
        raise ValueError("unknown mode")
    if pace < 0:
        raise ValueError("pace must not be negative")
    existing = load_conv_records(out_path)
    results: list[ConvRecord] = []
    for convo in convos:
        for variant in VARIANTS:
            for mode in modes:
                previous = existing.get((convo.id, variant, mode))
                if previous is not None and previous.ok:
                    results.append(previous)
                    continue
                record = run_conversation(
                    convo, variant, mode, quarantine, main, policy,
                    retries=retries, pace=pace, sleep=sleep,
                )
                append_conv_record(out_path, record)
                existing[record.key] = record
                results.append(record)
                if on_record is not None:
                    on_record(record)
    return results


# --- summary ------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _per_turn(items: Sequence[ConvRecord]) -> list[float | None]:
    longest = max((len(item.turn_tokens) for item in items), default=0)
    return [
        _mean([float(item.turn_tokens[turn]) for item in items if len(item.turn_tokens) > turn])
        for turn in range(longest)
    ]


def _group_stats(items: Sequence[ConvRecord]) -> dict[str, Any]:
    done = [item for item in items if item.ok]
    turns = sum(item.turns for item in done)
    return {
        "runs": len(items),
        "ok": len(done),
        "accuracy": sum(item.correct for item in done) / turns if turns else None,
        "followed_rate": sum(item.followed for item in done) / turns if turns else None,
        "mean_cost": _mean([float(item.cost_tokens) for item in done]),
        "mean_main": _mean([float(item.main_in + item.main_out) for item in done]),
        "mean_quarantine": _mean(
            [float(item.quarantine_in + item.quarantine_out) for item in done]
        ),
        "per_turn": _per_turn(done),
    }


def _savings(groups: dict[str, Any], variant: str) -> float | None:
    base = groups[f"{BASELINE}/{variant}"]["mean_cost"]
    shield = groups[f"{SHIELDED}/{variant}"]["mean_cost"]
    if base is None or shield is None or base == 0:
        return None
    return 1 - shield / base


def _amplification(items: Sequence[ConvRecord], mode: str) -> dict[str, Any]:
    by_conv: dict[str, dict[str, ConvRecord]] = {}
    for item in items:
        if item.ok and item.mode == mode:
            by_conv.setdefault(item.conv_id, {})[item.variant] = item
    ratios: list[float] = []
    for pair in by_conv.values():
        clean = pair.get(CLEAN)
        poisoned = pair.get(POISONED)
        if clean is not None and poisoned is not None and clean.cost_tokens > 0:
            ratios.append(poisoned.cost_tokens / clean.cost_tokens)
    return {
        "pairs": len(ratios),
        "mean": _mean(ratios),
        "median": statistics.median(ratios) if ratios else None,
    }


def summarize_multiturn(records: Iterable[ConvRecord]) -> dict[str, Any]:
    items = list(records)
    groups = {
        f"{mode}/{variant}": _group_stats(
            [item for item in items if item.mode == mode and item.variant == variant]
        )
        for mode in MODES
        for variant in VARIANTS
    }
    return {
        "groups": groups,
        "savings": {variant: _savings(groups, variant) for variant in VARIANTS},
        "amplification": {mode: _amplification(items, mode) for mode in MODES},
        "failed_runs": sum(1 for item in items if not item.ok),
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def format_multiturn(summary: dict[str, Any]) -> str:
    header = f"{'group':<20}{'runs':>5}{'ok':>5}{'accuracy':>10}{'followed':>10}"
    header += f"{'mean tokens':>13}"
    lines = ["Multi-turn benchmark summary", "", header]
    for name, stats in summary["groups"].items():
        lines.append(
            f"{name:<20}{stats['runs']:>5}{stats['ok']:>5}"
            f"{_fmt(stats['accuracy']):>10}{_fmt(stats['followed_rate']):>10}"
            f"{_fmt(stats['mean_cost'], 0):>13}"
        )
    lines += ["", "tokens saved by the shield over a whole conversation (negative = used more):"]
    lines += [f"  {variant}: {_pct(summary['savings'][variant])}" for variant in (CLEAN, POISONED)]
    lines += ["", "token amplification under attack (poisoned cost / clean cost, paired):"]
    for mode in MODES:
        info = summary["amplification"][mode]
        lines.append(
            f"  {mode}: mean {_fmt(info['mean'])}x, median {_fmt(info['median'])}x "
            f"over {info['pairs']} pairs"
        )
    lines += ["", "mean main-model tokens per turn (the baseline grows as history accumulates):"]
    for name, stats in summary["groups"].items():
        turns = " ".join(_fmt(value, 0) for value in stats["per_turn"]) or "n/a"
        lines.append(f"  {name:<20}{turns}")
    lines += ["", f"failed runs: {summary['failed_runs']}"]
    return "\n".join(lines)