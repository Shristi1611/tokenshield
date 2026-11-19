"""Two-section cross-check: read the document in parts and flag a fact the parts disagree on.

The typed schema stops instructions, not lies. A planted "the license is GPL" sentence becomes
license=gpl-3.0 even when the README also says MIT further down. This module reads the document in
`sections` parts, each by its own quarantined call, then merges the typed answers in code:

- bool fields: true if any section says true.
- int fields: the sum over sections, capped at the field's maximum.
- enum fields: values that mean "not stated" (none, other) are ignored. One distinct value wins.
  Two or more distinct values on a checked field give the reserved value `conflict`; on any other
  field the first section's value wins.
- token fields: the same rule as enum fields, with no default value (an unstated field is left out).

`conflict` is a fixed word, not content. It tells the main model that the document contradicts
itself on that field, so it can say so instead of repeating one side as fact.

A failure in any section fails the whole check closed (ExtractionFailed), with the tokens spent so
far. Costs: about one extra system prompt per extra section, not a second pass over the document.

Limits, stated plainly: the check only sees a contradiction when the true statement and the false
one land in different sections. A false claim placed next to the true one is not caught, and a
false claim in a document that states nothing true about that field is not caught either. It
turns some confidently wrong answers into flagged ones; it cannot make a wrong answer right.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from tokenshield.extractor import DEFAULT_MAX_CHARS, ExtractionFailed, LLMClient, extract
from tokenshield.schema import FactSchema, FieldType, TypedFacts, Value, content_hash

CONFLICT = "conflict"
CHECK_FIELDS = ("license", "declared_version")
NOT_STATED = ("none", "other")
MAX_SECTIONS = 6


@dataclass(frozen=True)
class CrossCheckResult:
    facts: TypedFacts
    conflicts: tuple[str, ...]
    sections: int
    input_tokens: int | None
    output_tokens: int | None
    truncated: bool


def split_sections(text: str, count: int) -> list[str]:
    """Split at line ends into `count` parts of near-equal length. Joining them gives the text."""
    if not 1 <= count <= MAX_SECTIONS:
        raise ValueError("sections must be between 1 and 6")
    cuts = [0]
    for index in range(1, count):
        target = max(cuts[-1], len(text) * index // count)
        newline = text.find("\n", max(target - 1, 0))
        cut = len(text) if newline == -1 else newline + 1
        if cut > cuts[-1]:
            cuts.append(cut)
    cuts.append(len(text))
    parts = [text[start:end] for start, end in pairwise(cuts) if end > start]
    return parts or [text]


def _add(total: int | None, extra: int | None) -> int | None:
    if total is None and extra is None:
        return None
    return (total or 0) + (extra or 0)


def _distinct(readings: list[Value]) -> list[Value]:
    seen: list[Value] = []
    for value in readings:
        if value not in seen:
            seen.append(value)
    return seen


def _merge_field(
    spec_name: str,
    spec_type: FieldType,
    choices: tuple[str, ...],
    maximum: float | None,
    required: bool,
    readings: list[Value],
    checked: bool,
) -> tuple[Value | None, bool]:
    """Merged value (None = leave the field out) and whether the sections conflict."""
    if spec_type is FieldType.BOOL:
        return any(value is True for value in readings), False
    if spec_type is FieldType.INT:
        total = sum(int(value) for value in readings)
        return (min(total, int(maximum)) if maximum is not None else total), False
    stated = [value for value in readings if value not in NOT_STATED]
    distinct = _distinct(stated)
    if len(distinct) > 1:
        if checked:
            return CONFLICT, True
        return distinct[0], False
    if distinct:
        return distinct[0], False
    if not readings:
        return None, False
    if spec_type is FieldType.ENUM:
        for fallback in NOT_STATED:
            if fallback in choices:
                return fallback, False
    return (readings[0], False) if required else (None, False)


def extract_crosschecked(
    client: LLMClient,
    schema: FactSchema,
    untrusted_text: str,
    *,
    sections: int = 2,
    max_chars: int = DEFAULT_MAX_CHARS,
    check_fields: tuple[str, ...] = CHECK_FIELDS,
    context: str = "",
) -> CrossCheckResult:
    parts = split_sections(untrusted_text, sections)
    spent_in: int | None = None
    spent_out: int | None = None
    readings: list[dict[str, Value]] = []
    truncated = False
    for part in parts:
        try:
            result = extract(client, schema, part, max_chars=max_chars, context=context)
        except ExtractionFailed as failure:
            raise ExtractionFailed(
                failure.reason,
                _add(spent_in, failure.input_tokens),
                _add(spent_out, failure.output_tokens),
            ) from None
        spent_in = _add(spent_in, result.input_tokens)
        spent_out = _add(spent_out, result.output_tokens)
        truncated = truncated or result.truncated
        readings.append(dict(result.facts.values))

    merged: list[tuple[str, Value]] = []
    conflicts: list[str] = []
    for spec in schema.fields:
        values = [reading[spec.name] for reading in readings if spec.name in reading]
        value, conflict = _merge_field(
            spec.name, spec.type, spec.choices, spec.max_value, spec.required, values,
            spec.name in check_fields,
        )
        if conflict:
            conflicts.append(spec.name)
        if value is not None:
            merged.append((spec.name, value))
    facts = TypedFacts(schema.name, content_hash(untrusted_text), tuple(merged))
    return CrossCheckResult(
        facts=facts,
        conflicts=tuple(conflicts),
        sections=len(parts),
        input_tokens=spent_in,
        output_tokens=spent_out,
        truncated=truncated,
    )
