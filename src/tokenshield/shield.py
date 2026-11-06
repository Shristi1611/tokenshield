"""Conversation-level shield: applies the Layer 1 gate to every message of a chat request.

Trust is decided per message. system, user and assistant messages are trusted by default and
tool messages are untrusted. A message may carry a `ts_kind` field, set by the calling harness,
to LOWER its trust (for example a file pasted into a user message). A `ts_kind` can never raise
trust: a tool message that claims to be a user message stays untrusted, and an unreadable
`ts_kind` is treated as unknown. Untrusted content leaves this module only as typed facts or as
a fixed placeholder, never as raw text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tokenshield.extractor import LLMClient
from tokenshield.gate import (
    NO_SCHEMA,
    Action,
    ContentItem,
    GatePolicy,
    SourceKind,
    apply_gate,
)
from tokenshield.meter import QUARANTINE, TokenMeter
from tokenshield.schema import content_hash

KIND_KEY = "ts_kind"
UNSUPPORTED_CONTENT = "unsupported_content"

_ROLE_KINDS = {
    "system": SourceKind.SYSTEM,
    "user": SourceKind.USER,
    "assistant": SourceKind.ASSISTANT,
}
_KNOWN_ROLES = frozenset({*_ROLE_KINDS, "tool"})


@dataclass(frozen=True)
class ItemReport:
    index: int
    role: str
    kind: SourceKind
    action: Action
    reason: str | None
    source_hash: str | None
    input_tokens: int | None
    output_tokens: int | None
    chars_in: int
    chars_out: int
    truncated: bool


@dataclass(frozen=True)
class ShieldOutcome:
    messages: tuple[dict[str, object], ...]
    items: tuple[ItemReport, ...]

    @property
    def untrusted_chars_in(self) -> int:
        return sum(item.chars_in for item in self.items if item.action is not Action.PASSED)

    @property
    def untrusted_chars_out(self) -> int:
        return sum(item.chars_out for item in self.items if item.action is not Action.PASSED)

    def count(self, action: Action) -> int:
        return sum(1 for item in self.items if item.action is action)


def _parse_kind(value: object) -> SourceKind | None:
    if isinstance(value, SourceKind):
        return value
    if isinstance(value, str):
        try:
            return SourceKind(value)
        except ValueError:
            return None
    return None


def _resolve_kind(message: Mapping[str, object], policy: GatePolicy) -> SourceKind:
    role = message.get("role")
    if not isinstance(role, str):
        return SourceKind.UNKNOWN
    if role in _ROLE_KINDS:
        base = _ROLE_KINDS[role]
    elif role == "tool":
        base = SourceKind.TOOL_OUTPUT
    else:
        return SourceKind.UNKNOWN
    if KIND_KEY not in message:
        return base
    declared = _parse_kind(message[KIND_KEY])
    if declared is None:
        return SourceKind.UNKNOWN
    if declared in policy.trusted:
        return base  # a claim of trust is ignored; only the role decides
    return declared


def _text_of(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                parts.append(part["text"])
            else:
                return None
        return "\n".join(parts)
    return None


def _unsupported(kind: SourceKind, content: object) -> tuple[str, str]:
    try:
        raw = json.dumps(content, sort_keys=True, default=str)
    except (TypeError, ValueError, RecursionError):
        raw = "unhashable"
    digest = content_hash(raw)
    text = f"[TS-BLOCKED kind={kind.value} source={digest[:12]} reason={UNSUPPORTED_CONTENT}]"
    return text, digest


def shield_messages(
    client: LLMClient,
    policy: GatePolicy,
    messages: Sequence[Mapping[str, object]],
    meter: TokenMeter | None = None,
) -> ShieldOutcome:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a list of messages")
    shielded: list[dict[str, object]] = []
    reports: list[ItemReport] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError("each message must be a mapping")
        kind = _resolve_kind(message, policy)
        raw_role = message.get("role")
        role = raw_role if isinstance(raw_role, str) and raw_role in _KNOWN_ROLES else "other"
        base = {key: value for key, value in message.items() if key not in (KIND_KEY, "content")}
        content = message.get("content")
        text = _text_of(content)
        chars_in = len(text) if text is not None else 0

        if kind in policy.trusted:
            passed = dict(base)
            if "content" in message:
                passed["content"] = content
            shielded.append(passed)
            reports.append(
                ItemReport(
                    index=index,
                    role=role,
                    kind=kind,
                    action=Action.PASSED,
                    reason=None,
                    source_hash=None,
                    input_tokens=None,
                    output_tokens=None,
                    chars_in=chars_in,
                    chars_out=chars_in,
                    truncated=False,
                )
            )
            continue

        if text is None:
            placeholder, digest = _unsupported(kind, content)
            shielded.append({**base, "content": placeholder})
            reports.append(
                ItemReport(
                    index=index,
                    role=role,
                    kind=kind,
                    action=Action.BLOCKED,
                    reason=UNSUPPORTED_CONTENT,
                    source_hash=digest,
                    input_tokens=None,
                    output_tokens=None,
                    chars_in=0,
                    chars_out=len(placeholder),
                    truncated=False,
                )
            )
            continue

        result = apply_gate(client, policy, ContentItem(kind, text))
        shielded.append({**base, "content": result.output_text})
        if meter is not None and result.reason != NO_SCHEMA:
            meter.record(
                f"{kind.value}:{result.source_hash[:12]}",
                QUARANTINE,
                result.action is Action.QUARANTINED,
                result.input_tokens,
                result.output_tokens,
            )
        reports.append(
            ItemReport(
                index=index,
                role=role,
                kind=kind,
                action=result.action,
                reason=result.reason,
                source_hash=result.source_hash,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                chars_in=chars_in,
                chars_out=len(result.output_text),
                truncated=result.truncated,
            )
        )
    return ShieldOutcome(tuple(shielded), tuple(reports))