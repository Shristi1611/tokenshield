"""Layer 1 gate: decides what the main model may see of each piece of content.

Trusted kinds pass unchanged. Every other kind is untrusted: its text is reduced to
schema-validated typed facts by the extractor, or replaced by a fixed placeholder. Raw
untrusted text never leaves this module. A kind with no schema is blocked (deny by default).

The kind of each item must come from the gateway (message role, tool metadata), never from
the content itself. Content must not be able to choose its own trust level.

Known gap: tokens spent on extractions that fail are not reported yet.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from tokenshield.extractor import DEFAULT_MAX_CHARS, ExtractionFailed, LLMClient, extract
from tokenshield.schema import FactSchema, content_hash

NO_SCHEMA = "no_schema"
UNKNOWN_REASON = "unknown"
_REASON_RE = re.compile(r"[a-z][a-z_]{0,31}")


class SourceKind(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL_OUTPUT = "tool_output"
    WEB = "web"
    FILE = "file"
    SKILL = "skill"
    UNKNOWN = "unknown"


class Action(str, Enum):
    PASSED = "passed"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"


DEFAULT_TRUSTED = frozenset({SourceKind.USER, SourceKind.SYSTEM, SourceKind.ASSISTANT})


@dataclass(frozen=True)
class ContentItem:
    kind: SourceKind
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SourceKind):
            raise TypeError("kind must be a SourceKind")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True)
class GatePolicy:
    schemas: Mapping[SourceKind, FactSchema] = field(default_factory=dict)
    trusted: frozenset[SourceKind] = DEFAULT_TRUSTED
    max_chars: int = DEFAULT_MAX_CHARS

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")
        schemas = dict(self.schemas)
        trusted = frozenset(self.trusted)
        if not all(isinstance(kind, SourceKind) for kind in (*schemas, *trusted)):
            raise TypeError("schema keys and trusted entries must be SourceKind values")
        if trusted.intersection(schemas):
            raise ValueError("a kind cannot be both trusted and schema-mapped")
        object.__setattr__(self, "schemas", MappingProxyType(schemas))
        object.__setattr__(self, "trusted", trusted)


@dataclass(frozen=True)
class GateResult:
    action: Action
    kind: SourceKind
    output_text: str
    source_hash: str
    reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False


def _blocked(kind: SourceKind, digest: str, reason: str) -> GateResult:
    # Reasons come from fixed constants; the pattern check is defense in depth.
    safe = reason if _REASON_RE.fullmatch(reason) else UNKNOWN_REASON
    text = f"[TS-BLOCKED kind={kind.value} source={digest[:12]} reason={safe}]"
    return GateResult(Action.BLOCKED, kind, text, digest, reason=safe)


def apply_gate(client: LLMClient, policy: GatePolicy, item: ContentItem) -> GateResult:
    digest = content_hash(item.text)
    if item.kind in policy.trusted:
        return GateResult(Action.PASSED, item.kind, item.text, digest)
    schema = policy.schemas.get(item.kind)
    if schema is None:
        return _blocked(item.kind, digest, NO_SCHEMA)
    try:
        result = extract(client, schema, item.text, max_chars=policy.max_chars)
    except ExtractionFailed as exc:
        return _blocked(item.kind, digest, exc.reason)
    return GateResult(
        Action.QUARANTINED,
        item.kind,
        result.facts.render(),
        digest,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        truncated=result.truncated,
    )