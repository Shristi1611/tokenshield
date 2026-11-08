"""Typed-fact schema: the only form in which untrusted content may reach the main model.

Free text is the channel prompt injection travels through. A typed fact has no free text:
every value is a bool, a bounded number, a choice from a fixed list, or a short token drawn
from a restricted character set (no spaces, no newlines). Anything else is rejected.

Limits: a typed fact can still be wrong (a lying page can claim license=mit), but it cannot
carry an instruction. Keep schemas small, because every field is information the main model
is allowed to see. A field's hint is trusted guidance for the quarantined model: it is written
by us, never taken from content, and limited to a safe character set.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

Value = bool | int | float | str

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9._:/@+\-]{1,64}")
_HINT_RE = re.compile(r"[A-Za-z0-9 .,:;_()/@+\-]{0,200}")


class SchemaViolation(ValueError):
    """Extracted data does not match its schema. Messages never echo untrusted values."""


class FieldType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    ENUM = "enum"
    TOKEN = "token"


def content_hash(text: str) -> str:
    """Stable identity of a piece of content (used for provenance and, later, Layer 3)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: FieldType
    required: bool = True
    choices: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    hint: str = ""

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"invalid field name: {self.name!r}")
        if _HINT_RE.fullmatch(self.hint) is None:
            raise ValueError(f"{self.name}: unsafe hint")
        if self.type is FieldType.ENUM:
            if not self.choices:
                raise ValueError(f"{self.name}: enum needs choices")
            for choice in self.choices:
                if not _TOKEN_RE.fullmatch(choice):
                    raise ValueError(f"{self.name}: unsafe enum choice {choice!r}")
        elif self.choices:
            raise ValueError(f"{self.name}: choices only allowed for enum")
        is_number = self.type in (FieldType.INT, FieldType.FLOAT)
        if (self.min_value is not None or self.max_value is not None) and not is_number:
            raise ValueError(f"{self.name}: bounds only allowed for int or float")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(f"{self.name}: min_value > max_value")

    def _in_range(self, value: float) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        return self.max_value is None or value <= self.max_value

    def validate(self, value: object) -> Value:
        kind = self.type
        if kind is FieldType.BOOL:
            if isinstance(value, bool):
                return value
        elif kind is FieldType.INT:
            if isinstance(value, int) and not isinstance(value, bool) and self._in_range(value):
                return value
        elif kind is FieldType.FLOAT:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    number = float(value)
                except OverflowError:
                    number = math.inf
                if math.isfinite(number) and self._in_range(number):
                    return number
        elif kind is FieldType.ENUM:
            if isinstance(value, str) and value in self.choices:
                return value
        elif kind is FieldType.TOKEN and isinstance(value, str) and _TOKEN_RE.fullmatch(value):
            return value
        raise SchemaViolation(f"{self.name}: value rejected (expected {kind.value})")


def _format(value: Value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class TypedFacts:
    schema: str
    source_hash: str
    values: tuple[tuple[str, Value], ...]

    def render(self) -> str:
        """One fixed, injection-safe block for the main model's context."""
        lines = [f"[TS-FACTS schema={self.schema} source={self.source_hash[:12]}]"]
        lines.extend(f"{key}={_format(value)}" for key, value in self.values)
        lines.append("[/TS-FACTS]")
        return "\n".join(lines)


@dataclass(frozen=True)
class FactSchema:
    name: str
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"invalid schema name: {self.name!r}")
        if not self.fields:
            raise ValueError("schema needs at least one field")
        names = [spec.name for spec in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field names")

    def validate(self, raw: object, source_text: str) -> TypedFacts:
        if not isinstance(raw, Mapping):
            raise SchemaViolation("extraction result is not a mapping")
        known = {spec.name for spec in self.fields}
        extra = [key for key in raw if key not in known]
        if extra:
            raise SchemaViolation(f"{len(extra)} unexpected field(s)")
        values: list[tuple[str, Value]] = []
        for spec in self.fields:
            if spec.name not in raw:
                if spec.required:
                    raise SchemaViolation(f"{spec.name}: missing required field")
                continue
            values.append((spec.name, spec.validate(raw[spec.name])))
        return TypedFacts(self.name, content_hash(source_text), tuple(values))


# Example schema: what a coding agent may learn from an untrusted README.
README_FACTS = FactSchema(
    name="readme_facts",
    fields=(
        FieldSpec("has_install_steps", FieldType.BOOL),
        FieldSpec("has_test_instructions", FieldType.BOOL),
        FieldSpec(
            "license",
            FieldType.ENUM,
            choices=("mit", "apache-2.0", "gpl-3.0", "bsd-3-clause", "other", "none"),
        ),
        FieldSpec(
            "primary_language",
            FieldType.ENUM,
            choices=("python", "javascript", "typescript", "go", "rust", "java", "other"),
        ),
        FieldSpec("num_code_blocks", FieldType.INT, min_value=0, max_value=10000),
        FieldSpec(
            "declared_version",
            FieldType.TOKEN,
            required=False,
            hint="Always include it when the text states a current release or version number.",
        ),
    ),
)