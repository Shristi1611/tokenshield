"""Token meter: records every model call, including failed ones, so no cost is hidden.

Totals are lower bounds when a provider omits token counts; `unreported_calls` says how many.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

QUARANTINE = "quarantine"
MAIN = "main"
_ROLES = frozenset({QUARANTINE, MAIN})


def _clean(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


@dataclass(frozen=True)
class CallRecord:
    label: str
    role: str
    ok: bool
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class MeterTotals:
    calls: int
    failed_calls: int
    unreported_calls: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []

    def record(
        self,
        label: str,
        role: str,
        ok: bool,
        input_tokens: object = None,
        output_tokens: object = None,
    ) -> None:
        if role not in _ROLES:
            raise ValueError("role must be 'quarantine' or 'main'")
        entry = CallRecord(label, role, bool(ok), _clean(input_tokens), _clean(output_tokens))
        with self._lock:
            self._records.append(entry)

    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def totals(self, role: str | None = None) -> MeterTotals:
        if role is not None and role not in _ROLES:
            raise ValueError("role must be 'quarantine' or 'main'")
        selected = [rec for rec in self.records() if role is None or rec.role == role]
        return MeterTotals(
            calls=len(selected),
            failed_calls=sum(1 for rec in selected if not rec.ok),
            unreported_calls=sum(
                1 for rec in selected if rec.input_tokens is None or rec.output_tokens is None
            ),
            input_tokens=sum(rec.input_tokens or 0 for rec in selected),
            output_tokens=sum(rec.output_tokens or 0 for rec in selected),
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()