"""Copy finished baseline conversations from an earlier results file into a new one.

The baseline agent never touches the shield or the quarantine model, so changes to those do not
affect it. Reusing its finished runs lets a re-run measure only the shielded side. Keep the old
file: it is the record of the first run. Deterministic failures (see FINAL_ERRORS) count as
finished, so an expensive run that cannot succeed is not repeated.
"""

from __future__ import annotations

from pathlib import Path

from tokenshield.benchmark import BASELINE
from tokenshield.multiturn import FINAL_ERRORS, ConvRecord, append_conv_record, load_conv_records


def _final(record: ConvRecord) -> bool:
    return record.ok or record.error in FINAL_ERRORS


def seed_baseline(source: Path, dest: Path) -> int:
    """Append finished baseline records from source that dest does not already have."""
    if source.resolve() == dest.resolve():
        raise ValueError("source and destination must differ")
    existing = load_conv_records(dest)
    copied = 0
    for record in load_conv_records(source).values():
        if record.mode != BASELINE or not _final(record):
            continue
        previous = existing.get(record.key)
        if previous is not None and _final(previous):
            continue
        append_conv_record(dest, record)
        copied += 1
    return copied