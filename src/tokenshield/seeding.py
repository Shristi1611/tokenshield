"""Copy finished baseline conversations from an earlier results file into a new one.

The baseline agent never touches the shield or the quarantine model, so changes to those do not
affect it. Reusing its finished runs lets a re-run measure only the shielded side. Keep the old
file: it is the record of the first run.
"""

from __future__ import annotations

from pathlib import Path

from tokenshield.benchmark import BASELINE
from tokenshield.multiturn import append_conv_record, load_conv_records


def seed_baseline(source: Path, dest: Path) -> int:
    """Append finished baseline records from source that dest does not already have."""
    if source.resolve() == dest.resolve():
        raise ValueError("source and destination must differ")
    existing = load_conv_records(dest)
    copied = 0
    for record in load_conv_records(source).values():
        if record.mode != BASELINE or not record.ok:
            continue
        previous = existing.get(record.key)
        if previous is not None and previous.ok:
            continue
        append_conv_record(dest, record)
        copied += 1
    return copied