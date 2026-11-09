"""List imperfect shielded runs from saved results files. Offline: no live calls, no page text.

  python scripts/list_blocked.py results/logbench.jsonl results/multiturn_pad150_v2.jsonl

For each shielded conversation that missed any answer or whose document was not turned into
facts, print the extraction action and the token counts. A blocked action with many quarantine
output tokens means a poisoned document pushed the quarantine model into prose and the extraction
failed closed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokenshield.benchmark import SHIELDED
from tokenshield.multiturn import ConvRecord, load_conv_records


def describe(record: ConvRecord) -> str:
    status = "ok" if record.ok else f"FAILED ({record.error})"
    parts = [
        f"{record.conv_id:<20}",
        f"{record.variant:<9}",
        f"{status:<20}",
        f"action={record.action}",
        f"correct={record.correct}/{record.turns}",
        f"quarantine_in/out={record.quarantine_in}/{record.quarantine_out}",
        f"total={record.cost_tokens}",
    ]
    return "  " + " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List imperfect shielded runs.")
    parser.add_argument("files", nargs="+", help="results files (.jsonl)")
    args = parser.parse_args(argv)
    missing = False
    for name in args.files:
        path = Path(name)
        if not path.exists():
            print(f"missing: {path}")
            missing = True
            continue
        print(f"== {path.name} ==")
        rows = [
            record
            for record in load_conv_records(path).values()
            if record.mode == SHIELDED
            and (not record.ok or record.action != "quarantined" or record.correct < record.turns)
        ]
        if not rows:
            print("  none")
            continue
        for record in sorted(rows, key=lambda item: (item.conv_id, item.variant)):
            print(describe(record))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())