"""Print the honest report for saved multi-turn results files. No live calls.

  python scripts/report.py results/multiturn_pad150_v2.jsonl results/logbench.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokenshield.multiturn import load_conv_records
from tokenshield.reporting import format_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report saved multi-turn results.")
    parser.add_argument("files", nargs="+", help="results files (.jsonl)")
    parser.add_argument("--seed", type=int, default=1, help="bootstrap seed")
    args = parser.parse_args(argv)
    missing = False
    for name in args.files:
        path = Path(name)
        if not path.exists():
            print(f"missing: {path}")
            missing = True
            continue
        print(format_report(path.name, load_conv_records(path).values(), seed=args.seed))
        print()
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())