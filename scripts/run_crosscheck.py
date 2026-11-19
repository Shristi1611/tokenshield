"""Run the cross-check evaluation on real READMEs. Resumable; paced for free-tier limits.

  python scripts/run_crosscheck.py --split tuning --limit 3 --out results/crosscheck.jsonl
  python scripts/run_crosscheck.py --split tuning --out results/crosscheck.jsonl
  python scripts/run_crosscheck.py --split heldout --out results/crosscheck.jsonl
  python scripts/run_crosscheck.py --summary --out results/crosscheck.jsonl

Protocol: run the tuning set first and adjust the design if you must. Then stop changing the code and
run the held-out set once. The held-out run is refused unless the tuning set has been run with the
same design (see the design id in the summary), so a held-out result cannot come from untested code.
Only quarantine calls are made (no main model). Prints no keys or document text.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokenshield.bifrost_client import BifrostClient, load_env_file
from tokenshield.crosscheck_eval import (
    DEFAULT_MAX_CHARS,
    DESIGN_ID,
    HELDOUT,
    METHODS,
    TUNING,
    EvalRecord,
    format_eval,
    load_eval_docs,
    load_eval_records,
    run_eval,
    summarize_eval,
)


def show(record: EvalRecord) -> None:
    status = "ok" if record.ok else f"FAILED ({record.error})"
    print(
        f"{record.doc:<28} {record.variant:<9} {record.method:<13} {record.position:<7} "
        f"{status:<22} license={record.license} flags={','.join(record.flags) or '-'} "
        f"tokens={record.tokens}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the cross-check evaluation.")
    parser.add_argument("--split", choices=(TUNING, HELDOUT), default=TUNING)
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated methods")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--pace", type=float, default=8.0, help="seconds to wait after each call")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--out", default="results/crosscheck.jsonl", help="results file")
    parser.add_argument("--summary", action="store_true", help="only print the saved summary")
    parser.add_argument("--force", action="store_true", help="run held-out without a tuning run")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.summary:
        print(format_eval(summarize_eval(load_eval_records(out_path).values())))
        return 0

    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    try:
        docs = load_eval_docs(args.split, max_chars=args.max_chars)
    except (ValueError, OSError) as error:
        print(f"Bad arguments: {error}")
        return 1
    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be at least 1")
            return 1
        docs = docs[: args.limit]
    if not docs:
        print("No documents found. Did you run the fetch scripts?")
        return 1

    if args.split == HELDOUT and not args.force:
        tuned = [
            record
            for record in load_eval_records(out_path).values()
            if record.split == TUNING and record.design == DESIGN_ID and record.ok
        ]
        if not tuned:
            print(
                f"Refusing: no tuning run with design {DESIGN_ID} in {out_path}. Run the tuning "
                "set with the current code first, or pass --force."
            )
            return 1

    env = load_env_file(Path(".env"))
    try:
        client = BifrostClient.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1

    print(
        f"Running {len(docs)} {args.split} documents, methods {', '.join(methods)}, "
        f"design {DESIGN_ID}, out {out_path}"
    )
    try:
        run_eval(
            docs, client, out_path, methods=methods, pace=args.pace, on_record=show,
        )
    except KeyboardInterrupt:
        print("Interrupted. Finished runs are saved; rerun the same command to resume.")
    except ValueError as error:
        print(f"Bad arguments: {error}")
        return 1

    print()
    print(format_eval(summarize_eval(load_eval_records(out_path).values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
