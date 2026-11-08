"""Run the multi-turn benchmark against the live gateway. Resumable and paced for free-tier limits.

  python scripts/run_multiturn.py --limit 1                                  (smoke run)
  python scripts/run_multiturn.py --limit 3 --out results/multiturn_pad0.jsonl
  python scripts/run_multiturn.py --limit 3 --pad 150 --pace 25 --out results/multiturn_pad150.jsonl
  python scripts/run_multiturn.py --summary --out results/multiturn_pad150.jsonl

--pad adds that many benign changelog lines to every README to make the documents longer.
--seed-baseline copies finished baseline runs from an earlier results file, so a re-run with
--modes shielded measures only the shielded side against the same baseline.
Finished conversations are skipped when you rerun, and failed ones are retried.
Nothing printed here contains keys or page text.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokenshield.benchmark import DEFAULT_MAIN_MODEL, default_policy
from tokenshield.bifrost_client import DEFAULT_BASE_URL, BifrostClient, load_env_file
from tokenshield.chat import BifrostChat
from tokenshield.multiturn import (
    ConvRecord,
    build_conversations,
    format_multiturn,
    load_conv_records,
    run_multiturn,
    summarize_multiturn,
)
from tokenshield.seeding import seed_baseline


def show(record: ConvRecord) -> None:
    status = "ok" if record.ok else f"FAILED ({record.error})"
    print(
        f"{record.conv_id:<22} {record.variant:<9} {record.mode:<9} {status:<22} "
        f"correct={record.correct}/{record.turns} followed={record.followed} "
        f"tokens={record.cost_tokens}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the multi-turn benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N READMEs")
    parser.add_argument("--pace", type=float, default=6.0, help="seconds to wait after each call")
    parser.add_argument("--pad", type=int, default=0, help="benign changelog lines to add")
    parser.add_argument("--out", default="results/multiturn.jsonl", help="results file")
    parser.add_argument("--modes", default="baseline,shielded", help="comma-separated modes")
    parser.add_argument("--seed-baseline", default=None, help="copy finished baseline runs from")
    parser.add_argument("--summary", action="store_true", help="only print the saved summary")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.summary:
        print(format_multiturn(summarize_multiturn(load_conv_records(out_path).values())))
        return 0

    try:
        convos = build_conversations(args.pad)
    except ValueError as error:
        print(f"Bad arguments: {error}")
        return 1
    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be at least 1")
            return 1
        convos = convos[: args.limit]
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())

    if args.seed_baseline:
        try:
            copied = seed_baseline(Path(args.seed_baseline), out_path)
        except ValueError as error:
            print(f"Bad arguments: {error}")
            return 1
        print(f"Copied {copied} finished baseline runs from {args.seed_baseline}")

    env = load_env_file(Path(".env"))
    try:
        quarantine = BifrostClient.from_env(env)
        chat = BifrostChat(
            env.get("TOKENSHIELD_BIFROST_URL", DEFAULT_BASE_URL),
            env.get("TOKENSHIELD_BIFROST_VK", ""),
            env.get("TOKENSHIELD_MAIN_MODEL", DEFAULT_MAIN_MODEL),
            max_tokens=2500,
        )
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1

    print(f"Running {len(convos)} READMEs (pad {args.pad}), modes {', '.join(modes)}, out {out_path}")
    try:
        run_multiturn(
            convos, quarantine, chat, default_policy(), out_path,
            modes=modes, pace=args.pace, on_record=show,
        )
    except KeyboardInterrupt:
        print("Interrupted. Finished conversations are saved; rerun the same command to resume.")
    except ValueError as error:
        print(f"Bad arguments: {error}")
        return 1

    print()
    print(format_multiturn(summarize_multiturn(load_conv_records(out_path).values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())