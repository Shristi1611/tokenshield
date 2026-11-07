"""Run the benchmark against the live gateway. Resumable and paced for Groq free-tier limits.

  python scripts/run_benchmark.py --limit 2     (smoke run: 2 tasks, 8 runs)
  python scripts/run_benchmark.py               (all 10 tasks, 40 runs)
  python scripts/run_benchmark.py --summary     (print the saved summary without running)

Results append to results/benchmark.jsonl. Finished runs are skipped when you rerun, and failed
runs are retried. Nothing printed here contains keys or page text.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokenshield.benchmark import (
    DEFAULT_MAIN_MODEL,
    RunRecord,
    build_tasks,
    default_policy,
    format_summary,
    load_records,
    run_benchmark,
    summarize,
)
from tokenshield.bifrost_client import DEFAULT_BASE_URL, BifrostClient, load_env_file
from tokenshield.chat import BifrostChat


def show(record: RunRecord) -> None:
    status = "ok" if record.ok else f"FAILED ({record.error})"
    print(
        f"{record.task_id:<18} {record.variant:<9} {record.mode:<9} {status:<22} "
        f"correct={record.correct} followed={record.followed} tokens={record.cost_tokens}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the tokenshield benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument("--pace", type=float, default=8.0, help="seconds to wait after each run")
    parser.add_argument("--out", default="results/benchmark.jsonl", help="results file")
    parser.add_argument("--modes", default="baseline,shielded", help="comma-separated modes")
    parser.add_argument("--summary", action="store_true", help="only print the saved summary")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.summary:
        print(format_summary(summarize(load_records(out_path).values())))
        return 0

    tasks = build_tasks()
    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be at least 1")
            return 1
        tasks = tasks[: args.limit]
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())

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

    print(f"Running {len(tasks)} tasks, modes {', '.join(modes)}, results in {out_path}")
    try:
        run_benchmark(
            tasks, quarantine, chat, default_policy(), out_path,
            modes=modes, pace=args.pace, on_record=show,
        )
    except KeyboardInterrupt:
        print("Interrupted. Finished runs are saved; rerun the same command to resume.")
    except ValueError as error:
        print(f"Bad arguments: {error}")
        return 1

    print()
    print(format_summary(summarize(load_records(out_path).values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())