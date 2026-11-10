"""Debug: run one quarantine extraction on a chosen benchmark document and show the result.

  python scripts/debug_extract.py --suite logs --index 6 --variant poisoned
  python scripts/debug_extract.py --suite readme --index 4 --variant clean --pad 150
  python scripts/debug_extract.py --suite flip --index 1 --variant poisoned

Prints only the expected answers (in question order), the extracted typed facts or a fixed
failure reason, and token counts. Never prints keys or page text. --repeat N runs it N times with
a 30 second pause, to see how often a failure happens.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from tokenshield.bifrost_client import BifrostClient, load_env_file
from tokenshield.catalog import LOG_FACTS
from tokenshield.extractor import ExtractionFailed, extract
from tokenshield.flipbench import build_flip_conversations
from tokenshield.logbench import build_log_conversations
from tokenshield.multiturn import Conversation, build_conversations
from tokenshield.schema import README_FACTS, FactSchema


def pick(suite: str, pad: int) -> tuple[tuple[Conversation, ...], FactSchema]:
    if suite == "logs":
        return build_log_conversations(), LOG_FACTS
    if suite == "flip":
        return build_flip_conversations(pad), README_FACTS
    return build_conversations(pad), README_FACTS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one quarantine extraction for debugging.")
    parser.add_argument("--suite", choices=("readme", "logs", "flip"), default="readme")
    parser.add_argument("--index", type=int, default=1, help="1-based position of the document")
    parser.add_argument("--variant", choices=("clean", "poisoned"), default="clean")
    parser.add_argument("--pad", type=int, default=150, help="padding lines (readme and flip)")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to run it")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be at least 1")
        return 1
    try:
        convos, schema = pick(args.suite, args.pad)
    except ValueError as error:
        print(f"Bad arguments: {error}")
        return 1
    if not 1 <= args.index <= len(convos):
        print(f"--index must be from 1 to {len(convos)}")
        return 1
    convo = convos[args.index - 1]
    text = convo.readme_poisoned if args.variant == "poisoned" else convo.readme_clean

    env = load_env_file(Path(".env"))
    try:
        client = BifrostClient.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1

    print(f"--- {convo.id} {args.variant} ({len(text)} chars), max_tokens {client.max_tokens}")
    print("expected:", " | ".join(convo.expected))
    for attempt in range(args.repeat):
        if attempt:
            time.sleep(30)
        try:
            result = extract(client, schema, text)
        except ExtractionFailed as error:
            tokens = f"{error.input_tokens} / {error.output_tokens}"
            print(f"FAILED: {error.reason}, tokens in/out: {tokens}")
            continue
        print(result.facts.render())
        print(f"tokens in/out: {result.input_tokens} / {result.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())