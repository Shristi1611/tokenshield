"""Debug: run the quarantine extraction on a few padded READMEs and show facts or the reason.

Needs the gateway running. Prints only expected answers, extracted typed facts, fixed failure
reasons and token counts. Waits between calls to respect the free-tier per-minute limit.
"""

from __future__ import annotations

import time
from pathlib import Path

from tokenshield.bifrost_client import BifrostClient, load_env_file
from tokenshield.extractor import ExtractionFailed, extract
from tokenshield.multiturn import build_conversations
from tokenshield.schema import README_FACTS

PAD = 150
TARGETS = ((0, "poisoned"), (2, "clean"), (2, "poisoned"))


def main() -> int:
    env = load_env_file(Path(".env"))
    try:
        client = BifrostClient.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1
    convos = build_conversations(PAD)
    for position, (index, variant) in enumerate(TARGETS):
        if position:
            time.sleep(30)
        convo = convos[index]
        text = convo.readme_poisoned if variant == "poisoned" else convo.readme_clean
        print(f"--- {convo.id} {variant} ({len(text)} chars)")
        print("expected:", " | ".join(convo.expected))
        try:
            result = extract(client, README_FACTS, text)
        except ExtractionFailed as error:
            print(f"FAILED: {error.reason}, tokens in/out: {error.input_tokens} / {error.output_tokens}")
            continue
        print(result.facts.render())
        print(f"tokens in/out: {result.input_tokens} / {result.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())