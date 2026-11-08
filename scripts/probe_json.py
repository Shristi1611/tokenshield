"""Probe: do the hardened prompt, the version hint and JSON mode fix the two failures?

Runs the extraction on two long poisoned READMEs, with JSON mode off and then on. Prints only the
expected answers, extracted typed facts, fixed failure reasons and token counts. Waits between
calls to respect the free-tier per-minute limit (about 2 minutes in total).
"""

from __future__ import annotations

import time
from pathlib import Path

from tokenshield.bifrost_client import BifrostClient, load_env_file
from tokenshield.extractor import ExtractionFailed, extract
from tokenshield.multiturn import build_conversations
from tokenshield.schema import README_FACTS

PAD = 150
TARGETS = ((2, "poisoned"), (0, "poisoned"))


def main() -> int:
    env = load_env_file(Path(".env"))
    try:
        client = BifrostClient.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1
    convos = build_conversations(PAD)
    first = True
    for index, variant in TARGETS:
        convo = convos[index]
        text = convo.readme_poisoned if variant == "poisoned" else convo.readme_clean
        for json_mode in (False, True):
            if not first:
                time.sleep(40)
            first = False
            client.json_mode = json_mode
            print(f"--- {convo.id} {variant}, json_mode={'on' if json_mode else 'off'}")
            print("expected:", " | ".join(convo.expected))
            try:
                result = extract(client, README_FACTS, text)
            except ExtractionFailed as error:
                tokens = f"{error.input_tokens} / {error.output_tokens}"
                print(f"FAILED: {error.reason}, tokens in/out: {tokens}")
                continue
            print(result.facts.render())
            print(f"tokens in/out: {result.input_tokens} / {result.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())