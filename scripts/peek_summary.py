"""Print the summary the summarize-first baseline writes for ONE poisoned real document.

  python scripts/peek_summary.py ripgrep
  python scripts/peek_summary.py jackson

Shows what the main model would be given in `summarized` mode. The text is model output and may
contain the planted attack. Makes one call; prints no keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tokenshield.baselines import SUMMARY_SYSTEM, summary_block, wrap_document
from tokenshield.bifrost_client import BifrostClient, load_env_file
from tokenshield.realdocs import build_real_conversations, load_documents


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "ripgrep"
    convos = build_real_conversations(load_documents(Path("data/realdocs")))
    convo = next((item for item in convos if name in item.id), None)
    if convo is None:
        print("no such document")
        return 1
    client = BifrostClient.from_env(load_env_file(Path(".env")))
    client.json_mode = False
    completion = client.complete(SUMMARY_SYSTEM, wrap_document(convo.readme_poisoned))
    print(f"{convo.id} | injection: {convo.injection_id}")
    print(summary_block(completion.text))
    print(f"tokens in/out: {completion.input_tokens}/{completion.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())