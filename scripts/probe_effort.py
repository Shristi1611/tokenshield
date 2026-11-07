"""Probe: does the gateway pass a reasoning-effort setting through, and does it cut tokens?

Sends the same extraction request three ways and prints only token counts. Needs the gateway
running and .env containing TOKENSHIELD_BIFROST_VK. Never prints keys or model text.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tokenshield.benchmark import PROJECTS, render_readme
from tokenshield.bifrost_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    BifrostError,
    _parse_completion,
    _urllib_transport,
    load_env_file,
)
from tokenshield.extractor import ExtractionFailed, _parse_json_object, build_system_prompt
from tokenshield.schema import README_FACTS, SchemaViolation

VARIANTS = (
    ("default", {}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("reasoning.effort=low", {"reasoning": {"effort": "low"}}),
)


def main() -> int:
    env = load_env_file(Path(".env"))
    key = env.get("TOKENSHIELD_BIFROST_VK", "")
    if not key:
        print("TOKENSHIELD_BIFROST_VK is missing from .env")
        return 1
    base = env.get("TOKENSHIELD_BIFROST_URL", DEFAULT_BASE_URL).rstrip("/")
    model = env.get("TOKENSHIELD_EXTRACTOR_MODEL", DEFAULT_MODEL)
    system = build_system_prompt(README_FACTS)
    user = f"<<<DATA probe0001>>>\n{render_readme(PROJECTS[0])}\n<<<END probe0001>>>"
    headers = {"Content-Type": "application/json", "x-bf-vk": key}
    print(f"model: {model}")
    for name, extra in VARIANTS:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            **extra,
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            status, text = _urllib_transport(f"{base}/v1/chat/completions", headers, body, 60.0)
        except OSError:
            print(f"{name:<24} network error")
            continue
        if not 200 <= status < 300:
            print(f"{name:<24} HTTP {status}")
            continue
        try:
            completion = _parse_completion(text)
        except BifrostError:
            print(f"{name:<24} unreadable response")
            continue
        try:
            README_FACTS.validate(_parse_json_object(completion.text), user)
            valid = "yes"
        except (ExtractionFailed, SchemaViolation):
            valid = "no"
        print(
            f"{name:<24} in={completion.input_tokens} out={completion.output_tokens} "
            f"valid_facts={valid}"
        )
        time.sleep(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())