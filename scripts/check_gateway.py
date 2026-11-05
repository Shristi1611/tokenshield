"""Smoke test: ask the local Bifrost gateway to list Groq models using the virtual key.

Reads the key from .env and never prints it. Prints only the HTTP status and a short summary.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://localhost:8080"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    env = load_env(Path(".env"))
    virtual_key = env.get("TOKENSHIELD_BIFROST_VK")
    if not virtual_key:
        print("TOKENSHIELD_BIFROST_VK is missing from .env")
        return 1

    request = urllib.request.Request(
        f"{BASE_URL}/v1/models?provider=groq",
        headers={"x-bf-vk": virtual_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}")
        print(error.read().decode("utf-8", errors="replace")[:300])
        return 1
    except urllib.error.URLError as error:
        print(f"Could not reach Bifrost: {error.reason}")
        return 1

    print(f"HTTP {status}")
    try:
        models = json.loads(body).get("data", [])
    except (ValueError, AttributeError):
        print("Response was not the expected JSON.")
        return 1
    print(f"{len(models)} models listed")
    for model in models[:15]:
        print(" -", model.get("id", "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())