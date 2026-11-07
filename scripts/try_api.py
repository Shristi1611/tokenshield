"""Call the running local API once and print the headline numbers.

Start scripts/run_api.py first. This makes live model calls, so wait until any benchmark
run has finished.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from tokenshield.benchmark import build_tasks

BASE = "http://127.0.0.1:8765"


def post(path: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main() -> int:
    task = build_tasks()[0]
    try:
        result = post(
            "/api/compare", {"question": task.question, "document": task.readme_poisoned}
        )
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}")
        return 1
    except urllib.error.URLError:
        print("Could not reach the API. Is scripts/run_api.py running?")
        return 1
    baseline = result["baseline"]
    shielded = result["shielded"]
    print(f"baseline total tokens: {baseline['total_tokens']}")
    print(f"shielded total tokens: {shielded['total_tokens']} (action: {shielded['action']})")
    print(f"saved tokens: {result['saved_tokens']}, fraction: {result['saved_fraction']}")
    print("what the main model saw instead of the README:")
    print(shielded["view"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())