"""Download real public READMEs for the real-document benchmark (raw.githubusercontent.com only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

CANDIDATES = (
    "psf/requests", "pallets/click", "pallets/flask", "Textualize/rich", "encode/httpx",
    "python-attrs/attrs", "Delgan/loguru", "pytest-dev/pytest", "tqdm/tqdm", "psf/black",
    "pydantic/pydantic", "httpie/cli", "expressjs/express", "chalk/chalk", "sindresorhus/execa",
    "mochajs/mocha", "fastify/fastify", "lodash/lodash", "date-fns/date-fns", "colinhacks/zod",
    "sindresorhus/got", "prettier/prettier", "spf13/cobra", "sirupsen/logrus", "stretchr/testify",
    "gorilla/mux", "go-chi/chi", "uber-go/zap", "labstack/echo", "gin-gonic/gin",
    "BurntSushi/ripgrep", "serde-rs/serde", "clap-rs/clap", "dtolnay/anyhow", "rayon-rs/rayon",
    "sharkdp/bat", "sharkdp/fd", "square/okhttp", "square/retrofit", "google/guava",
    "junit-team/junit5", "FasterXML/jackson-core",
)
BRANCHES = ("main", "master")
README_NAMES = ("README.md", "readme.md", "Readme.md")
TIMEOUT = 30.0


def _get(url: str) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": "tokenshield-eval"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read(2_000_000).decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _readme(repo: str) -> str | None:
    for branch in BRANCHES:
        for name in README_NAMES:
            text = _get(f"https://raw.githubusercontent.com/{repo}/{branch}/{name}")
            if text is not None:
                return text
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real READMEs for the benchmark.")
    parser.add_argument("--out", default="data/realdocs")
    parser.add_argument("--min-chars", type=int, default=2500)
    parser.add_argument("--max-chars", type=int, default=24000)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "readmes").mkdir(parents=True, exist_ok=True)
    fetched = []
    for repo in CANDIDATES:
        text = _readme(repo)
        if text is None:
            print(f"skip {repo}: no README found")
            continue
        if not args.min_chars <= len(text) <= args.max_chars:
            print(f"skip {repo}: {len(text)} chars outside range")
            continue
        slug = repo.replace("/", "__")
        (out / "readmes" / f"{slug}.md").write_text(text, encoding="utf-8", newline="")
        fetched.append(
            {"repo": repo, "slug": slug, "chars": len(text),
             "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        )
        print(f"ok   {repo}: {len(text)} chars")
    (out / "fetched.json").write_text(json.dumps(fetched, indent=2) + "\n", encoding="utf-8")
    print(f"\nfetched {len(fetched)} READMEs into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())