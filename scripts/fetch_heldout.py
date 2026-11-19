"""Download a SECOND set of real public READMEs, never used for tuning (the held-out set).

  python scripts/fetch_heldout.py

Only raw.githubusercontent.com is used. Files go to data/heldout/readmes. Each file is checked
against the SHA-256 recorded in data/heldout/labels.json when that file exists, and a changed
README is reported (the label may no longer match). Nothing printed contains document text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

CANDIDATES = (
    "django/django", "scrapy/scrapy", "sqlalchemy/sqlalchemy", "numpy/numpy",
    "scikit-learn/scikit-learn", "pallets/jinja", "pallets/werkzeug", "tiangolo/fastapi",
    "encode/starlette", "encode/uvicorn", "aio-libs/aiohttp", "python-pillow/Pillow",
    "pypa/pipenv", "pyenv/pyenv", "ansible/ansible", "ytdl-org/youtube-dl", "Textualize/textual",
    "more-itertools/more-itertools", "joblib/joblib", "paramiko/paramiko", "mitmproxy/mitmproxy",
    "getpelican/pelican", "mkdocs/mkdocs", "sphinx-doc/sphinx", "PyCQA/flake8", "PyCQA/pylint",
    "astral-sh/ruff", "astral-sh/uv", "hynek/structlog", "facebook/react", "vuejs/core",
    "sveltejs/svelte", "webpack/webpack", "rollup/rollup", "vitejs/vite", "eslint/eslint",
    "jestjs/jest", "axios/axios", "socketio/socket.io", "moment/moment", "jquery/jquery",
    "d3/d3", "chartjs/Chart.js", "ReactiveX/rxjs", "yargs/yargs", "tj/commander.js",
    "debug-js/debug", "winstonjs/winston", "pinojs/pino", "uuidjs/uuid", "typicode/json-server",
    "hashicorp/terraform", "prometheus/prometheus", "gohugoio/hugo", "caddyserver/caddy",
    "etcd-io/etcd", "junegunn/fzf", "cli/cli", "spf13/viper", "urfave/cli", "rs/zerolog",
    "jmoiron/sqlx", "go-gorm/gorm", "tokio-rs/tokio", "sharkdp/hyperfine", "starship/starship",
    "alacritty/alacritty", "bevyengine/bevy", "rust-lang/rustfmt", "rust-lang/mdBook",
    "spring-projects/spring-boot", "apache/commons-lang", "mockito/mockito", "ReactiveX/RxJava",
    "google/gson", "google/guice", "redis/redis", "curl/curl", "nlohmann/json", "fmtlib/fmt",
    "jqlang/jq", "BurntSushi/xsv", "ajeetdsouza/zoxide", "mozilla/pdf.js", "microsoft/TypeScript",
    "prettier/prettier", "psf/pyperclip", "pypa/pip", "pytest-dev/pluggy", "tox-dev/tox",
    "benbjohnson/litestream", "sirupsen/logrus", "uber-go/fx", "grpc/grpc-go",
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
    parser = argparse.ArgumentParser(description="Fetch the held-out READMEs.")
    parser.add_argument("--out", default="data/heldout")
    parser.add_argument("--min-chars", type=int, default=2500)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--only-labelled", action="store_true", help="fetch only labelled repos")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "readmes").mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.json"
    labelled: dict[str, dict] = {}
    if labels_path.exists():
        labelled = json.loads(labels_path.read_text(encoding="utf-8")).get("labels", {})
    fetched = []
    for repo in CANDIDATES:
        slug = repo.replace("/", "__")
        if args.only_labelled and slug not in labelled:
            continue
        text = _readme(repo)
        if text is None:
            print(f"skip {repo}: no README found")
            continue
        if not args.min_chars <= len(text) <= args.max_chars:
            print(f"skip {repo}: {len(text)} chars outside range")
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        note = ""
        pinned = labelled.get(slug, {}).get("sha256")
        if pinned and pinned != digest:
            note = "  CHANGED since it was labelled; check its license label"
        (out / "readmes" / f"{slug}.md").write_text(text, encoding="utf-8", newline="")
        fetched.append({"repo": repo, "slug": slug, "chars": len(text), "sha256": digest})
        print(f"ok   {repo}: {len(text)} chars{note}")
    (out / "fetched.json").write_text(json.dumps(fetched, indent=2) + "\n", encoding="utf-8")
    print(f"\nfetched {len(fetched)} READMEs into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
