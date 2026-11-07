"""Live check: run one Tavily search and show only safe summary facts.

Needs TAVILY_API_KEY in .env. Never prints the key or any page text.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from tokenshield.bifrost_client import load_env_file
from tokenshield.tavily_client import TavilyClient, TavilyError


def main() -> int:
    env = load_env_file(Path(".env"))
    try:
        client = TavilyClient.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1
    try:
        response = client.search("latest stable release of Python", max_results=3)
    except TavilyError as error:
        print(f"Tavily call failed: {error}")
        return 1
    print(f"{len(response.results)} results, {response.dropped} dropped")
    for item in response.results:
        domain = urlparse(item.url).netloc
        print(f" - {domain} score={item.score} content_chars={len(item.content)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())