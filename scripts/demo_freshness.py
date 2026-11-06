"""Live check of the freshness pipeline: real cheap model, real Tavily search.

Needs the gateway running and .env containing TOKENSHIELD_BIFROST_VK and TAVILY_API_KEY.
Never prints keys or page text. The answer below deliberately contains two stale claims.
Uses 2 results per claim and at most 2 checked claims, to stay inside Groq free-tier limits.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from tokenshield.bifrost_client import BifrostClient, load_env_file
from tokenshield.freshness import FreshnessError, check_answer
from tokenshield.meter import TokenMeter
from tokenshield.tavily_client import TavilyClient

ANSWER = (
    "The latest stable version of Python is 3.9. "
    "The latest stable version of Node.js is 14. "
    "Python was created by Guido van Rossum."
)


def main() -> int:
    env = load_env_file(Path(".env"))
    try:
        llm = BifrostClient.from_env(env)
        tavily = TavilyClient.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1

    meter = TokenMeter()
    try:
        report = check_answer(llm, llm, tavily, ANSWER, meter, max_checked=2, results_per_claim=2)
    except FreshnessError as error:
        print(f"Claim extraction failed closed: {error.reason}")
        return 1

    print(f"{len(report.claims)} claims found, {report.searches} Tavily searches")
    for item in report.claims:
        domains = sorted({urlparse(url).netloc for url in item.sources})
        score = "n/a" if item.score is None else f"{item.score:.2f}"
        print(f"- [{item.claim.category.value}] {item.claim.subject} = {item.claim.value}")
        print(f"    risk {item.risk:.1f} | {item.verdict.value} | freshness {score} | {item.note}")
        print(f"    sources: {', '.join(domains) if domains else 'none'}")
    totals = meter.totals()
    print(
        f"cheap-model calls: {totals.calls} ({totals.failed_calls} failed), "
        f"tokens in/out: {totals.input_tokens} / {totals.output_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())