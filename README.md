# TokenShield

**A trust boundary for LLM agents: typed facts in, token traps out.**

Agents read untrusted text: READMEs, build logs, web pages, search results. Any of it can tell the agent to do something else, to burn thousands of tokens doing it, or to state something false. TokenShield sits between that text and your main model. A cheap, quarantined model reads the untrusted text alone and may only return a small set of **typed, bounded fields**. The main model never sees the raw text, only a block of validated facts.

It runs as a **standalone, OpenAI-compatible gateway in front of [Bifrost](https://github.com/maximhq/bifrost)**. It is not a Bifrost plugin. A second pipeline uses **Tavily** to check whether an answer is still true.

Built for the Geek Room Global AI Hackathon (Bifrost track and Tavily track, one integrated project).

## What it does

| Layer | What it does | Status |
|---|---|---|
| 1. Typed-fact quarantine | Untrusted text goes to a quarantined model. Its JSON is checked against a strict schema (booleans, bounded numbers, enums, restricted tokens, no free text). Any error fails closed to a fixed `[TS-BLOCKED]` placeholder. | Built, benchmarked |
| 2. Conformal token budgets | Split-conformal threshold (alpha 0.1) learned from clean runs; runs above it are flagged. | Measured in the API; enforced per session in the gateway only |
| 3. Trap advisories | Shared warnings about known token traps. | Designed, not a headline result |
| Freshness check (Tavily) | Extracts time-sensitive claims from an answer, searches Tavily per claim, quarantines the results like any other untrusted text, and computes the verdict in code. | Works end to end; precision and recall **not measured** |

## Results

All results are **token counts** on the Groq free tier, on **templated documents** (READMEs and build logs generated from templates). Intervals are 95% bootstrap intervals over conversations: they show run-to-run spread, not real-world variance. No dollar savings are claimed.

**Long multi-turn conversations (10 READMEs x 5 turns, padded README):**

| | Plain agent | TokenShield |
|---|---|---|
| Mean tokens, clean | 9,881 | 4,028 |
| Mean tokens, attacked | 11,315 | 4,049 |
| Saved, clean | | +59.2% (58.9% to 59.5%) |
| Saved, attacked | | +64.2% (60.2% to 69.0%) |
| Turns hijacked under attack | 6 of 50 | 0 of 50 |
| Turns correct, attacked | 50 of 50 | 48 of 50 |

**Build logs (first 5 of 10 finished cleanly; the rest hit free-tier rate limits or a gateway outage and are counted as failed runs):** +57.3% clean (54.7% to 59.5%), +69.8% attacked (61.2% to 75.8%, 3 pairs). One plain-agent run hit an `http_413` after 24,252 tokens. One later log (l06) was blocked in the shielded poisoned run.

**Token amplification under attack** (attacked cost / clean cost): plain agent 1.08x to 2.28x depending on the set, shielded about 1.00x.

## Where it falls short

We report these on purpose.

- **Short tasks cost more.** On single-question tasks with short documents the shield used about 2.6x more tokens (the quarantine call is overhead). The savings come from long conversations, where a plain agent re-sends the document every turn.
- **It stops instructions, not lies.** On the answer-flip test (a planted false license or version, no instructions) the shield was *less* accurate under attack: 17 of 30 correct against 23 of 30 for the plain agent. A typed-fact quarantine cannot tell a planted false fact from a true one.
- **Denial of service.** If the quarantined model is hijacked into prose or an oversized reply, extraction fails closed and the main model sees only a placeholder. That is safe, but the user gets less.
- **Templated documents.** Real documents will vary more than our templates do.
- **Free-tier noise.** Rate limits cut some runs short. Failed runs are reported with their spend, not dropped.
- **Freshness is unmeasured.** There is no labelled precision/recall number for the Tavily check yet.
- **Layer 2 is enforced only in the gateway**, not in the API pipeline.

## How Bifrost and Tavily are used

- **Bifrost** runs in Docker on `localhost:8080`. Both the quarantine model and the main model are called through it with one virtual key (`x-bf-vk`). TokenShield's own gateway exposes `POST /v1/chat/completions` on `localhost:8766`, shields untrusted messages, forwards to Bifrost, and meters tokens per session (optional `X-Tokenshield-Session` header and budget). It refuses any request carrying an `Origin` header.
- **Tavily** supplies fresh evidence per claim. Search results are untrusted, so they pass through the same quarantine and typed schema as a README. At most 2 claims per answer and 2 results per claim are checked, to keep credits low.
- Whether a standalone layer in front of Bifrost counts for the Bifrost track's problem #4 has not been confirmed with the organizers.

## Run it

Requires Python 3.12, Docker, a Groq key configured in Bifrost, and a Tavily key.

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

:: Bifrost (settings and keys go in .env, which is gitignored)
docker run --name bifrost -p 8080:8080 --env-file .env -v "%cd%\bifrost-data:/app/data" maximhq/bifrost
:: later: docker start bifrost

python scripts\run_api.py        :: API on 127.0.0.1:8765
python scripts\run_gateway.py    :: gateway on 127.0.0.1:8766 (optional)

cd frontend
python -m http.server 3000       :: then open http://localhost:3000
```

The API only accepts pages served from `localhost:5173` or `localhost:3000`; for another port set `TOKENSHIELD_ALLOWED_ORIGINS`. Keys live only in `.env` and the Bifrost UI, never in the repo. The Tavily key is read as in `scripts\check_tavily.py`.

Tests and lint: `pytest -q` and `ruff check src tests scripts`.

Reproduce the benchmarks (each run spends free-tier tokens): `scripts\run_benchmark.py`, `run_multiturn.py`, `run_logbench.py`, `run_flipbench.py`, then `scripts\report.py results\<file>.jsonl`. Raw results are in `results/` and are kept as evidence, including failed and rate-limited attempts.

## Frontend

`frontend/index.html` is a single static page: attacks, defenses, a live simulator (plain agent vs TokenShield, through your local API), results, and details. If the API is not reachable it replays clearly labelled recorded runs from `results/`. Model output and web text are rendered as plain text, never as HTML.

## Layout

```
src/tokenshield/   schema, extractor, gate, shield, meter, chat, bifrost_client,
                   tavily_client, freshness, budget, benchmarks, reporting, api, gateway
scripts/           runners, probes, demos
tests/             unit tests for all of the above
results/           raw benchmark records (jsonl)
frontend/          the web page
```

No license file is included.