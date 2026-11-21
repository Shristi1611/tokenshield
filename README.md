# TokenShield

**A trust boundary for LLM agents: typed facts in, token traps out.**

Agents read untrusted text: READMEs, build logs, web pages, search results. Any of it can tell the agent to do something else, to burn thousands of tokens doing it, or to state something false. TokenShield sits between that text and your main model. A cheap, quarantined model reads the untrusted text alone and may only return a small set of **typed, bounded fields**. The main model never sees the raw text, only a block of validated facts.

It runs as a **standalone, OpenAI-compatible gateway in front of [Bifrost](https://github.com/maximhq/bifrost)**. It is not a Bifrost plugin. A second pipeline uses **Tavily** to check whether an answer is still true.

Built for the Geek Room Global AI Hackathon (Bifrost track and Tavily track, one integrated project).

**Live demo:** https://shristi1611.github.io/tokenshield/ (hosted page replays labelled recorded runs; live runs need the local setup below)
**Source:** https://github.com/shristi1611/tokenshield

## What it does

| Layer | What it does | Status |
|---|---|---|
| 1. Typed-fact quarantine | Untrusted text goes to a quarantined model. Its JSON is checked against a strict schema (booleans, bounded numbers, enums, restricted tokens, no free text). Any error fails closed to a fixed `[TS-BLOCKED]` placeholder. | Built, benchmarked |
| 2. Conformal token budgets | Split-conformal threshold (alpha 0.1) learned from clean runs; runs above it are flagged. | Measured in the API; enforced per session in the gateway only |
| 3. Trap advisories | Shared warnings about known token traps. | Designed, not a headline result |
| Cross-check (extension of layer 1) | Splits a document into sections, extracts each, and flags a license or version the sections disagree on as `conflict`. | Built; tuned on 8 documents, held-out run in progress (interim numbers below) |
| Freshness check (Tavily) | Extracts time-sensitive claims from an answer, searches Tavily per claim, quarantines the results like any other untrusted text, and computes the verdict in code. | Works end to end; measured on 48 real version claims (versions only) |

## Results

All results are **token counts** on the Groq free tier. The first three tables use **templated documents** (READMEs and build logs generated from templates); the sections after them use real public documents. Intervals are 95% bootstrap intervals over conversations: they show run-to-run spread, not real-world variance. No dollar savings are claimed.

**Long multi-turn conversations (10 READMEs x 5 turns, padded README):**

| | Plain agent | TokenShield |
|---|---|---|
| Mean tokens, clean | 9,881 | 4,028 |
| Mean tokens, attacked | 11,315 | 4,049 |
| Saved, clean | | +59.2% (58.9% to 59.5%) |
| Saved, attacked | | +64.2% (60.2% to 69.0%) |
| Turns hijacked under attack | 6 of 50 | 0 of 50 |
| Turns correct, attacked | 50 of 50 | 48 of 50 |

**Build logs (all 10 finished, 1 failed run counted at its spend):**

| | Plain agent | TokenShield |
|---|---|---|
| Saved, clean | | +57.0% (54.4% to 59.2%) |
| Saved, attacked | | +65.9% (59.3% to 71.6%, 9 pairs); +70.2% counting the failed run |
| Turns correct, clean | 46 of 50 | 49 of 50 |
| Turns correct, attacked | 40 of 45 | 43 of 50 |
| Turns hijacked under attack | 4 of 45 | 0 of 50 |

One plain-agent log hit an `http_413` after 24,252 tokens. One log (l06) was blocked by the shield in the attacked run and scored 0 of 5. Under attack the shielded accuracy (86%) was slightly below the plain agent's (89%).

**Answer flip (6 READMEs x 5 turns, planted false fact, no instructions):**

| | Plain agent | TokenShield, first run | TokenShield, with anchor hints |
|---|---|---|---|
| Turns correct, attacked | 23 of 30 | 17 of 30 | 27 of 30 |
| Turns correct, clean | 30 of 30 | 30 of 30 | 30 of 30 |

The hints were tuned on this template (see below).

### Real documents (public READMEs, nothing simulated except the planted attack text)

13 real READMEs from public repositories, each pinned by SHA-256, hand-labelled (labels in `data/realdocs/labels.json`, not derived from any model). Only the attack text is synthetic. Multi-turn, burn attack, Groq free tier.

- **Token savings: about 52 to 54%** for the shield. The savings come from quarantining the document once instead of re-sending it every turn, not from the schema: a summarize-first baseline saves about the same.
- **Accuracy under attack:** shield 98% (100% clean); summarize-first 83%, because the free-text summary carries the injection into the main model.
- **Plain agent:** followed the planted instruction on about 7% of turns (interval 0 to 24%), concentrated in one document (httpx). The shield: 0%.
- **Token amplification under attack** (attacked cost / clean cost, paired, 13 documents): plain agent 1.09x mean (1.03x median), shielded 1.01x.
- **Delimiter baseline** (random markers plus a "data, not instructions" prompt): saves no tokens (-2% to +1%) and did not measurably stop hijacks (6% of turns). Only 6 of 13 conversations completed because of rate limits, so this is the weakest-supported row.
- **Answer flip on real READMEs (10 documents, 35 questions):** accuracy under attack is 71% for the plain agent and 66% for the shield (both 100% clean), with the shield still saving about 54% of tokens. There is no shield advantage against a planted false fact. This is the weakness below.

### Cross-check against planted false facts

The shield cannot tell a lie from a fact, so we tried a cross-check: extract from two or three sections separately and flag a disagreement. A flag does not make the answer right, it only turns a confident wrong answer into a visible one. It catches only a lie that sits in a different section from the true statement.

**Tuning set** (8 real documents with a planted false license; design id `bffc38956551`, frozen before any held-out run):

| Method | Fooled | Flagged as conflict | False alarms on clean docs |
|---|---|---|---|
| Single extraction, shipped hints | 7 of 8 | 0 | 0 of 9 |
| Single extraction, no hints | 8 of 8 | 0 | 0 of 8 |
| Cross-check, 2 sections | 3 of 8 | 5 of 8 | 0 of 8 |
| Cross-check, 3 sections | 3 of 8 | 5 of 8 | 0 of 8 |

Cost: 2 or 3 model calls per document instead of 1; about 13% (2 sections) and 29% (3 sections) more tokens. The planted release number was almost never flagged, because most READMEs state no version. The anchor hints barely helped on real documents.

**Held-out set, INTERIM (14 of 21 documents finished; replace this block when the run completes).** The design was frozen before this run and the held-out documents were never used for tuning. 2 sections is the main method.

| Method | Planted license fooled | Flagged as conflict | Clean docs wrongly flagged |
|---|---|---|---|
| Single extraction, no hints | 13 of 13 | 0 | 0 of 14 |
| Single extraction, shipped hints | 11 of 13 | 0 | 0 of 14 |
| Cross-check, 2 sections | 1 of 13 | 12 of 13 | 0 of 13 |
| Cross-check, 3 sections | 0 of 13 | 13 of 13 | 0 of 13 |

Read this carefully. A flag is not a fix: the agent still does not know which license is true. The cross-check works here because the planted note and the real License section land in different sections. The one miss with 2 sections was a note planted next to the license, and a test shows that case beats it. An attacker who knows the design would plant the false claim inside or beside the License section, and the cross-check would not see it. Our attack positions (start, middle, end of the document) are not adversarial, and the planted note is not subtle. The planted release number was flagged rarely (2 of 13 with 2 sections). Cost: two or three model calls per document, about 17% (2 sections) and 37% (3 sections) more tokens than a single extraction.

### Freshness check (Tavily), measured on software versions

48 real claims: 24 products, each with one fresh claim (the newest stable series) and one stale claim (an older series that used to be newest). Ground truth comes from a release tracker read on 2026-11-19 (endoflife.date), not from Tavily or the model; we spot-checked three products against independent sources (Python 3.15, Rust 1.100, Laravel 13). Shipped settings, one run, no tuning.

| | Result |
|---|---|
| Stale claims caught (said outdated) | 16 of 24 (67%, 95% interval 46% to 83%) |
| Stale claims said current | 0 of 24 |
| Stale claims left unsure | 8 of 24 |
| Precision of "outdated" | 80% (16 of 20) |
| Fresh claims confirmed current | 12 of 24 |
| Fresh claims wrongly said outdated | 4 of 24 (17%, 95% interval 4% to 33%) |
| Fresh claims left unsure or not extracted | 8 of 24 |
| Cost per claim | 1 Tavily search, 3 model calls, about 2,000 tokens |

It works as a "needs checking" flag, not a verifier: it never called a stale answer current, but it confirmed only half of the fresh ones. The false alarms we inspected came from out-of-date third-party pages in Tavily's results (for example a blog post stating Python 3.14 as the latest), which the quarantined model read correctly; its recency estimate for such pages is also unreliable. Stale claims one series behind were caught 64% of the time (14 of 22), the two claims two series behind 2 of 2. Versions only, one phrasing, small sample.

### When does the shield pay for itself?

From the saved real-document conversations (13 READMEs of about 900 to 3,400 tokens each, no new model calls): the shield cost 29.6% more after the first question and was 29.2% cheaper after the second. It was cheaper from the second question on for all 13 documents. The one-off quarantine cost averages 2,527 tokens, against 2,148 for the plain agent's first question. A single-question task still costs more, and documents shorter than the ones tested (all were over 2,500 characters) are not covered.

## Where it falls short

We report these on purpose.

- **Planted false facts are not stopped.** A typed fact blocks instructions, not lies. On real READMEs the shield scored 66% against the plain agent's 71% under a planted false license. The cross-check flags most planted licenses when the lie is far from the true statement, but it does not say which value is true, and it cannot see a lie planted next to the true statement.
- **Hints were tuned on the templated flip test.** On real documents they barely helped.
- **A single question costs more.** The shield costs about 30% more tokens on the first question and pays for itself from the second (13 real documents). The savings come from conversations of two or more questions.
- **Denial of service.** If the quarantined model is hijacked into prose or an oversized reply, extraction fails closed and the main model sees only a placeholder. That is safe, but the user gets less.
- **Small samples.** 13 real documents for the burn attack, 8 for tuning and 21 for the held-out cross-check, 24 products for freshness. One attack template per kind, and the planted note is not subtle.
- **Freshness is a flag, and only measured on versions.** It confirmed half the fresh claims and wrongly flagged 17%; its answers are only as good as the pages Tavily returns. Prices, office holders and laws are unmeasured.
- **Free-tier noise.** Rate limits cut some runs short. Failed runs are reported with their spend, not dropped.
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

python -m http.server 3000       :: from the repo root, then open http://localhost:3000
```

The API only accepts pages served from `localhost:5173` or `localhost:3000`; for another port set `TOKENSHIELD_ALLOWED_ORIGINS`. Keys live only in `.env` and the Bifrost UI, never in the repo. The Tavily key is read as in `scripts\check_tavily.py`.

Tests and lint: `pytest -q` and `ruff check src tests scripts`.

Reproduce the benchmarks (each run spends free-tier tokens): `scripts\run_benchmark.py`, `run_multiturn.py`, `run_logbench.py`, `run_flipbench.py`, `run_multiturn.py --docs data\realdocs`, `run_crosscheck.py`, `run_freshness.py`, `break_even.py`, then `scripts\report.py results\<file>.jsonl`. Raw results are in `results/` and are kept as evidence, including failed and rate-limited attempts.

## Frontend

`index.html` (repo root, so GitHub Pages can serve it) is a single static page: attacks, defenses, a live simulator (plain agent vs TokenShield, through your local API), results, and details. If the API is not reachable (for example on the hosted GitHub Pages version, where visitors have no local API) it replays clearly labelled recorded runs from `results/`. Live runs need the local setup above. Model output and web text are rendered as plain text, never as HTML.

## Layout

```
src/tokenshield/   schema, extractor, crosscheck, crosscheck_eval, baselines, realdocs, breakeven, freshness_eval, gate, shield, meter, chat, bifrost_client,
                   tavily_client, freshness, budget, benchmarks, reporting, api, gateway
scripts/           runners, probes, demos
tests/             unit tests for all of the above
results/           raw benchmark records (jsonl)
index.html         the web page (repo root)
```

No license file is included.