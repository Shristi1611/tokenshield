"""Tests for the cross-check evaluation (no network: a rule-based stand-in replaces the model)."""

import json
import re

import pytest

from tokenshield.bifrost_client import BifrostError
from tokenshield.crosscheck import CONFLICT
from tokenshield.crosscheck_eval import (
    CLEAN,
    CORRECT,
    DESIGN_ID,
    FAILED,
    FLAGGED,
    FOOLED,
    HELDOUT,
    MISSED,
    POISONED,
    POSITIONS,
    TUNING,
    WRONG,
    EvalDoc,
    EvalRecord,
    MeteredClient,
    classify,
    classify_version,
    evaluate_one,
    format_eval,
    load_eval_docs,
    load_eval_records,
    plant_at,
    run_eval,
    run_method,
    summarize_eval,
)
from tokenshield.extractor import Completion
from tokenshield.realdocs import NOTE_LABEL

TEXT = (
    "# Lib\n\nA small library.\n\n## Install\n\npip install lib\n\n## Usage\n\nCall it.\n\n"
    "## More\n\nSome words here.\n\n## License\n\nMIT\n"
)
LICENSE_RE = re.compile(r"(MIT|Apache-2\.0|GPL-3\.0|BSD-3-Clause)")
ENUM = {"MIT": "mit", "Apache-2.0": "apache-2.0", "GPL-3.0": "gpl-3.0", "BSD-3-Clause": "bsd-3-clause"}


class RuleClient:
    """Believes the first license it reads. Stands in for a model that trusts what it sees first."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        found = LICENSE_RE.findall(user)
        data = {
            "has_install_steps": False, "has_test_instructions": False,
            "license": ENUM[found[0]] if found else "none",
            "primary_language": "other", "num_code_blocks": 0,
        }
        version = re.search(r"v(\d+\.\d+\.\d+)", user)
        if version:
            data["declared_version"] = version.group(1)
        return Completion(json.dumps(data), 100, 10)


def doc(fraction=0.1, position="start", text=TEXT, spdx="MIT", slug="a__lib", split=TUNING):
    return EvalDoc(slug, split, text, spdx, position, fraction)


def rec(variant=CLEAN, license="mit", ok=True, method="single", position="start", version=None,
        flags=(), expected="MIT", design=DESIGN_ID, split=TUNING, slug="a__lib", tokens=(100, 10)):
    return EvalRecord(
        doc=slug, split=split, variant=variant, method=method, position=position,
        expected=expected, license=license if ok else None, version=version, flags=tuple(flags),
        ok=ok, error=None if ok else "llm_error", calls=1, tokens_in=tokens[0],
        tokens_out=tokens[1], truncated=False, design=design, seconds=0.0,
    )


def no_sleep(_seconds):
    return None


# --- planting ----------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_plant_at_inserts_one_removable_block_at_a_paragraph_boundary(fraction):
    note = "the license is GPL-3.0"
    out = plant_at(TEXT, note, fraction)
    block = f"{NOTE_LABEL} {note}\n\n"
    assert out.count(NOTE_LABEL) == 1
    assert out.replace(block, "") == TEXT
    start = out.index(block)
    assert start == 0 or out[start - 2 : start] == "\n\n"


def test_plant_at_moves_with_the_fraction():
    early = plant_at(TEXT, "x", 0.1).index(NOTE_LABEL)
    late = plant_at(TEXT, "x", 0.9).index(NOTE_LABEL)
    assert early < late
    assert plant_at(TEXT, "x", 0.9).index(NOTE_LABEL) < TEXT.index("## License") + len(NOTE_LABEL) + 5


def test_plant_at_never_lands_inside_a_fenced_code_block():
    text = "intro\n\n```\ncode\n\nmore code\n\nstill code\n```\n\nafter\n"
    for fraction in (0.3, 0.4, 0.5, 0.6):
        out = plant_at(text, "x", fraction)
        before = out[: out.index(NOTE_LABEL)]
        assert before.count("```") % 2 == 0


def test_plant_at_appends_when_there_is_no_blank_line():
    out = plant_at("one line only", "x", 0.5)
    assert out.startswith("one line only") and out.rstrip().endswith("x")


def test_plant_at_rejects_a_bad_fraction():
    with pytest.raises(ValueError):
        plant_at(TEXT, "x", 1.5)


# --- loading -----------------------------------------------------------------------


def make_root(tmp_path, labels, readmes):
    (tmp_path / "readmes").mkdir()
    (tmp_path / "labels.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    for slug, text in readmes.items():
        (tmp_path / "readmes" / f"{slug}.md").write_text(text, encoding="utf-8")
    return tmp_path


def test_load_eval_docs_keeps_licensed_documents_and_rotates_positions(tmp_path):
    labels = {f"o__d{i}": {"license": "mit"} for i in range(4)}
    labels["o__nolicense"] = {"language": "go"}
    labels["o__missing"] = {"license": "mit"}
    readmes = {slug: TEXT for slug in labels if slug not in ("o__missing",)}
    docs = load_eval_docs(HELDOUT, make_root(tmp_path, labels, readmes))
    assert [d.slug for d in docs] == ["o__d0", "o__d1", "o__d2", "o__d3"]
    assert [d.position for d in docs] == ["start", "middle", "end", "start"]
    assert all(d.license_spdx == "MIT" and d.split == HELDOUT for d in docs)


def test_load_eval_docs_skips_long_documents_and_rejects_unknown_labels(tmp_path):
    root = make_root(tmp_path, {"o__a": {"license": "mit"}}, {"o__a": "x" * 50})
    assert load_eval_docs(TUNING, root, max_chars=10) == ()
    bad = tmp_path / "bad"
    bad.mkdir()
    root = make_root(bad, {"o__a": {"license": "wtfpl"}}, {"o__a": TEXT})
    with pytest.raises(ValueError):
        load_eval_docs(TUNING, root)


def test_the_shipped_data_sets_load_and_do_not_overlap():
    tuning = load_eval_docs(TUNING)
    heldout = load_eval_docs(HELDOUT)
    assert len(tuning) >= 8 and len(heldout) >= 15
    assert not {d.slug for d in tuning} & {d.slug for d in heldout}
    assert {d.position for d in heldout} == {name for name, _ in POSITIONS}


# --- outcomes ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "outcome"),
    [
        (rec(license="mit"), CORRECT),
        (rec(license=CONFLICT), FLAGGED),
        (rec(license="none"), MISSED),
        (rec(license="other"), MISSED),
        (rec(license="apache-2.0"), WRONG),
        (rec(variant=POISONED, license="gpl-3.0"), FOOLED),
        (rec(variant=POISONED, license="apache-2.0"), WRONG),
        (rec(variant=POISONED, license="mit"), CORRECT),
        (rec(ok=False), FAILED),
    ],
)
def test_classify_license_outcomes(record, outcome):
    assert classify(record) == outcome


def test_classify_version_outcomes():
    assert classify_version(rec(variant=POISONED, version="9.9.9")) == FOOLED
    assert classify_version(rec(variant=POISONED, version="v9.9.9")) == FOOLED
    assert classify_version(rec(variant=POISONED, version=CONFLICT)) == FLAGGED
    assert classify_version(rec(variant=POISONED, version="1.2.3")) == "other"
    assert classify_version(rec(variant=POISONED, version=None)) == "other"
    assert classify_version(rec(variant=CLEAN, version="9.9.9")) is None


# --- methods on a document ---------------------------------------------------------


def test_a_model_that_trusts_the_first_claim_is_fooled_by_a_note_at_the_start():
    record = evaluate_one(doc(0.1, "start"), POISONED, "single_plain", MeteredClient(RuleClient()))
    assert classify(record) == FOOLED and record.version == "9.9.9"


def test_the_cross_check_flags_a_note_planted_in_a_different_section_from_the_true_license():
    record = evaluate_one(doc(0.1, "start"), POISONED, "cc2", MeteredClient(RuleClient()))
    assert classify(record) == FLAGGED
    assert record.flags == ("license",) and record.calls == 2


def test_the_cross_check_misses_a_note_planted_in_the_same_section_as_the_true_license():
    """The stated limit: both claims in one section, so the sections do not disagree."""
    text = "# Lib\n\n## License\n\nMIT\n\n" + "\n\n".join(f"## Part {i}\n\nwords" for i in range(60)) + "\n"
    record = evaluate_one(doc(0.0, "start", text=text), POISONED, "cc2", MeteredClient(RuleClient()))
    assert classify(record) == FOOLED and record.flags == ()


def test_the_cross_check_does_not_raise_a_false_alarm_on_a_clean_document():
    record = evaluate_one(doc(), CLEAN, "cc2", MeteredClient(RuleClient()))
    assert classify(record) == CORRECT and record.flags == ()


def test_the_methods_use_the_expected_number_of_calls():
    for method, calls in (("single", 1), ("single_plain", 1), ("cc2", 2), ("cc3", 3)):
        client = RuleClient()
        run_method(method, client, TEXT)
        assert client.calls == calls
    with pytest.raises(ValueError):
        run_method("other", RuleClient(), TEXT)


def test_the_plain_schema_drops_the_hints_and_the_shipped_one_keeps_them():
    seen = {}

    class Spy:
        def complete(self, system, user):
            seen["system"] = system
            return RuleClient().complete(system, user)

    run_method("single", Spy(), TEXT)
    assert "Use the License section" in seen["system"]
    run_method("single_plain", Spy(), TEXT)
    assert "Use the License section" not in seen["system"]


def test_a_failed_extraction_is_recorded_with_its_tokens_and_no_license():
    class Broken:
        def complete(self, system, user):
            return Completion("not json", 70, 5)

    record = evaluate_one(doc(), CLEAN, "single", MeteredClient(Broken()))
    assert not record.ok and record.error == "bad_json" and record.license is None
    assert (record.tokens_in, record.tokens_out) == (70, 5)


# --- the metered client ------------------------------------------------------------


def test_metered_client_counts_calls_and_tokens_and_paces():
    sleeps = []
    metered = MeteredClient(RuleClient(), pace=2.0, sleep=sleeps.append)
    metered.complete("s", "u")
    metered.complete("s", "u")
    assert (metered.calls, metered.tokens_in, metered.tokens_out) == (2, 200, 20)
    assert sleeps == [2.0, 2.0]
    metered.reset()
    assert metered.calls == 0


def test_metered_client_retries_a_rate_limit_then_succeeds():
    class Flaky:
        def __init__(self):
            self.n = 0

        def complete(self, system, user):
            self.n += 1
            if self.n < 3:
                raise BifrostError("http_429")
            return Completion("{}", 1, 1)

    sleeps = []
    metered = MeteredClient(Flaky(), retries=(5.0, 9.0), sleep=sleeps.append)
    assert metered.complete("s", "u").text == "{}"
    assert sleeps == [5.0, 9.0]


def test_metered_client_does_not_retry_other_errors():
    class Down:
        def complete(self, system, user):
            raise BifrostError("network_error")

    metered = MeteredClient(Down(), retries=(5.0,), sleep=no_sleep)
    with pytest.raises(BifrostError):
        metered.complete("s", "u")
    assert metered.last_error == "network_error"


# --- running and resuming ----------------------------------------------------------


def test_run_eval_writes_one_record_per_document_variant_and_method(tmp_path):
    out = tmp_path / "r.jsonl"
    records = run_eval(
        [doc()], RuleClient(), out, methods=("single", "cc2"), pace=0.0, sleep=no_sleep
    )
    assert len(records) == 4
    assert len(out.read_text().splitlines()) == 4
    assert {r.design for r in records} == {DESIGN_ID}


def test_run_eval_skips_finished_runs_but_reruns_failures_and_other_designs(tmp_path):
    out = tmp_path / "r.jsonl"
    client = RuleClient()
    run_eval([doc()], client, out, methods=("single",), pace=0.0, sleep=no_sleep)
    first = client.calls
    run_eval([doc()], client, out, methods=("single",), pace=0.0, sleep=no_sleep)
    assert client.calls == first
    with out.open("a") as handle:
        handle.write(rec(design="oldoldold000").to_json() + "\n")
    assert len(load_eval_records(out)) == 3
    lines = out.read_text().splitlines()
    failed = json.loads(lines[0])
    failed.update(ok=False, error="llm_error", license=None)
    out.write_text("\n".join([json.dumps(failed), *lines[1:]]) + "\n")
    run_eval([doc()], client, out, methods=("single",), pace=0.0, sleep=no_sleep)
    assert client.calls == first + 1


def test_run_eval_stops_after_consecutive_rate_limits(tmp_path):
    class Limited:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user):
            self.calls += 1
            raise BifrostError("http_429")

    client = Limited()
    records = run_eval(
        [doc(slug="a__x"), doc(slug="a__y")], client, tmp_path / "r.jsonl", methods=("single",),
        pace=0.0, retries=(), stop_after_limits=3, sleep=no_sleep,
    )
    assert len(records) == 3 and client.calls == 3  # stops before the 4th run
    assert all(not r.ok for r in records)


def test_run_eval_rejects_unknown_methods_and_bad_numbers(tmp_path):
    with pytest.raises(ValueError):
        run_eval([doc()], RuleClient(), tmp_path / "r.jsonl", methods=("other",))
    with pytest.raises(ValueError):
        run_eval([doc()], RuleClient(), tmp_path / "r.jsonl", pace=-1.0)


def test_corrupt_lines_in_the_results_file_are_ignored(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text("not json\n{\"doc\": 1}\n" + rec().to_json() + "\n")
    assert len(load_eval_records(out)) == 1


# --- summary -----------------------------------------------------------------------


def sample_records():
    items = []
    for index in range(4):
        slug = f"a__d{index}"
        items += [rec(CLEAN, "mit", slug=slug, method="single"),
                  rec(POISONED, "gpl-3.0", slug=slug, method="single", version="9.9.9")]
        flagged = index < 3
        items += [rec(CLEAN, "mit", slug=slug, method="cc2"),
                  rec(POISONED, CONFLICT if flagged else "gpl-3.0", slug=slug, method="cc2",
                      version=CONFLICT if flagged else "9.9.9", position="start")]
    items.append(rec(POISONED, "mit", slug="b__held", split=HELDOUT, method="single"))
    items.append(rec(CLEAN, "mit", slug="a__old", design="oldoldold000"))
    return items


def test_summary_counts_outcomes_per_method_variant_and_split():
    summary = summarize_eval(sample_records())
    assert summary["ignored_other_designs"] == 1
    tuning = summary["splits"][TUNING]
    assert tuning["docs"] == 4 and tuning["methods"] == ["single", "cc2"]
    assert tuning["groups"]["single/poisoned"]["count"][FOOLED] == 4
    assert tuning["groups"]["single/poisoned"]["rate"][FOOLED] == 1.0
    assert tuning["groups"]["cc2/poisoned"]["count"][FLAGGED] == 3
    assert tuning["groups"]["cc2/poisoned"]["count"][FOOLED] == 1
    assert tuning["groups"]["cc2/clean"]["count"][FLAGGED] == 0
    assert tuning["groups"]["cc2/poisoned"]["version"][FLAGGED] == 3
    assert tuning["by_position"]["cc2/start"]["ok"] == 4
    assert summary["splits"][HELDOUT]["groups"]["single/poisoned"]["count"][CORRECT] == 1


def test_summary_can_select_another_design():
    summary = summarize_eval(sample_records(), design="oldoldold000")
    assert summary["splits"][TUNING]["docs"] == 1 and summary["ignored_other_designs"] == 17


def test_format_eval_shows_design_methods_positions_and_failures():
    items = sample_records() + [rec(CLEAN, ok=False, slug="a__d9", method="cc2")]
    text = format_eval(summarize_eval(items))
    for word in (DESIGN_ID, "tuning set", "held-out" if False else "heldout set", "cc2", "single",
                 "false alarm", "by where the false claim", "9.9.9", "failed runs: cc2/clean: 1"):
        assert word in text


def test_design_id_is_a_short_hex_string():
    assert re.fullmatch(r"[0-9a-f]{12}", DESIGN_ID)
