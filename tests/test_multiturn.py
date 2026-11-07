"""Tests for the multi-turn benchmark."""

import json
from dataclasses import asdict

import pytest

from tokenshield.benchmark import (
    BASELINE,
    CLEAN,
    INJECTION_LABEL,
    POISONED,
    PROJECTS,
    SHIELDED,
    SYSTEM_PROMPT,
    default_policy,
)
from tokenshield.bifrost_client import BifrostError
from tokenshield.extractor import Completion
from tokenshield.multiturn import (
    FORMAT_SUFFIX,
    QUARANTINE_ERROR,
    ConvRecord,
    append_conv_record,
    build_conversations,
    format_multiturn,
    initial_messages,
    load_conv_records,
    pad_readme,
    run_conversation,
    run_multiturn,
    summarize_multiturn,
)
from tokenshield.shield import KIND_KEY

CONVOS = build_conversations()
CONVO = CONVOS[0]
POLICY = default_policy()
ANSWERS = ["ANSWER: MIT", "ANSWER: python", "ANSWER: yes", "ANSWER: 1.2.3", "ANSWER: yes"]


class FakeMain:
    """Stands in for the main model. Replies are Completions or BifrostErrors; the last repeats."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, model=None, max_tokens=None):
        self.calls.append([dict(message) for message in messages])
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeQuarantine:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply


def valid_json():
    return json.dumps(
        {
            "has_install_steps": True,
            "has_test_instructions": True,
            "license": "mit",
            "primary_language": "python",
            "num_code_blocks": 2,
            "declared_version": "v1.2.3",
        }
    )


def good_quarantine():
    return FakeQuarantine(Completion(valid_json(), 300, 60))


def scripted_main():
    return FakeMain(*[Completion(text, 100, 10) for text in ANSWERS])


def no_sleep(_seconds):
    return None


def zero_clock():
    return 0.0


def run(variant, mode, quarantine=None, main=None, **kwargs):
    kwargs.setdefault("sleep", no_sleep)
    kwargs.setdefault("clock", zero_clock)
    return run_conversation(
        CONVO, variant, mode, quarantine or good_quarantine(), main or scripted_main(), POLICY,
        **kwargs,
    )


# --- conversations ------------------------------------------------------------------


def test_ten_conversations_with_five_in_schema_turns_each():
    assert len(CONVOS) == 10
    assert len({convo.id for convo in CONVOS}) == 10
    assert all(len(convo.questions) == 5 for convo in CONVOS)
    assert CONVO.expected == ("MIT", "python", "yes", "1.2.3", "yes")


def test_poisoned_readmes_carry_the_injection_and_clean_ones_do_not():
    for convo in CONVOS:
        assert INJECTION_LABEL not in convo.readme_clean
        assert INJECTION_LABEL in convo.readme_poisoned


def test_initial_messages_mark_the_document_only_for_the_shield():
    baseline = initial_messages(CONVO, CLEAN, BASELINE)
    shielded = initial_messages(CONVO, POISONED, SHIELDED)
    assert baseline[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert baseline[1] == {"role": "user", "content": CONVO.readme_clean}
    assert shielded[1][KIND_KEY] == "file"
    assert shielded[1]["content"] == CONVO.readme_poisoned


def test_unknown_variant_and_mode_are_rejected():
    with pytest.raises(ValueError, match="variant"):
        initial_messages(CONVO, "other", BASELINE)
    with pytest.raises(ValueError, match="mode"):
        initial_messages(CONVO, CLEAN, "other")
    with pytest.raises(ValueError, match="mode"):
        run(CLEAN, "other")


def test_padding_inserts_a_changelog_before_contributing():
    text = "# x\n\n## Contributing\n\nhi\n"
    padded = pad_readme(text, 3)
    assert padded.count("\n- ") == 3
    assert padded.index("## Changelog") < padded.index("## Contributing")
    assert padded.endswith("hi\n")
    assert pad_readme(text, 0) == text


def test_padded_conversations_are_longer_but_keep_the_answers():
    padded = build_conversations(40)[0]
    assert len(padded.readme_clean) > len(CONVO.readme_clean) + 1000
    assert padded.expected == CONVO.expected
    assert "## Changelog" in padded.readme_poisoned
    assert INJECTION_LABEL in padded.readme_poisoned


def test_padding_cannot_be_negative():
    with pytest.raises(ValueError, match="lines"):
        pad_readme("x", -1)
    with pytest.raises(ValueError, match="pad_lines"):
        build_conversations(-1)


# --- running a conversation ---------------------------------------------------------


def test_baseline_history_grows_and_always_carries_the_raw_readme():
    main = scripted_main()
    record = run(CLEAN, BASELINE, main=main)
    assert [len(call) for call in main.calls] == [3, 5, 7, 9, 11]
    for call in main.calls:
        assert call[1]["content"] == CONVO.readme_clean
        assert KIND_KEY not in call[1]
    assert main.calls[0][2]["content"] == CONVO.questions[0] + FORMAT_SUFFIX
    assert main.calls[4][3] == {"role": "assistant", "content": ANSWERS[0]}
    assert main.calls[4][5] == {"role": "assistant", "content": ANSWERS[1]}
    assert (record.ok, record.turns, record.correct, record.followed) == (True, 5, 5, 0)
    assert (record.main_in, record.main_out) == (500, 50)
    assert record.turn_tokens == (110,) * 5
    assert (record.quarantine_calls, record.action, record.retry_tokens) == (0, None, 0)


def test_shielded_quarantines_once_and_resends_only_the_facts():
    quarantine = good_quarantine()
    main = scripted_main()
    record = run(CLEAN, SHIELDED, quarantine, main)
    assert len(quarantine.calls) == 1
    assert [len(call) for call in main.calls] == [3, 5, 7, 9, 11]
    for call in main.calls:
        assert call[1]["content"].startswith("[TS-FACTS")
        assert "fastgrid" not in str(call)
    assert record.correct == 5
    assert (record.quarantine_in, record.quarantine_out, record.quarantine_calls) == (300, 60, 1)
    assert record.action == "quarantined"
    assert record.total_tokens == 910


def test_the_injection_never_reaches_the_shielded_main_model():
    main = scripted_main()
    run(POISONED, SHIELDED, main=main)
    assert all("prime numbers" not in str(call) for call in main.calls)
    base = scripted_main()
    run(POISONED, BASELINE, main=base)
    assert all("prime numbers" in str(call) for call in base.calls)


def test_a_long_reply_is_counted_and_stays_in_the_history():
    long_reply = "ANSWER: MIT\n" + "extra line\n" * 5
    main = FakeMain(Completion(long_reply, 100, 10), *[Completion(t, 100, 10) for t in ANSWERS[1:]])
    record = run(CLEAN, BASELINE, main=main)
    assert record.followed == 1
    assert record.correct == 5
    assert main.calls[1][3]["content"] == long_reply


def test_a_blocked_extraction_is_not_retried():
    quarantine = FakeQuarantine(Completion("not json", 10, 5))
    main = scripted_main()
    record = run(CLEAN, SHIELDED, quarantine, main)
    assert record.ok is True
    assert record.action == "blocked"
    assert (record.quarantine_in, record.quarantine_out) == (10, 5)
    assert len(quarantine.calls) == 1
    assert main.calls[0][1]["content"].startswith("[TS-BLOCKED")


def test_rate_limits_are_retried_without_losing_the_turn():
    main = FakeMain(BifrostError("http_429"), *[Completion(t, 100, 10) for t in ANSWERS])
    sleeps = []
    record = run(CLEAN, BASELINE, main=main, sleep=sleeps.append)
    assert record.ok is True
    assert sleeps == [20.0]
    assert record.turn_tokens == (110,) * 5
    assert len(main.calls) == 6


def test_retries_run_out_and_the_failure_is_recorded():
    sleeps = []
    record = run(CLEAN, BASELINE, main=FakeMain(BifrostError("http_429")), sleep=sleeps.append)
    assert (record.ok, record.error, record.turns) == (False, "http_429", 0)
    assert sleeps == [20.0, 45.0, 90.0]


def test_other_errors_are_not_retried():
    sleeps = []
    record = run(CLEAN, BASELINE, main=FakeMain(BifrostError("http_401")), sleep=sleeps.append)
    assert (record.ok, record.error, record.turns) == (False, "http_401", 0)
    assert sleeps == []


def test_a_failure_part_way_keeps_what_was_spent():
    main = FakeMain(
        Completion(ANSWERS[0], 100, 10), Completion(ANSWERS[1], 100, 10), BifrostError("http_401")
    )
    record = run(CLEAN, BASELINE, main=main)
    assert (record.ok, record.error, record.turns, record.correct) == (False, "http_401", 2, 2)
    assert record.turn_tokens == (110, 110)
    assert record.main_in == 200


def test_quarantine_failures_are_retried_and_then_recorded():
    quarantine = FakeQuarantine(RuntimeError("busy"))
    main = scripted_main()
    sleeps = []
    record = run(CLEAN, SHIELDED, quarantine, main, sleep=sleeps.append)
    assert (record.ok, record.error, record.turns) == (False, QUARANTINE_ERROR, 0)
    assert sleeps == [20.0, 45.0, 90.0]
    assert len(quarantine.calls) == 4
    assert main.calls == []


def test_a_quarantine_retry_can_succeed():
    quarantine = FakeQuarantine(RuntimeError("busy"), Completion(valid_json(), 300, 60))
    sleeps = []
    record = run(CLEAN, SHIELDED, quarantine, sleep=sleeps.append)
    assert (record.ok, record.action, record.quarantine_calls) == (True, "quarantined", 1)
    assert sleeps == [20.0]


def test_pace_waits_after_every_model_call():
    baseline_sleeps = []
    run(CLEAN, BASELINE, sleep=baseline_sleeps.append, pace=1.5)
    assert baseline_sleeps == [1.5] * 5
    shielded_sleeps = []
    run(CLEAN, SHIELDED, sleep=shielded_sleeps.append, pace=1.5)
    assert shielded_sleeps == [1.5] * 6


# --- records and the runner ---------------------------------------------------------


def crec(**overrides):
    values = {
        "conv_id": "c01",
        "variant": CLEAN,
        "mode": BASELINE,
        "ok": True,
        "error": None,
        "turns": 5,
        "correct": 5,
        "followed": 0,
        "main_in": 400,
        "main_out": 100,
        "quarantine_in": 0,
        "quarantine_out": 0,
        "quarantine_calls": 0,
        "unreported": 0,
        "retry_tokens": 0,
        "turn_tokens": (80, 90, 100, 110, 120),
        "action": None,
        "seconds": 1.0,
    }
    values.update(overrides)
    return ConvRecord(**values)


def test_record_tokens_and_key():
    record = crec(quarantine_in=5, quarantine_out=7, retry_tokens=3)
    assert record.total_tokens == 512
    assert record.cost_tokens == 515
    assert record.key == ("c01", CLEAN, BASELINE)


def test_records_round_trip_through_the_results_file(tmp_path):
    path = tmp_path / "results" / "multi.jsonl"
    record = crec()
    append_conv_record(path, record)
    assert load_conv_records(path) == {record.key: record}


def test_the_latest_record_wins(tmp_path):
    path = tmp_path / "multi.jsonl"
    append_conv_record(path, crec(ok=False, error="http_429"))
    append_conv_record(path, crec())
    assert load_conv_records(path)[("c01", CLEAN, BASELINE)].ok is True


def test_unreadable_lines_are_ignored(tmp_path):
    bad_turns = json.dumps({**asdict(crec()), "turn_tokens": "x"})
    path = tmp_path / "multi.jsonl"
    lines = ["not json", "", '{"conv_id": "x"}', "[1]", bad_turns, crec().to_json()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert list(load_conv_records(path)) == [("c01", CLEAN, BASELINE)]


def test_a_missing_results_file_is_empty(tmp_path):
    assert load_conv_records(tmp_path / "missing.jsonl") == {}


def test_the_runner_resumes_and_skips_finished_conversations(tmp_path):
    out = tmp_path / "results" / "multi.jsonl"
    main = FakeMain(Completion("ANSWER: x", 50, 10))
    quarantine = good_quarantine()
    first = run_multiturn(CONVOS[:2], quarantine, main, POLICY, out, pace=0, sleep=no_sleep)
    assert len(first) == 8
    assert len(main.calls) == 40
    assert len(quarantine.calls) == 4
    second = run_multiturn(CONVOS[:3], quarantine, main, POLICY, out, pace=0, sleep=no_sleep)
    assert len(second) == 12
    assert len(main.calls) == 60
    assert len(out.read_text(encoding="utf-8").splitlines()) == 12


def test_failed_conversations_are_retried_on_the_next_invocation(tmp_path):
    out = tmp_path / "multi.jsonl"
    first = run_multiturn(
        CONVOS[:1], good_quarantine(), FakeMain(BifrostError("http_401")), POLICY, out,
        pace=0, sleep=no_sleep,
    )
    assert all(not record.ok for record in first)
    good = FakeMain(Completion("ANSWER: x", 50, 10))
    second = run_multiturn(CONVOS[:1], good_quarantine(), good, POLICY, out, pace=0, sleep=no_sleep)
    assert all(record.ok for record in second)
    assert len(good.calls) == 20
    assert len(load_conv_records(out)) == 4
    assert len(out.read_text(encoding="utf-8").splitlines()) == 8


def test_the_runner_can_run_one_mode_and_reports_new_records(tmp_path):
    seen = []
    records = run_multiturn(
        CONVOS[:1], good_quarantine(), scripted_main(), POLICY, tmp_path / "m.jsonl",
        modes=(BASELINE,), pace=0, sleep=no_sleep, on_record=seen.append,
    )
    assert [record.mode for record in records] == [BASELINE, BASELINE]
    assert len(seen) == 2


def test_the_runner_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        run_multiturn(
            CONVOS[:1], good_quarantine(), scripted_main(), POLICY, tmp_path / "m.jsonl",
            modes=("other",),
        )
    with pytest.raises(ValueError, match="pace"):
        run_multiturn(
            CONVOS[:1], good_quarantine(), scripted_main(), POLICY, tmp_path / "m.jsonl", pace=-1
        )


# --- summary ------------------------------------------------------------------------


def dataset():
    return [
        crec(conv_id="c01", variant=CLEAN, mode=BASELINE),
        crec(conv_id="c01", variant=POISONED, mode=BASELINE, main_in=1400, main_out=100,
             correct=4, followed=3, turn_tokens=(200, 250, 300, 350, 400)),
        crec(conv_id="c01", variant=CLEAN, mode=SHIELDED, main_in=100, main_out=50,
             quarantine_in=200, quarantine_out=100, turn_tokens=(30,) * 5),
        crec(conv_id="c01", variant=POISONED, mode=SHIELDED, main_in=100, main_out=60,
             quarantine_in=220, quarantine_out=120, turn_tokens=(32,) * 5),
    ]


def test_summary_groups_savings_and_amplification():
    summary = summarize_multiturn(dataset())
    groups = summary["groups"]
    assert groups["baseline/clean"]["mean_cost"] == 500
    assert groups["baseline/poisoned"]["mean_cost"] == 1500
    assert groups["shielded/clean"]["mean_cost"] == 450
    assert groups["baseline/poisoned"]["accuracy"] == pytest.approx(0.8)
    assert groups["baseline/poisoned"]["followed_rate"] == pytest.approx(0.6)
    assert groups["baseline/clean"]["per_turn"] == [80, 90, 100, 110, 120]
    assert groups["shielded/clean"]["mean_quarantine"] == 300
    assert summary["savings"][CLEAN] == pytest.approx(0.1)
    assert summary["savings"][POISONED] == pytest.approx(1 - 500 / 1500)
    assert summary["amplification"][BASELINE]["mean"] == pytest.approx(3.0)
    assert summary["amplification"][SHIELDED]["mean"] == pytest.approx(500 / 450)


def test_summary_ignores_failed_runs_but_counts_them():
    items = dataset() + [crec(conv_id="c99", ok=False, error="http_401")]
    summary = summarize_multiturn(items)
    assert summary["failed_runs"] == 1
    assert summary["groups"]["baseline/clean"]["runs"] == 2
    assert summary["groups"]["baseline/clean"]["ok"] == 1


def test_summary_counts_retry_tokens():
    summary = summarize_multiturn([crec(retry_tokens=100)])
    assert summary["groups"]["baseline/clean"]["mean_cost"] == 600


def test_summary_of_nothing_is_all_none():
    summary = summarize_multiturn([])
    assert summary["groups"]["baseline/clean"]["mean_cost"] is None
    assert summary["savings"][CLEAN] is None
    assert summary["groups"]["baseline/clean"]["per_turn"] == []


def test_formatted_summary_has_every_section():
    text = format_multiturn(summarize_multiturn(dataset()))
    for label in (
        "Multi-turn benchmark summary",
        "baseline/clean",
        "shielded/poisoned",
        "tokens saved by the shield",
        "token amplification under attack",
        "main-model tokens per turn",
        "failed runs: 0",
    ):
        assert label in text


def test_formatted_empty_summary_does_not_crash():
    assert "n/a" in format_multiturn(summarize_multiturn([]))


def test_projects_are_unchanged_by_the_conversation_builder():
    assert len(PROJECTS) == len(CONVOS)