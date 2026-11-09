"""Tests that a benchmark run stops early when the free-tier limit keeps failing."""

import pytest

from tokenshield.benchmark import BASELINE, SHIELDED, default_policy
from tokenshield.bifrost_client import BifrostError
from tokenshield.extractor import Completion
from tokenshield.multiturn import (
    LIMIT_ERRORS,
    STOP_AFTER_LIMITS,
    build_conversations,
    load_conv_records,
    run_multiturn,
)

CONVOS = build_conversations()
OK = Completion("ANSWER: x", 50, 10)
LIMIT = BifrostError("http_429")


class ScriptedMain:
    """Plays a script of replies in order; the last one repeats."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, *, model=None, max_tokens=None):
        self.calls += 1
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        return step


class UnusedQuarantine:
    def complete(self, system, user):
        raise AssertionError("not used")


class BusyQuarantine:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        raise RuntimeError("busy")


def no_sleep(_seconds):
    return None


def run(main, out, quarantine=None, **kwargs):
    kwargs.setdefault("modes", (BASELINE,))
    return run_multiturn(
        CONVOS[:3], quarantine or UnusedQuarantine(), main, default_policy(), out,
        retries=(), pace=0, sleep=no_sleep, **kwargs,
    )


def test_the_defaults():
    assert STOP_AFTER_LIMITS == 3
    assert "http_429" in LIMIT_ERRORS
    assert "http_401" not in LIMIT_ERRORS


def test_three_rate_limit_failures_in_a_row_stop_the_run(tmp_path):
    main = ScriptedMain(LIMIT)
    out = tmp_path / "m.jsonl"
    results = run(main, out)
    assert len(results) == 3
    assert main.calls == 3
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_a_success_resets_the_streak(tmp_path):
    main = ScriptedMain(LIMIT, OK, OK, OK, OK, OK, LIMIT)
    results = run(main, tmp_path / "m.jsonl")
    assert [record.ok for record in results] == [False, True, False, False, False]
    assert main.calls == 9


def test_stopping_can_be_disabled(tmp_path):
    main = ScriptedMain(LIMIT)
    results = run(main, tmp_path / "m.jsonl", stop_after_limits=0)
    assert len(results) == 6
    assert main.calls == 6


def test_other_failures_never_stop_the_run(tmp_path):
    main = ScriptedMain(BifrostError("http_401"))
    results = run(main, tmp_path / "m.jsonl")
    assert len(results) == 6
    assert main.calls == 6


def test_a_custom_threshold_is_respected(tmp_path):
    results = run(ScriptedMain(LIMIT), tmp_path / "m.jsonl", stop_after_limits=2)
    assert len(results) == 2


def test_failed_runs_are_saved_so_a_rerun_retries_them(tmp_path):
    out = tmp_path / "m.jsonl"
    run(ScriptedMain(LIMIT), out)
    assert all(not record.ok for record in load_conv_records(out).values())
    results = run(ScriptedMain(OK), out)
    assert len(results) == 6
    assert all(record.ok for record in results)


def test_quarantine_failures_count_too(tmp_path):
    quarantine = BusyQuarantine()
    main = ScriptedMain(OK)
    results = run(main, tmp_path / "m.jsonl", quarantine=quarantine, modes=(SHIELDED,))
    assert len(results) == 3
    assert {record.error for record in results} == {"quarantine_llm_error"}
    assert quarantine.calls == 3
    assert main.calls == 0


def test_a_negative_threshold_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="stop_after_limits"):
        run(ScriptedMain(OK), tmp_path / "m.jsonl", stop_after_limits=-1)