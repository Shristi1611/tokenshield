"""Tests that deterministic failures are final and are not rerun on resume."""

from tokenshield.benchmark import BASELINE, default_policy
from tokenshield.bifrost_client import BifrostError
from tokenshield.multiturn import (
    FINAL_ERRORS,
    build_conversations,
    load_conv_records,
    run_multiturn,
)

CONVOS = build_conversations()


class FailingMain:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def chat(self, messages, *, model=None, max_tokens=None):
        self.calls += 1
        raise BifrostError(self.error)


class UnusedQuarantine:
    def complete(self, system, user):
        raise AssertionError("the baseline never calls the quarantine model")


def no_sleep(_seconds):
    return None


def run_once(main, out):
    return run_multiturn(
        CONVOS[:1], UnusedQuarantine(), main, default_policy(), out,
        modes=(BASELINE,), pace=0, sleep=no_sleep,
    )


def test_request_too_large_is_a_final_error():
    assert "http_413" in FINAL_ERRORS
    assert "http_429" not in FINAL_ERRORS
    assert "http_401" not in FINAL_ERRORS


def test_a_request_too_large_outcome_is_recorded_and_not_rerun(tmp_path):
    out = tmp_path / "m.jsonl"
    main = FailingMain("http_413")
    first = run_once(main, out)
    assert [record.error for record in first] == ["http_413", "http_413"]
    assert main.calls == 2
    second = run_once(main, out)
    assert main.calls == 2
    assert len(second) == 2
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    assert all(not record.ok for record in load_conv_records(out).values())


def test_other_failures_are_still_retried_on_the_next_invocation(tmp_path):
    out = tmp_path / "m.jsonl"
    main = FailingMain("http_401")
    run_once(main, out)
    assert main.calls == 2
    run_once(main, out)
    assert main.calls == 4