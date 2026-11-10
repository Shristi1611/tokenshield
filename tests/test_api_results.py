"""Tests for the /api/results endpoint."""

import json

from tokenshield.api import RESULT_SUITES, ApiService
from tokenshield.benchmark import default_policy
from tokenshield.multiturn import ConvRecord, append_conv_record


def crec(conv_id, variant, mode, cost):
    return ConvRecord(
        conv_id=conv_id, variant=variant, mode=mode, ok=True, error=None, turns=5, correct=5,
        followed=0, main_in=cost - 100, main_out=100, quarantine_in=0, quarantine_out=0,
        quarantine_calls=0, unreported=0, retry_tokens=0, turn_tokens=(), action=None,
        seconds=1.0,
    )


def make(tmp_path):
    return ApiService(object(), object(), None, default_policy(), tmp_path / "benchmark.jsonl")


def write_suite(tmp_path, name):
    path = tmp_path / name
    for conv in ("c01", "c02"):
        append_conv_record(path, crec(conv, "clean", "baseline", 500))
        append_conv_record(path, crec(conv, "clean", "shielded", 200))
        append_conv_record(path, crec(conv, "poisoned", "baseline", 600))
        append_conv_record(path, crec(conv, "poisoned", "shielded", 200))
    return path


def get(tmp_path):
    return make(tmp_path).handle("GET", "/api/results", None)


def test_the_suite_labels_are_the_documented_ones():
    assert set(RESULT_SUITES) == {"readme_long", "readme_short", "logs", "flip"}


def test_without_files_nothing_is_available(tmp_path):
    assert get(tmp_path) == (200, {"available": False, "suites": {}})


def test_a_saved_suite_is_summarised(tmp_path):
    write_suite(tmp_path, "multiturn_pad150_v2.jsonl")
    status, body = get(tmp_path)
    assert status == 200
    assert body["available"] is True
    suite = body["suites"]["readme_long"]
    assert suite["file"] == "multiturn_pad150_v2.jsonl"
    assert suite["savings"]["clean"]["conversations"] == 2
    assert suite["savings"]["clean"]["saving"] == 0.6
    assert set(suite["groups"]) == {
        "baseline/clean", "baseline/poisoned", "shielded/clean", "shielded/poisoned",
    }
    json.dumps(body)


def test_the_second_log_run_is_preferred(tmp_path):
    write_suite(tmp_path, "logbench.jsonl")
    write_suite(tmp_path, "logbench_v2.jsonl")
    assert get(tmp_path)[1]["suites"]["logs"]["file"] == "logbench_v2.jsonl"


def test_the_first_log_run_is_the_fallback(tmp_path):
    write_suite(tmp_path, "logbench.jsonl")
    assert get(tmp_path)[1]["suites"]["logs"]["file"] == "logbench.jsonl"


def test_unlisted_files_are_ignored(tmp_path):
    write_suite(tmp_path, "other.jsonl")
    assert get(tmp_path)[1] == {"available": False, "suites": {}}


def test_unreadable_files_are_skipped(tmp_path):
    (tmp_path / "flipbench.jsonl").write_text("not json\n{}\n", encoding="utf-8")
    assert get(tmp_path)[1] == {"available": False, "suites": {}}


def test_only_get_is_allowed(tmp_path):
    result = make(tmp_path).handle("POST", "/api/results", {})
    assert result == (405, {"error": "method_not_allowed"})