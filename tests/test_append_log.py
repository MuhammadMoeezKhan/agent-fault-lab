"""Tests for the append-only log observer.

The observer exists because a replay oracle compares content and so cannot see
an effect whose only durable trace carries a timestamp. The property that
matters is therefore narrow and testable: two runs that appended the same
number of lines must compare equal however their timestamps differ, and a run
that appended one line more must not.
"""
from __future__ import annotations

import json
import os

from agentfaultlab.observers import AppendLogObserver
from agentfaultlab.observers.base import equal


def _write(root: str, name: str, records: list[dict]) -> None:
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_two_runs_with_different_timestamps_compare_equal(tmp_path):
    once = tmp_path / "once"
    again = tmp_path / "again"
    for root, stamp in ((once, "2026-10-26T06:31:19.031Z"), (again, "2026-10-26T18:02:55.887Z")):
        os.makedirs(root)
        _write(str(root), "audit.jsonl", [{"at": stamp, "action": "queued", "id": "a"}])
    observer = AppendLogObserver(str(once), drop_keys=("at",))
    other = AppendLogObserver(str(again), drop_keys=("at",))
    assert equal(observer.snapshot(), other.snapshot())


def test_one_append_more_is_visible(tmp_path):
    once = tmp_path / "once"
    twice = tmp_path / "twice"
    os.makedirs(once)
    os.makedirs(twice)
    _write(str(once), "audit.jsonl", [{"at": "t1", "id": "a"}])
    _write(str(twice), "audit.jsonl", [{"at": "t1", "id": "a"}, {"at": "t2", "id": "a"}])
    left = AppendLogObserver(str(once), drop_keys=("at",)).snapshot()
    right = AppendLogObserver(str(twice), drop_keys=("at",)).snapshot()
    assert left["audit.jsonl"]["lines"] == 1
    assert right["audit.jsonl"]["lines"] == 2
    assert not equal(left, right)


def test_content_still_distinguishes_a_different_write_of_the_same_size(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    os.makedirs(left_root)
    os.makedirs(right_root)
    _write(str(left_root), "audit.jsonl", [{"at": "t1", "id": "a"}])
    _write(str(right_root), "audit.jsonl", [{"at": "t2", "id": "b"}])
    left = AppendLogObserver(str(left_root), drop_keys=("at",)).snapshot()
    right = AppendLogObserver(str(right_root), drop_keys=("at",)).snapshot()
    assert left["audit.jsonl"]["lines"] == right["audit.jsonl"]["lines"] == 1
    assert not equal(left, right)


def test_a_file_that_is_not_json_is_named_count_only(tmp_path):
    os.makedirs(tmp_path / "logs")
    with open(tmp_path / "logs" / "server.log", "w", encoding="utf-8") as handle:
        handle.write("started at 06:31\nqueued a\n")
    state = AppendLogObserver(str(tmp_path), drop_keys=("at",)).snapshot()
    entry = state[os.path.join("logs", "server.log")]
    assert entry["mode"] == "count-only"
    assert entry["lines"] == 2
    # The weaker comparison is named rather than hidden: without this, a
    # report could present a count-only file as a content comparison.
    assert "content" not in entry


def test_only_the_named_patterns_are_read(tmp_path):
    _write(str(tmp_path), "audit.jsonl", [{"at": "t", "id": "a"}])
    with open(tmp_path / "memory.md", "w", encoding="utf-8") as handle:
        handle.write("not a log\n")
    state = AppendLogObserver(str(tmp_path), patterns=("*.jsonl",)).snapshot()
    assert list(state) == ["audit.jsonl"]


def test_a_missing_log_is_absent_rather_than_an_error(tmp_path):
    assert AppendLogObserver(str(tmp_path / "nothing-here")).snapshot() == {}


def test_without_declared_keys_the_comparison_is_count_only(tmp_path):
    _write(str(tmp_path), "audit.jsonl", [{"at": "t1", "id": "a"}])
    state = AppendLogObserver(str(tmp_path)).snapshot()
    assert state["audit.jsonl"]["mode"] == "count-only"
