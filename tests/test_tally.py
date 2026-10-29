"""Tests for counting files whose names cannot be compared.

Some servers write one file per call with a timestamp and a random suffix in
the filename, so every trial produces a different path and a filesystem
observer never reproduces a baseline. Excluding such files was the earlier
answer and the preregistration records it as a known bias, pointing the same
way as the one `JsonDocObserver` removed: a server that stamps its filenames
tends to be one keeping an audit trail on purpose. For at least one server the
stamped file is the whole effect.

Both directions are pinned here. A repeat must still show, and two trials that
each wrote one equivalent file must still compare equal however the files were
named.
"""
from __future__ import annotations

import json
import os

from agentfaultlab.observers.tally import DirectoryTallyObserver


def _write(root, name, payload):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    with open(path, "w", encoding="utf-8") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle)
    return path


def test_two_runs_with_differently_named_equivalent_files_compare_equal(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write(a, "run-2026-10-29T02-48-39-073Z-5b3e82c1.json", {"score": 88, "at": "t1"})
    _write(b, "run-2026-10-29T03-11-02-114Z-9fa0c7de.json", {"score": 88, "at": "t2"})
    one = DirectoryTallyObserver(a, drop_keys=("at",)).snapshot()
    two = DirectoryTallyObserver(b, drop_keys=("at",)).snapshot()
    assert one == two


def test_a_second_file_is_a_duplicate_even_with_an_unrelated_name(tmp_path):
    root = str(tmp_path / "x")
    _write(root, "run-aaaa.json", {"score": 88, "at": "t1"})
    once = DirectoryTallyObserver(root, drop_keys=("at",)).snapshot()
    _write(root, "run-bbbb.json", {"score": 88, "at": "t2"})
    twice = DirectoryTallyObserver(root, drop_keys=("at",)).snapshot()
    assert once != twice
    assert once["count"] == 1 and twice["count"] == 2


def test_identical_records_are_counted_not_collapsed(tmp_path):
    # Writing the same record twice under two names is exactly the duplicate
    # this observer exists to see, so the payloads must not become a set.
    root = str(tmp_path / "y")
    _write(root, "one.json", {"v": 1})
    first = DirectoryTallyObserver(root).snapshot()
    _write(root, "two.json", {"v": 1})
    second = DirectoryTallyObserver(root).snapshot()
    assert first["payloads"] != second["payloads"]


def test_a_changed_value_still_differs_after_dropping(tmp_path):
    a, b = str(tmp_path / "p"), str(tmp_path / "q")
    _write(a, "r1.json", {"score": 88, "at": "t"})
    _write(b, "r2.json", {"score": 91, "at": "t"})
    assert (
        DirectoryTallyObserver(a, drop_keys=("at",)).snapshot()
        != DirectoryTallyObserver(b, drop_keys=("at",)).snapshot()
    )


def test_the_dropped_keys_are_recorded_in_the_snapshot(tmp_path):
    root = str(tmp_path / "d")
    _write(root, "r.json", {"a": 1})
    snapshot = DirectoryTallyObserver(root, drop_keys=("b", "c.d")).snapshot()
    # A reader can see what was given up, and widening the projection changes
    # the report rather than passing unnoticed.
    assert snapshot["dropped"] == ["b", "c.d"]


def test_a_missing_directory_is_absent_rather_than_an_error(tmp_path):
    snapshot = DirectoryTallyObserver(str(tmp_path / "nope")).snapshot()
    assert snapshot["__present__"] is False


def test_an_unparsable_file_is_compared_by_bytes_not_skipped(tmp_path):
    root = str(tmp_path / "u")
    _write(root, "broken.json", "{not json")
    snapshot = DirectoryTallyObserver(root).snapshot()
    # Skipping it would make a malformed file look like no file at all.
    assert snapshot["count"] == 1
    assert snapshot["payloads"][0].startswith("unparsed:")


def test_only_matching_patterns_are_counted(tmp_path):
    root = str(tmp_path / "m")
    _write(root, "keep.json", {"v": 1})
    _write(root, "ignore.log", "noise")
    snapshot = DirectoryTallyObserver(root, patterns=("*.json",)).snapshot()
    assert snapshot["count"] == 1


def test_nested_directories_are_walked(tmp_path):
    root = str(tmp_path / "n")
    _write(os.path.join(root, "deep"), "r.json", {"v": 1})
    assert DirectoryTallyObserver(root).snapshot()["count"] == 1


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert "\u2014" not in handle.read()
