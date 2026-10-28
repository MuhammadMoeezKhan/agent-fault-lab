"""Tests for whether a duplicate is visible through a server's read tools.

This is the study's own question, and it is not the same as whether the
effect is visible at all. One server keys its record and appends to a history
inside it: its read tool answers with one submission after one application
and after two, so a client can confirm the write landed and cannot discover
that it landed twice.

The measurement is confounded by time. S1 and S2 are separate trials at
different wall clock times, so any timestamp in a read tool's output differs
whether or not anything was duplicated. Comparing raw output reported that
server's duplicate as visible when what had moved was the clock, which is why
the answer is three-valued and the keys to ignore are declared in the spec.
"""
from __future__ import annotations

from agentfaultlab.audit import _duplicate_visible, _strip_keys


def test_identical_output_means_the_duplicate_is_hidden():
    once = '{"count": 1, "items": ["a"]}'
    twice = '{"count": 1, "items": ["a"]}'
    # Reliable without any declaration: a duplicate that showed would have
    # made these differ.
    assert _duplicate_visible(once, twice, ()) is False


def test_a_moving_timestamp_alone_is_not_a_visible_duplicate():
    once = '{"count": 1, "updated": "2026-10-28T16:59:50Z"}'
    twice = '{"count": 1, "updated": "2026-10-28T17:02:11Z"}'
    # With the stamp declared, the real answer comes through.
    assert _duplicate_visible(once, twice, ("updated",)) is False


def test_a_real_difference_counts_when_drops_are_declared():
    once = '{"count": 1, "updated": "t1"}'
    twice = '{"count": 2, "updated": "t2"}'
    assert _duplicate_visible(once, twice, ("updated",)) is True


def test_a_difference_with_nothing_declared_is_unknown():
    once = '{"count": 1, "updated": "t1"}'
    twice = '{"count": 1, "updated": "t2"}'
    # Could be a duplicate, could be the clock. Recorded as unknown and left
    # out of the denominator rather than resolved in either direction, which
    # is the whole reason this returns None instead of True.
    assert _duplicate_visible(once, twice, ()) is None


def test_a_missing_probe_output_is_unknown():
    assert _duplicate_visible(None, '{"a": 1}', ("x",)) is None
    assert _duplicate_visible('{"a": 1}', None, ("x",)) is None


def test_dropping_reaches_nested_structures_and_lists():
    payload = {"a": 1, "ts": "t", "rows": [{"v": 1, "ts": "t"}, {"v": 2, "ts": "u"}]}
    assert _strip_keys(payload, ("ts",)) == {
        "a": 1,
        "rows": [{"v": 1}, {"v": 2}],
    }


def test_a_list_losing_an_entry_still_differs_after_dropping():
    once = '{"history": [{"status": "s", "at": "t1"}]}'
    twice = '{"history": [{"status": "s", "at": "t1"}, {"status": "s", "at": "t2"}]}'
    # The surviving fields of the second entry are identical to the first, so
    # only the list length carries the duplicate. It must survive the drop.
    assert _duplicate_visible(once, twice, ("at",)) is True


def test_non_json_output_is_compared_as_text():
    assert _duplicate_visible("one row", "one row", ()) is False
    assert _duplicate_visible("one row", "two rows", ("ts",)) is True


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert "\u2014" not in handle.read()
