"""Tests for counting compare-and-swap preconditions apart from dedup keys.

These are two different mechanisms and the study's recommendation depends on
not confusing them. An idempotency key makes a repeat safe by absorbing it
and answering success. A precondition makes a repeat safe by refusing it,
because the state it names has moved, and the caller gets back an error it
cannot distinguish from a real failure. Pooling them would report a safety
property while hiding its cost.
"""
from __future__ import annotations

from agentfaultlab.audit import (
    Fixture,
    _dedup_argument,
    _precondition_argument,
)
from agentfaultlab.mcp.session import Tool


def _tool(**properties):
    return Tool(
        name="edit_file",
        description="",
        input_schema={"type": "object", "properties": properties},
        annotations={},
    )


def test_a_precondition_is_not_counted_as_a_deduplication_key():
    tool = _tool(path={"type": "string"}, expected_hash={"type": "string"})
    fixture = Fixture(tool="edit_file", arguments={"path": "p", "expected_hash": "ab"})
    # The dedup detector must stay silent: no server in the sample takes an
    # idempotency key, and a precondition must not quietly make that false.
    assert _dedup_argument(tool, fixture) is None
    name, supplied = _precondition_argument(tool, fixture)
    assert name == "expected_hash"
    assert supplied is True


def test_offering_a_precondition_and_passing_one_are_counted_apart():
    tool = _tool(path={"type": "string"}, expected_hash={"type": "string"})
    # The same tool, called without the guard. The ecosystem fact (the tool
    # offers one) and the row fact (this call used one) differ, and the
    # difference is the whole comparison.
    without = Fixture(tool="edit_file", arguments={"path": "p"})
    name, supplied = _precondition_argument(tool, without)
    assert name == "expected_hash"
    assert supplied is False


def test_a_precondition_passed_but_absent_from_the_schema_still_counts():
    # A server may accept a guard it never advertises, which is exactly the
    # undeclared-mechanism case the study keeps finding.
    fixture = Fixture(tool="x", arguments={"if_match": "etag-1"})
    name, supplied = _precondition_argument(None, fixture)
    assert name == "if_match"
    assert supplied is True


def test_an_unrelated_argument_is_not_a_precondition():
    tool = _tool(path={"type": "string"}, content={"type": "string"})
    fixture = Fixture(tool="edit_file", arguments={"path": "p", "content": "c"})
    assert _precondition_argument(tool, fixture) == (None, False)


def test_hyphens_and_case_do_not_hide_a_guard():
    fixture = Fixture(tool="x", arguments={"If-Match": "v1"})
    name, supplied = _precondition_argument(None, fixture)
    assert name == "If-Match"
    assert supplied is True


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert "\u2014" not in handle.read()
