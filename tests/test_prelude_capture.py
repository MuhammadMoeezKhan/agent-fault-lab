"""Tests for capturing a value out of a prelude reply.

A prepare-then-commit tool mints a token in the prepare call and requires it
in the commit. The token is random per call, so a static spec cannot name it,
and without capture the whole shape is unauditable. That shape matters more
than one server: a token handed back by the server is the mechanism that
would make a retry safe, so being unable to audit it would leave the study
silent on the one fix it keeps pointing at.

The boundaries pinned here are the ones that would produce a wrong verdict
rather than an error: a capture that quietly fails would call the tool with a
literal "{tx}" and the row would read as a server refusal.
"""
from __future__ import annotations

import pytest

from agentfaultlab.audit import (
    CaptureError,
    Fixture,
    capture_from,
    extract,
    _substitute,
)
from agentfaultlab.spec import SpecError, from_document


def test_structured_content_is_preferred_over_the_text_body():
    result = {
        "structuredContent": {"transactionId": "tx_structured"},
        "content": [{"type": "text", "text": '{"transactionId": "tx_text"}'}],
    }
    # Both carry it; the structured one is what the server declared.
    assert capture_from(result, "transactionId") == "tx_structured"


def test_a_json_text_body_is_read_when_there_is_no_structured_content():
    result = {"content": [{"type": "text", "text": '{"transactionId": "tx_text"}'}]}
    assert capture_from(result, "transactionId") == "tx_text"


def test_structured_content_missing_the_path_falls_through_to_the_text():
    result = {
        "structuredContent": {"other": 1},
        "content": [{"type": "text", "text": '{"transactionId": "tx_text"}'}],
    }
    assert capture_from(result, "transactionId") == "tx_text"


def test_a_missing_path_raises_rather_than_returning_none():
    # Returning None would be substituted as "None" and the tool would be
    # called with a plausible-looking wrong token.
    with pytest.raises(CaptureError):
        capture_from({"structuredContent": {"a": 1}}, "transactionId")
    with pytest.raises(CaptureError):
        capture_from(None, "transactionId")


def test_dotted_and_indexed_paths():
    payload = {"result": {"ids": [{"value": "first"}, {"value": "second"}]}}
    assert extract(payload, "result.ids.1.value") == "second"
    with pytest.raises(CaptureError):
        extract(payload, "result.ids.9.value")


def test_a_whole_string_placeholder_keeps_the_captured_type():
    # A numeric id must not become the string "12": a server that validates
    # its own token type would refuse, and the row would read as a refusal.
    assert _substitute("{tx}", "/root", {"tx": 12}) == 12
    assert _substitute("id-{tx}", "/root", {"tx": 12}) == "id-12"


def test_substitution_handles_root_and_bindings_together():
    out = _substitute(
        {"path": "{root}/f", "token": "{tx}", "list": ["{tx}"]},
        "/trial",
        {"tx": "abc"},
    )
    assert out == {"path": "/trial/f", "token": "abc", "list": ["abc"]}


def test_a_server_prelude_cannot_capture():
    document = {
        "server": {
            "name": "s",
            "command": ["x"],
            "observer": {"kind": "filesystem"},
            "prelude": [["prepare", {}, {"tx": "transactionId"}]],
        },
        "fixtures": [{"tool": "commit", "arguments": {"id": "{tx}"}}],
    }
    # It runs once before any fixture, so there is nothing to bind into and
    # the binding would be shared across fixtures that each need their own.
    with pytest.raises(SpecError):
        from_document(document)


def test_a_fixture_prelude_capture_is_parsed_in_both_spellings():
    document = {
        "server": {"name": "s", "command": ["x"], "observer": {"kind": "filesystem"}},
        "fixtures": [
            {
                "tool": "commit",
                "arguments": {"id": "{tx}"},
                "prelude": [["prepare", {}, {"tx": "transactionId"}]],
            },
            {
                "tool": "commit",
                "id": "dict-form",
                "arguments": {"id": "{tx}"},
                "prelude": [
                    {
                        "tool": "prepare",
                        "arguments": {},
                        "capture": {"tx": "transactionId"},
                    }
                ],
            },
        ],
    }
    _, fixtures = from_document(document)
    for fixture in fixtures:
        assert fixture.prelude == (("prepare", {}, {"tx": "transactionId"}),)
        assert fixture.prelude_captures == [{"tx": "transactionId"}]


def test_a_capture_path_must_be_a_non_empty_string():
    document = {
        "server": {"name": "s", "command": ["x"], "observer": {"kind": "filesystem"}},
        "fixtures": [
            {
                "tool": "commit",
                "arguments": {},
                "prelude": [["prepare", {}, {"tx": ""}]],
            }
        ],
    }
    with pytest.raises(SpecError):
        from_document(document)


def test_the_fixture_resolves_arguments_against_the_bindings():
    fixture = Fixture(tool="commit", arguments={"id": "{tx}", "p": "{root}/x"})
    assert fixture.resolve("/trial", {"tx": "abc"}) == {"id": "abc", "p": "/trial/x"}


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert "\u2014" not in handle.read()
