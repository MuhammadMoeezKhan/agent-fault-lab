"""Aiming the fault at a later call of the tool under test.

Some servers prepare and commit through one tool name: a preview call mints a
single-use token and an apply call consumes it. The preview has to run in the
fixture's prelude, so the apply is the second call of the tool in the trial.
With the fault fixed at the first occurrence the preview would take it and
the trial would abort. These tests pin the contract of `fault_occurrence`:
the default is the preregistered first call, a spec must account for every
prelude call to the tool under test or be refused, and the fault then lands
on the fixture's own application.
"""

import pytest

from agentfaultlab.audit import Audit, Fixture
from agentfaultlab.observers import FilesystemObserver
from agentfaultlab.spec import SpecError
from agentfaultlab.spec import load as load_spec

from test_mcp_audit import make_target


def _target():
    return make_target(
        observer=lambda root: FilesystemObserver(
            root, ignore=(".keys.json", ".tokens.json", ".DS_Store")
        )
    )


def _staged_fixture(occurrence: int = 2) -> Fixture:
    return Fixture(
        tool="staged_append",
        arguments={
            "path": "notes.txt",
            "line": "one",
            "mode": "apply",
            "confirm_token": "{token}",
        },
        prelude=(
            (
                "staged_append",
                {"path": "notes.txt", "line": "one", "mode": "preview"},
                {"token": "confirmToken"},
            ),
        ),
        probes=(("read_file", {"path": "notes.txt"}),),
        fault_occurrence=occurrence,
    )


def _spec(tmp_path, fixture_extra: str) -> str:
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"server": {"name": "x", "command": ["true"], "observer": {"kind": "filesystem"}},'
        ' "fixtures": [{"tool": "t", "arguments": {}, '
        '"prelude": [["t", {"mode": "preview"}, {"token": "confirmToken"}]]'
        + fixture_extra
        + "}]}"
    )
    return str(spec)


def test_the_first_call_is_the_default_and_the_preregistered_target():
    assert Fixture(tool="append_line").fault_occurrence == 1


def test_a_prelude_call_to_the_tool_under_test_is_still_refused_by_default(tmp_path):
    with pytest.raises(SpecError, match="fault_occurrence"):
        load_spec(_spec(tmp_path, ""))


def test_a_spec_that_accounts_for_the_prelude_call_loads(tmp_path):
    _target, fixtures = load_spec(_spec(tmp_path, ', "fault_occurrence": 2'))
    assert fixtures[0].fault_occurrence == 2


def test_a_count_that_disagrees_with_the_prelude_is_refused(tmp_path):
    with pytest.raises(SpecError, match="must agree"):
        load_spec(_spec(tmp_path, ', "fault_occurrence": 3'))


def test_zero_is_refused(tmp_path):
    with pytest.raises(SpecError, match="counted from 1"):
        load_spec(_spec(tmp_path, ', "fault_occurrence": 0'))


def test_a_server_prelude_call_cannot_be_accounted_for(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"server": {"name": "x", "command": ["true"], "observer": {"kind": "filesystem"},'
        ' "prelude": [["t", {}]]},'
        ' "fixtures": [{"tool": "t", "arguments": {}, "fault_occurrence": 2,'
        ' "prelude": [["t", {}]]}]}'
    )
    with pytest.raises(SpecError, match="server prelude"):
        load_spec(str(spec))


def test_the_fault_lands_on_the_apply_and_the_token_absorbs_the_retry():
    audit = Audit(_target(), repeats=1, settle_s=0.0)
    rows = [v.as_row() for v in audit.run_tool(_staged_fixture(), modes=("F1",))]
    (row,) = rows
    # The preview ran unfaulted, so the trial did not abort; the apply
    # committed and lost its reply; the retry reused the spent token and was
    # refused inside a success frame, so the line landed once.
    assert "prelude call" not in row["notes"]
    assert row["verdict"] == "SAFE"
    assert row["multiplicities"] == "once"
    assert "fault aimed at call 2 of staged_append" in row["notes"]


def test_aimed_at_the_first_call_the_preview_takes_the_fault():
    # The failure the field exists to prevent, shown directly: with the
    # default occurrence the prelude's preview is the faulted call.
    audit = Audit(_target(), repeats=1, settle_s=0.0)
    rows = [v.as_row() for v in audit.run_tool(_staged_fixture(1), modes=("F1",))]
    assert "prelude call staged_append failed" in rows[0]["notes"]


def test_the_test_file_writes_no_em_dashes():
    with open(__file__, encoding="utf-8") as handle:
        assert chr(0x2014) not in handle.read()
