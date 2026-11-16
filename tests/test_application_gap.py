"""A declared pause between applications, for servers that name by the clock.

pdf-it names an output it places itself by the second it was written, so two
applications inside one second overwrite each other and two a second apart
leave two files. Whether a repeat lands twice is then decided by where the
second boundary falls. These tests pin the contract of the field that lets a
spec declare the client's pacing: zero changes nothing, a gap is applied
before every application after the first and before a retry, and every row
from such a fixture says so.
"""

import time

from agentfaultlab.audit import Audit, Fixture
from agentfaultlab.spec import load as load_spec

from test_mcp_audit import make_target


def test_zero_is_the_default_and_the_preregistered_pacing():
    assert Fixture(tool="append_line", arguments={}).application_gap_s == 0.0


def test_the_gap_separates_the_two_clean_applications():
    audit = Audit(make_target(), repeats=1, settle_s=0.0)
    fixture = Fixture(
        tool="append_line",
        arguments={"path": "{root}/notes.txt", "line": "one"},
        application_gap_s=0.4,
    )
    started = time.monotonic()
    baselines = audit.baselines(fixture)
    elapsed = time.monotonic() - started
    # One gap in clean_twice; clean_once and clean_once_again have none.
    assert elapsed >= 0.4
    assert baselines["repeat_distinguishable"]


def test_every_row_from_a_gapped_fixture_carries_the_value():
    audit = Audit(make_target(), repeats=1, settle_s=0.0)
    fixture = Fixture(
        tool="append_line",
        arguments={"path": "{root}/notes.txt", "line": "one"},
        application_gap_s=0.1,
    )
    rows = [verdict.as_row() for verdict in audit.run_tool(fixture, modes=("F7", "F1"))]
    assert rows and all("application gap 0.1s declared" in row["notes"] for row in rows)


def test_an_ungapped_fixture_says_nothing_about_pacing():
    audit = Audit(make_target(), repeats=1, settle_s=0.0)
    fixture = Fixture(
        tool="append_line", arguments={"path": "{root}/notes.txt", "line": "one"}
    )
    rows = [verdict.as_row() for verdict in audit.run_tool(fixture, modes=("F7",))]
    assert all("application gap" not in row["notes"] for row in rows)


def test_a_spec_can_declare_it(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"server": {"name": "x", "command": ["true"], "observer": {"kind": "filesystem"}},'
        ' "fixtures": [{"tool": "t", "arguments": {}, "application_gap_s": 1.1}]}'
    )
    _target, fixtures = load_spec(str(spec))
    assert fixtures[0].application_gap_s == 1.1
