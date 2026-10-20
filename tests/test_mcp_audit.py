"""Tests for the MCP fault proxy, the observers and the replay oracle.

Everything here runs against `fake_server.py`, so the suite needs no network,
no API key and no npx. The point of these tests is that the audit's verdicts
are checked against a server whose behaviour is known by construction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from agentfaultlab.audit import (
    DUPLICATED,
    MECHANISM_ABSORBED,
    MECHANISM_UNKNOWN,
    MECHANISM_NONE,
    MECHANISM_REJECTED,
    SAFE,
    Audit,
    Fixture,
    ServerTarget,
    classify,
)
from agentfaultlab.mcp import FaultPlan, ServerSession, taxonomy
from agentfaultlab.mcp.injector import Action, FaultInjector, truncate_list_argument
from agentfaultlab.observers import FilesystemObserver, GitObserver, SqliteObserver, digest

FAKE_SERVER = os.path.join(os.path.dirname(__file__), "fake_server.py")


def make_target(**overrides) -> ServerTarget:
    defaults = dict(
        name="fake",
        command=[sys.executable, FAKE_SERVER],
        observer=lambda root: FilesystemObserver(root, ignore=(".keys.json", ".DS_Store")),
        timeout_s=10.0,
    )
    defaults.update(overrides)
    return ServerTarget(**defaults)


# ------------------------------------------------------------------- taxonomy

def test_every_mode_is_reachable_by_id_and_name():
    for id_ in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        mode = taxonomy.get(id_)
        assert taxonomy.get(mode.name) is mode
        assert mode.provenance


def test_unknown_mode_names_the_known_ones():
    with pytest.raises(KeyError, match="F1"):
        taxonomy.get("F99")


# -------------------------------------------------------------------- planning

def test_plan_parses_tool_and_occurrences():
    plan = FaultPlan.parse("F1:write_file@2,3")
    assert plan.fault.id == "F1"
    assert plan.tool == "write_file"
    assert plan.occurrences == (2, 3)


def test_plan_parses_all_as_a_probability_of_one():
    plan = FaultPlan.parse("F6:append_line@all")
    assert plan.occurrences == ()
    assert plan.probability == 1.0


def test_plan_rejects_both_occurrences_and_probability():
    with pytest.raises(ValueError, match="not both"):
        FaultPlan(mode="F1", probability=0.5, occurrences=(1,))


def test_injector_only_hits_the_named_occurrence():
    injector = FaultInjector(FaultPlan.parse("F1:append_line@2"))
    call = {"method": "tools/call", "params": {"name": "append_line", "arguments": {}}}
    other = {"method": "tools/call", "params": {"name": "read_file", "arguments": {}}}
    assert not injector.decide(other).injected
    assert not injector.decide(call).injected
    assert injector.decide(call).injected
    assert not injector.decide(call).injected


def test_injector_ignores_everything_that_is_not_a_tool_call():
    injector = FaultInjector(FaultPlan.parse("F1"))
    assert not injector.decide({"method": "tools/list", "id": 1}).injected


def test_probabilistic_injection_is_seeded():
    def draws(seed: int) -> list[bool]:
        injector = FaultInjector(
            FaultPlan(mode="F1", occurrences=(), probability=0.5, seed=seed)
        )
        call = {"method": "tools/call", "params": {"name": "t", "arguments": {}}}
        return [injector.decide(call).injected for _ in range(20)]

    assert draws(7) == draws(7)
    assert draws(7) != draws(8)


def test_truncate_picks_the_longest_list_argument():
    message = {
        "method": "tools/call",
        "params": {"name": "t", "arguments": {"a": [1], "lines": [1, 2, 3], "x": "s"}},
    }
    trimmed, detail = truncate_list_argument(message, keep=1)
    assert trimmed["params"]["arguments"]["lines"] == [1]
    assert detail == {"argument": "lines", "from": 3, "to": 1}
    # The original is untouched, so a retry can send the full call.
    assert message["params"]["arguments"]["lines"] == [1, 2, 3]


def test_truncate_reports_nothing_to_do():
    message = {"method": "tools/call", "params": {"name": "t", "arguments": {"a": "s"}}}
    assert truncate_list_argument(message, keep=1)[1] == {}


# -------------------------------------------------------------------- observers

def test_filesystem_observer_sees_content_not_mtime(tmp_path):
    observer = FilesystemObserver(str(tmp_path))
    target = tmp_path / "a.txt"
    target.write_text("one")
    first = observer.snapshot()
    os.utime(target, (0, 0))
    assert observer.snapshot() == first
    target.write_text("two")
    assert observer.snapshot() != first


def test_filesystem_observer_is_canonical_across_two_roots(tmp_path):
    for name in ("left", "right"):
        root = tmp_path / name
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "a.txt").write_text("same")
    left = FilesystemObserver(str(tmp_path / "left")).snapshot()
    right = FilesystemObserver(str(tmp_path / "right")).snapshot()
    assert digest(left) == digest(right)


def test_sqlite_observer_counts_duplicate_rows(tmp_path):
    import sqlite3

    path = tmp_path / "app.db"
    observer = SqliteObserver(str(path))
    assert observer.snapshot() == {"__present__": False}
    connection = sqlite3.connect(path)
    connection.execute("create table charges (id integer, amount integer)")
    connection.commit()
    once = observer.snapshot()
    connection.execute("insert into charges values (1, 10)")
    connection.commit()
    twice_start = observer.snapshot()
    connection.execute("insert into charges values (1, 10)")
    connection.commit()
    connection.close()
    assert once != twice_start != observer.snapshot()
    assert observer.snapshot()["charges"]["rows"] == [[1, 10], [1, 10]]


def test_git_observer_ignores_commit_hashes_but_not_commits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                       capture_output=True)

    git("init", "-q")
    (repo / "a.txt").write_text("one")
    git("add", "a.txt")
    git("commit", "-qm", "first")
    observer = GitObserver(str(repo))
    first = observer.snapshot()
    assert first["commit_count"] == 1
    git("commit", "-qm", "empty", "--allow-empty")
    second = observer.snapshot()
    assert second["commit_count"] == 2
    assert ["empty", first["commits"][0][1], 1] in second["commits"]


def test_classify_prefers_once_then_twice_then_none():
    assert classify({"a": 1}, {}, {"a": 1}, {"a": 2}) == "once"
    assert classify({"a": 2}, {}, {"a": 1}, {"a": 2}) == "twice"
    assert classify({}, {}, {"a": 1}, {"a": 2}) == "none"
    assert classify({"z": 9}, {}, {"a": 1}, {"a": 2}) == "other"


# ---------------------------------------------------------------------- session

def test_session_lists_the_fake_server_tools(tmp_path):
    with ServerSession([sys.executable, FAKE_SERVER], cwd=str(tmp_path)) as session:
        names = {tool.name for tool in session.list_tools()}
        assert {"append_line", "read_file", "secret_append"} <= names
        assert session.server_info["serverInfo"]["name"] == "fake-server"


def test_hint_accessors_read_annotations(tmp_path):
    with ServerSession([sys.executable, FAKE_SERVER], cwd=str(tmp_path)) as session:
        tools = {tool.name: tool for tool in session.list_tools()}
    assert tools["secret_append"].read_only_hint is True
    assert tools["append_line"].idempotent_hint is True
    assert tools["read_file"].destructive_hint is None
    assert tools["append_many"].list_argument_names() == ["lines"]


def test_f1_commits_the_write_but_tells_the_caller_nothing(tmp_path):
    plan = FaultPlan.parse("F1:append_line@1")
    with ServerSession(
        [sys.executable, FAKE_SERVER], cwd=str(tmp_path), plan=plan
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    assert call.timed_out is True
    assert call.caller_saw_success is False
    # The server did the work, and the audit can prove it two ways.
    assert call.withheld_result is not None
    assert (tmp_path / "log.txt").read_text() == "x\n"


def test_f6_tells_the_caller_success_and_changes_nothing(tmp_path):
    plan = FaultPlan.parse("F6:append_line@1")
    with ServerSession(
        [sys.executable, FAKE_SERVER], cwd=str(tmp_path), plan=plan
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    assert call.caller_saw_success is True
    assert not (tmp_path / "log.txt").exists()


def test_f7_never_reaches_the_server(tmp_path):
    plan = FaultPlan.parse("F7:append_line@1")
    with ServerSession(
        [sys.executable, FAKE_SERVER], cwd=str(tmp_path), plan=plan
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    assert call.error is not None
    assert not (tmp_path / "log.txt").exists()


def test_f2_applies_the_effect_twice(tmp_path):
    plan = FaultPlan.parse("F2:append_line@1")
    with ServerSession(
        [sys.executable, FAKE_SERVER], cwd=str(tmp_path), plan=plan
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
        session.call_tool("read_file", {"path": "log.txt"})
    assert call.caller_saw_success is True
    assert (tmp_path / "log.txt").read_text() == "x\nx\n"


def test_f3_lands_after_the_caller_has_given_up(tmp_path):
    plan = FaultPlan(mode="F3", tool="append_line", delay_s=0.3)
    with ServerSession(
        [sys.executable, FAKE_SERVER], cwd=str(tmp_path), plan=plan
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
        assert call.timed_out is True
        assert not (tmp_path / "log.txt").exists()
        import time

        time.sleep(1.2)
        assert (tmp_path / "log.txt").read_text() == "x\n"


def test_f5_commits_a_prefix_of_the_batch(tmp_path):
    plan = FaultPlan(mode="F5", tool="append_many", keep=1)
    with ServerSession(
        [sys.executable, FAKE_SERVER], cwd=str(tmp_path), plan=plan
    ) as session:
        call = session.call_tool(
            "append_many", {"path": "log.txt", "lines": ["a", "b", "c"]}
        )
    assert call.timed_out is True
    assert (tmp_path / "log.txt").read_text() == "a\n"


def test_trace_records_calls_and_messages(tmp_path):
    from agentfaultlab.mcp import TraceWriter, iter_trace

    path = tmp_path / "trace.jsonl"
    with TraceWriter(str(path), context={"server": "fake"}) as trace:
        with ServerSession(
            [sys.executable, FAKE_SERVER],
            cwd=str(tmp_path),
            plan=FaultPlan.parse("F1:append_line@1"),
            trace=trace,
        ) as session:
            session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    records = list(iter_trace(str(path)))
    assert records[0]["event"] == "open"
    calls = [record for record in records if record["event"] == "call"]
    assert calls[0]["fault"] == "F1:lost_response_after_commit"
    assert calls[0]["timed_out"] is True
    assert any(record.get("method") == "tools/call" for record in records)


# ------------------------------------------------------------------------ audit

@pytest.mark.parametrize("mode", ["F1", "F2", "F3"])
def test_audit_catches_the_duplicating_tool(mode):
    audit = Audit(make_target(), repeats=2, settle_s=0.5)
    fixture = Fixture(
        tool="append_line",
        arguments={"path": "log.txt", "line": "charge"},
        probes=(("read_file", {"path": "log.txt"}),),
    )
    tools = {tool.name: tool for tool in audit.list_tools()}
    verdict = audit.run_tool(fixture, tools["append_line"], modes=[mode])[0]
    assert verdict.verdict == DUPLICATED
    assert verdict.mutating is True
    assert verdict.confirmable is True
    assert any("idempotentHint" in note for note in verdict.annotation_violations)


def test_audit_passes_the_tool_with_a_key():
    audit = Audit(make_target(), repeats=2)
    fixture = Fixture(
        tool="append_with_key",
        arguments={"path": "log.txt", "line": "charge", "idempotency_key": "k-1"},
    )
    tools = {tool.name: tool for tool in audit.list_tools()}
    verdict = audit.run_tool(fixture, tools["append_with_key"], modes=["F1"])[0]
    assert verdict.verdict == SAFE
    assert verdict.dedup_argument == "idempotency_key"
    # The key makes a repeat a no-op, which is indistinguishable from state.
    assert verdict.repeat_distinguishable is False


def test_audit_says_when_an_overwrite_cannot_show_a_duplicate():
    audit = Audit(make_target(), repeats=1)
    fixture = Fixture(tool="write_file", arguments={"path": "a.txt", "text": "hello"})
    verdict = audit.run_tool(fixture, modes=["F1"])[0]
    # A duplicated overwrite leaves no duplicate effect, so this is a pass,
    # but the trial could not have detected one and the row says so.
    assert verdict.verdict == SAFE
    assert verdict.repeat_distinguishable is False
    assert any("not distinguishable" in note for note in verdict.notes)


def test_audit_flags_the_unconfirmable_tool_that_claims_read_only():
    audit = Audit(make_target(), repeats=2)
    fixture = Fixture(
        tool="secret_append",
        arguments={"line": "charge"},
        probes=(("list_files", {}),),
    )
    tools = {tool.name: tool for tool in audit.list_tools()}
    verdict = audit.run_tool(fixture, tools["secret_append"], modes=["F1"])[0]
    assert verdict.verdict == DUPLICATED
    assert verdict.confirmable is False
    assert verdict.caller_left_unsure is True
    assert "readOnlyHint:true but the backend changed" in verdict.annotation_violations


def test_safe_by_rejection_is_not_reported_as_the_same_pass_as_dedup():
    """A refused repeat keeps the state right but leaves the agent guessing."""
    audit = Audit(make_target(), repeats=2)
    tools = {tool.name: tool for tool in audit.list_tools()}

    rejecting = audit.run_tool(
        Fixture(tool="create_file", arguments={"path": "a.txt", "text": "x"}),
        tools["create_file"],
        modes=["F1"],
    )[0]
    assert rejecting.verdict == SAFE
    assert rejecting.safety_mechanism == MECHANISM_REJECTED
    assert any("refused the repeat" in note for note in rejecting.notes)

    # A tool that really deduplicates leaves one application looking exactly
    # like two, so state cannot prove the repeat was absorbed rather than
    # quietly refused. The verdict is SAFE and the mechanism is honest about
    # not being decidable; the declared key is reported separately.
    deduplicating = audit.run_tool(
        Fixture(
            tool="append_with_key",
            arguments={"path": "log.txt", "line": "x", "idempotency_key": "k-1"},
        ),
        tools["append_with_key"],
        modes=["F1"],
    )[0]
    assert deduplicating.verdict == SAFE
    assert deduplicating.safety_mechanism == MECHANISM_UNKNOWN
    assert not deduplicating.repeat_distinguishable
    assert deduplicating.dedup_argument


def test_absorption_is_claimed_only_when_two_applications_would_have_shown():
    """`absorbed_repeat` needs state evidence: S2 must differ from S1."""
    audit = Audit(make_target(), repeats=1)
    tools = {tool.name: tool for tool in audit.list_tools()}
    absorbing = audit.run_tool(
        Fixture(tool="write_file", arguments={"path": "a.txt", "text": "x"}),
        tools["write_file"],
        modes=["F1"],
    )[0]
    # write_file overwrites, so one application and two look the same and the
    # mechanism is not decidable from state either.
    assert absorbing.verdict == SAFE
    assert not absorbing.repeat_distinguishable
    assert absorbing.safety_mechanism in (MECHANISM_UNKNOWN,)


def test_duplicating_tool_has_no_safety_mechanism():
    audit = Audit(make_target(), repeats=1)
    verdict = audit.run_tool(
        Fixture(tool="append_line", arguments={"path": "log.txt", "line": "x"}),
        modes=["F1"],
    )[0]
    assert verdict.verdict == DUPLICATED
    assert verdict.safety_mechanism == MECHANISM_NONE


def test_control_fault_is_safe_for_the_duplicating_tool():
    audit = Audit(make_target(), repeats=2)
    fixture = Fixture(tool="append_line", arguments={"path": "log.txt", "line": "x"})
    verdict = audit.run_tool(fixture, modes=["F7"])[0]
    assert verdict.verdict == SAFE


def test_audit_rows_are_flat_and_serialisable():
    audit = Audit(make_target(), repeats=1)
    fixture = Fixture(tool="append_line", arguments={"path": "log.txt", "line": "x"})
    row = audit.run_tool(fixture, modes=["F1"])[0].as_row()
    assert json.loads(json.dumps(row))["tool"] == "append_line"


# ------------------------------------------------------------------------ proxy

def proxy_command(fault: str, trace: str | None = None) -> list[str]:
    """The proxy, run as a real subprocess, wrapping the fake server."""
    command = [sys.executable, "-m", "agentfaultlab.cli", "proxy", "--fault", fault]
    if trace:
        command += ["--trace", trace]
    return command + ["--", sys.executable, FAKE_SERVER]


def test_proxy_passes_the_handshake_and_tool_list_through(tmp_path):
    with ServerSession(proxy_command("F1:nothing@1"), cwd=str(tmp_path)) as session:
        assert session.server_info["serverInfo"]["name"] == "fake-server"
        assert "append_line" in {tool.name for tool in session.list_tools()}


def test_proxy_withholds_the_response_but_the_write_lands(tmp_path):
    with ServerSession(
        proxy_command("F1:append_line@1"), cwd=str(tmp_path), timeout_s=2.0
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    assert call.timed_out is True
    assert (tmp_path / "log.txt").read_text() == "x\n"


def test_proxy_only_hits_the_named_occurrence(tmp_path):
    with ServerSession(
        proxy_command("F1:append_line@1"), cwd=str(tmp_path), timeout_s=2.0
    ) as session:
        first = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
        second = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    assert first.timed_out is True
    assert second.caller_saw_success is True
    assert (tmp_path / "log.txt").read_text() == "x\nx\n"


def test_proxy_synthesises_success_without_touching_the_backend(tmp_path):
    with ServerSession(
        proxy_command("F6:append_line@1"), cwd=str(tmp_path), timeout_s=5.0
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    assert call.caller_saw_success is True
    assert not (tmp_path / "log.txt").exists()


def test_proxy_duplicate_delivery_gives_the_client_one_answer(tmp_path):
    with ServerSession(
        proxy_command("F2:append_line@1"), cwd=str(tmp_path), timeout_s=5.0
    ) as session:
        call = session.call_tool("append_line", {"path": "log.txt", "line": "x"})
        # The shadow copy's answer must not surface as a second response, so a
        # following call still matches its own request.
        echo = session.call_tool("read_file", {"path": "log.txt"})
    assert call.caller_saw_success is True
    assert echo.result["content"][0]["text"] == "x\nx\n"


def test_proxy_trace_records_the_injection_and_the_withheld_response(tmp_path):
    from agentfaultlab.mcp import iter_trace

    trace_path = tmp_path / "proxy.jsonl"
    with ServerSession(
        proxy_command("F1:append_line@1", str(trace_path)),
        cwd=str(tmp_path),
        timeout_s=2.0,
    ) as session:
        session.call_tool("append_line", {"path": "log.txt", "line": "x"})
    import time

    time.sleep(0.3)
    events = [record["event"] for record in iter_trace(str(trace_path))]
    assert "inject" in events
    assert "withheld" in events


def test_sqlite_observer_ignores_create_statement_formatting(tmp_path):
    """Regression: raw CREATE text made identical tables read as different.

    An agent asked to create a table writes whatever casing it likes, which
    was landing real episodes in the `other` bucket.
    """
    import sqlite3

    from agentfaultlab.observers import SqliteObserver, digest

    digests = []
    for index, statement in enumerate(
        [
            "create table refunds (id text, amount integer)",
            "CREATE TABLE refunds (id TEXT, amount INTEGER)",
            "CREATE  TABLE   refunds\n  ( id  Text ,\n amount  Integer )",
        ]
    ):
        path = tmp_path / f"db{index}.sqlite"
        connection = sqlite3.connect(path)
        connection.executescript(statement)
        connection.commit()
        connection.close()
        digests.append(digest(SqliteObserver(str(path)).snapshot()))
    assert len(set(digests)) == 1, digests


def test_sqlite_observer_still_sees_a_real_schema_change(tmp_path):
    import sqlite3

    from agentfaultlab.observers import SqliteObserver, digest

    digests = []
    for index, statement in enumerate(
        [
            "create table refunds (id text, amount integer)",
            "create table refunds (id text, amount text)",
            "create table refunds (id text, amount integer, note text)",
            "create table refunds (id text primary key, amount integer)",
        ]
    ):
        path = tmp_path / f"db{index}.sqlite"
        connection = sqlite3.connect(path)
        connection.executescript(statement)
        connection.commit()
        connection.close()
        digests.append(digest(SqliteObserver(str(path)).snapshot()))
    assert len(set(digests)) == 4, digests


# -------------------------------------------------------------------- prelude

def test_prelude_runs_before_the_snapshot_so_its_effect_is_part_of_s0():
    """A prelude is session setup, not an effect under test.

    memorix refuses every project-scoped tool until `memorix_session_start`
    has bound the session to a git-backed root, so some servers cannot be
    audited at all without this. Its writes have to land before the first
    snapshot, or every trial would score the prelude's own effect.
    """
    target = make_target(
        prelude=[("append_line", {"path": "{root}/prelude.txt", "line": "bound"})]
    )
    audit = Audit(target, repeats=1, settle_s=0.0)
    fixture = Fixture(
        tool="append_line",
        arguments={"path": "{root}/notes.txt", "line": "one"},
    )
    baselines = audit.baselines(fixture)
    s1 = baselines["s1"]
    assert any(path.endswith("prelude.txt") for path in s1)
    # S1 differs from S0 only by the fixture's own file.
    s0 = baselines["s0"]
    assert set(s1) - set(s0) == {"notes.txt"}


def test_a_failing_prelude_aborts_the_trial_instead_of_reading_as_a_lost_effect():
    target = make_target(
        prelude=[("append_line", {"path": "/does/not/exist/x.txt", "line": "x"})]
    )
    audit = Audit(target, repeats=1, settle_s=0.0)
    fixture = Fixture(
        tool="append_line", arguments={"path": "{root}/notes.txt", "line": "one"}
    )
    result, _, _, _ = audit._trial(
        fixture, condition="clean_once", plan=None, applications=1,
        retry_if_unsure=False,
    )
    assert result.error and "prelude call append_line failed" in result.error


# ------------------------------------------------------- sqlite ignore_columns

def _db(tmp_path, statements):
    import sqlite3

    path = tmp_path / "app.db"
    connection = sqlite3.connect(path)
    for statement in statements:
        connection.execute(statement)
    connection.commit()
    connection.close()
    return str(path)


def test_sqlite_observer_can_ignore_volatile_columns(tmp_path):
    path = _db(
        tmp_path,
        [
            "create table notes (id integer primary key, body text, created_at text)",
            "insert into notes (body, created_at) values ('one', '2026-01-01T00:00:00')",
        ],
    )
    strict = SqliteObserver(path).snapshot()
    relaxed = SqliteObserver(path, ignore_columns=("id", "created_at")).snapshot()
    assert strict["notes"]["rows"] == [[1, "one", "2026-01-01T00:00:00"]]
    assert relaxed["notes"]["rows"] == [["one"]]
    assert relaxed["notes"]["ignored_columns"] == ["id", "created_at"]
    assert [column[0] for column in relaxed["notes"]["columns"]] == ["body"]


def test_ignoring_a_row_id_still_shows_a_duplicate_as_two_rows(tmp_path):
    path = _db(
        tmp_path,
        [
            "create table notes (id integer primary key, body text)",
            "insert into notes (body) values ('one')",
            "insert into notes (body) values ('one')",
        ],
    )
    snapshot = SqliteObserver(path, ignore_columns=("id",)).snapshot()
    assert snapshot["notes"]["rows"] == [["one"], ["one"]]


def test_ignore_columns_can_be_qualified_by_table(tmp_path):
    path = _db(
        tmp_path,
        [
            "create table a (seen text, body text)",
            "create table b (seen text, body text)",
            "insert into a values ('t', 'x')",
            "insert into b values ('t', 'y')",
        ],
    )
    snapshot = SqliteObserver(path, ignore_columns=("a.seen",)).snapshot()
    assert snapshot["a"]["rows"] == [["x"]]
    assert snapshot["b"]["rows"] == [["t", "y"]]


def test_an_unreproducible_baseline_yields_no_verdict():
    """Two identical clean runs must agree, or the oracle has no baseline.

    mushroomdb keeps its graph in an append only log whose bytes differ
    between two identical runs, so comparing a faulted end state against S1
    or S2 produced DIVERGED rows that looked like findings. The audit now
    says UNPROVEN and runs no fault trials.
    """
    from agentfaultlab.audit import UNPROVEN

    audit = Audit(make_target(), repeats=2)
    verdicts = audit.run_tool(
        Fixture(
            tool="append_stamped_line",
            arguments={"path": "log.txt", "line": "charge"},
        ),
        modes=["F1", "F2"],
    )
    assert [verdict.verdict for verdict in verdicts] == [UNPROVEN, UNPROVEN]
    assert all(
        any("not observable canonically" in note for note in verdict.notes)
        for verdict in verdicts
    )
    # No trials ran, so there are no multiplicities to report.
    assert all(verdict.multiplicities == [] for verdict in verdicts)


def test_unproven_rows_are_left_out_of_the_rates():
    from agentfaultlab.audit import UNPROVEN
    from agentfaultlab.report import summarise

    audit = Audit(make_target(), repeats=1)
    rows = audit.run_tool(
        Fixture(tool="append_line", arguments={"path": "log.txt", "line": "x"}),
        modes=["F1"],
    ) + audit.run_tool(
        Fixture(
            tool="append_stamped_line",
            arguments={"path": "stamped.txt", "line": "x"},
        ),
        modes=["F1"],
    )
    stats = summarise(rows)
    assert stats["unproven"] == 1
    # One duplicating fixture of one testable fixture, not of two.
    assert stats["duplicate_rate"].startswith("100.0% (1/1)")
