"""Tests for resolving a glob in a sqlite observer's path.

Some servers keep their store under a directory named for the machine or the
account that created it. Writing that name into a spec makes the spec work on
one computer only, and puts the operator's account name in a public file. A
glob segment fixes both. The risk it introduces is matching the wrong store,
or matching several and quietly taking one, so that is what is pinned here.
"""
from __future__ import annotations

import os
import sqlite3

from agentfaultlab.observers.sqlite import SqliteObserver


def _db(path, rows=("a",)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("create table notes (body text)")
    c.executemany("insert into notes values (?)", [(r,) for r in rows])
    c.commit()
    c.close()
    return path


def test_a_glob_finds_a_store_under_a_machine_named_directory(tmp_path):
    _db(str(tmp_path / "memory" / "auditor-somehost" / "log.sqlite3"), ("one",))
    observer = SqliteObserver(str(tmp_path / "memory" / "*" / "log.sqlite3"))
    snapshot = observer.snapshot()
    assert snapshot.get("__present__") is not False
    assert "one" in repr(snapshot)


def test_an_exact_path_still_works_and_is_not_globbed(tmp_path):
    path = _db(str(tmp_path / "plain" / "log.sqlite3"))
    observer = SqliteObserver(path)
    assert observer.resolve() == os.path.abspath(path)


def test_a_missing_store_reports_absent_rather_than_raising(tmp_path):
    observer = SqliteObserver(str(tmp_path / "memory" / "*" / "log.sqlite3"))
    assert observer.snapshot() == {"__present__": False}


def test_several_matches_are_refused_and_recorded(tmp_path):
    _db(str(tmp_path / "memory" / "one" / "log.sqlite3"), ("x",))
    _db(str(tmp_path / "memory" / "two" / "log.sqlite3"), ("y",))
    observer = SqliteObserver(str(tmp_path / "memory" / "*" / "log.sqlite3"))
    snapshot = observer.snapshot()
    # Not silently the first by sort order: the ambiguity goes into the
    # snapshot, so baselines differ and the row comes out unproven instead of
    # confidently wrong about whichever store happened to sort first.
    assert snapshot["__present__"] is False
    assert sorted(snapshot["__ambiguous__"]) == ["log.sqlite3", "log.sqlite3"]


def test_a_store_appearing_later_is_picked_up_by_the_same_observer(tmp_path):
    # The snapshot is taken before and after the call, and the store does not
    # exist until the server writes it, so resolution has to happen per
    # snapshot rather than once at construction.
    observer = SqliteObserver(str(tmp_path / "memory" / "*" / "log.sqlite3"))
    assert observer.snapshot() == {"__present__": False}
    _db(str(tmp_path / "memory" / "auditor-host" / "log.sqlite3"), ("later",))
    assert observer.snapshot().get("__present__") is not False


def _store(path, rows):
    connection = sqlite3.connect(path)
    connection.execute("create table messages (id text, text text)")
    connection.executemany("insert into messages values (?, ?)", rows)
    connection.commit()
    connection.close()


def test_unnamed_mode_reads_every_match_and_ignores_the_names(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _store(tmp_path / "a" / "x1.sqlite", [("1", "hello")])
    _store(tmp_path / "a" / "x2.sqlite", [("1", "bye")])
    _store(tmp_path / "b" / "zz.sqlite", [("1", "bye")])
    _store(tmp_path / "b" / "yy.sqlite", [("1", "hello")])
    first = SqliteObserver(str(tmp_path / "a" / "*.sqlite"), unnamed=True).snapshot()
    second = SqliteObserver(str(tmp_path / "b" / "*.sqlite"), unnamed=True).snapshot()
    assert first == second
    assert len(first["__unnamed__"]) == 2


def test_unnamed_mode_sees_a_second_store_as_a_different_state(tmp_path):
    _store(tmp_path / "one.sqlite", [("1", "hello")])
    observer = SqliteObserver(str(tmp_path / "*.sqlite"), unnamed=True)
    once = observer.snapshot()
    _store(tmp_path / "two.sqlite", [("1", "hello")])
    assert observer.snapshot() != once


def test_unnamed_mode_with_no_match_is_absent(tmp_path):
    observer = SqliteObserver(str(tmp_path / "*.sqlite"), unnamed=True)
    assert observer.snapshot() == {"__present__": False}


def test_a_spec_can_ask_for_unnamed_mode(tmp_path):
    from agentfaultlab.spec import load

    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"server": {"name": "x", "command": ["true"], "observer": '
        '{"kind": "sqlite", "path": "p/*.sqlite", "unnamed": true}},'
        ' "fixtures": [{"tool": "t", "arguments": {}}]}'
    )
    target, _ = load(str(spec))
    assert target.observer(str(tmp_path)).unnamed is True


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert "\u2014" not in handle.read()
