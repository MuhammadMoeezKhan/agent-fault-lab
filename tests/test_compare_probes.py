"""Tests for comparing two probe runs candidate by candidate.

The startability rate is the breadth result this study leans on hardest, and
it was measured before a containment defect was fixed. The comparison exists
to find out whether that mattered, so the thing it must never do is report
agreement when individual candidates moved, or compare two runs that do not
cover the same candidates.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "compare_probes.py")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load():
    spec = importlib.util.spec_from_file_location("compare_probes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle)
    return path


def _started(name, tools=None):
    return {
        "server": name,
        "package": name,
        "attempts": [{"command": [name], "started": True}],
        "started_via": "registry",
        "tools": tools if tools is not None else [{"name": "t", "readOnlyHint": True}],
    }


def _failed(name, error):
    return {
        "server": name,
        "package": name,
        "attempts": [{"command": [name], "started": False, "error": error}],
        "started_via": None,
    }


def test_a_candidate_that_starts_after_the_fix_is_reported(tmp_path, capsys):
    mod = _load()
    before = _write(str(tmp_path / "b" / "1.json"), [_failed("a/x", "ENOENT /var/folders/zz/T")])
    after = _write(str(tmp_path / "a" / "1.json"), [_started("a/x")])
    mod.main(["--before", before, "--after", after])
    out = capsys.readouterr().out
    assert "changed class: 1 of 1" in out
    # Both error texts are shown, because a changed outcome needs attributing
    # and the class alone does not say why it moved.
    assert "ENOENT" in out
    assert "did-not-start" in out


def test_no_change_says_the_threat_is_retired(tmp_path, capsys):
    mod = _load()
    before = _write(str(tmp_path / "b" / "1.json"), [_started("a/x"), _failed("a/y", "boom")])
    after = _write(str(tmp_path / "a" / "1.json"), [_started("a/x"), _failed("a/y", "boom")])
    mod.main(["--before", before, "--after", after])
    out = capsys.readouterr().out
    assert "No candidate changed class" in out
    assert "changed class: 0 of 2" in out


def test_runs_covering_different_candidates_are_called_out(tmp_path, capsys):
    mod = _load()
    before = _write(str(tmp_path / "b" / "1.json"), [_started("a/x")])
    after = _write(str(tmp_path / "a" / "1.json"), [_started("a/y")])
    mod.main(["--before", before, "--after", after])
    out = capsys.readouterr().out
    # Rates from non-overlapping runs are meaningless, so this has to be loud
    # rather than a footnote.
    assert "NOT COMPARABLE" in out
    assert "only in the original: a/x" in out
    assert "only in the re-probe: a/y" in out


def test_a_regression_is_reported_as_readily_as_an_improvement(tmp_path, capsys):
    mod = _load()
    # The fix wraps every launch in a container-side timeout, which could in
    # principle break a server. A comparison that only looked for improvements
    # would miss that, so both directions must surface.
    before = _write(str(tmp_path / "b" / "1.json"), [_started("a/x")])
    after = _write(str(tmp_path / "a" / "1.json"), [_failed("a/x", "exit 127")])
    mod.main(["--before", before, "--after", after])
    out = capsys.readouterr().out
    assert "changed class: 1 of 1" in out
    assert "exit 127" in out


def test_several_batch_files_pool_into_one_run(tmp_path, capsys):
    mod = _load()
    b1 = _write(str(tmp_path / "b" / "1.json"), [_started("a/x")])
    b2 = _write(str(tmp_path / "b" / "2.json"), [_started("a/y")])
    a1 = _write(str(tmp_path / "a" / "1.json"), [_started("a/x")])
    a2 = _write(str(tmp_path / "a" / "2.json"), [_started("a/y")])
    mod.main(["--before", b1, b2, "--after", a1, a2])
    out = capsys.readouterr().out
    assert "2 before, 2 after, 2 in both" in out


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert "\u2014" not in handle.read()
