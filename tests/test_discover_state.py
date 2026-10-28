"""The discovery script's containment guard.

Fixture discovery is the step that starts a server nobody has audited yet, so
it is the step hard rule 9 exists for, and it was the last place in the
toolchain that still launched on the host with nothing but an isolated HOME.
Both incidents behind the rule happened during exactly this kind of probe: a
skill file written into the host agent's configuration, and four auto-commits
into this repository, neither from a tool call.

A default that runs on the host is therefore a bug, not a convenience, and
these tests pin the refusal.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "discover_state.py")


def _run(*args):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_running_without_a_container_is_refused():
    done = _run("--runner", "npx", "--package", "whatever", "--schema")
    assert done.returncode != 0
    assert "hard rule 9" in done.stderr


def test_the_refusal_names_both_ways_out():
    done = _run("--runner", "npx", "--package", "whatever", "--schema")
    assert "--contain" in done.stderr
    assert "--on-host" in done.stderr


def test_on_host_is_an_explicit_opt_in_not_a_default():
    """It exists for reproducing an audit already collected on the host.

    What matters is that a caller has to type it. The flag is absent from no
    command line by accident, which is the property the refusal above buys.
    """
    source = open(SCRIPT, encoding="utf-8").read()
    assert '"--on-host"' in source
    assert "action=\"store_true\"" in source
    # And it is not given a default that turns it on.
    assert "--on-host\", default=True" not in source


def test_the_trial_root_is_mounted_not_the_working_directory():
    """The mount has to be the root so one diff sees HOME and the workdir.

    A server that keeps its state under HOME (memorix writes
    $HOME/.memorix/data/memorix.db) and one that writes beside its working
    directory must both show up, or discovery reports "nothing changed on
    disk" for a server that changed plenty.
    """
    source = open(SCRIPT, encoding="utf-8").read()
    assert "mount=root" in source
    assert "cwd=work" in source


@pytest.mark.parametrize("flag", ["--contain", "--build"])
def test_the_flags_exist(flag):
    done = _run("--help")
    assert flag in done.stdout


def test_no_em_dashes_in_the_script():
    assert "—" not in open(SCRIPT, encoding="utf-8").read()


def test_a_seed_file_can_be_written_before_the_server_starts():
    """A data-oriented server needs an input file that exists before the call.

    Without one the tool refuses, the diff shows nothing, and the fixture
    looks like a server with no durable state. The flag takes relpath=text
    so the seed is part of the command that produced the result.
    """
    done = _run("--help")
    assert "--seed-file" in done.stdout
    assert "RELPATH=TEXT" in done.stdout


def test_the_seed_is_written_into_the_working_directory_not_the_mount_root():
    """It has to be where the server is started, not beside the throwaway HOME.

    The mount root holds both, and a seed written at the root is not where a
    tool given a relative path will look for it.
    """
    source = open(SCRIPT, encoding="utf-8").read()
    assert "os.path.join(work, relative)" in source


def test_repeated_seed_file_flags_accumulate():
    """Several --seed-file flags must all take effect.

    With argparse's default behaviour for nargs="*", a repeated flag replaces
    the earlier one instead of adding to it, so only the last seed was
    written. Nothing failed loudly: the server simply reported that its input
    was missing, and the candidate looked like it needed a fixture shape the
    harness could not build. It was nearly parked on that basis.
    """
    spec = importlib.util.spec_from_file_location("discover_state", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import tempfile

    work = tempfile.mkdtemp()
    written = module.seed_files(
        work, ["a.txt=one", "b/c.txt=two", "d.txt=line\\nline"]
    )
    assert [name for name, _ in written] == ["a.txt", "b/c.txt", "d.txt"]
    # Each file is on disk, the nested one had its parent created, and the
    # escape is turned into a real newline.
    assert open(os.path.join(work, "a.txt")).read() == "one"
    assert open(os.path.join(work, "b", "c.txt")).read() == "two"
    assert open(os.path.join(work, "d.txt")).read() == "line\nline"


def test_the_seed_file_flag_is_configured_to_accumulate():
    """A repeated --seed-file must add rather than replace.

    argparse's default for nargs="*" replaces, so only the last seed was
    written and nothing failed loudly: the server reported its input missing
    and the candidate looked unfixturable. It was nearly parked on that basis.
    """
    source = open(SCRIPT, encoding="utf-8").read()
    flag = source[source.index('"--seed-file"'):]
    flag = flag[: flag.index("help=")]
    assert 'action="extend"' in flag
