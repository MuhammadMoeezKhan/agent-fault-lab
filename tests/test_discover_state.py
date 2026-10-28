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
