"""Tests for the startability funnel's classifier.

The funnel is the breadth result the paper leans on hardest, and every row of
it is a number nobody can check by eye once the draw passes a hundred
candidates. Two mistakes are already on the record: a hand tally that put
nine servers in the annotating row where the recorded JSON holds ten, and a
"did not start" row that counted a server which started fine and then failed
`tools/list` because the network was off. Both are classification boundaries,
so the boundaries are pinned here.
"""
from __future__ import annotations

import importlib.util
import os
import sys

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "probe_funnel.py",
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)


def _load():
    spec = importlib.util.spec_from_file_location("probe_funnel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _started(tools):
    return {
        "server": "s",
        "started_via": "registry",
        "attempts": [{"started": True, "tools": tools}],
    }


def _failed(error, stderr=()):
    return {
        "server": "s",
        "started_via": None,
        "attempts": [{"started": False, "error": error, "stderr_tail": list(stderr)}],
    }


def test_no_attempt_is_an_install_failure_not_a_launch_failure():
    funnel = _load()
    assert funnel.classify({"server": "s", "attempts": []}) == "not-installed"


def test_a_tool_without_read_only_true_is_mutating():
    funnel = _load()
    assert funnel.classify(_started([{"readOnlyHint": None}])) == "mutating"
    assert funnel.classify(_started([{"readOnlyHint": False}])) == "mutating"
    assert funnel.classify(_started([{"readOnlyHint": True}])) == "read-only"


def test_a_server_with_no_tools_at_all_is_not_counted_as_mutating():
    funnel = _load()
    assert funnel.classify(_started([])) == "read-only"


def test_a_tools_list_network_failure_is_network_bound_not_a_launch_failure():
    """The case that moved two servers out of the headline rate.

    `initialize` succeeded and `tools/list` failed with `fetch failed`, which
    says the server needs the network to describe itself. That is the
    containment profile excluding it by design, not the registry publishing a
    command that does not work.
    """
    funnel = _load()
    record = _failed("SessionError: tools/list failed: {'message': 'fetch failed'}")
    assert funnel.classify(record) == "network-bound"


def test_launch_failure_buckets_prefer_the_specific_diagnosis():
    funnel = _load()
    cases = {
        "a module the package does not ship (MODULE_NOT_FOUND)": _failed(
            "exited with status 1", ["code: 'MODULE_NOT_FOUND'"]
        ),
        "an entry point not runnable as published": _failed(
            "exited with status 2", ["/usr/local/bin/x: 1: import: not found"]
        ),
        "printed its own usage: a subcommand or flag the registry omits": _failed(
            "exited with status 2", ["usage: x [-h] [--workspace WORKSPACE]"]
        ),
        "exited 0 in silence, never answering initialize": _failed(
            "the server exited with status 0 without answering. "
            "Last stderr: nothing on stderr"
        ),
        "a vendored binary needing a newer glibc than the image": _failed(
            "exited with status 1", ["version `GLIBC_2.39' not found"]
        ),
    }
    for expected, record in cases.items():
        assert funnel.failure_cause(record) == expected, expected


def test_writes_only_is_not_counted_as_a_retry_annotation():
    """`readOnlyHint: false` answers a different question than the study asks.

    It tells a client the tool writes. It says nothing about whether a repeat
    applies the effect twice, which is the whole subject of the audit, so it
    cannot be pooled with `idempotentHint` into one "annotates" rate.
    """
    funnel = _load()
    writes_only = {"readOnlyHint": False, "idempotentHint": None, "destructiveHint": None}
    retry = {"readOnlyHint": False, "idempotentHint": True, "destructiveHint": None}
    assert funnel.classify(_started([writes_only])) == "mutating"
    assert funnel.classify(_started([retry])) == "mutating"
    # The split itself is asserted through the report, since that is where the
    # two classes are counted.
    assert "writes-only" in open(SCRIPT, encoding="utf-8").read()


def test_rate_carries_an_interval_and_survives_a_zero_denominator():
    funnel = _load()
    assert "95% CI" in funnel.rate(1, 10)
    assert funnel.rate(0, 0) == "n/a (0)"
