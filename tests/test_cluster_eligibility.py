"""Tests for the per-arm clean-complete cluster count.

This number decides whether the primary sign test has any power, so the way
it is computed needs pinning rather than trusting. Three boundaries have
already proved easy to get wrong by hand: a refused batch whose episodes
recorded `state=none` looks exactly like a client failing the task, a cluster
nobody has measured on an arm looks exactly like one the arm cannot do, and
the clean-complete count of one client is not a denominator for a comparison
against another.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "cluster_eligibility.py",
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

FIELDS = ("task", "agent", "condition", "arm", "multiplicity", "task_success")

HOSTED = "claude-code 2.1.311 (model sonnet, tool timeout 20000ms, clean config)"
LOCAL = "ollama qwen3:8b"


def _load():
    spec = importlib.util.spec_from_file_location("cluster_eligibility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _row(task, agent=HOSTED, condition="clean", arm="baseline",
         multiplicity="once", task_success="True"):
    return {
        "task": task,
        "agent": agent,
        "condition": condition,
        "arm": arm,
        "multiplicity": multiplicity,
        "task_success": task_success,
    }


def test_a_refused_batch_is_excluded_with_its_reason(tmp_path):
    mod = _load()
    good = _write(str(tmp_path / "hosted-20261028" / "a.csv"), [_row("t1")])
    bad = _write(str(tmp_path / "hosted-20261027" / "a.csv"),
                 [_row("t1", multiplicity="none", task_success="False")])
    outcomes, skipped, rows = mod.collect([good, bad])
    # The refused rows are reported, not silently dropped, and they do not
    # turn an eligible cluster into a conflict.
    assert rows == 1
    assert len(skipped) == 1 and skipped[0][1] == 1
    assert "refused" in skipped[0][2]
    assert mod.verdicts(outcomes)[("hosted", "t1")] == "eligible"


def test_completion_reads_the_end_state_and_ignores_the_claim(tmp_path):
    """Eligibility asks whether the arm did the work, nothing else.

    The first version of this also required `task_success`, which is
    `landed_once and claim == success` and so folds in the claim parser. One
    local-arm cluster landed its write exactly once, reported the right
    computed answer, and the parser read the message as asserting nothing, so
    a cluster the arm had completed was counted as one it could not. The
    mistake understated the weaker arm, which is the direction that deserves
    the most suspicion, so the two axes are kept apart here.
    """
    mod = _load()
    path = _write(str(tmp_path / "b.csv"), [
        _row("landed-but-unparsed-claim", task_success="False"),
        _row("wrong-state-but-claimed", multiplicity="other"),
        _row("landed-and-claimed"),
    ])
    verdict = mod.verdicts(mod.collect([path])[0])
    assert verdict[("hosted", "landed-but-unparsed-claim")] == "eligible"
    assert verdict[("hosted", "wrong-state-but-claimed")] == "not-complete"
    assert verdict[("hosted", "landed-and-claimed")] == "eligible"


def test_disagreeing_batches_are_a_conflict_not_a_maximum(tmp_path):
    mod = _load()
    one = _write(str(tmp_path / "one.csv"), [_row("flaky")])
    two = _write(str(tmp_path / "two.csv"),
                 [_row("flaky", multiplicity="other", task_success="False")])
    verdict = mod.verdicts(mod.collect([one, two])[0])
    # Finishing a task in one batch and not another is a property of the
    # cluster worth reporting, so it must not be resolved by taking the best
    # run and counting the cluster as eligible.
    assert verdict[("hosted", "flaky")] == "conflict"


def test_only_the_clean_baseline_cell_counts(tmp_path):
    mod = _load()
    path = _write(str(tmp_path / "c.csv"), [
        _row("t", condition="f6"),
        _row("t", arm="unconfirmed"),
    ])
    outcomes, _, rows = mod.collect([path])
    # A faulted episode or a different policy arm says nothing about whether
    # the clean baseline establishes the comparison.
    assert rows == 0
    assert outcomes == {}


def test_the_client_axis_is_not_the_policy_arm_column(tmp_path):
    mod = _load()
    assert mod.client_of({"agent": HOSTED}) == "hosted"
    assert mod.client_of({"agent": LOCAL}) == "local"
    assert mod.client_of({"agent": "some other runner"}) == "unknown"
    # The same client across batches with different tool timeouts is one
    # client, not several, or every arm would hold a single cluster.
    other = "claude-code 2.1.311 (model sonnet, tool timeout 9000ms)"
    assert mod.client_of({"agent": other}) == mod.client_of({"agent": HOSTED})


def test_an_unmeasured_cluster_is_neither_eligible_nor_failed(tmp_path):
    mod = _load()
    path = _write(str(tmp_path / "d.csv"), [
        _row("shared", agent=HOSTED),
        _row("shared", agent=LOCAL),
        _row("hosted-only", agent=HOSTED),
    ])
    verdict = mod.verdicts(mod.collect([path])[0])
    assert ("local", "hosted-only") not in verdict
    assert verdict[("hosted", "hosted-only")] == "eligible"


def test_the_script_writes_no_em_dashes():
    with open(SCRIPT) as handle:
        assert "\u2014" not in handle.read()
