import math
import random

import pytest

from agentfaultlab import (
    HIGH, MEDIUM, NONE, POLICIES, BillingLedger, Episode, FaultProfile, IdempotentRetry,
    NaiveRetry, ToolTimeout, VerifiedIdempotent, VerifyBeforeRetry, pass_hat_k, run_experiment,
    summarize,
)
from agentfaultlab.metrics import wilson_interval


def ledger(profile, seed=1):
    return BillingLedger(profile, random.Random(seed))


# --- faults -----------------------------------------------------------------

def test_profile_rejects_bad_probability():
    with pytest.raises(ValueError):
        FaultProfile("bad", p_timeout=1.5)


# --- tools: each failure mode, forced ----------------------------------------

def test_no_faults_charges_once():
    led = ledger(NONE)
    led.charge("A", 100)
    assert led.final_state("A") == (1, True)


def test_timeout_after_commit_commits_and_raises():
    led = ledger(FaultProfile("t", p_timeout=1.0, p_commit_given_timeout=1.0))
    with pytest.raises(ToolTimeout):
        led.charge("A", 100)
    assert led.final_state("A")[0] == 1  # the caller saw an error, the write happened


def test_timeout_before_commit_leaves_nothing():
    led = ledger(FaultProfile("t", p_timeout=1.0, p_commit_given_timeout=0.0))
    with pytest.raises(ToolTimeout):
        led.charge("A", 100)
    assert led.final_state("A")[0] == 0


def test_idempotency_key_replay_creates_no_second_charge():
    led = ledger(FaultProfile("t", p_timeout=1.0, p_commit_given_timeout=1.0))
    for _ in range(3):
        with pytest.raises(ToolTimeout):
            led.charge("A", 100, idempotency_key="k")
    assert led.final_state("A")[0] == 1


def test_delayed_visibility_hides_then_reveals():
    led = ledger(FaultProfile("v", p_delayed_visibility=1.0, visibility_lag=2))
    led.charge("A", 100)
    assert led.get_charge("A")["count"] == 0
    assert led.get_charge("A")["count"] == 0
    assert led.get_charge("A")["count"] == 1


def test_partial_success_answers_ok_but_is_unfulfilled():
    led = ledger(FaultProfile("p", p_partial_success=1.0))
    assert led.charge("A", 100) == {"status": "ok"}
    assert led.final_state("A") == (1, False)
    led.complete_fulfillment("A")
    assert led.final_state("A") == (1, True)


# --- agents -----------------------------------------------------------------

def test_every_policy_passes_the_clean_eval():
    for policy in POLICIES:
        eps = run_experiment(policy, NONE, tasks=20, trials=8)
        assert all(e.success for e in eps), policy.name


def test_naive_retry_double_charges_after_a_lost_response():
    led = ledger(FaultProfile("t", p_timeout=1.0, p_commit_given_timeout=1.0))
    assert NaiveRetry().run(led, "A", 100) == "failure"
    assert led.final_state("A")[0] == 3  # one timeout, three charges


def test_idempotent_retry_never_duplicates_but_can_misreport():
    led = ledger(FaultProfile("t", p_timeout=1.0, p_commit_given_timeout=1.0))
    assert IdempotentRetry().run(led, "A", 100) == "failure"
    assert led.final_state("A") == (1, True)  # state is right, the report is wrong


def test_verify_before_retry_is_fooled_by_delayed_visibility():
    prof = FaultProfile("d", p_timeout=1.0, p_commit_given_timeout=1.0,
                        p_delayed_visibility=1.0, visibility_lag=2)
    led = ledger(prof)
    VerifyBeforeRetry().run(led, "A", 100)
    assert led.final_state("A")[0] >= 2  # the single read missed the commit


def test_verified_idempotent_survives_the_same_faults_and_repairs_partials():
    prof = FaultProfile("all", p_timeout=1.0, p_commit_given_timeout=1.0,
                        p_delayed_visibility=1.0, visibility_lag=2, p_partial_success=1.0)
    led = ledger(prof)
    assert VerifiedIdempotent().run(led, "A", 100) == "success"
    assert led.final_state("A") == (1, True)


def test_naive_retry_reports_success_on_a_partial_success():
    led = ledger(FaultProfile("p", p_partial_success=1.0))
    assert NaiveRetry().run(led, "A", 100) == "success"
    assert led.final_state("A") == (1, False)  # silent failure


def test_unconfirmed_when_the_ledger_never_shows_the_charge():
    prof = FaultProfile("x", p_timeout=1.0, p_commit_given_timeout=0.0)
    led = ledger(prof)
    assert VerifiedIdempotent().run(led, "A", 100) == "unconfirmed"


# --- metrics ----------------------------------------------------------------

def ep(task, trial, report="success", count=1, fulfilled=True):
    return Episode("p", "f", task, trial, report, count, fulfilled)


def test_pass_hat_k_all_success_and_none():
    assert pass_hat_k([ep(0, t) for t in range(8)], 8) == 1.0
    assert pass_hat_k([ep(0, t, count=2) for t in range(8)], 8) == 0.0


def test_pass_hat_k_matches_the_combinatorial_estimate():
    eps = [ep(0, t) for t in range(6)] + [ep(0, t, count=2) for t in range(6, 8)]
    assert pass_hat_k(eps, 2) == pytest.approx(math.comb(6, 2) / math.comb(8, 2))


def test_pass_hat_k_needs_enough_trials():
    with pytest.raises(ValueError):
        pass_hat_k([ep(0, 0)], 8)


def test_flags():
    assert ep(0, 0, count=2).duplicate
    assert ep(0, 0, fulfilled=False).silent_failure
    assert not ep(0, 0, report="unconfirmed", fulfilled=False).silent_failure
    assert not ep(0, 0, report="failure", count=1).success


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(20, 100)
    assert lo < 0.20 < hi


# --- experiments --------------------------------------------------------------

def test_runs_are_reproducible():
    a = run_experiment(NaiveRetry(), MEDIUM, tasks=30, trials=8, seed=7)
    b = run_experiment(NaiveRetry(), MEDIUM, tasks=30, trials=8, seed=7)
    assert a == b


def test_faults_hurt_the_naive_policy_and_the_verified_policy_holds():
    naive = summarize(run_experiment(NaiveRetry(), HIGH, tasks=100, trials=8), k=8)
    solid = summarize(run_experiment(VerifiedIdempotent(), HIGH, tasks=100, trials=8), k=8)
    assert naive["duplicate_action_rate"] > 0.05
    assert solid["duplicate_action_rate"] == 0.0
    assert solid["silent_failure_rate"] == 0.0
    assert solid["task_success"] > naive["task_success"]
