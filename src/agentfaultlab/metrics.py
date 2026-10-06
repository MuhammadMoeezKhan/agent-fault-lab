"""Metrics worth reporting beside task success."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    policy: str
    profile: str
    task: int
    trial: int
    report: str  # "success" | "failure" | "unconfirmed"
    count: int  # ground-truth charges for the invoice
    fulfilled: bool  # ground-truth fulfillment

    @property
    def state_correct(self) -> bool:
        return self.count == 1 and self.fulfilled

    @property
    def success(self) -> bool:
        return self.report == "success" and self.state_correct

    @property
    def duplicate(self) -> bool:
        return self.count >= 2

    @property
    def silent_failure(self) -> bool:
        """The agent said success while the real state is wrong."""
        return self.report == "success" and not self.state_correct


def rate(episodes: list[Episode], predicate) -> float:
    return sum(1 for e in episodes if predicate(e)) / len(episodes) if episodes else 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pass_hat_k(episodes: list[Episode], k: int) -> float:
    """tau-bench style pass^k: the chance a task succeeds on k independent trials in a row.

    For each task with n trials and c successes, the unbiased estimate is
    C(c, k) / C(n, k); the result is the mean over tasks.
    """
    by_task: dict[int, list[Episode]] = defaultdict(list)
    for e in episodes:
        by_task[e.task].append(e)
    scores = []
    for trials in by_task.values():
        n, c = len(trials), sum(1 for e in trials if e.success)
        if n < k:
            raise ValueError(f"need at least k={k} trials per task, got {n}")
        scores.append(math.comb(c, k) / math.comb(n, k))
    return sum(scores) / len(scores)


def summarize(episodes: list[Episode], k: int = 8) -> dict:
    n = len(episodes)
    dup = sum(1 for e in episodes if e.duplicate)
    silent = sum(1 for e in episodes if e.silent_failure)
    ok = sum(1 for e in episodes if e.success)
    return {
        "episodes": n,
        "task_success": ok / n,
        "duplicate_action_rate": dup / n,
        "duplicate_ci": wilson_interval(dup, n),
        "silent_failure_rate": silent / n,
        "silent_ci": wilson_interval(silent, n),
        "unconfirmed_rate": rate(episodes, lambda e: e.report == "unconfirmed"),
        f"pass_hat_{k}": pass_hat_k(episodes, k),
    }
