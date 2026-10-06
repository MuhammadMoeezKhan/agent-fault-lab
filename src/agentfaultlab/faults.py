"""Fault profiles: how often a tool misbehaves, and in which of three ways.

The three failure modes follow the non-atomic failures described in
Mansoor, Phadke, Rana (2026), "Verified Tool Calls Improve LLM Agent
Reliability Under Non-Atomic Failures" (arXiv:2608.02645):

* timeout after commit: the write happens, the response is lost
* delayed visibility: the write happened, but read-backs miss it for a while
* partial success: the write happens, one of its effects does not

The low/medium/high probabilities reuse that paper's Table 1 so results are
comparable in shape. Their fourth fault, stale conflicts, is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultProfile:
    name: str
    # Per call: the response is lost and the caller sees a timeout.
    p_timeout: float = 0.0
    # Given a lost response, did the write already commit? A timeout cannot
    # tell the caller which, which is the whole problem.
    p_commit_given_timeout: float = 0.5
    # Per committed write: read-backs miss it for `visibility_lag` reads.
    p_delayed_visibility: float = 0.0
    visibility_lag: int = 2
    # Per committed write: the charge is recorded but fulfillment is not applied.
    p_partial_success: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "p_timeout",
            "p_commit_given_timeout",
            "p_delayed_visibility",
            "p_partial_success",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {value}")
        if self.visibility_lag < 0:
            raise ValueError("visibility_lag must be >= 0")


NONE = FaultProfile("none")
LOW = FaultProfile("low", 0.05, 0.5, 0.05, 2, 0.03)
MEDIUM = FaultProfile("medium", 0.20, 0.5, 0.15, 2, 0.10)
HIGH = FaultProfile("high", 0.35, 0.5, 0.25, 2, 0.15)

PROFILES = {p.name: p for p in (NONE, LOW, MEDIUM, HIGH)}
