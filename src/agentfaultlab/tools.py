"""A small, synthetic billing ledger: the tool an agent calls.

It is deliberately generic. It models a state-changing tool with an optional
idempotency key, a read-back, and an idempotent repair call, and it keeps the
ground-truth state so metrics can compare what the agent reported with what
actually happened.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .faults import FaultProfile


class ToolTimeout(Exception):
    """The response was lost. The caller cannot tell whether the write committed."""


@dataclass
class Charge:
    invoice: str
    amount: int
    fulfilled: bool
    hidden_reads_left: int  # read-backs that still miss this charge


class BillingLedger:
    def __init__(self, profile: FaultProfile, rng: random.Random) -> None:
        self.profile = profile
        self.rng = rng
        self.charges: list[Charge] = []  # ground truth
        self._by_key: dict[str, Charge] = {}
        self.calls = 0

    def charge(self, invoice: str, amount: int, idempotency_key: str | None = None) -> dict:
        self.calls += 1
        p = self.profile
        timed_out = self.rng.random() < p.p_timeout

        if idempotency_key is not None and idempotency_key in self._by_key:
            # Server-side replay: same key, no new effect.
            if timed_out:
                raise ToolTimeout("response lost")
            return {"status": "ok"}

        commits = (not timed_out) or (self.rng.random() < p.p_commit_given_timeout)
        if commits:
            fulfilled = self.rng.random() >= p.p_partial_success
            lag = p.visibility_lag if self.rng.random() < p.p_delayed_visibility else 0
            charge = Charge(invoice, amount, fulfilled, lag)
            self.charges.append(charge)
            if idempotency_key is not None:
                self._by_key[idempotency_key] = charge

        if timed_out:
            raise ToolTimeout("response lost")
        # A partial success still answers "ok": the failure is silent.
        return {"status": "ok"}

    def get_charge(self, invoice: str) -> dict:
        """Read-back. Charges inside their visibility lag are missed, once per read."""
        visible = 0
        fulfilled = True
        for c in self.charges:
            if c.invoice != invoice:
                continue
            if c.hidden_reads_left > 0:
                c.hidden_reads_left -= 1
                continue
            visible += 1
            fulfilled = fulfilled and c.fulfilled
        return {"count": visible, "fulfilled": fulfilled and visible > 0}

    def complete_fulfillment(self, invoice: str) -> None:
        """Idempotent repair: apply the missing effect, never create a new charge."""
        for c in self.charges:
            if c.invoice == invoice:
                c.fulfilled = True

    def final_state(self, invoice: str) -> tuple[int, bool]:
        """Ground truth, bypassing faults: (charge count, all fulfilled)."""
        mine = [c for c in self.charges if c.invoice == invoice]
        return len(mine), bool(mine) and all(c.fulfilled for c in mine)
