"""Scripted agent policies for the billing task: charge an invoice exactly once.

These are deterministic stand-ins for how an LLM agent behaves around a
tool call, so the harness runs with no API keys. To test a real agent, wrap it
in any object with the same `name` and `run(ledger, invoice, amount)` shape,
which returns one of "success", "failure" or "unconfirmed".
"""

from __future__ import annotations

from typing import Protocol

from .tools import BillingLedger, ToolTimeout

MAX_ATTEMPTS = 3
SUCCESS, FAILURE, UNCONFIRMED = "success", "failure", "unconfirmed"


class AgentPolicy(Protocol):
    name: str

    def run(self, ledger: BillingLedger, invoice: str, amount: int) -> str: ...


class NaiveRetry:
    """Binary contract: a timeout means failure, so retry."""

    name = "naive-retry"

    def run(self, ledger: BillingLedger, invoice: str, amount: int) -> str:
        for _ in range(MAX_ATTEMPTS):
            try:
                ledger.charge(invoice, amount)
                return SUCCESS
            except ToolTimeout:
                continue
        return FAILURE


class VerifyBeforeRetry:
    """On a timeout, read the ledger once before retrying (no idempotency key)."""

    name = "verify-before-retry"

    def run(self, ledger: BillingLedger, invoice: str, amount: int) -> str:
        for _ in range(MAX_ATTEMPTS):
            try:
                ledger.charge(invoice, amount)
                return SUCCESS
            except ToolTimeout:
                if ledger.get_charge(invoice)["count"] >= 1:
                    return SUCCESS
        return FAILURE


class IdempotentRetry:
    """Retry with a stable idempotency key: retries are safe, outcomes are not checked."""

    name = "idempotency-key"

    def run(self, ledger: BillingLedger, invoice: str, amount: int) -> str:
        key = f"charge:{invoice}"
        for _ in range(MAX_ATTEMPTS):
            try:
                ledger.charge(invoice, amount, idempotency_key=key)
                return SUCCESS
            except ToolTimeout:
                continue
        return FAILURE


class VerifiedIdempotent:
    """Idempotency key, then verify the postcondition and repair it.

    Reports success only when the ledger confirms exactly one fulfilled charge.
    Otherwise it reports the third state, "unconfirmed", instead of guessing.
    """

    name = "idempotency-key+verify"
    POLLS = 3

    def run(self, ledger: BillingLedger, invoice: str, amount: int) -> str:
        key = f"charge:{invoice}"
        for _ in range(MAX_ATTEMPTS):
            try:
                ledger.charge(invoice, amount, idempotency_key=key)
            except ToolTimeout:
                pass
            state = self._poll(ledger, invoice)
            if state["count"] >= 1:
                if not state["fulfilled"]:
                    ledger.complete_fulfillment(invoice)
                    state = ledger.get_charge(invoice)
                if state["count"] == 1 and state["fulfilled"]:
                    return SUCCESS
                return UNCONFIRMED
        return UNCONFIRMED

    def _poll(self, ledger: BillingLedger, invoice: str) -> dict:
        state = ledger.get_charge(invoice)
        for _ in range(self.POLLS - 1):
            if state["count"] >= 1:
                break
            state = ledger.get_charge(invoice)
        return state


POLICIES = [NaiveRetry(), VerifyBeforeRetry(), IdempotentRetry(), VerifiedIdempotent()]
