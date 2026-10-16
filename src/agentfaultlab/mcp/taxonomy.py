"""The fault taxonomy for the agent-tool boundary, with provenance.

Every mode here is one that published work or a tracked protocol issue says
happens in practice. The column that matters for ground truth is
`effect_expected`: what the backend should look like afterwards if the tool
is well behaved. The audit compares that against what an observer actually
sees in the backend, never against what the tool or the agent says.

Provenance, by id:

* F1 lost response after commit: Mansoor, Phadke, Rana (arXiv:2608.02645),
  "timeout after commit"; LIMBO (arXiv:2609.29095) `timeout_post`;
  modelcontextprotocol/modelcontextprotocol issue 3394 (no request-level
  deduplication in the protocol).
* F2 duplicate delivery: LIMBO `duplicate_delivery`.
* F3 late commit: LIMBO `timeout_late`.
* F4 concurrent twin calls: the effectfence probe; classical concurrency
  testing. Clients that parallelise on `idempotentHint` produce this shape.
* F5 partial batch: LIMBO `partial_timeout`; the partial-success fault in
  agent-fault-lab v0.1 (Mansoor et al. Table 1).
* F6 silent OK: "Fabrication After Tool Failure" (45.3% of episodes report
  success after a failed call); SilentProbe.
* F7 error before commit: the control. A retry here must be safe, so any
  duplicate effect under F7 is a harness bug, not a server finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: What the backend should hold after the fault, for a well behaved tool.
#: "once" means the effect applied exactly once, "none" means not at all,
#: "unknown" means the protocol genuinely cannot say (the caller is entitled
#: to either, and a tool is only faulted for making it unknowable).
EffectExpectation = Literal["once", "none", "unknown"]


@dataclass(frozen=True)
class FaultMode:
    id: str
    name: str
    description: str
    #: What a correct tool leaves behind after the *client retries once*.
    effect_expected_after_retry: EffectExpectation
    #: True if the server receives the call at all.
    reaches_server: bool
    provenance: str
    #: Modes that need a per-call shape (for example a list argument).
    requires: tuple[str, ...] = field(default=())


LOST_RESPONSE = FaultMode(
    id="F1",
    name="lost_response_after_commit",
    description=(
        "Forward the call, wait for the server's response, then drop it. The "
        "client sees a timeout and cannot tell whether the write happened."
    ),
    effect_expected_after_retry="once",
    reaches_server=True,
    provenance="Mansoor et al. 2608.02645; LIMBO timeout_post; MCP issue 3394",
)

DUPLICATE_DELIVERY = FaultMode(
    id="F2",
    name="duplicate_delivery",
    description="Deliver the same call twice, under two request ids.",
    effect_expected_after_retry="once",
    reaches_server=True,
    provenance="LIMBO duplicate_delivery",
)

LATE_COMMIT = FaultMode(
    id="F3",
    name="late_commit",
    description=(
        "Hold the call past the client's timeout, then forward it. The client "
        "has already given up and may have retried."
    ),
    effect_expected_after_retry="once",
    reaches_server=True,
    provenance="LIMBO timeout_late",
)

CONCURRENT_TWIN = FaultMode(
    id="F4",
    name="concurrent_twin",
    description="Send two identical calls at the same time.",
    effect_expected_after_retry="once",
    reaches_server=True,
    provenance="effectfence probe; clients that parallelise on idempotentHint",
)

PARTIAL_BATCH = FaultMode(
    id="F5",
    name="partial_batch",
    description=(
        "For a call with a list argument, forward a prefix of the list and "
        "drop the response, so part of the work commits and the caller is "
        "told nothing."
    ),
    effect_expected_after_retry="once",
    reaches_server=True,
    provenance="LIMBO partial_timeout; agent-fault-lab v0.1 partial success",
    requires=("list_argument",),
)

SILENT_OK = FaultMode(
    id="F6",
    name="silent_ok",
    description=(
        "Do not forward the call. Return an empty success to the client, so "
        "it believes a write happened that never did."
    ),
    effect_expected_after_retry="none",
    reaches_server=False,
    provenance="Fabrication After Tool Failure; SilentProbe",
)

ERROR_BEFORE_COMMIT = FaultMode(
    id="F7",
    name="error_before_commit",
    description=(
        "Control. Fail the call without forwarding it, so a retry must leave "
        "the effect applied exactly once."
    ),
    effect_expected_after_retry="once",
    reaches_server=False,
    provenance="control condition",
)

MODES: dict[str, FaultMode] = {}
for _mode in (
    LOST_RESPONSE,
    DUPLICATE_DELIVERY,
    LATE_COMMIT,
    CONCURRENT_TWIN,
    PARTIAL_BATCH,
    SILENT_OK,
    ERROR_BEFORE_COMMIT,
):
    MODES[_mode.id] = _mode
    MODES[_mode.name] = _mode


def get(mode: str) -> FaultMode:
    """Look a mode up by id (`F1`) or by name (`lost_response_after_commit`)."""
    try:
        return MODES[mode]
    except KeyError:
        known = sorted({m.id for m in MODES.values()})
        raise KeyError(f"unknown fault mode {mode!r}; known ids: {', '.join(known)}") from None


#: Modes the audit runs on every mutating tool. F4 and F5 are conditional:
#: F4 only where the client policy under test parallelises, F5 only where a
#: tool takes a list.
AUDIT_SEQUENCE = ("F7", "F1", "F2", "F3")
