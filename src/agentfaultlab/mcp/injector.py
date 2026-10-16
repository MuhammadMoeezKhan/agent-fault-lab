"""Which calls get a fault, and what the fault does to them.

The decision is deliberately deterministic: a plan names a tool and the
occurrence numbers to hit, so a whole audit run reproduces from the plan plus
the server's commit hash. Probabilistic injection is available for the
consequence study, where a seeded random draw is what you want, and the draw
is recorded in the trace either way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from . import taxonomy
from .jsonrpc import Message, tool_name


class Action(Enum):
    """What the transport should do with a request."""

    PASS = "pass"
    #: Forward, wait for the response, then withhold it from the caller.
    FORWARD_AND_DROP_RESPONSE = "forward_and_drop_response"
    #: Forward twice under two request ids; the caller sees the first answer.
    FORWARD_TWICE = "forward_twice"
    #: Forward both copies at once; the caller sees whichever lands first.
    FORWARD_CONCURRENT = "forward_concurrent"
    #: Hold the request, then forward it after the caller has given up.
    DELAY_THEN_FORWARD = "delay_then_forward"
    #: Truncate a list argument, forward, then withhold the response.
    TRUNCATE_AND_DROP_RESPONSE = "truncate_and_drop_response"
    #: Never forward; answer the caller with an empty success.
    SYNTHESISE_SUCCESS = "synthesise_success"
    #: Never forward; answer the caller with an error.
    SYNTHESISE_ERROR = "synthesise_error"


_ACTION_BY_MODE = {
    "F1": Action.FORWARD_AND_DROP_RESPONSE,
    "F2": Action.FORWARD_TWICE,
    "F3": Action.DELAY_THEN_FORWARD,
    "F4": Action.FORWARD_CONCURRENT,
    "F5": Action.TRUNCATE_AND_DROP_RESPONSE,
    "F6": Action.SYNTHESISE_SUCCESS,
    "F7": Action.SYNTHESISE_ERROR,
}


@dataclass(frozen=True)
class Decision:
    action: Action
    mode: Optional[taxonomy.FaultMode] = None
    #: Seconds to hold the request, for F3.
    delay_s: float = 0.0
    #: How many list elements to keep, for F5.
    keep: int = 1
    #: Which list argument F5 truncated, once known.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def injected(self) -> bool:
        return self.action is not Action.PASS

    def describe(self) -> Optional[str]:
        return None if self.mode is None else f"{self.mode.id}:{self.mode.name}"


PASS = Decision(Action.PASS)


@dataclass
class FaultPlan:
    """One fault mode, aimed at one tool, on chosen occurrences of the call.

    `occurrences` counts matching `tools/call` requests from 1. An empty
    tuple with `probability` set switches to a seeded random draw per call.
    `tool=None` matches any tool.
    """

    mode: str
    tool: Optional[str] = None
    occurrences: tuple[int, ...] = (1,)
    probability: float = 0.0
    delay_s: float = 2.0
    keep: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        self.fault = taxonomy.get(self.mode)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")
        if self.probability and self.occurrences:
            raise ValueError("set occurrences or probability, not both")
        if not self.probability and not self.occurrences:
            raise ValueError("a plan needs occurrences or a probability")
        if self.keep < 0:
            raise ValueError("keep must be >= 0")

    @classmethod
    def parse(cls, spec: str, **overrides: Any) -> "FaultPlan":
        """Parse `F1`, `F1:write_file`, or `F1:write_file@2` (also `@1,3`).

        `@all` hits every matching call.
        """
        mode, _, rest = spec.partition(":")
        tool, _, where = rest.partition("@")
        kwargs: dict[str, Any] = {"mode": mode.strip(), "tool": tool.strip() or None}
        if where.strip() == "all":
            kwargs["occurrences"] = ()
            kwargs["probability"] = 1.0
        elif where.strip():
            kwargs["occurrences"] = tuple(int(n) for n in where.split(",") if n.strip())
        kwargs.update(overrides)
        return cls(**kwargs)


class FaultInjector:
    """Applies one plan to a stream of requests, counting matches as it goes."""

    def __init__(self, plan: Optional[FaultPlan] = None) -> None:
        self.plan = plan
        self._matches = 0
        self._rng = random.Random(plan.seed if plan else 0)
        self.injections: list[dict[str, Any]] = []

    @property
    def matches(self) -> int:
        """How many requests the plan's tool filter has matched so far."""
        return self._matches

    def decide(self, message: Message) -> Decision:
        plan = self.plan
        if plan is None or message.get("method") != "tools/call":
            return PASS
        name = tool_name(message)
        if plan.tool is not None and name != plan.tool:
            return PASS
        self._matches += 1
        if plan.occurrences:
            hit = self._matches in plan.occurrences
        else:
            hit = self._rng.random() < plan.probability
        if not hit:
            return PASS
        decision = Decision(
            action=_ACTION_BY_MODE[plan.fault.id],
            mode=plan.fault,
            delay_s=plan.delay_s,
            keep=plan.keep,
        )
        self.injections.append(
            {"occurrence": self._matches, "tool": name, "mode": plan.fault.id}
        )
        return decision


def truncate_list_argument(message: Message, keep: int) -> tuple[Message, dict[str, Any]]:
    """Return a copy of a `tools/call` with its longest list argument cut down.

    F5 needs a call that does several things, and in MCP that shows up as a
    list-valued argument (paths, edits, rows, ids). The longest list is the
    one most likely to be the batch. Returns the message unchanged, and an
    empty detail, when no list argument has more than `keep` elements.
    """
    params = message.get("params")
    if not isinstance(params, dict):
        return message, {}
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        return message, {}
    candidates = [
        (key, value)
        for key, value in arguments.items()
        if isinstance(value, list) and len(value) > keep
    ]
    if not candidates:
        return message, {}
    key, value = max(candidates, key=lambda kv: len(kv[1]))
    trimmed = dict(message)
    trimmed["params"] = dict(params)
    trimmed["params"]["arguments"] = dict(arguments)
    trimmed["params"]["arguments"][key] = value[:keep]
    return trimmed, {"argument": key, "from": len(value), "to": keep}
