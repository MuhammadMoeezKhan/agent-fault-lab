"""Effect observers: ground truth read from the backend, not from the tool.

An observer answers one question about a backend: what is in it right now?
It returns a canonical snapshot, meaning two snapshots of two backends that
hold the same state compare equal. That is the whole requirement, because the
audit classifies an effect by replay rather than by a hand written oracle:

1. run the operation once on a clean backend and snapshot it      (S1)
2. run it twice on a clean backend and snapshot it                (S2)
3. run the faulted trial on a clean backend and snapshot it       (Sf)

Sf == S1 means the effect applied exactly once, Sf == S2 means twice, Sf
equal to the clean snapshot means not at all, and anything else is reported
as `other` and inspected by hand. No step asks the tool or the agent what
happened, and nothing has to be known in advance about what the tool does.

Canonicalisation is therefore the one thing an observer must get right:
content digests instead of mtimes, sorted rows instead of insertion order,
commit messages and tree digests instead of commit hashes. Anything that
changes run to run without the effect changing has to be dropped here, or
every trial lands in `other`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

#: A canonical, JSON-serialisable view of a backend.
Snapshot = dict[str, Any]


@runtime_checkable
class Observer(Protocol):
    """Reads one backend's state."""

    name: str

    def snapshot(self) -> Snapshot:
        """The backend's current state, canonicalised."""
        ...


def digest(snapshot: Snapshot) -> str:
    """A stable digest of a snapshot, for compact logging and comparison."""
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def equal(left: Snapshot, right: Snapshot) -> bool:
    return digest(left) == digest(right)


def diff(before: Snapshot, after: Snapshot) -> dict[str, Any]:
    """A readable difference between two snapshots, for reports and traces."""
    added, removed, changed = {}, {}, {}
    for key in after.keys() | before.keys():
        if key not in before:
            added[key] = after[key]
        elif key not in after:
            removed[key] = before[key]
        elif before[key] != after[key]:
            changed[key] = {"before": before[key], "after": after[key]}
    return {"added": added, "removed": removed, "changed": changed}


def changed(before: Snapshot, after: Snapshot) -> bool:
    return not equal(before, after)


class CompositeObserver:
    """Several observers over one backend, snapshotted together.

    A single MCP server often touches more than one kind of state (a git
    server writes files and refs), and a classification is only sound if the
    snapshot covers everything the tool can reach.
    """

    def __init__(self, *observers: Observer, name: str = "composite") -> None:
        self.observers = observers
        self.name = name

    def snapshot(self) -> Snapshot:
        return {obs.name: obs.snapshot() for obs in self.observers}
