"""Append-only logs, compared by how much they grew rather than by content.

A replay oracle compares snapshots by content, which makes it blind to an
effect whose only durable trace carries a wall clock timestamp. stashpad is
the case that forced this: its write tools append one line per call to
`audit.jsonl` and `feedback.jsonl`, each stamped to the millisecond, so a
clean baseline never reproduces, every row comes out UNPROVEN, and the
fixture teaches nothing. Excluding those paths buys a verdict on the keyed
state at a price that had to be written into threats to validity: a repeat of
`stashpad_feedback` appends a second line nobody can see, so the verdict reads
safe-by-absorption while the log in fact grew.

The bias is not random. Append-only audit logs are commonest in the servers
that take safety seriously, so excluding them understates duplication among
careful servers.

This observer recovers the part of such a file that is comparable. Two things
are recorded per file:

- **`lines`**, the count. This is the measure that matters for duplication:
  one application against two differ by the number of appends, whatever the
  timestamps say.
- **`content`**, a digest over the lines with declared keys dropped, when
  every line parses as JSON and `drop_keys` is given. That tells a duplicate
  apart from an unrelated write of the same size.

When a file's lines do not all parse as JSON, or no keys were declared, the
entry records `mode: "count-only"`. The weaker treatment is named in the
snapshot rather than hidden in it, so a report cannot quietly present a
count-only comparison as a content comparison.

What this observer must never do is strip a field that an effect could change.
`drop_keys` is declared per spec, like an observer projection (preregistration
6.3), and a key named there is one the audit has committed to not measuring.
"""

from __future__ import annotations

import hashlib
import json
import os
from fnmatch import fnmatch
from typing import Any, Iterable, Sequence

from .base import Snapshot


class AppendLogObserver:
    """Snapshots append-only log files by line count, and by scrubbed content.

    `patterns` are fnmatch patterns against the path relative to `root`, so a
    spec names the logs explicitly instead of this guessing which files are
    logs. Nothing matching is not an error: a log that no call has created yet
    is simply absent from the snapshot, which is also how its first line shows
    up as a change.
    """

    def __init__(
        self,
        root: str,
        *,
        patterns: Sequence[str] = ("*.jsonl", "*.log"),
        drop_keys: Iterable[str] = (),
        name: str = "append-log",
        max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.root = os.path.abspath(root)
        self.patterns = tuple(patterns)
        self.drop_keys = frozenset(drop_keys)
        self.name = name
        self.max_bytes = max_bytes

    def _matches(self, relative: str) -> bool:
        return any(
            fnmatch(relative, pattern) or fnmatch(os.path.basename(relative), pattern)
            for pattern in self.patterns
        )

    def snapshot(self) -> Snapshot:
        state: Snapshot = {}
        if not os.path.isdir(self.root):
            return state
        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, self.root)
                if not self._matches(relative) or os.path.islink(path):
                    continue
                state[relative] = self._entry(path)
        return state

    def _entry(self, path: str) -> dict[str, Any]:
        try:
            if os.path.getsize(path) > self.max_bytes:
                return {"mode": "too-large", "bytes": os.path.getsize(path)}
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = [line for line in handle.read().split("\n") if line.strip()]
        except OSError as error:
            return {"mode": "unreadable", "error": type(error).__name__}
        entry: dict[str, Any] = {"lines": len(lines)}
        scrubbed = self._scrub(lines) if self.drop_keys else None
        if scrubbed is None:
            entry["mode"] = "count-only"
            return entry
        entry["mode"] = "scrubbed-content"
        entry["dropped"] = sorted(self.drop_keys)
        payload = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))
        entry["content"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return entry

    def _scrub(self, lines: Sequence[str]) -> list[Any] | None:
        """Every line as JSON with the declared keys dropped, or None.

        All or nothing on purpose. A file where only some lines parse would
        give a digest over a mixture of scrubbed objects and raw text, which
        compares equal or unequal for reasons no reader could reconstruct.
        """
        out: list[Any] = []
        for line in lines:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(record, dict):
                record = {
                    key: value
                    for key, value in record.items()
                    if key not in self.drop_keys
                }
            out.append(record)
        return out
