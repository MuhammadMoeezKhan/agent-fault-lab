"""JSON documents, compared with declared keys dropped.

The replay oracle compares snapshots by content, so a server whose durable
state is one JSON document with a timestamp inside it can never reproduce a
clean baseline: every trial differs, every row reads UNPROVEN, and the fixture
teaches nothing. Three servers in the sample keep state that way, and the
first two were handled by excluding the file, which costs a verdict and biases
the result in a direction worth naming: a server that records when it wrote
something tends to be a server that was built carefully, so excluding such
files understates duplication among careful servers.

`AppendLogObserver` solved this for newline-delimited logs, where the count of
lines is the thing a duplicate changes. This is the other shape: a single
document, pretty printed, keyed by name, where a repeat overwrites the same key
and the only differences are a timestamp and a counter.

What it records per file is the document with the declared keys removed, so
two trials that stored the same payload compare equal, and a trial that stored
it twice does not. `drop_keys` entries come in two forms:

  - a bare name, such as `updated`, removes that key wherever it appears, at
    any depth. Timestamps are usually sprinkled through a document rather than
    collected at the top, so this is the common case.
  - a dotted path, such as `statistics.total_calculations`, removes exactly
    that location and nothing else. Use it when a name means one thing in one
    place and something measurable in another.

Both are declared per spec, like any observer projection (preregistration
6.3), and a key named there is one the audit has committed to not measuring.
That commitment is the whole safeguard, so the snapshot records which keys
were dropped alongside the digest: a reader can see what was given up, and a
spec cannot quietly widen the projection without the report changing.

A file that does not parse as JSON is recorded as `mode: "unparsed"` with its
digest, which still compares, rather than being skipped. Skipping it would
make a malformed document look like no document at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from fnmatch import fnmatch
from typing import Any, Iterable, Sequence

from .base import Snapshot


def _strip(value: Any, names: frozenset[str], paths: frozenset[str], trail: str = "") -> Any:
    """`value` with dropped keys removed, recursively.

    `names` match a key at any depth; `paths` match the dotted trail exactly.
    Lists keep their order and their length, because both are observable: a
    duplicate that appends to a list is exactly what this has to catch.
    """
    if isinstance(value, dict):
        out = {}
        for key, inner in value.items():
            here = f"{trail}.{key}" if trail else key
            if key in names or here in paths:
                continue
            out[key] = _strip(inner, names, paths, here)
        return out
    if isinstance(value, list):
        return [_strip(item, names, paths, trail) for item in value]
    return value


class JsonDocObserver:
    """Snapshots JSON documents, with declared keys projected away.

    `patterns` are fnmatch patterns against the path relative to `root`, so a
    spec names the documents explicitly instead of this guessing which files
    hold state. A document that no call has created yet is simply absent from
    the snapshot, which is also how its first appearance shows up as a change.
    """

    def __init__(
        self,
        root: str,
        *,
        patterns: Sequence[str] = ("*.json",),
        drop_keys: Iterable[str] = (),
        name: str = "json-doc",
        max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.root = os.path.abspath(root)
        self.patterns = tuple(patterns)
        drops = tuple(drop_keys)
        self.drop_keys = tuple(drops)
        self.names = frozenset(key for key in drops if "." not in key)
        self.paths = frozenset(key for key in drops if "." in key)
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
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError as error:
            return {"mode": "unreadable", "error": type(error).__name__}
        try:
            document = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError):
            return {
                "mode": "unparsed",
                "content": hashlib.sha256(raw).hexdigest()[:16],
            }
        stripped = _strip(document, self.names, self.paths)
        payload = json.dumps(stripped, sort_keys=True, separators=(",", ":"))
        return {
            "mode": "projected" if self.drop_keys else "verbatim",
            "dropped": sorted(self.drop_keys),
            "content": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        }
