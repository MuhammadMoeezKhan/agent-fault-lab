"""Files whose names cannot be compared, counted by their contents instead.

Some servers write one file per call and put a timestamp and a random suffix
in the **filename**: `security-2026-10-29T02-48-39-073Z-5b3e82c1.json`. Every
trial therefore produces a different path, so a filesystem observer never
reproduces a clean baseline and every row reads UNPROVEN.

`stashpad`'s handoff files were handled by excluding them outright, and the
preregistration records that exclusion as a known bias. It is the same bias
`JsonDocObserver` was built to remove, pointing the other way: a server that
stamps its filenames tends to be one keeping an audit trail on purpose, so
excluding those files understates duplication among exactly the servers that
are trying hardest to be accountable. And for at least one server the
stamped file **is** the effect: there is nothing else to measure.

So this observer never looks at a name. It records how many files match, and
the multiset of their projected contents. Two trials that each wrote one
equivalent file compare equal however their files were named; a trial that
wrote two does not.

`drop_keys` works as it does for `JsonDocObserver`, bare names at any depth or
dotted paths, and is declared per spec (preregistration 6.3). The dropped keys
go into the snapshot beside the digests, so a reader can see what was given up
and a spec cannot quietly widen the projection without the report changing.

What this gives up, stated plainly: file names, and therefore any effect whose
only trace is a rename. A spec that needs names compared should use the
filesystem observer, and one that needs both should compose the two.
"""

from __future__ import annotations

import collections
import fnmatch
import hashlib
import json
import os
from typing import Any, Iterable, Optional

from .base import Snapshot
from .json_doc import _strip

#: Beyond this a file is recorded by size and digest rather than parsed, so a
#: server that writes something enormous cannot stall a snapshot.
MAX_BYTES = 4 * 1024 * 1024


class DirectoryTallyObserver:
    """Counts matching files and digests their projected contents."""

    def __init__(
        self,
        root: str,
        *,
        patterns: Iterable[str] = ("*.json",),
        drop_keys: Iterable[str] = (),
        name: str = "tally",
        max_bytes: int = MAX_BYTES,
    ) -> None:
        self.root = os.path.abspath(root)
        self.patterns = tuple(patterns)
        drops = tuple(drop_keys)
        self.names = frozenset(key for key in drops if "." not in key)
        self.paths = frozenset(key for key in drops if "." in key)
        self.dropped = tuple(sorted(drops))
        self.name = name
        self.max_bytes = max_bytes

    def _matches(self, filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.patterns)

    def _payload(self, path: str) -> str:
        try:
            raw = open(path, "rb").read(self.max_bytes + 1)
        except OSError as exc:
            return f"unreadable:{type(exc).__name__}"
        if len(raw) > self.max_bytes:
            return f"too-large:{len(raw)}:{hashlib.sha256(raw).hexdigest()[:16]}"
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Still compared, by bytes. Skipping it would make a malformed
            # file look like no file at all.
            return "unparsed:" + hashlib.sha256(raw).hexdigest()[:16]
        canonical = json.dumps(
            _strip(document, self.names, self.paths),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "projected:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:16]

    def snapshot(self) -> Snapshot:
        if not os.path.isdir(self.root):
            return {"__present__": False, "dropped": list(self.dropped)}
        payloads: list[str] = []
        for base, _dirs, files in os.walk(self.root):
            for filename in sorted(files):
                if self._matches(filename):
                    payloads.append(self._payload(os.path.join(base, filename)))
        # Sorted, so the order the walk happened to produce never matters, and
        # counted, so two equivalent files are not collapsed into one: writing
        # the same record twice under two names is exactly the duplicate this
        # observer exists to see.
        tally = collections.Counter(payloads)
        return {
            "count": len(payloads),
            "payloads": sorted(f"{digest}*{n}" for digest, n in tally.items()),
            "dropped": list(self.dropped),
        }
