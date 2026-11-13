"""Zip-container documents, compared entry by entry with declared keys dropped.

A whole class of writers in the sample produce a zip container rather than a
plain file: `.xmind`, `.docx`, `.xlsx`, `.pptx` and the ODF formats are all
zip archives holding XML or JSON. Compared by bytes, none of them can ever
reproduce a clean baseline, for two reasons that have nothing to do with what
the tool did:

  - a zip stores a modification time per entry, so the same content written a
    second later is a different file,
  - compression is not required to be deterministic across runs.

That is enough to make every row UNPROVEN. The first such server in the sample
(an outliner writing `.xmind`) looked unobservable for exactly this reason,
and it is not: unpacking the archive showed **two** moving values in one of
its three entries, a created and a modified timestamp, with everything else
byte-identical, including the document body that carries the actual effect.

So this observer records, per archive, the sorted entry names and a digest of
each entry's projected content. Entry names are part of the state: an archive
that gained an entry changed. JSON entries are projected through the same
`drop_keys` contract as `JsonDocObserver`, so a declared timestamp is dropped
wherever it appears; non-JSON entries are compared by their uncompressed
bytes, which are stable even when the container's framing is not.

Two kinds of drop are declared. `drop_keys` projects a key out of a JSON
entry. `drop_entries` names a whole entry, which is what a `.docx` needs: two
identical writes differ in `dcterms:created` and `dcterms:modified` inside
`docProps/core.xml`, that entry is XML rather than JSON, and parsing OOXML to
reach two tags would widen this observer's contract for one format. A dropped
entry keeps its name in the comparison, so an archive that gained or lost it
is still a changed archive, and loses only its content.

The drops are declared per spec and derived the way preregistration 6.3
requires: by diffing two identical clean runs and naming what moved, never by
reading a key's name and guessing. The snapshot records the drop list beside
the digest, so a reader sees what was given up and a spec cannot widen the
projection without the report changing.

What this deliberately does not do is normalise content it was not told to.
An entry whose body carries a generated identifier stays non-reproducible and
the fixture stays UNPROVEN, which is the honest outcome: the alternative is an
observer that quietly launders away the differences a duplicate would show.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import zipfile
from typing import Any, Iterable, Sequence

from .base import Snapshot
from .json_doc import _strip

#: Beyond this an archive is recorded by size and digest rather than opened,
#: so a server that writes something enormous cannot stall a snapshot.
MAX_BYTES = 32 * 1024 * 1024


class ZipDocObserver:
    """Snapshots zip-container documents as entry names plus projected digests."""

    def __init__(
        self,
        root: str,
        *,
        patterns: Sequence[str] = ("*.xmind", "*.docx", "*.xlsx", "*.pptx"),
        drop_keys: Iterable[str] = (),
        drop_entries: Iterable[str] = (),
        name: str = "zip-doc",
        max_bytes: int = MAX_BYTES,
    ) -> None:
        self.root = os.path.abspath(root)
        self.patterns = tuple(patterns)
        drops = tuple(drop_keys)
        self.drop_keys = drops
        self.drop_entries = tuple(drop_entries)
        self.names = frozenset(key for key in drops if "." not in key)
        self.paths = frozenset(key for key in drops if "." in key)
        self.name = name
        self.max_bytes = max_bytes

    def _matches(self, relative: str) -> bool:
        return any(
            fnmatch.fnmatch(relative, pattern)
            or fnmatch.fnmatch(os.path.basename(relative), pattern)
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
            size = os.path.getsize(path)
            if size > self.max_bytes:
                return {"mode": "too-large", "bytes": size}
        except OSError as error:
            return {"mode": "unreadable", "error": type(error).__name__}
        try:
            with zipfile.ZipFile(path) as archive:
                members = sorted(archive.namelist())
                entries = {
                    member: self._member(archive, member)
                    for member in members
                    if not self._dropped(member)
                }
        except (OSError, zipfile.BadZipFile):
            # Not a zip after all, or truncated mid-write. Compared by bytes
            # rather than skipped: a half-written archive is an effect, and
            # skipping it would make it look like no file at all.
            try:
                with open(path, "rb") as handle:
                    raw = handle.read()
            except OSError as error:
                return {"mode": "unreadable", "error": type(error).__name__}
            return {
                "mode": "not-a-zip",
                "content": hashlib.sha256(raw).hexdigest()[:16],
            }
        return {
            "mode": "projected" if (self.drop_keys or self.drop_entries) else "verbatim",
            "dropped": sorted(self.drop_keys),
            "dropped_entries": sorted(self.drop_entries),
            "members": members,
            "entries": entries,
        }

    def _dropped(self, member: str) -> bool:
        """Whether this entry is one the spec declared out of the comparison.

        Declared whole-entry drops exist because `drop_keys` can only project
        a JSON entry, and the moving values are not always in JSON. A `.docx`
        written twice from the same input differs in exactly two values,
        `dcterms:created` and `dcterms:modified` in `docProps/core.xml`, and
        that entry is XML. Parsing OOXML to drop two tags would widen this
        observer's contract for one format; naming the entry does not.

        The cost is stated plainly: dropping an entry gives up every effect
        that would only have shown up inside it. That is acceptable only when
        two identical clean runs show the entry moving on its own and the
        entries carrying the effect (`word/document.xml` for a `.docx`) are
        byte-identical, which is the 6.3 procedure. The entry name is still
        compared, because it stays in `members`: an archive that gained or
        lost the entry altogether is still a changed archive.
        """
        return any(
            member == pattern or fnmatch.fnmatch(member, pattern)
            for pattern in self.drop_entries
        )

    def _member(self, archive: zipfile.ZipFile, member: str) -> str:
        if member.endswith("/"):
            return "dir"
        try:
            raw = archive.read(member)
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            return f"unreadable:{type(error).__name__}"
        if len(raw) > self.max_bytes:
            return f"too-large:{len(raw)}"
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            # XML and binary entries compare by their uncompressed bytes,
            # which do not carry the container's framing.
            return "bytes:" + hashlib.sha256(raw).hexdigest()[:16]
        canonical = json.dumps(
            _strip(document, self.names, self.paths),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "json:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
