"""PDF documents, compared by their bytes with declared trailer keys given up.

A PDF renderer stamps every file it writes with a creation date and a file
identifier, so two renders of the same input are never byte identical and a
byte observer reads every repeat as a change. The comparison here keeps the
whole file and blanks only the values of keys a spec declares, which is the
same declared-projection contract as `drop_keys` on the JSON observers
(preregistration 6.3): the spec says what the audit commits to not measuring,
and nothing else is given up.

A declared key's value may be written directly (`/CreationDate (D:...)`,
`/ID [<...> <...>]`) or as an indirect reference (`/CreationDate 17 0 R`), in
which case the referenced object's body is blanked instead. Values are
replaced by a fixed marker rather than removed, and the replacement is the
same in both files being compared, so a file that gained or lost a page, an
object or a byte of content still differs.

Objects inside compressed object streams are not reached. A spec that needs
one will see its fixture come back UNPROVEN, which is the honest outcome, and
the observer would then need extending rather than the spec loosening.

With `tally` set, file names are not compared: the snapshot is the count of
matching files and a multiset of their projected digests. That is for servers
that name each output by the time it was written, where a second application
leaves a second file under a new name and the names themselves say nothing.
"""
from __future__ import annotations

import collections
import fnmatch
import hashlib
import os
import re
from typing import Iterable

from .base import Snapshot

MAX_BYTES = 32 * 1024 * 1024

_MARK = b"(afl-dropped)"


def _key(name: str) -> bytes:
    return re.escape(name.encode("ascii"))


def project(raw: bytes, keys: Iterable[str]) -> bytes:
    """Blank the values of the declared keys, direct or indirect."""
    out = raw
    for name in keys:
        key = _key(name)
        # Indirect: `/Key N G R`. Blank the referenced object's body, then the
        # reference itself is left as it was, since the numbering is part of
        # the file's structure and is the same in both renders.
        for number, generation in re.findall(
            rb"/" + key + rb"\s+(\d+)\s+(\d+)\s+R", out
        ):
            out = re.sub(
                rb"(\b" + number + rb"\s+" + generation + rb"\s+obj\b)(.*?)(\bendobj\b)",
                lambda m: m.group(1) + b" " + _MARK + b" " + m.group(3),
                out,
                count=1,
                flags=re.S,
            )
        # Direct: a string, a hex string or an array.
        out = re.sub(
            rb"(/" + key + rb")\s*(\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[[^\]]*\])",
            lambda m: m.group(1) + b" " + _MARK,
            out,
            flags=re.S,
        )
    return out


class PdfDocObserver:
    """Snapshots PDF files, with declared keys projected away."""

    def __init__(
        self,
        root: str,
        *,
        patterns: Iterable[str] = ("*.pdf",),
        drop_keys: Iterable[str] = (),
        tally: bool = False,
        name: str = "pdf-doc",
        max_bytes: int = MAX_BYTES,
    ) -> None:
        self.root = os.path.abspath(root)
        self.patterns = tuple(patterns)
        self.drop_keys = tuple(sorted(drop_keys))
        self.tally = tally
        self.name = name
        self.max_bytes = max_bytes

    def _matches(self, relative: str) -> bool:
        return any(
            fnmatch.fnmatch(relative, pattern)
            or fnmatch.fnmatch(os.path.basename(relative), pattern)
            for pattern in self.patterns
        )

    def _payload(self, path: str) -> str:
        try:
            if os.path.getsize(path) > self.max_bytes:
                return f"too-large:{os.path.getsize(path)}"
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError as exc:
            return f"unreadable:{type(exc).__name__}"
        mode = "projected" if self.drop_keys else "verbatim"
        projected = project(raw, self.drop_keys)
        return f"{mode}:" + hashlib.sha256(projected).hexdigest()[:16]

    def snapshot(self) -> Snapshot:
        found: dict[str, str] = {}
        if os.path.isdir(self.root):
            for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
                dirnames.sort()
                for filename in sorted(filenames):
                    path = os.path.join(directory, filename)
                    relative = os.path.relpath(path, self.root)
                    if os.path.islink(path) or not self._matches(relative):
                        continue
                    found[relative] = self._payload(path)
        dropped = list(self.drop_keys)
        if self.tally:
            counts = collections.Counter(found.values())
            return {
                "count": len(found),
                "payloads": sorted(f"{d}*{n}" for d, n in counts.items()),
                "dropped": dropped,
            }
        return {"files": found, "dropped": dropped}
