"""Filesystem state: every path under a root, by content digest."""

from __future__ import annotations

import hashlib
import os
from fnmatch import fnmatch
from typing import Iterable

from .base import Snapshot

#: Paths that change without any tool touching them.
DEFAULT_IGNORE = (
    ".git/index",
    ".git/logs/*",
    ".git/COMMIT_EDITMSG",
    "*.pyc",
    "__pycache__/*",
    ".DS_Store",
    "*.db-journal",
    "*.db-wal",
    "*.db-shm",
)


class FilesystemObserver:
    """Snapshots a directory tree as path to content digest.

    Mode bits are recorded because a chmod is an effect; mtimes and sizes are
    not, since the digest already covers content and mtimes move on their own.
    Symlinks are recorded by target rather than followed, so a link swap shows
    up and a loop cannot hang the snapshot.
    """

    def __init__(
        self,
        root: str,
        *,
        ignore: Iterable[str] = DEFAULT_IGNORE,
        name: str = "filesystem",
        max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.root = os.path.abspath(root)
        self.ignore = tuple(ignore)
        self.name = name
        self.max_bytes = max_bytes

    def _ignored(self, relative: str) -> bool:
        return any(fnmatch(relative, pattern) for pattern in self.ignore)

    def snapshot(self) -> Snapshot:
        state: Snapshot = {}
        if not os.path.isdir(self.root):
            return state
        for directory, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames.sort()
            filenames.sort()
            for name in dirnames:
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, self.root)
                if self._ignored(relative):
                    continue
                if os.path.islink(path):
                    state[relative] = self._link(path)
                else:
                    state[relative] = {"kind": "dir", "mode": self._mode(path)}
            for name in filenames:
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, self.root)
                if self._ignored(relative):
                    continue
                state[relative] = self._file(path)
        return state

    def _file(self, path: str) -> dict[str, object]:
        if os.path.islink(path):
            return self._link(path)
        try:
            size = os.path.getsize(path)
            if size > self.max_bytes:
                # Too big to digest cheaply. Size alone still detects an
                # append, which is the duplicate shape that matters.
                return {"kind": "file", "size": size, "digest": "oversize"}
            hasher = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 16), b""):
                    hasher.update(chunk)
            return {
                "kind": "file",
                "digest": hasher.hexdigest()[:16],
                "size": size,
                "mode": self._mode(path),
            }
        except (OSError, ValueError) as exc:
            return {"kind": "unreadable", "error": type(exc).__name__}

    def _link(self, path: str) -> dict[str, object]:
        try:
            return {"kind": "symlink", "target": os.readlink(path)}
        except OSError as exc:
            return {"kind": "unreadable", "error": type(exc).__name__}

    @staticmethod
    def _mode(path: str) -> str:
        try:
            return oct(os.stat(path, follow_symlinks=False).st_mode & 0o7777)
        except OSError:
            return "?"
