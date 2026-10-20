"""Git state: refs, commit history and the working tree, by content."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import subprocess
from typing import Optional

from .base import Snapshot


class GitObserver:
    """Snapshots a git repository without using commit hashes as identity.

    A commit hash includes a timestamp, so two runs of the same commit differ
    by hash while being the same effect. Commits are therefore recorded as
    (subject, tree digest, parent count), which is stable across runs and
    still distinguishes one commit from two identical ones: a duplicated
    commit shows up as the same pair twice.

    The staged and unstaged views come from `git status --porcelain`, so a
    tool that stages without committing is still visible.

    `ignore` names paths whose contents change on every run and carry no
    effect. @contentrain/mcp auto-commits a `.contentrain/context.json` that
    differs between two identical runs, which made every tree digest differ
    and left the server unauditable. With those paths excluded, a commit's
    identity is computed from the blob ids of everything else instead of from
    git's own tree hash. What was excluded is recorded in the snapshot.
    """

    def __init__(
        self,
        repo: str,
        *,
        name: str = "git",
        max_commits: int = 200,
        git: str = "git",
        ignore: Optional[tuple[str, ...]] = None,
        ignore_empty_commits: bool = False,
    ) -> None:
        self.repo = os.path.abspath(repo)
        self.name = name
        self.max_commits = max_commits
        self.git = git
        self.ignore = tuple(ignore or ())
        self.ignore_empty_commits = ignore_empty_commits

    def _ignored(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.ignore)

    def _tree_digest(self, commit: str) -> str:
        """A commit's content, by blob id, with the ignored paths left out."""
        listing = self._run("ls-tree", "-r", commit) or ""
        entries = []
        for line in listing.splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            parts = meta.split()
            blob = parts[2] if len(parts) > 2 else ""
            if self._ignored(path):
                continue
            entries.append(f"{path}\x00{blob}")
        return hashlib.sha256(
            "\n".join(sorted(entries)).encode("utf-8")
        ).hexdigest()[:12]

    def _run(self, *args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                [self.git, "-C", self.repo, *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    def snapshot(self) -> Snapshot:
        if not os.path.isdir(os.path.join(self.repo, ".git")):
            return {"__repo__": False}
        state: Snapshot = {"__repo__": True}

        head = self._run("symbolic-ref", "--quiet", "--short", "HEAD")
        state["head"] = head.strip() if head else "detached"

        refs = self._run("for-each-ref", "--format=%(refname)") or ""
        state["refs"] = sorted(line.strip() for line in refs.splitlines() if line.strip())

        log = (
            self._run(
                "log",
                f"--max-count={self.max_commits}",
                "--format=%s%x1f%T%x1f%p%x1f%H%x1f%P",
                "--all",
            )
            or ""
        )
        raw = []
        for line in log.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            subject = parts[0]
            tree = parts[1][:12] if len(parts) > 1 else ""
            parents = len(parts[2].split()) if len(parts) > 2 and parts[2].strip() else 0
            commit = parts[3].strip() if len(parts) > 3 else ""
            parent_ids = parts[4].split() if len(parts) > 4 else []
            if self.ignore and commit:
                tree = self._tree_digest(commit)
            raw.append(
                {
                    "subject": subject,
                    "tree": tree,
                    "parents": parents,
                    "commit": commit,
                    "parent_ids": parent_ids,
                }
            )

        trees = {entry["commit"]: entry["tree"] for entry in raw if entry["commit"]}
        commits = []
        empty = 0
        for entry in raw:
            if (
                self.ignore_empty_commits
                and len(entry["parent_ids"]) == 1
                and trees.get(entry["parent_ids"][0]) == entry["tree"]
            ):
                # A commit whose tree matches its parent's changed nothing
                # that is being observed, so it is history noise rather than
                # an effect. @contentrain/mcp commits a context file after
                # every call and commits a vocabulary save twice when the
                # call is retried: the second commit has the same tree as the
                # first, so the content is idempotent exactly as the tool
                # declares, and counting that commit as a duplicated effect
                # would have been a false annotation violation.
                empty += 1
                continue
            commits.append([entry["subject"], entry["tree"], entry["parents"]])
        if self.ignore_empty_commits:
            # Deliberately not recorded in the snapshot: the number of
            # dropped commits is itself different after one application and
            # two, so putting it in the state would reintroduce exactly the
            # difference this is meant to remove.
            del empty
        # Sorted, because `--all` orders branches by date and a duplicate
        # commit must compare equal regardless of which order it is listed in.
        state["commits"] = sorted(commits, key=lambda c: (c[0], c[1], c[2]))
        state["commit_count"] = len(commits)

        status = self._run("status", "--porcelain=v1", "--untracked-files=all") or ""
        state["status"] = sorted(
            line
            for line in status.splitlines()
            if line.strip() and not self._ignored(line[3:].strip().strip('"'))
        )
        if self.ignore:
            state["ignored_paths"] = list(self.ignore)

        stash = self._run("stash", "list", "--format=%gd %s") or ""
        state["stash"] = [line.strip() for line in stash.splitlines() if line.strip()]
        return state
