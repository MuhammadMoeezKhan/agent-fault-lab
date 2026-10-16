"""Git state: refs, commit history and the working tree, by content."""

from __future__ import annotations

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
    """

    def __init__(
        self, repo: str, *, name: str = "git", max_commits: int = 200, git: str = "git"
    ) -> None:
        self.repo = os.path.abspath(repo)
        self.name = name
        self.max_commits = max_commits
        self.git = git

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
                "--format=%s%x1f%T%x1f%p",
                "--all",
            )
            or ""
        )
        commits = []
        for line in log.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            subject = parts[0]
            tree = parts[1][:12] if len(parts) > 1 else ""
            parents = len(parts[2].split()) if len(parts) > 2 and parts[2].strip() else 0
            commits.append([subject, tree, parents])
        # Sorted, because `--all` orders branches by date and a duplicate
        # commit must compare equal regardless of which order it is listed in.
        state["commits"] = sorted(commits, key=lambda c: (c[0], c[1], c[2]))
        state["commit_count"] = len(commits)

        status = self._run("status", "--porcelain=v1", "--untracked-files=all") or ""
        state["status"] = sorted(line for line in status.splitlines() if line.strip())

        stash = self._run("stash", "list", "--format=%gd %s") or ""
        state["stash"] = [line.strip() for line in stash.splitlines() if line.strip()]
        return state
