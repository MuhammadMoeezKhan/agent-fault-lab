"""Audit specs: a JSON file that says how to build a server and what to call.

One file describes one server, so a spec is the unit of contribution: a tool
author can send a spec for their own server and the audit runs it unchanged.
Everything that could differ between machines (paths, ports) is written as
`{root}`, which is replaced with a fresh temporary directory per trial.

    {
      "server": {
        "name": "reference-filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "{root}"],
        "version": "2026.9.1",
        "observer": {"kind": "filesystem"},
        "setup": {"files": {"notes.txt": "seed\n"}}
      },
      "fixtures": [
        {
          "tool": "write_file",
          "arguments": {"path": "{root}/notes.txt", "content": "hello"},
          "probes": [["read_text_file", {"path": "{root}/notes.txt"}]]
        }
      ]
    }

The setup block is deliberately small: files, a git repository, and SQL run
against a SQLite file. Anything more is a Python spec, which the loader also
accepts (a module exposing `TARGET` and `FIXTURES`).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from typing import Any, Callable, Optional

from .audit import Fixture, ServerTarget
from .observers import CompositeObserver, FilesystemObserver, GitObserver, SqliteObserver

#: A deterministic identity, so a git-backed trial is byte stable across runs.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "agent-fault-lab",
    "GIT_AUTHOR_EMAIL": "audit@example.invalid",
    "GIT_COMMITTER_NAME": "agent-fault-lab",
    "GIT_COMMITTER_EMAIL": "audit@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}


class SpecError(ValueError):
    """The spec cannot be understood, or names something that is not there."""


def _observer_factory(config: dict[str, Any]) -> Callable[[str], Any]:
    kind = config.get("kind")
    if kind == "filesystem":
        ignore = config.get("ignore")
        return lambda root: FilesystemObserver(
            os.path.join(root, config.get("path", "")),
            ignore=tuple(ignore) if ignore is not None else FilesystemObserver(root).ignore,
            name=config.get("name", "filesystem"),
        )
    if kind == "sqlite":
        if "path" not in config:
            raise SpecError("a sqlite observer needs a path")
        return lambda root: SqliteObserver(
            os.path.join(root, config["path"]), name=config.get("name", "sqlite")
        )
    if kind == "git":
        return lambda root: GitObserver(
            os.path.join(root, config.get("path", "")), name=config.get("name", "git")
        )
    if kind == "composite":
        parts = [_observer_factory(part) for part in config.get("parts", [])]
        if not parts:
            raise SpecError("a composite observer needs parts")
        return lambda root: CompositeObserver(*(part(root) for part in parts))
    raise SpecError(f"unknown observer kind {kind!r}")


def _setup_factory(config: Optional[dict[str, Any]]) -> Optional[Callable[[str], None]]:
    if not config:
        return None

    def setup(root: str) -> None:
        for relative, content in (config.get("files") or {}).items():
            path = os.path.join(root, relative)
            os.makedirs(os.path.dirname(path) or root, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        for relative in config.get("directories") or []:
            os.makedirs(os.path.join(root, relative), exist_ok=True)
        git = config.get("git")
        if git:
            repo = os.path.join(root, git.get("path", ""))
            os.makedirs(repo, exist_ok=True)
            environment = {**os.environ, **GIT_ENV}
            for args in (
                ["init", "-q", "-b", "main"],
                *( [["add", "-A"], ["commit", "-qm", git.get("message", "seed")]]
                   if git.get("commit", True) else [] ),
            ):
                subprocess.run(
                    ["git", "-C", repo, *args],
                    check=True,
                    capture_output=True,
                    env=environment,
                )
        for entry in config.get("sqlite") or []:
            path = os.path.join(root, entry["path"])
            os.makedirs(os.path.dirname(path) or root, exist_ok=True)
            connection = sqlite3.connect(path)
            try:
                connection.executescript(entry.get("sql", ""))
                connection.commit()
            finally:
                connection.close()

    return setup


def load(path: str) -> tuple[ServerTarget, list[Fixture]]:
    """Load a spec from JSON, or from a Python module exposing TARGET/FIXTURES."""
    if path.endswith(".py"):
        return _load_python(path)
    with open(path, encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path}: {exc}") from exc
    return from_document(document)


def from_document(document: dict[str, Any]) -> tuple[ServerTarget, list[Fixture]]:
    server = document.get("server")
    if not isinstance(server, dict):
        raise SpecError("the spec needs a 'server' object")
    for required in ("name", "command", "observer"):
        if required not in server:
            raise SpecError(f"server.{required} is required")
    target = ServerTarget(
        name=server["name"],
        command=list(server["command"]),
        observer=_observer_factory(server["observer"]),
        setup=_setup_factory(server.get("setup")),
        env={**GIT_ENV, **(server.get("env") or {})},
        cwd_is_root=bool(server.get("cwd_is_root", True)),
        version=str(server.get("version", "")),
        timeout_s=float(server.get("timeout_s", 20.0)),
    )
    fixtures = []
    for raw in document.get("fixtures") or []:
        if "tool" not in raw:
            raise SpecError("every fixture needs a 'tool'")
        fixtures.append(
            Fixture(
                tool=raw["tool"],
                arguments=raw.get("arguments") or {},
                probes=tuple(
                    (name, args) for name, args in (raw.get("probes") or [])
                ),
                note=raw.get("note", ""),
            )
        )
    if not fixtures:
        raise SpecError("the spec needs at least one fixture")
    return target, fixtures


def _load_python(path: str) -> tuple[ServerTarget, list[Fixture]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("afl_spec", path)
    if spec is None or spec.loader is None:
        raise SpecError(f"{path} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = getattr(module, "TARGET", None)
    fixtures = getattr(module, "FIXTURES", None)
    if not isinstance(target, ServerTarget) or not fixtures:
        raise SpecError(f"{path} must define TARGET and FIXTURES")
    return target, list(fixtures)
