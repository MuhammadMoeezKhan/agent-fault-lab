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

The setup block is deliberately small: files, a git repository (optionally
with files staged but uncommitted, which a commit tool needs, or left
untracked, which a staging tool needs), and SQL run
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
from .observers import (
    AppendLogObserver,
    CompositeObserver,
    FilesystemObserver,
    GitObserver,
    SqliteObserver,
)

#: A fixed identity, so a commit does not fail for want of a user.name on a
#: machine that has none.
#:
#: Commit *dates* are deliberately not pinned. GitObserver identifies a commit
#: by (subject, tree, parent count) rather than by hash, so a moving timestamp
#: cannot make two equivalent trials differ, and pinning the dates actively
#: breaks servers that read them through a library: GitPython rejects the ISO
#: 8601 form with a colon in the offset and fails the commit outright. Pin them
#: in a spec's own `env` if an observer is ever added that digests .git objects.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "agent-fault-lab",
    "GIT_AUTHOR_EMAIL": "audit@example.invalid",
    "GIT_COMMITTER_NAME": "agent-fault-lab",
    "GIT_COMMITTER_EMAIL": "audit@example.invalid",
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
        tables = config.get("tables")
        return lambda root: SqliteObserver(
            os.path.join(root, config["path"]),
            name=config.get("name", "sqlite"),
            tables=tuple(tables) if tables else None,
            ignore_columns=tuple(config.get("ignore_columns") or ()),
        )
    if kind == "git":
        ignore = config.get("ignore")
        return lambda root: GitObserver(
            os.path.join(root, config.get("path", "")),
            name=config.get("name", "git"),
            ignore=tuple(ignore) if ignore else None,
            ignore_empty_commits=bool(config.get("ignore_empty_commits", False)),
        )
    if kind == "append-log":
        # The logs are named by the spec, never guessed, and so are the keys
        # the audit commits to not measuring (preregistration 6.3).
        return lambda root: AppendLogObserver(
            os.path.join(root, config.get("path", "")),
            patterns=tuple(config.get("patterns") or ("*.jsonl", "*.log")),
            drop_keys=tuple(config.get("drop_keys") or ()),
            name=config.get("name", "append-log"),
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

            def run(*args: str) -> None:
                completed = subprocess.run(
                    ["git", "-C", repo, *args],
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                if completed.returncode != 0:
                    # A seed commit with nothing staged is the common mistake,
                    # so say that rather than surfacing a CalledProcessError
                    # from somewhere inside trial setup.
                    detail = (completed.stderr or completed.stdout).strip()
                    raise SpecError(
                        f"setup step `git {' '.join(args)}` failed in {repo}: "
                        f"{detail or completed.returncode}"
                    )

            run("init", "-q", "-b", git.get("branch", "main"))
            if git.get("commit", True):
                run("add", "-A")
                run("commit", "-qm", git.get("message", "seed"))
            # Files written and staged but left uncommitted. A commit tool can
            # only be audited when there is something staged for it to commit,
            # and whether a repeat finds the index empty is the behaviour under
            # test.
            def write(relative: str, content: str) -> None:
                path = os.path.join(repo, relative)
                os.makedirs(os.path.dirname(path) or repo, exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)

            staged = git.get("stage") or {}
            for relative, content in staged.items():
                write(relative, content)
            if staged:
                run("add", *staged.keys())
            # Written after the seed commit and left untracked, so a tool that
            # stages files has something to stage. Anything in `setup.files`
            # would have been swept into the seed commit instead.
            for relative, content in (git.get("untracked") or {}).items():
                write(relative, content)
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


def _cwd_is_root(server: dict[str, Any]) -> bool:
    """Guard the one setting that can let a server out of the trial root.

    With `cwd_is_root: false` the server inherits the directory the harness
    was started from. A server that writes relative to its working directory
    then writes there: @contentrain/mcp, pointed at a subdirectory while the
    process ran in the engine repository, initialised itself in that
    repository and auto-committed four commits onto its main branch. Opting
    out now has to be explicit about it.
    """
    if server.get("cwd_is_root", True):
        return True
    if server.get("allow_process_cwd"):
        return False
    raise SpecError(
        "cwd_is_root: false lets the server write into the directory the "
        "harness was started from, outside the trial root. If that is really "
        "wanted, set allow_process_cwd: true as well; otherwise put the "
        "fixture repository at the trial root (setup.git.path: \"\") and "
        "leave cwd_is_root alone."
    )


def _contained(server: dict[str, Any]) -> bool:
    """A spec file cannot opt a third-party server out of its container.

    Rule 9 is not a default to be overridden by the thing being audited. The
    two incidents that produced the rule, a skill file written into the real
    `~/.claude` and four auto-commits into the engine repository, both came
    from a server doing something on startup that no spec predicted, so the
    spec is exactly the wrong place to trust. The only uncontained servers are
    first-party fixtures constructed in code (tests/fake_server.py), which
    pass `contained=False` to ServerTarget directly and never go through a
    spec file.
    """
    if "contained" not in server:
        return True
    if server.get("contained"):
        return True
    raise SpecError(
        "contained: false is not allowed in a spec. Rule 9 runs every "
        "third-party server inside a container, and a server that misbehaves "
        "on startup is precisely the case the container exists for. If this "
        "is first-party code, build the ServerTarget in Python and pass "
        "contained=False there, with a comment saying why it is first-party."
    )


def _validate_preludes(
    *,
    target_prelude: tuple[tuple[str, dict[str, Any]], ...],
    fixtures: list[Fixture],
) -> None:
    """Refuse a prelude that calls a tool under test.

    A prelude runs inside the trial, before the first snapshot, so the fault
    is already armed. If a prelude calls the fixture's own tool it consumes
    the fault's first occurrence: the prelude itself fails, the trial aborts,
    and the row comes out BROKEN or, worse, NOT_MUTATING, which reads as a
    finding about the server rather than a mistake in the spec. That happened
    once, on a server whose delete needed an index the index tool had to
    create, and it cost an hour to see.

    The constraint was documented on `Target.prelude` from the start and
    enforced nowhere, which is the worst of both: a reader of the spec format
    could know the rule and a writer of a spec could still break it silently.
    """
    server_tools = {tool for tool, _ in target_prelude}
    for fixture in fixtures:
        if fixture.tool in server_tools:
            raise SpecError(
                f"the server prelude calls {fixture.tool!r}, which fixture "
                f"{fixture.identity!r} is testing. A prelude runs with the "
                f"fault already armed, so it would consume the fault's first "
                f"occurrence and the fixture's verdict would be meaningless. "
                f"Move that call into the fixtures that need it, as their own "
                f"prelude."
            )
        own = {tool for tool, _ in fixture.prelude}
        if fixture.tool in own:
            raise SpecError(
                f"fixture {fixture.identity!r} has {fixture.tool!r} in its "
                f"own prelude, which is the tool it is testing. Same reason: "
                f"the prelude call would take the fault."
            )


def _prelude(raw: Any) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Parse `prelude`: calls every trial makes before the fixture.

    Accepts `[["tool", {...}], ...]` or `[{"tool": ..., "arguments": {...}}]`.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SpecError("server.prelude must be a list of calls")
    calls: list[tuple[str, dict[str, Any]]] = []
    for entry in raw:
        if isinstance(entry, dict):
            tool = entry.get("tool")
            arguments = entry.get("arguments") or {}
        elif isinstance(entry, (list, tuple)) and entry:
            tool = entry[0]
            arguments = entry[1] if len(entry) > 1 else {}
        else:
            raise SpecError(f"unreadable prelude call: {entry!r}")
        if not isinstance(tool, str) or not tool:
            raise SpecError(f"a prelude call needs a tool name: {entry!r}")
        if not isinstance(arguments, dict):
            raise SpecError(f"prelude arguments for {tool} must be an object")
        calls.append((tool, arguments))
    return tuple(calls)


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
        prelude=_prelude(server.get("prelude")),
        cwd_is_root=_cwd_is_root(server),
        contained=_contained(server),
        container_command=tuple(server.get("container_command", ()) or ()),
        image=str(server.get("image", "agent-fault-lab:audit")),
        version=str(server.get("version", "")),
        timeout_s=float(server.get("timeout_s", 20.0)),
        quiesce_s=float(server.get("quiesce_s", 0.0)),
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
                prelude=_prelude(raw.get("prelude")),
                note=raw.get("note", ""),
                id=raw.get("id", ""),
            )
        )
    _assign_ids(fixtures)
    if not fixtures:
        raise SpecError("the spec needs at least one fixture")
    _validate_preludes(target_prelude=_prelude(server.get("prelude")), fixtures=fixtures)
    return target, fixtures


def load_tasks(path: str) -> tuple[ServerTarget, list["AgentTask"]]:
    """Load a task file: a pointer to a server spec plus the agent tasks.

    The spec path is resolved relative to the task file, so a task file and
    the spec it names travel together.
    """
    from .consequence import AgentTask

    with open(path, encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path}: {exc}") from exc
    spec_path = document.get("spec")
    if not spec_path:
        raise SpecError(f"{path}: a task file needs a 'spec' pointing at a server")
    if not os.path.isabs(spec_path):
        # Try, in order: next to the task file, relative to the working
        # directory, and relative to the task file's parent directory. That
        # covers a task file kept beside its spec and one kept in a tasks/
        # subdirectory referring to the spec by its repository path.
        here = os.path.dirname(os.path.abspath(path))
        candidates = [
            os.path.join(here, spec_path),
            os.path.abspath(spec_path),
            os.path.normpath(os.path.join(here, "..", spec_path)),
            os.path.normpath(os.path.join(here, "..", "..", spec_path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                spec_path = candidate
                break
        else:
            raise SpecError(
                f"{path}: cannot find the server spec {document['spec']!r}; "
                f"looked in {', '.join(candidates)}"
            )
    target, _ = load(spec_path)

    tasks = []
    seen: set[str] = set()
    for raw in document.get("tasks") or []:
        for required in ("id", "instruction", "reference", "target_tool"):
            if required not in raw:
                raise SpecError(f"{path}: a task is missing '{required}'")
        if raw["id"] in seen:
            raise SpecError(f"{path}: duplicate task id {raw['id']!r}")
        seen.add(raw["id"])
        reference = tuple(
            (name, arguments) for name, arguments in raw["reference"]
        )
        if not reference:
            raise SpecError(f"{path}: task {raw['id']} has an empty reference")
        tasks.append(
            AgentTask(
                id=raw["id"],
                instruction=raw["instruction"],
                reference=reference,
                target_tool=raw["target_tool"],
                confirmable=raw.get("confirmable"),
                note=raw.get("note", ""),
            )
        )
    if not tasks:
        raise SpecError(f"{path}: no tasks")
    return target, tasks


def _assign_ids(fixtures: list[Fixture]) -> None:
    """Give every fixture a unique id, defaulting to the tool name.

    Two fixtures for one tool would otherwise aggregate together in reports,
    which is wrong whenever they exercise different semantics.
    """
    used: set[str] = set()
    for fixture in fixtures:
        if fixture.id:
            if fixture.id in used:
                raise SpecError(f"duplicate fixture id {fixture.id!r}")
            used.add(fixture.id)
    counters: dict[str, int] = {}
    for fixture in fixtures:
        if fixture.id:
            continue
        candidate = fixture.tool
        while candidate in used:
            counters[fixture.tool] = counters.get(fixture.tool, 1) + 1
            candidate = f"{fixture.tool}#{counters[fixture.tool]}"
        fixture.id = candidate
        used.add(candidate)


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
