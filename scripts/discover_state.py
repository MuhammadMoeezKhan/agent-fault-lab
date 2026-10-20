#!/usr/bin/env python3
"""Find out where a server keeps its state, by calling one tool and diffing.

A fixture needs three things the registry does not say: the exact arguments a
mutating tool requires, where the effect lands, and whether that place can be
observed canonically. This script answers all three for one tool at a time.

It starts the server with its own isolated HOME (so a probe never writes into
the real one) while keeping the shared npm cache, so a launch costs seconds
rather than a cold install. Then it snapshots the tree, calls the tool once,
and prints what changed, including anything the server wrote under its HOME.

`--schema` prints the tool's input schema and calls nothing, which is how to
work out the arguments before writing a fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from agentfaultlab.mcp.session import ServerSession  # noqa: E402

#: Files every npm or uvx launch writes, which are noise in a state diff.
NOISE = (
    "/.npm/", "/.cache/", "/.config/configstore/", "/.npmrc", "/.bun/",
    "/.local/share/uv/", "/.local/state/uv/", "/.DS_Store", "/.node_repl_history",
)


def walk(root: str) -> dict[str, str]:
    """Every file under `root`, by content digest. Directories are recorded too."""
    state: dict[str, str] = {}
    for current, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d != ".git" or True]
        for name in directories:
            path = os.path.join(current, name)
            state[os.path.relpath(path, root) + "/"] = "dir"
        for name in files:
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root)
            if any(part in "/" + relative for part in NOISE):
                continue
            try:
                with open(path, "rb") as handle:
                    state[relative] = hashlib.sha256(handle.read()).hexdigest()[:12]
            except OSError as error:
                state[relative] = f"unreadable: {error.errno}"
    return state


def diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        path for path in set(before) & set(after) if before[path] != after[path]
    )
    return {"added": added, "removed": removed, "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runner", required=True, choices=["npx", "uvx"])
    parser.add_argument("--package", required=True)
    parser.add_argument("--argv", nargs="*", default=[], help="server arguments")
    parser.add_argument("--tool", help="the tool to call once")
    parser.add_argument("--arguments", default="{}", help="JSON arguments for it")
    parser.add_argument("--schema", action="store_true", help="print schemas only")
    parser.add_argument("--probe", nargs="*", default=[],
                        help="read-only tools to call after, as tool:JSON")
    parser.add_argument("--before", nargs="*", default=[],
                        help="calls to make before the diff (session binding, "
                             "a project root), as tool:JSON")
    parser.add_argument("--git", action="store_true",
                        help="make the working directory a git repository with "
                             "one commit, which some servers require")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--keep", action="store_true", help="do not delete the tree")
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="afl-state-")
    home = os.path.join(root, "home")
    work = os.path.join(root, "work")
    os.makedirs(home)
    os.makedirs(work)

    env = dict(os.environ)
    env["HOME"] = home
    # The shared npm cache is what keeps an isolated HOME affordable.
    env.setdefault("npm_config_cache", os.path.expanduser("~/.npm"))
    env["NO_COLOR"] = "1"

    command = (
        [args.runner, "-y", args.package] if args.runner == "npx"
        else [args.runner, args.package]
    )
    command += [item.replace("{work}", work) for item in args.argv]
    if args.git:
        import subprocess

        identity = [
            "-c", "user.name=Audit Fixture",
            "-c", "user.email=fixture@example.invalid",
        ]
        subprocess.run(["git", "init", "-q", "-b", "main", work], check=True)
        with open(os.path.join(work, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("audit fixture repository\n")
        subprocess.run(["git", "-C", work, "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", work] + identity + ["commit", "-q", "-m", "seed"],
            check=True,
        )

    print(f"command: {' '.join(command)}")
    print(f"root: {root}")

    try:
        with ServerSession(command, cwd=work, env=env, timeout_s=args.timeout) as session:
            tools = {tool.name: tool for tool in session.list_tools()}
            if args.schema:
                names = [args.tool] if args.tool else sorted(tools)
                for name in names:
                    tool = tools.get(name)
                    if tool is None:
                        print(f"{name}: no such tool")
                        continue
                    print(f"\n=== {name}")
                    print(f"annotations: {json.dumps(tool.annotations)}")
                    print(json.dumps(tool.input_schema, indent=2)[:4000])
                return 0

            if not args.tool:
                parser.error("--tool is required unless --schema is given")
            for spec in args.before:
                name, _, raw = spec.partition(":")
                pre = session.call_tool(
                    name, json.loads((raw or "{}").replace("{work}", work))
                )
                body = json.dumps(pre.result or pre.error or {})
                print(f"before {name}: saw_success={pre.caller_saw_success} "
                      f"{body[:400]}")
            before = walk(root)
            arguments = json.loads(args.arguments.replace("{work}", work))
            call = session.call_tool(args.tool, arguments)
            after = walk(root)
            print(f"\ncall: {args.tool} {json.dumps(arguments)[:300]}")
            print(f"caller saw success: {call.caller_saw_success}")
            text = json.dumps(call.result or call.error or {})
            print(f"reply: {text[:600]}")
            changes = diff(before, after)
            print("\nstate diff (paths relative to the root):")
            for kind, paths in changes.items():
                for path in paths:
                    print(f"  {kind:<8} {path}")
            if not any(changes.values()):
                print("  nothing changed on disk")
            for spec in args.probe:
                name, _, raw = spec.partition(":")
                probe_call = session.call_tool(
                    name, json.loads((raw or "{}").replace("{work}", work))
                )
                body = json.dumps(probe_call.result or probe_call.error or {})
                print(f"\nprobe {name}: {body[:600]}")
    finally:
        if args.keep:
            print(f"\nkept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
