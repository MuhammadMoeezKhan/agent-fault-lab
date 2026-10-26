#!/usr/bin/env python3
"""Start a server from its registry metadata and list what it declares.

Nothing is called. The probe reads `tools/list` only, so a server can be
planned for (or ruled out) from its declared tools and annotations rather than
from its README.

Two things it measures, both of which belong in the paper:

1. **Startability.** Whether the command the registry describes actually
   speaks MCP. Of the first candidates tried, three hung at `initialize`
   because their entry point needs a subcommand the registry does not list,
   and one declares a `bun run` script that is not runnable from the published
   package at all. "Could not be started from its own metadata" is a result.
2. **Annotation coverage.** How many mutating tools declare `idempotentHint`
   at all. A server that declares nothing gives a client nothing to decide a
   retry with, which is the gap the fault study is about.

Each server runs inside a throwaway tree, with `HOME` moved into it and only
`npm_config_cache` pointing back at the real download cache, so a launch stays
fast without the server being able to write into the real home.

That isolation was added after a probe did damage. An earlier version left
`HOME` alone, on the argument that npx needs its cache there. Starting
codex-supervisor-mcp and asking it for nothing but `tools/list` then installed
a skill file into `~/.claude/skills/`, in the real home directory, with no tool
call made and nothing asked. A server that edits the host agent's own
configuration when it starts is a finding in its own right, and it is also the
reason nothing here runs with the real HOME any more.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from agentfaultlab.mcp.session import ServerSession  # noqa: E402

#: Servers that keep their state in a fixed home directory need it moved, or
#: a probe would write into the user's real store. Values take `{root}`.
STATE_ENV = {
    "tabellio": {"TABELLIO_STORE": "{root}/store"},
    "@plur-ai/mcp": {"PLUR_HOME": "{root}/plur"},
}

#: Fallbacks tried when the registry's own command does not answer. Keeping
#: these separate from the registry argv is the point: the difference between
#: "runs as documented" and "runs only if you guess a subcommand" is a result.
FALLBACK_ARGV = {
    "mushroomdb": [["mcp", "{root}/db"]],
    "xlsx-for-ai": [["mcp"], ["serve"], ["--mcp"]],
    "tabellio": [["mcp"], ["serve"]],
    "@cyanheads/git-mcp-server": [[]],
}


def load_rows(path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("__snapshot__") or record.get("__trailer__"):
                continue
            rows[record.get("name", "")] = record
    return rows


def command_for(row: dict[str, Any], argv: list[str], root: str) -> list[str]:
    runner = row["runner"]
    base = [runner, "-y", row["package"]] if runner == "npx" else [runner, row["package"]]
    return base + [item.replace("{root}", root) for item in argv]


def probe_once(
    command: list[str],
    root: str,
    env_extra: dict[str, str],
    timeout_s: float,
    image: str | None = None,
) -> dict[str, Any]:
    """Start a candidate and read tools/list. Calls no tool.

    With `image`, the candidate runs inside the container (rule 9). That is
    the only mode allowed for a server that has not been audited before: the
    two startup incidents that produced the rule, a skill file written into
    the host agent's configuration and four auto-commits into this
    repository, both happened during a probe exactly like this one, with no
    tool ever called.
    """
    env = dict(os.environ)
    home = os.path.join(root, "home")
    os.makedirs(home, exist_ok=True)
    env["HOME"] = home
    env.setdefault("npm_config_cache", os.path.expanduser("~/.npm"))
    env.update({key: value.replace("{root}", root) for key, value in env_extra.items()})
    env["NO_COLOR"] = "1"
    containment = None
    if image:
        from agentfaultlab.container import Containment, installed_argv

        containment = Containment(image=image, mount=root)
        command = installed_argv(command)
        # The container carries its own HOME on a tmpfs, and the host npm
        # cache is meaningless inside it.
        env = {
            key: value
            for key, value in env.items()
            if key not in ("HOME", "npm_config_cache")
        }
    session = ServerSession(
        command, cwd=root, env=env, timeout_s=timeout_s, containment=containment
    )
    record: dict[str, Any] = {"command": command, "contained": bool(image)}
    try:
        with session:
            tools = session.list_tools()
            record["server_info"] = session.server_info
            record["started"] = True
            record["tools"] = [
                {
                    "name": tool.name,
                    "readOnlyHint": tool.read_only_hint,
                    "idempotentHint": tool.idempotent_hint,
                    "destructiveHint": tool.destructive_hint,
                    "properties": sorted(
                        (tool.input_schema.get("properties") or {}).keys()
                    ),
                    "description": (tool.description or "").strip()[:200],
                }
                for tool in tools
            ]
    except Exception as error:
        record["started"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        record["stderr_tail"] = list(getattr(session, "stderr_tail", []) or [])[-10:]
    return record


def build_probe_image(
    tag: str, candidates: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Install the candidates into the audit image, once, with network.

    A candidate that cannot be installed here cannot be audited at all under
    rule 9, which is a funnel category of its own rather than a failure to
    start: @wonderwhy-er/desktop-commander depends on fsevents, a macOS-only
    native module that node-gyp cannot build on Linux.

    The two runners are kept apart deliberately. An earlier version passed
    every candidate to `npm install -g`, so three PyPI packages came back as
    "cannot be installed in the audit image" without ever having been tried
    as Python packages. That is a funnel category being reported for a
    candidate that belongs in a different one, which is worse than a crash.
    """
    from agentfaultlab.container import build_image, require

    require()
    npm = [row["package"] for row in candidates if row["runner"] == "npx"]
    pypi = [row["package"] for row in candidates if row["runner"] == "uvx"]
    other = sorted({row["runner"] for row in candidates} - {"npx", "uvx"})
    if other:
        raise SystemExit(f"no install path for runner(s): {', '.join(other)}")
    build_image(tag, packages=npm, py_packages=pypi, quiet=False)
    installed: list[str] = []
    failed: list[str] = []
    for name, sink in (("installed", installed), ("failed", failed)):
        done = subprocess.run(
            ["docker", "run", "--rm", "--network=none", tag, "cat", f"/opt/afl/{name}"],
            capture_output=True,
            text=True,
        )
        sink.extend(line for line in done.stdout.split() if line)
    return {"installed": installed, "failed": failed}


def probe(
    row: dict[str, Any],
    timeout_s: float,
    allow_fallback: bool,
    image: str | None = None,
) -> dict[str, Any]:
    package = row["package"]
    env_extra = STATE_ENV.get(package, {})
    result: dict[str, Any] = {
        "server": row.get("name", ""),
        "package": package,
        "runner": row["runner"],
        "registry_argv": row.get("launch_argv") or [],
        "registry_needs_input": row.get("launch_needs_input") or [],
        "attempts": [],
    }
    with tempfile.TemporaryDirectory(prefix="afl-probe-") as root:
        os.makedirs(os.path.join(root, "db"), exist_ok=True)
        os.makedirs(os.path.join(root, "store"), exist_ok=True)
        candidates: list[tuple[str, list[str]]] = [
            ("registry", list(result["registry_argv"]))
        ]
        if allow_fallback:
            candidates += [
                ("guessed", argv) for argv in FALLBACK_ARGV.get(package, [])
            ]
        for source, argv in candidates:
            argv = [
                item.replace("{db_dir}", "{root}/db").replace("{store}", "{root}/store")
                for item in argv
            ]
            attempt = probe_once(
                command_for(row, argv, root), root, env_extra, timeout_s, image
            )
            attempt["argv_source"] = source
            result["attempts"].append(attempt)
            if attempt["started"]:
                result["started_via"] = source
                result["tools"] = attempt["tools"]
                result["server_info"] = attempt.get("server_info", {})
                break
        else:
            result["started_via"] = None
    return result


def annotation_coverage(tools: list[dict[str, Any]]) -> dict[str, int]:
    mutating = [tool for tool in tools if tool["readOnlyHint"] is not True]
    return {
        "tools": len(tools),
        "mutating_or_unmarked": len(mutating),
        "with_idempotent_hint": sum(
            1 for tool in mutating if tool["idempotentHint"] is not None
        ),
        "with_read_only_hint": sum(
            1 for tool in tools if tool["readOnlyHint"] is not None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("frame", help="a frame-all.jsonl written by sample_frame.py")
    parser.add_argument("names", nargs="+", help="registry server names to probe")
    parser.add_argument("--out", help="write the full result as JSON here")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--contain",
        metavar="IMAGE",
        help="probe inside this container image (rule 9). Required for any "
             "server not audited before",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="build --contain's image first, installing the candidates' "
             "packages, and report which could not be installed at all",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="try only the registry's own command, so startability is measured "
             "as documented",
    )
    args = parser.parse_args()

    # Checked before anything else is read or built. The rule is the reason
    # this script exists in this form, so it should fail on the flag rather
    # than after doing work that implies the flag was fine.
    if not args.contain:
        print(
            "refusing to probe on the host: rule 9 requires a container. "
            "Pass --contain IMAGE (add --build the first time).",
            file=sys.stderr,
        )
        return 2

    rows = load_rows(args.frame)
    if args.build:
        if not args.contain:
            print("--build needs --contain IMAGE", file=sys.stderr)
            return 2
        wanted = [rows[n] for n in args.names if n in rows]
        outcome = build_probe_image(args.contain, wanted)
        print(f"\nimage {args.contain}")
        print(f"  installed: {len(outcome['installed'])} {outcome['installed']}")
        print(f"  cannot install: {len(outcome['failed'])} {outcome['failed']}")
        uninstallable = set(outcome["failed"])
    else:
        uninstallable = set()
    results = []
    for name in args.names:
        row = rows.get(name)
        if row is None:
            print(f"{name}: not in the frame", file=sys.stderr)
            continue
        print(f"\n=== {name} ({row['package']})", flush=True)
        if row["package"] in uninstallable:
            print("  cannot be installed in the audit image, so not auditable")
            results.append(
                {
                    "server": name,
                    "package": row["package"],
                    "started_via": None,
                    "uninstallable": True,
                }
            )
            continue
        result = probe(row, args.timeout, not args.no_fallback, args.contain)
        results.append(result)
        if result["started_via"] is None:
            for attempt in result["attempts"]:
                print(f"  did not start: {' '.join(attempt['command'])}", flush=True)
                print(f"    {attempt.get('error')}", flush=True)
            continue
        coverage = annotation_coverage(result["tools"])
        print(
            f"  started via {result['started_via']} argv; "
            f"{coverage['tools']} tools, "
            f"{coverage['with_idempotent_hint']} of "
            f"{coverage['mutating_or_unmarked']} mutating tools declare "
            f"idempotentHint",
            flush=True,
        )
        result["coverage"] = coverage
        for tool in result["tools"]:
            if tool["readOnlyHint"] is True:
                continue
            print(
                f"    {tool['name']:<32} idem={tool['idempotentHint']} "
                f"destr={tool['destructiveHint']} ro={tool['readOnlyHint']}",
                flush=True,
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
