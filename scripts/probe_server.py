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

Each server runs with its working directory inside a throwaway tree. `HOME` is
NOT replaced: npx and uvx keep their caches there, and a throwaway HOME makes
every launch a cold install (which is how the first run took three minutes per
server and still timed out). State directories are redirected per server where
the server declares an environment variable for them.
"""
from __future__ import annotations

import argparse
import json
import os
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
    command: list[str], root: str, env_extra: dict[str, str], timeout_s: float
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update({key: value.replace("{root}", root) for key, value in env_extra.items()})
    env["NO_COLOR"] = "1"
    session = ServerSession(command, cwd=root, env=env, timeout_s=timeout_s)
    record: dict[str, Any] = {"command": command}
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


def probe(row: dict[str, Any], timeout_s: float, allow_fallback: bool) -> dict[str, Any]:
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
                command_for(row, argv, root), root, env_extra, timeout_s
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
        "--no-fallback",
        action="store_true",
        help="try only the registry's own command, so startability is measured "
             "as documented",
    )
    args = parser.parse_args()

    rows = load_rows(args.frame)
    results = []
    for name in args.names:
        row = rows.get(name)
        if row is None:
            print(f"{name}: not in the frame", file=sys.stderr)
            continue
        print(f"\n=== {name} ({row['package']})", flush=True)
        result = probe(row, args.timeout, not args.no_fallback)
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
