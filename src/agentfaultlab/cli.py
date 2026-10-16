"""The command line: `simulate`, `proxy` and `check`.

`simulate` is the v0.1 scripted-policy grid, kept as it was. `proxy` puts a
fault between a real client and a real server. `check` runs the conformance
audit from a spec file and is the command a tool author would put in CI.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import report, spec
from .audit import Audit
from .mcp import taxonomy
from .mcp.session import TraceWriter


def _split_server_command(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split our own arguments from the server command after `--`."""
    arguments = list(argv)
    if "--" not in arguments:
        return arguments, []
    index = arguments.index("--")
    return arguments[:index], arguments[index + 1 :]


def _add_proxy(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "proxy",
        help="relay this process's stdio to an MCP server, injecting a fault",
        description=(
            "Point a real MCP client at this command instead of the server. "
            "The server command follows a bare --."
        ),
    )
    parser.add_argument(
        "--fault",
        help="fault spec, for example F1:write_file@1, F6:any@all (see --list-faults)",
    )
    parser.add_argument("--trace", help="write a JSONL trace of every message here")
    parser.add_argument("--delay", type=float, default=2.0, help="F3 hold, in seconds")
    parser.add_argument("--keep", type=int, default=1, help="F5 batch prefix length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cwd", help="working directory for the server")


def _add_check(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "check",
        help="audit a server's tools against the fault taxonomy",
        description=(
            "Runs each fixture under each fault mode and classifies the "
            "backend's end state against clean baselines. Exits 1 if any "
            "tool duplicates an effect, loses one, or contradicts its hints."
        ),
    )
    parser.add_argument("spec", help="path to an audit spec (.json or .py)")
    parser.add_argument(
        "--faults",
        default=",".join(taxonomy.AUDIT_SEQUENCE),
        help="comma separated fault ids to run",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--tool", action="append", help="limit to this tool (repeatable)")
    parser.add_argument("--out", help="write the Markdown report here")
    parser.add_argument("--csv", help="write the rows here")
    parser.add_argument("--jsonl", help="write the rows as JSONL here")
    parser.add_argument("--trace", help="write a JSONL trace of every message here")
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit 0, even when findings are reported",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-fault-lab",
        description="Fault injection and state-checked audits for tool-calling agents.",
    )
    parser.add_argument(
        "--list-faults", action="store_true", help="print the fault taxonomy and exit"
    )
    subparsers = parser.add_subparsers(dest="command")

    simulate = subparsers.add_parser(
        "simulate", help="the scripted-policy grid from v0.1 (no servers, no models)"
    )
    simulate.add_argument("--tasks", type=int, default=200)
    simulate.add_argument("--trials", type=int, default=16)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--k", type=int, default=8)
    simulate.add_argument("--p-commit", type=float, default=0.5)
    simulate.add_argument("--out", default="results")

    _add_proxy(subparsers)
    _add_check(subparsers)
    return parser


def _list_faults() -> int:
    print("Fault taxonomy (ids are stable; see taxonomy.py for provenance)\n")
    seen = set()
    for mode in taxonomy.MODES.values():
        if mode.id in seen:
            continue
        seen.add(mode.id)
        print(f"{mode.id}  {mode.name}")
        print(f"    {mode.description}")
        print(f"    after one retry a correct tool leaves the effect: "
              f"{mode.effect_expected_after_retry}")
        print(f"    provenance: {mode.provenance}\n")
    return 0


def _run_check(args: argparse.Namespace) -> int:
    target, fixtures = spec.load(args.spec)
    if args.tool:
        wanted = set(args.tool)
        fixtures = [fixture for fixture in fixtures if fixture.tool in wanted]
        if not fixtures:
            print(f"no fixture matches {sorted(wanted)}", file=sys.stderr)
            return 2
    modes = [mode.strip() for mode in args.faults.split(",") if mode.strip()]
    for mode in modes:
        taxonomy.get(mode)

    trace = (
        TraceWriter(args.trace, context={"server": target.name, "spec": args.spec})
        if args.trace
        else None
    )
    audit = Audit(target, repeats=args.repeats, settle_s=args.settle, trace=trace)
    try:
        tools = {tool.name: tool for tool in audit.list_tools()}
        verdicts = []
        for fixture in fixtures:
            if fixture.tool not in tools:
                print(
                    f"warning: {target.name} does not expose {fixture.tool!r}; skipped",
                    file=sys.stderr,
                )
                continue
            verdicts.extend(audit.run_tool(fixture, tools[fixture.tool], modes=modes))
    finally:
        if trace is not None:
            trace.close()

    if not verdicts:
        print("nothing was audited: no fixture matched a tool the server exposes",
              file=sys.stderr)
        return 2
    markdown = report.to_markdown(verdicts, audit.server_info)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    else:
        print(markdown)
    if args.csv:
        report.to_csv(verdicts, args.csv)
    if args.jsonl:
        report.to_jsonl(verdicts, args.jsonl)
    return 0 if args.exit_zero else report.exit_code(verdicts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # `agent-fault-lab --tasks 10` kept working when subcommands arrived.
    if raw and raw[0].startswith("-") and raw[0] not in ("-h", "--help", "--list-faults"):
        raw = ["simulate", *raw]
    ours, server_command = _split_server_command(raw)
    parser = build_parser()
    args = parser.parse_args(ours)

    if getattr(args, "list_faults", False):
        return _list_faults()
    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        return 2

    if command == "simulate":
        from .run import run_simulation

        return run_simulation(args)
    if command == "proxy":
        if not server_command:
            print("a server command is required after --", file=sys.stderr)
            return 2
        from .mcp.proxy import run_proxy

        return run_proxy(
            server_command,
            fault=args.fault,
            trace_path=args.trace,
            delay_s=args.delay,
            keep=args.keep,
            seed=args.seed,
            cwd=args.cwd,
        )
    if command == "check":
        return _run_check(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
