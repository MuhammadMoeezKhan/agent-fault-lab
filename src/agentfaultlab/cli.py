"""The command line: `simulate`, `proxy` and `check`.

`simulate` is the v0.1 scripted-policy grid, kept as it was. `proxy` puts a
fault between a real client and a real server. `check` runs the conformance
audit from a spec file and is the command a tool author would put in CI.
"""

from __future__ import annotations

import argparse
import os
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


def _add_consequence(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "consequence",
        help="run agent episodes against a server through the fault proxy",
        description=(
            "Gives a real agent a task on a real server, injects a fault, and "
            "scores what landed against what the agent claimed. The Claude "
            "Code arm is capped by the budget approved in DECISIONS.md Q3."
        ),
    )
    parser.add_argument("tasks", help="path to a task file (specs/tasks/*.json)")
    parser.add_argument(
        "--agent",
        default="ollama",
        help="ollama[:model], or claude[:model]",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=300.0,
        help=(
            "seconds to wait for one local-model reply. A thinking model can "
            "spend minutes on a single turn, and a reply that never arrives "
            "costs the whole episode, so raise this rather than lose cells"
        ),
    )
    parser.add_argument(
        "--conditions",
        default="clean,F1",
        help="comma separated: clean, or a fault id aimed at each task's target tool",
    )
    parser.add_argument("--arms", default="baseline", help="baseline,unconfirmed_allowed")
    parser.add_argument("--repeats", type=int, default=1, help="episodes per cell")
    parser.add_argument("--task", action="append", help="limit to this task id")
    parser.add_argument("--out", help="write the Markdown summary here")
    parser.add_argument("--csv", help="write the episode rows here")
    parser.add_argument("--trace-dir", help="write a proxy trace per episode here")
    parser.add_argument(
        "--ledger",
        default=os.path.expanduser("~/.agent-fault-lab/episodes.json"),
        help="where the daily episode budget is tracked",
    )
    parser.add_argument("--daily-cap", type=int, default=35)
    parser.add_argument(
        "--any-hour",
        action="store_true",
        help="skip the off-hours check (only if the approval has changed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the episode plan and the budget, and run nothing",
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
    _add_consequence(subparsers)
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


def _build_agent(descriptor: str, timeout_s: float = 300.0):
    """`ollama`, `ollama:qwen3:8b`, `claude`, or `claude:sonnet`."""
    kind, _, model = descriptor.partition(":")
    kind = kind.strip().lower()
    if kind == "ollama":
        from .agents_local import OllamaAgent, available

        if not available():
            raise SystemExit(
                "the local model server is not reachable at 127.0.0.1:11434. "
                "Start it with `ollama serve`."
            )
        return OllamaAgent(model=model or "qwen3:8b", timeout_s=timeout_s)
    if kind in ("claude", "claude-code"):
        from .consequence import ClaudeCodeAgent

        return ClaudeCodeAgent(model=model or "sonnet")
    raise SystemExit(f"unknown agent {descriptor!r}; use ollama[:model] or claude[:model]")


def _run_consequence(args: argparse.Namespace) -> int:
    from . import spec
    from .consequence import (
        ConsequenceRunner,
        EpisodeBudget,
        BudgetError,
        summarise,
    )

    target, tasks = spec.load_tasks(args.tasks)
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.id in wanted]
        if not tasks:
            print(f"no task matches {sorted(wanted)}", file=sys.stderr)
            return 2
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for condition in conditions:
        if condition != "clean":
            taxonomy.get(condition)

    planned = len(tasks) * len(conditions) * len(arms) * args.repeats
    agent = _build_agent(args.agent, timeout_s=args.agent_timeout)
    # Only the subscription arm draws on a rate limit that is shared with his
    # own interactive work, so only that arm is metered.
    meters = args.agent.split(":")[0].lower() in ("claude", "claude-code")
    budget = (
        EpisodeBudget(
            args.ledger, daily_cap=args.daily_cap, enforce_hours=not args.any_hour
        )
        if meters
        else None
    )

    print(f"agent: {agent.name}")
    print(
        f"plan: {len(tasks)} tasks x {len(conditions)} conditions x {len(arms)} arms "
        f"x {args.repeats} repeats = {planned} episodes"
    )
    if budget is not None:
        print(
            f"budget: {budget.used_today()} used today, {budget.remaining_today()} "
            f"left of {budget.daily_cap}; in window: {budget.in_window()}"
        )
        try:
            budget.check(planned)
        except BudgetError as exc:
            print(f"refusing to start: {exc}", file=sys.stderr)
            return 3
    if args.dry_run:
        for task in tasks:
            print(f"  {task.id}: target={task.target_tool}")
        return 0

    runner = ConsequenceRunner(target, budget=budget, trace_dir=args.trace_dir)
    results = []
    for task in tasks:
        for condition in conditions:
            fault = None if condition == "clean" else f"{condition}:{task.target_tool}@1"
            for arm in arms:
                for _ in range(args.repeats):
                    result = runner.run_episode(
                        task, agent, condition=condition, fault=fault, arm=arm
                    )
                    results.append(result)
                    flag = "ERR" if result.error else ""
                    print(
                        f"  {task.id:28s} {condition:6s} {arm:20s} "
                        f"state={result.multiplicity:6s} claim={result.claim:12s} "
                        f"{flag}"
                    )

    rows = summarise(results)
    lines = [
        "# Consequence study",
        "",
        "State comes from an observer reading the backend. The claim comes from "
        "the agent's final message, read by a keyword parser that still needs "
        "hand validation, so treat claim-derived rates as provisional.",
        "",
        "Rates are over completed episodes. An episode whose agent or server "
        "never answered is counted under errored and left out of every "
        "denominator, so a cell with few completed episodes carries little "
        "weight however clean its rates look.",
        "",
        "| agent | arm | condition | attempted | completed | task success "
        "| duplicate | false success | honest unconfirmed | false failure "
        "| errored |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent']} | {row['arm']} | {row['condition']} "
            f"| {row['episodes']} | {row['completed']} "
            f"| {row['task_success']} | {row['duplicate_effect']} "
            f"| {row['false_success']} | {row['honest_unconfirmed']} "
            f"| {row['false_failure']} | {row['errored']} |"
        )
    markdown = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    else:
        print()
        print(markdown)
    if args.csv:
        import csv as _csv

        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(
                handle, fieldnames=list(results[0].as_row())
            )
            writer.writeheader()
            for result in results:
                writer.writerow(result.as_row())
    return 0


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
    if command == "consequence":
        return _run_consequence(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
