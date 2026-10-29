"""The command line: `simulate`, `proxy` and `check`.

`simulate` is the v0.1 scripted-policy grid, kept as it was. `proxy` puts a
fault between a real client and a real server. `check` runs the conformance
audit from a spec file and is the command a tool author would put in CI.
"""

from __future__ import annotations

import argparse
import re
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
    parser.add_argument(
        "--contain",
        metavar="IMAGE",
        help=(
            "run the server inside this container image instead of on the "
            "host (rule 9: third-party servers are untrusted code)"
        ),
    )
    parser.add_argument(
        "--mount",
        help="host directory to mount as the container's workdir (default: --cwd)",
    )


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
        "--image",
        metavar="IMAGE",
        help=(
            "container image to run the server in, overriding the spec's "
            "(default: agent-fault-lab:audit). There is no flag for running a "
            "third-party server outside a container: rule 9 does not have an "
            "off switch"
        ),
    )
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
        "--config-dir",
        help=(
            "config directory for the hosted-agent arm, holding only a login. "
            "Keeps the operator's own CLAUDE.md, skills, plugins and hooks out "
            "of the measured session, which --strict-mcp-config does not do"
        ),
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=300.0,
        help=(
            "deadline for one local-model turn. The loop streams, so this "
            "tells a server that stopped answering apart from a model that is "
            "still producing tokens and never finishing. The second is "
            "recorded as non-termination, and raising this does not recover "
            "it: the same six episodes failed at 300s and at 900s"
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
    parser.add_argument(
        "--client-binary",
        metavar="PATH",
        help=(
            "the hosted client executable to measure, instead of whatever is "
            "on PATH. The client on PATH updates itself (2.1.310, then .311, "
            "then .312 within three days), so pinning a version means "
            "installing it somewhere of our own: `npm install --prefix DIR "
            "@anthropic-ai/claude-code@VERSION` and point this at "
            "DIR/node_modules/.bin/claude. Nothing in the operator's own "
            "installation is touched"
        ),
    )
    parser.add_argument(
        "--client-version",
        metavar="VERSION",
        help=(
            "refuse to run unless the hosted client reports exactly this "
            "version (DECISIONS.md 2026-10-27). An unpinned client is an "
            "unrecorded independent variable: it updated itself between two "
            "batches of one experiment and moved a result"
        ),
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "print no verdict, state, claim or summary table for this batch, "
            "so its messages can still be hand labelled blind. Rows and "
            "traces are written as usual; read them with label_claims.py "
            "after labelling"
        ),
    )
    parser.add_argument(
        "--disable-slash-commands",
        action="store_true",
        help=(
            "pass --disable-slash-commands to the hosted agent, so the account's "
            "synced skills do not reach the measured session (DECISIONS.md "
            "Q22-A: the main run uses this; the pilot batches did not)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="append",
        metavar="CSV",
        default=None,
        help=(
            "a CSV of episodes already run, whose cells this batch skips. "
            "Pass the batch's own --csv to continue a run that was stopped by "
            "the cap, by a client refusal or by the machine. An errored "
            "episode counts as run: a reproducible failure is something to "
            "look at, not work to repeat every night out of the same cap"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help=(
            "stop cleanly after N episodes. The nightly cap is the reason "
            "this exists: without it a long plan is refused outright by the "
            "budget check instead of running the part that fits"
        ),
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
    if args.image:
        target.image = args.image
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
    # The JSONL first: it is the form the pooling script reads and the only
    # complete record of a run, so it should not be the thing a failure in
    # another writer destroys.
    if args.jsonl:
        report.to_jsonl(verdicts, args.jsonl)
    if args.csv:
        report.to_csv(verdicts, args.csv)
    return 0 if args.exit_zero else report.exit_code(verdicts)


def _build_agent(
    descriptor: str,
    timeout_s: float = 300.0,
    config_dir: Optional[str] = None,
    disable_slash_commands: bool = False,
    expect_version: Optional[str] = None,
    binary: Optional[str] = None,
):
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

        return ClaudeCodeAgent(
            model=model or "sonnet",
            config_dir=config_dir,
            disable_slash_commands=disable_slash_commands,
            expect_version=expect_version,
            **({"binary": binary} if binary else {}),
        )
    raise SystemExit(f"unknown agent {descriptor!r}; use ollama[:model] or claude[:model]")


#: "resets 12:30am (America/Chicago)" -> "00:30", so the timing guard knows
#: when not to start. Only the client's own wording is parsed; anything else
#: leaves the guard with nothing, which is better than a wrong time.
_RESET_PATTERN = re.compile(r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.I)


def _reset_time(message: str) -> Optional[str]:
    match = _RESET_PATTERN.search(message)
    if not match:
        return None
    hour = int(match.group(1)) % 12
    minute = int(match.group(2) or 0)
    if match.group(3).lower() == "pm":
        hour += 12
    return f"{hour:02d}:{minute:02d}"


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

    from .consequence import (
        ClientRefused,
        ClientVersionMismatch,
        RowWriter,
        plan_cells,
        recorded_cells,
        remaining_cells,
    )

    grid = plan_cells(tasks, conditions, arms, args.repeats)
    already = recorded_cells(args.resume or [])
    cells = remaining_cells(grid, already)
    skipped = len(grid) - len(cells)
    if args.limit is not None and args.limit >= 0:
        cells = cells[: args.limit]
    planned = len(cells)

    try:
        agent = _build_agent(
            args.agent,
            timeout_s=args.agent_timeout,
            config_dir=args.config_dir,
            disable_slash_commands=args.disable_slash_commands,
            expect_version=args.client_version,
            binary=args.client_binary,
        )
    except ClientVersionMismatch as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return 3
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
        f"grid: {len(tasks)} tasks x {len(conditions)} conditions x {len(arms)} arms "
        f"x {args.repeats} repeats = {len(grid)} episodes"
    )
    if skipped:
        print(f"resumed: {skipped} already on disk, {len(grid) - skipped} left")
    if args.limit is not None and planned < len(grid) - skipped:
        print(f"limit: running {planned} of them this time")
    print(f"this batch: {planned} episodes")
    if not planned:
        print("nothing left to run")
        return 0
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
        for cell in cells:
            print(f"  {cell.task:28s} {cell.condition:6s} {cell.arm:20s} #{cell.repeat}")
        return 0

    # Q25 guard 1: one trivial question before any approved episode is spent.
    # Not an episode, not in the ledger, and it costs nothing if the client is
    # healthy. Thirty episodes were spent on a session-limit message before
    # this existed.
    if meters and hasattr(agent, "preflight"):
        try:
            reply = agent.preflight()
        except ClientRefused as exc:
            print(f"refusing to start: {exc}", file=sys.stderr)
            if budget is not None:
                budget.record_refusal(resets_at=_reset_time(str(exc)))
            print(
                "recorded as a client refusal, not as an episode. The "
                "approval says to stop for the night rather than retry.",
                file=sys.stderr,
            )
            return 4
        print(f"preflight: {reply.strip()[:60]!r}")

    runner = ConsequenceRunner(target, budget=budget, trace_dir=args.trace_dir)
    results = []
    by_id = {task.id: task for task in tasks}
    # Rows are appended as each episode finishes, so a run stopped by the cap,
    # a refusal or the machine keeps what it spent and can be resumed from this
    # same file.
    writer = RowWriter(args.csv) if args.csv else None
    try:
        for cell in cells:
            task = by_id[cell.task]
            condition, arm = cell.condition, cell.arm
            fault = None if condition == "clean" else f"{condition}:{task.target_tool}@1"
            try:
                result = runner.run_episode(
                    task, agent, condition=condition, fault=fault, arm=arm,
                )
            except ClientRefused as exc:
                print(f"\nstopping the batch: {exc}", file=sys.stderr)
                if results:
                    print(
                        f"{len(results)} episode(s) ran before this one; they "
                        "are kept in the CSV, and the refused episode is not "
                        "counted as one. Resume with --resume on that file.",
                        file=sys.stderr,
                    )
                return 4
            results.append(result)
            if writer is not None:
                writer.write(result)
            flag = "ERR" if result.error else ""
            if args.blind:
                # A batch that will be hand labelled must not print its own
                # answers. Set D was run without this and the log paired every
                # episode with the parser's verdict, its observed state, the
                # task, the condition and the arm, which is exactly the set of
                # fields the worksheet withholds. Reading that log to check the
                # batch had finished destroyed the blindness of 30 messages.
                print(
                    f"  {task.id:28s} {condition:6s} {arm:20s} "
                    f"episode recorded {flag}"
                )
            else:
                print(
                    f"  {task.id:28s} {condition:6s} {arm:20s} "
                    f"state={result.multiplicity:6s} "
                    f"claim={result.claim:12s} {flag}"
                )
    finally:
        if writer is not None:
            writer.close()

    if args.blind:
        done = sum(1 for r in results if not r.error)
        print(
            f"\nblind batch: {done} of {len(results)} episodes completed. "
            "No verdicts printed. Build the worksheet with "
            "`label_claims.py worksheet`, label it, then score it."
        )
        if args.csv:
            # Already written, row by row, as the episodes finished. Rewriting
            # the file here would truncate whatever a resumed run was
            # continuing.
            print(f"rows: {args.csv} (do not read before labelling)")
        return 0

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
        "**This table covers the episodes this invocation ran, not the whole "
        "run.** A resumed batch appends to the CSV and summarises only its own "
        "part, so pool the CSV rather than adding these tables together.",
        "",
        "| agent | arm | condition | attempted | completed | task success "
        "| duplicate | false success | honest unconfirmed | false failure "
        "| contradictory | errored |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent']} | {row['arm']} | {row['condition']} "
            f"| {row['episodes']} | {row['completed']} "
            f"| {row['task_success']} | {row['duplicate_effect']} "
            f"| {row['false_success']} | {row['honest_unconfirmed']} "
            f"| {row['false_failure']} | {row['contradictory']} "
            f"| {row['errored']} |"
        )
    markdown = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    else:
        print()
        print(markdown)
    if args.csv:
        print(f"rows: {args.csv} ({len(results)} appended)")
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
            contain=args.contain,
            mount=args.mount,
        )
    if command == "check":
        return _run_check(args)
    if command == "consequence":
        return _run_consequence(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
