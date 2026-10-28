#!/usr/bin/env python3
"""Pool `probe_server.py` result files into the startability funnel.

The funnel is a result in its own right, and it is the cheapest one the study
has: one `tools/list` call per candidate, no fixture, no fault injection. So
it needs the same discipline as the audit. This script derives every stage
from the recorded JSON rather than from a reader's tally, and prints Wilson
intervals, because a rate quoted without one invites the reader to treat 44%
of 150 and 44% of 10 as the same claim.

Four stages are decided mechanically:

- **not installed**: the package never installed in the probe image, so the
  candidate was never launched. Recorded with no attempt at all.
- **did not start**: launched from the command the registry publishes (plus
  whatever fallbacks the prober tried) and never answered `initialize`.
- **read-only**: started, and every tool declares `readOnlyHint: true`.
- **mutating**: started with at least one tool that does not.

A tool counts as mutating when it does not declare `readOnlyHint: true`,
which is all a client knows about it before calling.

One stage cannot be decided mechanically: whether a started server is a
hosted proxy that needs the network and credentials at call time. That is a
judgement about what the tools do, so this script does not guess. It lists
the mutating candidates for inspection and takes the decided names from
`--network-bound`, so the classification is auditable as a file rather than
remembered.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentfaultlab.metrics import wilson_interval  # noqa: E402


def _text(record: dict) -> str:
    return " ".join(
        (attempt.get("error") or "") + " " + " ".join(attempt.get("stderr_tail") or [])
        for attempt in record.get("attempts") or []
    )


def classify(record: dict) -> str:
    attempts = record.get("attempts") or []
    if not attempts:
        return "not-installed"
    if not record.get("started_via"):
        # A server that answers `initialize` and then fails `tools/list` with
        # a network error has not failed to start: it has told us it needs the
        # network to describe itself, which is the network-bound row, not the
        # startability one. Counting it as a launch failure would inflate the
        # headline rate with a candidate the containment profile excludes by
        # design.
        if "fetch failed" in _text(record) or "ENOTFOUND" in _text(record):
            return "network-bound"
        return "did-not-start"
    tools = next(
        (a.get("tools") or [] for a in attempts if a.get("started")),
        [],
    )
    mutating = [t for t in tools if t.get("readOnlyHint") is not True]
    return "mutating" if mutating else "read-only"


#: Launch-failure buckets, in the order they are tested. The order matters:
#: a package whose entry point is unrunnable also exits non-zero, and the
#: more specific diagnosis is the useful one.
FAILURE_CAUSES = (
    (
        ("MODULE_NOT_FOUND", "Cannot find module"),
        "a module the package does not ship (MODULE_NOT_FOUND)",
    ),
    (
        ("GLIBC_",),
        "a vendored binary needing a newer glibc than the image",
    ),
    (
        ("Syntax error", "import: not found", "cannot execute"),
        "an entry point not runnable as published",
    ),
    (
        ("usage:", "Usage:", "Examples:", "--help"),
        "printed its own usage: a subcommand or flag the registry omits",
    ),
    (
        ("nothing on stderr",),
        "exited 0 in silence, never answering initialize",
    ),
    (
        ("status 1",),
        "exited 1 with a message that names no cause",
    ),
)


def failure_cause(record: dict) -> str:
    """A coarse bucket for why a launch failed, from the recorded stderr.

    Buckets are declared in `FAILURE_CAUSES` rather than written inline so
    that re-running this on a larger draw cannot quietly reclassify an
    earlier count: a changed boundary is a changed table here, visible in the
    diff, instead of a number that moved for no recorded reason.
    """
    text = _text(record)
    for needles, title in FAILURE_CAUSES:
        if any(needle in text for needle in needles):
            return title
    return "other startup failure"


def rate(successes: int, n: int) -> str:
    if not n:
        return "n/a (0)"
    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} = {successes / n:.0%}, 95% CI [{low:.0%}, {high:.0%}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="+", help="probe_server.py --out JSON files")
    parser.add_argument(
        "--network-bound",
        help=(
            "a file of server names, one per line, decided by inspection to "
            "be hosted proxies or to need the network and credentials at "
            "call time; they are reported in their own row and taken out of "
            "the mutating row"
        ),
    )
    parser.add_argument("--label", default="pooled")
    args = parser.parse_args()

    network_bound: set[str] = set()
    if args.network_bound and os.path.exists(args.network_bound):
        with open(args.network_bound, encoding="utf-8") as handle:
            network_bound = {
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("#")
            }

    records: dict[str, dict] = {}
    for path in args.results:
        with open(path, encoding="utf-8") as handle:
            for record in json.load(handle):
                records[record["server"]] = record

    stages = collections.Counter()
    causes = collections.Counter()
    mutating: list[tuple[str, int, int, int, int]] = []
    for name, record in records.items():
        stage = classify(record)
        if stage == "mutating" and name in network_bound:
            stage = "network-bound"
        stages[stage] += 1
        if stage == "did-not-start":
            causes[failure_cause(record)] += 1
        if stage == "mutating":
            tools = next(
                (a.get("tools") or [] for a in record["attempts"] if a.get("started")),
                [],
            )
            muts = [t for t in tools if t.get("readOnlyHint") is not True]
            retry = [
                t
                for t in muts
                if t.get("idempotentHint") is not None
                or t.get("destructiveHint") is not None
            ]
            writes_only = [
                t
                for t in muts
                if t not in retry and t.get("readOnlyHint") is False
            ]
            mutating.append(
                (name, len(tools), len(muts), len(retry), len(writes_only))
            )

    total = len(records)
    print(f"# Probe funnel: {args.label}, {total} candidates\n")
    order = [
        ("did-not-start", "did not start from the command the registry publishes"),
        ("not-installed", "cannot be installed in the audit image"),
        ("read-only", "started, every tool declared read only"),
        ("network-bound", "hosted proxy, or needs the network and credentials at call time"),
        ("mutating", "started with at least one tool not declared read only"),
    ]
    print("| stage | servers | rate of all candidates |")
    print("|---|---|---|")
    for key, title in order:
        print(f"| {title} | {stages[key]} | {rate(stages[key], total)} |")
    print()

    if causes:
        print("## Why the launches failed\n")
        print("| cause | count |")
        print("|---|---|")
        for cause, count in causes.most_common():
            print(f"| {cause} | {count} |")
        print()

    print("## Annotation coverage, among servers with a mutating tool\n")
    print(
        "`retry` counts a tool declaring `idempotentHint` or "
        "`destructiveHint`: the two hints that bear on whether repeating a "
        "call is safe, which is the question this study asks. `writes-only` "
        "counts a tool that declares `readOnlyHint: false` and neither of "
        "the other two: it has told the client that it writes and nothing "
        "about what a repeat would do. The weaker class is reported "
        "separately rather than folded into either side.\n"
    )
    retry_servers = [row for row in mutating if row[3]]
    writes_only = [row for row in mutating if not row[3] and row[4]]
    print(
        f"- servers annotating retry on at least one mutating tool: "
        f"{rate(len(retry_servers), len(mutating))}"
    )
    print(
        f"- servers declaring only that a tool writes: "
        f"{rate(len(writes_only), len(mutating))}"
    )
    print(
        f"- servers declaring nothing on any mutating tool: "
        f"{rate(len(mutating) - len(retry_servers) - len(writes_only), len(mutating))}"
    )
    tool_total = sum(row[2] for row in mutating)
    print(
        f"- mutating tools with a retry annotation: "
        f"{rate(sum(row[3] for row in mutating), tool_total)}"
    )
    print()
    print("| server | tools | mutating | retry | writes-only |")
    print("|---|---|---|---|---|")
    for name, tools, muts, retry, rofalse in sorted(mutating, key=lambda row: -row[2]):
        print(f"| {name} | {tools} | {muts} | {retry} | {rofalse} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
