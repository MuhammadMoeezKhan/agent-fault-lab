#!/usr/bin/env python3
"""Pool the rows from an audit batch into one summary, server by server.

The per-server reports answer "what does this server do". The pilot's question
is "what does the sample do", and that needs one denominator across servers,
with the per-server rates next to it so a single server cannot carry a rate.

Rows come from the JSONL the audit writes, so this reads results and never
runs a server.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from agentfaultlab.audit import DUPLICATED, PARTIAL, UNPROVEN  # noqa: E402
from agentfaultlab.metrics import wilson_interval  # noqa: E402

CONTROL = "F7"


def load(directory: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def fixtures(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["server"], row["tool"], row["fixture"]), []).append(row)
    return grouped


def rate(numerator: int, denominator: int, *, interval: bool = False) -> str:
    """A rate with its denominator, and on request a 95% Wilson interval.

    The interval is printed for the pooled numbers because the denominators
    here are small: four of four is 100% with a lower bound of 51%, and a
    reader who sees only the 100% is being misled by a sample of four.
    """
    if denominator == 0:
        return "n/a (0)"
    body = f"{numerator / denominator:.0%} ({numerator}/{denominator})"
    if not interval:
        return body
    low, high = wilson_interval(numerator, denominator)
    return f"{body}, 95% CI [{low:.0%}, {high:.0%}]"


def describe(rows: list[dict[str, Any]], *, intervals: bool = True) -> dict[str, Any]:
    """Rates for one set of rows.

    `intervals` is off for the per-server table: a server contributing one
    testable fixture would print "100%, 95% CI [21%, 100%]", which looks like
    a measurement rather than the single observation it is. The intervals
    belong on the pooled numbers.
    """
    grouped = fixtures(rows)
    proven = {
        key: group
        for key, group in grouped.items()
        if not all(row["verdict"] == UNPROVEN for row in group)
    }
    unproven = len(grouped) - len(proven)
    mutating = {k: g for k, g in proven.items() if g[0]["mutating"]}
    testable = {
        k: g for k, g in mutating.items()
        if any(row["repeat_distinguishable"] for row in g)
    }
    duplicating = {
        k for k, g in testable.items()
        if any(row["verdict"] == DUPLICATED and row["fault"] != CONTROL for row in g)
    }
    violations = sorted(
        {
            f"{k[0]}/{k[1]} ({k[2]})"
            for k, g in proven.items()
            if any(row["annotation_violations"] for row in g)
        }
    )
    with_hint = {
        k for k, g in mutating.items() if g[0]["idempotent_hint"] is not None
    }
    rejected = {
        k for k, g in mutating.items()
        if any(row["safety_mechanism"] == "backend_rejected_repeat" for row in g)
    }
    # Undecidable from state, rather than merely unlabelled: no row for this
    # fixture ever showed a refusal or an absorption, and at least one safe
    # row had to leave the mechanism open. Counting the note text instead
    # caught fixtures that did show a refusal under another fault mode.
    undecidable = {
        k for k, g in mutating.items()
        if not any(
            row["safety_mechanism"] in ("backend_rejected_repeat", "absorbed_repeat")
            for row in g
        )
        and any(
            row["verdict"] == "SAFE" and row["safety_mechanism"] == "unknown"
            for row in g
        )
    }
    unconfirmable = {
        k for k, g in mutating.items() if any(row["confirmable"] is False for row in g)
    }
    partial = {
        k for k, g in mutating.items() if any(row["verdict"] == PARTIAL for row in g)
    }
    control_failures = [
        row for row in rows
        if row["fault"] == CONTROL and row["verdict"] == DUPLICATED
    ]
    return {
        "fixtures": len(grouped),
        "unproven": unproven,
        "mutating": len(mutating),
        "testable": len(testable),
        "duplicating": len(duplicating),
        "duplicate_rate": rate(len(duplicating), len(testable), interval=intervals),
        "testable_share": rate(len(testable), len(mutating), interval=intervals),
        "declares_idempotent_hint": rate(len(with_hint), len(mutating)),
        "safe_by_rejection": rate(len(rejected), len(mutating)),
        "mechanism_undecidable": rate(len(undecidable), len(mutating)),
        "unconfirmable": rate(len(unconfirmable), len(mutating)),
        "annotation_violations": violations,
        "control_failures": len(control_failures),
        "partial": rate(len(partial), len(mutating)),
        "dedup_argument": rate(
            len([k for k, g in mutating.items() if g[0]["dedup_argument"]]),
            len(mutating),
            interval=intervals,
        ),
        # Counted apart from the key above. A key absorbs a repeat and answers
        # success; a precondition refuses it and answers an error the caller
        # cannot tell from a real failure. Offering one and passing one are
        # also counted apart, because the same tool with and without it is the
        # sharpest comparison in the study.
        # Counted over fixtures that have a read path at all, because a
        # fixture with no probe cannot answer the question either way and
        # pooling it with the ones that can would understate the gap.
        # Only fixtures where a duplicate exists in the state can answer this.
        # Where the repeat is absorbed, S1 equals S2 and the read tools of
        # course show no difference, so counting those as "hidden" would turn
        # a trivial truth into a headline.
        "duplicate_hidden": rate(
            len([
                k for k, g in mutating.items()
                if str(g[0].get("repeat_distinguishable")).lower() == "true"
                and str(g[0].get("duplicate_visible")).lower() == "false"
            ]),
            len([
                k for k, g in mutating.items()
                if str(g[0].get("repeat_distinguishable")).lower() == "true"
                and str(g[0].get("duplicate_visible")).lower() in ("true", "false")
            ]),
            interval=intervals,
        ),
        "offers_precondition": rate(
            len([k for k, g in mutating.items() if g[0].get("precondition_argument")]),
            len(mutating),
            interval=intervals,
        ),
        "uses_precondition": rate(
            len([
                k for k, g in mutating.items()
                if str(g[0].get("precondition_supplied")).lower() in ("true", "1")
            ]),
            len(mutating),
            interval=intervals,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", help="a batch directory of audit JSONL files")
    parser.add_argument("--out", help="write the Markdown summary here")
    args = parser.parse_args()

    rows = load(args.directory)
    if not rows:
        print(f"no rows in {args.directory}", file=sys.stderr)
        return 1

    pooled = describe(rows)
    servers = sorted({row["server"] for row in rows})

    lines = [
        "# Pooled audit summary",
        "",
        f"Rows: {len(rows)} over {len(servers)} servers, from "
        f"`{os.path.basename(os.path.abspath(args.directory))}`.",
        "",
        "The unit is a (server, tool, fixture) triple. A fixture counts as "
        "duplicating if any fault other than the control applied its effect "
        "twice. Fixtures whose clean baselines did not reproduce are UNPROVEN "
        "and appear in no denominator.",
        "",
        f"- fixtures: {pooled['fixtures']}, of which unproven: {pooled['unproven']}",
        f"- mutating: {pooled['mutating']}, of which a duplicate could be seen "
        f"at all: {pooled['testable']} ({pooled['testable_share']})",
        f"- **duplicate effect under at least one fault: "
        f"{pooled['duplicate_rate']}** of the testable mutating fixtures",
        f"- declares idempotentHint at all: "
        f"{pooled['declares_idempotent_hint']} of mutating fixtures",
        f"- takes a caller supplied deduplication key: {pooled['dedup_argument']}",
        f"- a duplicate is invisible through the server's own read tools: "
        f"{pooled['duplicate_hidden']} (of the testable fixtures whose read "
        f"path can answer)",
        f"- offers a compare-and-swap precondition: {pooled['offers_precondition']}",
        f"- and the fixture passed one: {pooled['uses_precondition']}",
        f"- safe because the backend refused the repeat: "
        f"{pooled['safe_by_rejection']}",
        f"- safe, but absorbed or refused cannot be told apart from state: "
        f"{pooled['mechanism_undecidable']}",
        f"- unconfirmable through the server's own read tools: "
        f"{pooled['unconfirmable']}",
        f"- partial over-application under F5, where a list argument was "
        f"tested: {pooled['partial']}",
        f"- control (F7) failures, which would invalidate the run: "
        f"{pooled['control_failures']}",
        "",
    ]
    if pooled["annotation_violations"]:
        lines += ["Annotation violations:", ""]
        lines += [f"- {entry}" for entry in pooled["annotation_violations"]]
    else:
        lines += ["No annotation violation survived inspection.", ""]

    lines += [
        "",
        "## By server",
        "",
        "| server | fixtures | unproven | testable | duplicating | declares idempotentHint | refused repeat |",
        "|---|---|---|---|---|---|---|",
    ]
    for server in servers:
        subset = [row for row in rows if row["server"] == server]
        stats = describe(subset, intervals=False)
        lines.append(
            f"| {server} | {stats['fixtures']} | {stats['unproven']} | "
            f"{stats['testable']} | {stats['duplicate_rate']} | "
            f"{stats['declares_idempotent_hint']} | {stats['safe_by_rejection']} |"
        )

    body = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(body)
        print(f"wrote {args.out}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
