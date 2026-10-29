#!/usr/bin/env python3
"""Pool consequence CSVs into the per-arm clean-complete cluster count.

The power analysis asks for 12 to 15 task clusters, and the number it wants
is not how many clusters were designed. A cluster only contributes to the
sign test on an arm that can finish it with no fault injected. If the clean
baseline cannot do the task, the fault has nothing to suppress, both arms
record a row of zeros, and the sign test reads a tie where the truth is that
the client could not do the work. Counting designed clusters is the wrong
denominator, so this script derives the right one per arm.

Eligibility is mechanical and asks exactly one question: did the arm do the
work. A cluster is clean-complete when a clean, baseline, no-fault episode
left the backend in the state of exactly one application
(`multiplicity == "once"`). Any other end state means the clean condition did
not establish the baseline the faulted conditions are compared against.

**It deliberately does not consult `task_success`**, and the first version of
this script did, which was wrong. `task_success` is `landed_once and claim ==
success`, so it folds in the claim parser, and the parser is a separate
instrument the preregistration already treats as provisional and
hand-validated. One local-arm cluster recorded `state=once` with a correct
computed answer in its message, and the parser read that message as asserting
nothing, so the cluster was counted as one the arm could not complete. The
arm had completed it. Eligibility is about capability; what the agent then
said about it is a different axis measured separately, and mixing them
understated the weaker arm.

Two things this script refuses to do quietly:

- **Pool a batch that is known bad.** Provenance exclusions are declared in
  `EXCLUDED` with the reason, so a directory that was aborted or refused
  cannot drift back into a denominator. The excluded rows are counted and
  printed, never silently dropped.
- **Resolve a disagreement by itself.** If one batch completes a cluster on
  an arm and another does not, that is a flaky cluster and a finding, not an
  input to a maximum. Those are reported as `conflict` and left out of the
  eligible count, because an arm that finishes a task four times in five is
  not the same evidence as one that finishes it every time.

One thing this script computes but does not promote. The declared primary
test is a sign test **within** a client arm, comparing a faulted condition
against that arm's own clean baseline, and the preregistration says plainly
that a cluster excluded on one client is still reported for the other. So the
binding number per test is each arm's own clean-complete count, and the
intersection across clients is **not** a power denominator for anything
declared. It is printed because the descriptive client comparison reads
better on a common cluster set, and because a cluster one client can do and
another cannot is a finding about the client.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentfaultlab.metrics import wilson_interval  # noqa: E402

# Declared, with the reason, so an excluded batch cannot drift back into a
# denominator by being forgotten. Matched against any path component.
EXCLUDED = (
    (
        "hosted-20261027",
        "every episode refused at the session reset and recorded state=none; "
        "a refusal is not a failure to do the task",
    ),
    (
        "audit-20261025-aborted",
        "batch aborted part way, rows describe an incomplete run",
    ),
)

# The end state a clean baseline must reach to make the cluster usable.
COMPLETE_MULTIPLICITY = "once"


def excluded_reason(path: str) -> str | None:
    parts = os.path.normpath(path).split(os.sep)
    for name, reason in EXCLUDED:
        if name in parts:
            return reason
    return None


def client_of(row: dict) -> str:
    """Name the client from the recorded agent string.

    This is the client axis (hosted Claude Code against a local open-weight
    model), not the `arm` column, which is the policy axis the preregistration
    pairs on. Both are called arms in conversation; keeping them apart here
    is the whole point of the function name.

    The agent column carries the full client description, which is what makes
    a run reproducible, but it also carries a tool timeout and a config note
    that differ between batches of the same arm. Keying on the raw string
    would split one arm into several.
    """
    agent = (row.get("agent") or "").lower()
    if "claude-code" in agent:
        return "hosted"
    if "ollama" in agent or "qwen" in agent:
        return "local"
    return "unknown"


def is_clean_baseline(row: dict) -> bool:
    return row.get("condition") == "clean" and row.get("arm") == "baseline"


def completed(row: dict) -> bool:
    """State only. See the module docstring for why the claim is not read."""
    return row.get("multiplicity") == COMPLETE_MULTIPLICITY


def collect(paths: list[str]) -> tuple[dict, list, int]:
    """Return {(client, task): set(outcomes)}, the excluded notes, and a row count."""
    outcomes: dict = collections.defaultdict(set)
    skipped: list = []
    rows_read = 0
    for path in paths:
        reason = excluded_reason(path)
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if is_clean_baseline(r)]
        if reason:
            if rows:
                skipped.append((path, len(rows), reason))
            continue
        for row in rows:
            rows_read += 1
            outcomes[(client_of(row), row["task"])].add(
                "complete" if completed(row) else "incomplete"
            )
    return outcomes, skipped, rows_read


def verdicts(outcomes: dict) -> dict:
    out: dict = {}
    for key, seen in outcomes.items():
        if seen == {"complete"}:
            out[key] = "eligible"
        elif seen == {"incomplete"}:
            out[key] = "not-complete"
        else:
            out[key] = "conflict"
    return out


def rate(count: int, total: int) -> str:
    if not total:
        return "n/a"
    low, high = wilson_interval(count, total)
    return "%d/%d (%.0f%%, CI [%.0f%%, %.0f%%])" % (
        count,
        total,
        100.0 * count / total,
        100.0 * low,
        100.0 * high,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="consequence CSV files to pool")
    parser.add_argument(
        "--designed",
        type=int,
        default=None,
        help="number of clusters designed, for the denominator the power "
        "analysis should not use",
    )
    args = parser.parse_args(argv)

    outcomes, skipped, rows_read = collect(args.csv)
    verdict = verdicts(outcomes)

    clients = sorted({client for client, _ in verdict})
    tasks = sorted({task for _, task in verdict})

    print("Clean-complete clusters per arm")
    print("=" * 64)
    print("rows pooled: %d over %d files" % (rows_read, len(args.csv)))
    for path, count, reason in skipped:
        print("excluded: %s (%d clean rows): %s" % (path, count, reason))
    print()

    print("%-34s %s" % ("cluster", "  ".join("%-12s" % c for c in clients)))
    for task in tasks:
        cells = []
        for client in clients:
            cells.append("%-12s" % verdict.get((client, task), "-"))
        print("%-34s %s" % (task, "  ".join(cells)))
    print()

    eligible: dict = {}
    for client in clients:
        measured = [t for t in tasks if (client, t) in verdict]
        good = [t for t in measured if verdict[(client, t)] == "eligible"]
        eligible[client] = set(good)
        unmeasured = [t for t in tasks if (client, t) not in verdict]
        print("%s client" % client)
        print("  measured clean:  %d clusters" % len(measured))
        print("  clean-complete:  %s of those measured" % rate(len(good), len(measured)))
        print("  sign test runs on this arm's own count: %d" % len(good))
        if args.designed:
            print("  against designed: %s" % rate(len(good), args.designed))
        conflicts = [t for t in measured if verdict[(client, t)] == "conflict"]
        if conflicts:
            print("  conflicting, excluded from eligible: %s" % ", ".join(conflicts))
        if unmeasured:
            print(
                "  never measured clean, so neither eligible nor excluded: %s"
                % ", ".join(unmeasured)
            )
    print()

    if len(clients) >= 2:
        both = set.intersection(*(eligible[c] for c in clients))
        print("clean-complete on every client: %d clusters" % len(both))
        print("  Descriptive only. No declared test uses this as a")
        print("  denominator, and a cluster never measured on one client")
        print("  depresses it without any client having failed anything.")
        for client in clients:
            only = sorted(eligible[client] - both)
            if only:
                print("  %s only: %s" % (client, ", ".join(only)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
