#!/usr/bin/env python3
"""Compare two probe runs of the same candidates, candidate by candidate.

A probe run classifies each candidate as not installed, network bound, did not
start, read only, or mutating, and the startability rate derived from it is the
breadth result this study leans on hardest. That rate was measured before a
defect in the containment profile was fixed: the host's temporary directory was
forwarded into the container, naming a path that does not exist inside it, so a
server using it failed on a path the harness invented and was recorded as
having failed to start.

A failure caused by the profile and recorded as a property of the server
inflates the did-not-start row. This script answers whether it did, by
re-running the same candidates in the same order under the fixed profile and
diffing the outcomes rather than re-deriving a rate and comparing summaries.
Two rates can agree while individual candidates move in both directions, and a
summary comparison would hide that.

What it prints:

- every candidate whose class changed, with both error texts, so each one can
  be attributed rather than assumed;
- the counts per class in both runs, and the startability rate in both with
  Wilson intervals;
- candidates present in one run and not the other, which should be none and
  would mean the two runs are not comparable.

It takes no view on which run is right. A changed outcome is a fact needing an
explanation, and some explanations are not the profile: a package resolved at
build time can differ between runs because `npx` and `uvx` fetch the latest
version, so a candidate can legitimately change behaviour between two nights.
That possibility is why the output includes the error text and not just the
class.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentfaultlab.metrics import wilson_interval  # noqa: E402

CLASSIFIER = os.path.join(os.path.dirname(__file__), "probe_funnel.py")


def _load_classifier():
    """Reuse probe_funnel's classifier rather than restating its rules.

    The boundaries between not-installed, network-bound and did-not-start are
    pinned by that module's tests, and a second copy of them here would be a
    second thing to keep correct.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe_funnel", CLASSIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_run(paths: list[str]) -> dict[str, dict]:
    """Candidate name to its record, pooled across a run's batch files."""
    records: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for record in json.load(handle):
                records[record["server"]] = record
    return records


def error_text(record: dict) -> str:
    parts = []
    for attempt in record.get("attempts") or []:
        if attempt.get("error"):
            parts.append(attempt["error"])
        for line in attempt.get("stderr_tail") or []:
            parts.append(line)
    return " | ".join(parts)[:300] or "(no error recorded)"


def startable(record: dict) -> bool:
    return bool(record.get("started_via"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs="+", required=True, help="original run's JSON")
    parser.add_argument("--after", nargs="+", required=True, help="re-probe's JSON")
    args = parser.parse_args(argv)

    funnel = _load_classifier()
    before = read_run(args.before)
    after = read_run(args.after)

    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    shared = sorted(set(before) & set(after))

    print("Probe comparison")
    print("=" * 72)
    print("candidates: %d before, %d after, %d in both" % (len(before), len(after), len(shared)))
    if only_before or only_after:
        # Not a detail: if the two runs cover different candidates then the
        # rates are not comparable and nothing below should be quoted.
        print()
        print("NOT COMPARABLE: the runs cover different candidates")
        for name in only_before:
            print("  only in the original: %s" % name)
        for name in only_after:
            print("  only in the re-probe: %s" % name)
    print()

    changed = []
    for name in shared:
        was, now = funnel.classify(before[name]), funnel.classify(after[name])
        if was != now:
            changed.append((name, was, now))

    print("changed class: %d of %d" % (len(changed), len(shared)))
    for name, was, now in changed:
        print()
        print("  %s" % name)
        print("    %-14s -> %-14s" % (was, now))
        print("    before: %s" % error_text(before[name]))
        print("    after:  %s" % error_text(after[name]))
    print()

    for label, run in (("original", before), ("re-probe", after)):
        counts = collections.Counter(funnel.classify(r) for r in run.values())
        started = sum(1 for r in run.values() if startable(r))
        total = len(run)
        low, high = wilson_interval(started, total)
        print("%s: %s" % (label, dict(sorted(counts.items()))))
        print(
            "  started: %d/%d = %.0f%%, CI [%.0f%%, %.0f%%]"
            % (started, total, 100.0 * started / total, 100.0 * low, 100.0 * high)
        )

    if not changed:
        print()
        print("No candidate changed class. The did-not-start row is unaffected")
        print("by the profile fix, and the threat is retired by measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
