"""Audit reports: a CSV of rows, and a Markdown summary a human can read."""

from __future__ import annotations

import csv
import json
from typing import Any, Iterable, Optional, Sequence

from .mcp import taxonomy

from .audit import (
    DUPLICATED,
    MECHANISM_ABSORBED,
    MECHANISM_REJECTED,
    NOT_MUTATING,
    PARTIAL,
    SAFE,
    UNPROVEN,
    ToolVerdict,
)

#: F7 fails before the server is reached, so a duplicate there is a harness
#: bug rather than a finding. It is reported separately and never counted in
#: a duplicate rate.
CONTROL_FAULT = "F7"

COLUMNS = (
    "server",
    "tool",
    "fixture",
    "fault",
    "verdict",
    "safety_mechanism",
    "multiplicities",
    "mutating",
    "repeat_distinguishable",
    "confirmable",
    "caller_left_unsure",
    "dedup_argument",
    "read_only_hint",
    "idempotent_hint",
    "annotation_violations",
    "deterministic",
    "notes",
    # Added with containment. A row that does not say whether its server ran
    # inside the container cannot be read a month later, and the audit now
    # mixes host-collected pre-rule rows with contained ones.
    "contained",
    "image",
    # Added 2026-10-28, when the first compare-and-swap guard appeared. Kept
    # apart from dedup_argument: a key absorbs a repeat and answers success, a
    # precondition refuses it and answers an error.
    "precondition_argument",
    "precondition_supplied",
)


def to_csv(verdicts: Sequence[ToolVerdict], path: str) -> None:
    """Write the rows, with every field a row carries.

    The column list used to be a second, hand-maintained copy of the row's
    keys, and it fell behind: `contained` and `image` were added to the row
    and not here, so `DictWriter` raised on the first verdict of every audit
    that produced one. The Markdown report was written before the CSV and the
    JSONL after it, which means the exception destroyed the JSONL, the one
    form the pooling script reads. Unknown keys are now an explicit failure
    at the top rather than a crash per row, and the writer takes the row's
    own keys, so the two cannot drift again.
    """
    rows = [verdict.as_row() for verdict in verdicts]
    fieldnames = list(COLUMNS)
    extra = [key for row in rows for key in row if key not in COLUMNS]
    if extra:
        raise ValueError(
            "audit rows carry fields the report does not list: "
            + ", ".join(sorted(set(extra)))
        )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_jsonl(verdicts: Sequence[ToolVerdict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(verdict.as_row(), separators=(",", ":")) + "\n")


def summarise(verdicts: Iterable[ToolVerdict]) -> dict[str, Any]:
    """Headline rates, with the denominators stated rather than implied.

    Rates are per tool, not per row: a tool audited under four fault modes
    produces four rows, and counting rows would multiply one finding into
    several. A tool counts as duplicating if any non-control fault made the
    effect land twice.

    The duplicate denominator is tools where a duplicate could have been seen
    at all: mutating, and with one application distinguishable from two.
    Including the rest would dilute the rate with tools the method cannot
    speak about, which would understate the problem.
    """
    rows = list(verdicts)
    # Keyed by fixture, not by tool: one tool with two fixtures is two units
    # of analysis, because they can behave differently under the same fault.
    by_tool: dict[tuple[str, str, str], list[ToolVerdict]] = {}
    for row in rows:
        by_tool.setdefault((row.server, row.tool, row.fixture), []).append(row)

    # A fixture whose baselines were not reproducible is left out of every
    # denominator: the method has nothing to say about it, and counting it as
    # safe would be the same mistake as counting it as a finding.
    by_tool = {
        key: group
        for key, group in by_tool.items()
        if not all(row.verdict == UNPROVEN for row in group)
    }
    mutating = {key: group for key, group in by_tool.items() if group[0].mutating}
    testable = {
        key: group
        for key, group in mutating.items()
        if any(row.repeat_distinguishable for row in group)
    }
    duplicated = {
        key
        for key, group in testable.items()
        if any(row.verdict == DUPLICATED and row.fault != CONTROL_FAULT for row in group)
    }
    unconfirmable = {
        key for key, group in mutating.items() if any(row.confirmable is False for row in group)
    }
    with_dedup = {key for key, group in mutating.items() if group[0].dedup_argument}
    offers_precondition = {
        key for key, group in mutating.items() if group[0].precondition_argument
    }
    uses_precondition = {
        key for key, group in mutating.items() if group[0].precondition_supplied
    }
    violating = {key for key, group in by_tool.items() if any(row.annotation_violations for row in group)}
    flaky = [row for row in rows if not row.deterministic]
    unproven = [row for row in rows if row.verdict == UNPROVEN]

    per_fault = {}
    for fault in sorted({row.fault for row in rows if row.fault != CONTROL_FAULT}):
        subject = [
            row
            for row in rows
            if row.fault == fault
            and row.mutating
            and row.repeat_distinguishable
            and row.verdict != UNPROVEN
        ]
        hits = [row for row in subject if row.verdict == DUPLICATED]
        per_fault[fault] = _rate(len(hits), len(subject))

    safe_tools = {
        key for key, group in mutating.items() if any(row.verdict == SAFE for row in group)
    }
    rejected = {
        key
        for key, group in mutating.items()
        if any(row.safety_mechanism == MECHANISM_REJECTED for row in group)
    }
    absorbed = {
        key
        for key, group in mutating.items()
        if any(row.safety_mechanism == MECHANISM_ABSORBED for row in group)
    }

    control_failures = [
        row for row in rows if row.fault == CONTROL_FAULT and row.verdict == DUPLICATED
    ]

    return {
        "verdicts": len(rows),
        "tools": len({(row.server, row.tool) for row in rows}),
        "fixtures": len(by_tool),
        "mutating": len(mutating),
        "duplicate_testable": len(testable),
        "duplicated": len(duplicated),
        "duplicate_rate": _rate(len(duplicated), len(testable)),
        "duplicate_rate_by_fault": per_fault,
        "unconfirmable": len(unconfirmable),
        "unconfirmable_rate": _rate(len(unconfirmable), len(mutating)),
        "with_dedup_argument": len(with_dedup),
        "dedup_rate": _rate(len(with_dedup), len(mutating)),
        "offers_precondition": len(offers_precondition),
        "offers_precondition_rate": _rate(len(offers_precondition), len(mutating)),
        "uses_precondition": len(uses_precondition),
        "uses_precondition_rate": _rate(len(uses_precondition), len(mutating)),
        "safe_tools": len(safe_tools),
        "safe_by_rejection": len(rejected),
        "safe_by_rejection_rate": _rate(len(rejected), len(safe_tools)),
        "safe_by_absorption": len(absorbed),
        "annotation_violations": len(violating),
        "nondeterministic": len(flaky),
        "unproven": len(unproven),
        "partial": len([row for row in rows if row.verdict == PARTIAL]),
        "control_failures": len(control_failures),
    }


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a (0 tools)"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def to_markdown(
    verdicts: Sequence[ToolVerdict], server_info: Optional[dict[str, Any]] = None
) -> str:
    stats = summarise(verdicts)
    lines = [
        "# Conformance audit",
        "",
        "Verdicts come from comparing the backend's end state against two "
        "clean baselines (one application, two applications). No verdict "
        "reads what the tool or the agent reported. Rates are per tool.",
        "",
        f"- tools audited: {stats['tools']}, over {stats['fixtures']} fixtures, "
        f"of which mutating: {stats['mutating']}",
        f"- duplicate effect under at least one fault: {stats['duplicate_rate']} "
        f"of the {stats['duplicate_testable']} mutating "
        f"{'fixture' if stats['duplicate_testable'] == 1 else 'fixtures'} where "
        "one application is distinguishable from two",
        f"- unconfirmable through the server's own read tools: "
        f"{stats['unconfirmable_rate']} of mutating fixtures",
        f"- taking a deduplication argument: {stats['dedup_rate']} of mutating "
        "fixtures",
        f"- safe only because the backend refused the repeat: "
        f"{stats['safe_by_rejection_rate']} of the mutating fixtures that came "
        "out safe (the agent still gets an error it cannot interpret)",
        f"- tools contradicting their own annotations: {stats['annotation_violations']}",
        f"- nondeterministic rows (treat as unproven): {stats['nondeterministic']}",
        f"- rows with no verdict because the clean baselines were not "
        f"reproducible: {stats['unproven']}",
        f"- partial over-application under F5 (a prefix landed, then the "
        f"retry applied the whole list): {stats['partial']}",
    ]
    if server_info:
        reported = (server_info.get("serverInfo") or {}) if server_info else {}
        lines += [
            "",
            f"Server as it identified itself: `{reported.get('name', 'unknown')}` "
            f"version `{reported.get('version', 'unknown')}`, protocol "
            f"`{server_info.get('protocolVersion', 'unknown')}`.",
        ]
    if stats["control_failures"]:
        lines.append(
            f"- **{stats['control_failures']} control (F7) rows duplicated, which "
            "should be impossible: treat this run as a harness bug, not a finding**"
        )
    if stats["duplicate_rate_by_fault"]:
        lines += ["", "By fault mode (share of testable mutating tools):", ""]
        for fault, rate in stats["duplicate_rate_by_fault"].items():
            mode = taxonomy.get(fault)
            lines.append(f"- {fault} ({mode.name}): {rate}")
    lines += [
        "",
        "| server | fixture | tool | fault | verdict | how | repeats "
        "| confirmable | hints |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in verdicts:
        hints = []
        if row.read_only_hint is not None:
            hints.append(f"readOnly={str(row.read_only_hint).lower()}")
        if row.idempotent_hint is not None:
            hints.append(f"idempotent={str(row.idempotent_hint).lower()}")
        flag = " (violated)" if row.annotation_violations else ""
        confirmable = {True: "yes", False: "no", None: "not probed"}[row.confirmable]
        lines.append(
            f"| {row.server} | {row.fixture} | `{row.tool}` | {row.fault} "
            f"| {row.verdict} "
            f"| {row.safety_mechanism} | {','.join(row.multiplicities)} | {confirmable} "
            f"| {' '.join(hints) or 'none'}{flag} |"
        )
    violations = [row for row in verdicts if row.annotation_violations]
    if violations:
        lines += ["", "## Annotation violations", ""]
        for row in violations:
            for violation in row.annotation_violations:
                lines.append(
                    f"- `{row.server}` / `{row.tool}` ({row.fixture}, {row.fault}): "
                    f"{violation}"
                )
    # Notes repeat across a tool's fault rows, so they are listed once per tool.
    seen: dict[str, list[str]] = {}
    for row in verdicts:
        for note in row.notes:
            seen.setdefault(row.fixture, [])
            if note not in seen[row.fixture]:
                seen[row.fixture].append(note)
    if seen:
        lines += ["", "## Notes", ""]
        for fixture, notes in seen.items():
            lines.append(f"- {fixture}: {'; '.join(notes)}")
    return "\n".join(lines) + "\n"


def exit_code(verdicts: Sequence[ToolVerdict]) -> int:
    """0 when nothing was found, 1 when something was. For CI use."""
    for row in verdicts:
        if row.verdict not in (SAFE, NOT_MUTATING) or row.annotation_violations:
            return 1
    return 0
