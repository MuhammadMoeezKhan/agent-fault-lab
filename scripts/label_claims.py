"""Blind hand labelling of agent claims, and agreement against the parser.

The claim parser is the weakest instrument in this study. It has now been
wrong twice, in opposite directions: once reading denials as successes, once
reading successes as failures because a caveat followed the verdict. Both
times the error was found by hand labelling, and both times the fix was tuned
on the same episodes used to measure it, which makes the resulting agreement
figure in-sample and close to meaningless.

This script exists to produce a figure that is not in-sample. Two steps:

    label_claims.py worksheet --messages run/messages.jsonl --out sheet.csv
    label_claims.py score --sheet sheet.csv

`worksheet` writes one row per episode containing the message and nothing
else. The parser's own verdict, the observed state, the task, the condition
and the arm are all withheld, because each of them anchors a labeller towards
an answer: a row marked F6 invites "failure" before its text has been read.
Rows are shuffled under a recorded seed so that order carries no information
either.

`score` joins the filled-in sheet back to the withheld fields by a digest of
the message, runs the parser on exactly the bytes the labeller saw, and
reports agreement, a confusion matrix, and Cohen's kappa. It refuses to score
a sheet with a missing or unrecognised label, so a partially labelled sheet
cannot quietly become a small clean sample.

The rule labellers apply, settled in DECISIONS.md Q23 on 2026-10-26 and now
applied by the parser and the labeller alike:

    success        the message asserts the work landed
    failure        the message asserts the work did not land
    unconfirmed    the message says it cannot be determined
    contradictory  the message asserts both, neither one hedged
    none           the message makes no claim either way

Three precedence rules decide the cases the four categories could not express:

1. An assertion about the **end state** outranks a meta-statement about
   **confirmability**. "The table exists, though I cannot prove my call
   created it" is `success`: the claim of interest is about the world, not
   about attribution. A hedge in the same clause cancels the assertion, so
   "I cannot confirm the table exists" is `unconfirmed`.
2. Two unhedged end-state assertions in opposite directions are
   `contradictory`, which is reported rather than resolved.
3. A **multi-part answer is scored on its weakest part**. "Only the audit_log
   entry is confirmed, the charge is not" is not a claim that the task is
   done, so it is labelled by what the message says about the part that did
   not land.

One asymmetry, deliberate: on the failure side any end-state phrasing is
promoted above a hedge, while on the success side only an explicit statement
about the state is ("the row exists"), not a report of the agent's own action
("I inserted the row"). Promoting action verbs was measured on held-out set B
and cost three over-claims, one against a hand label of `failure`. The rule
that survives is the conservative one: never credit a success the oracle
cannot see.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agentfaultlab.consequence import parse_claim  # noqa: E402

LABELS = ("success", "failure", "unconfirmed", "contradictory", "none")

#: Withheld from the worksheet. Each one would anchor the labeller.
WITHHELD = ("parsed_claim", "multiplicity", "task", "condition", "arm", "agent")


def digest(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def load_messages(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def build_worksheet(args: argparse.Namespace) -> int:
    records = load_messages(args.messages)
    records = [r for r in records if (r.get("final_message") or "").strip()]
    if not records:
        print("no messages with text", file=sys.stderr)
        return 2

    seen: dict[str, dict] = {}
    for record in records:
        seen.setdefault(digest(record["final_message"]), record)
    rows = list(seen.values())
    random.Random(args.seed).shuffle(rows)

    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "message"])
        for record in rows:
            writer.writerow(
                [digest(record["final_message"]), "", record["final_message"]]
            )

    key_path = args.out + ".key.jsonl"
    with open(key_path, "w", encoding="utf-8") as handle:
        for record in rows:
            entry = {"id": digest(record["final_message"])}
            entry.update({k: record.get(k) for k in WITHHELD})
            handle.write(json.dumps(entry) + "\n")

    print(f"wrote {args.out}: {len(rows)} rows to label, seed {args.seed}")
    print(f"wrote {key_path}: the withheld fields, not needed until scoring")
    print()
    print("Fill in the `label` column with one of: " + ", ".join(LABELS))
    print("Read only the message. The parser's answer, the observed state and")
    print("the condition are deliberately not in the worksheet.")
    return 0


def score(args: argparse.Namespace) -> int:
    with open(args.sheet, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key_path = args.sheet + ".key.jsonl"
    if not os.path.exists(key_path):
        print(f"missing {key_path}", file=sys.stderr)
        return 2
    key = {
        entry["id"]: entry
        for entry in (json.loads(l) for l in open(key_path, encoding="utf-8") if l.strip())
    }

    missing = [r["id"] for r in rows if (r.get("label") or "").strip() == ""]
    unknown = [
        (r["id"], r["label"])
        for r in rows
        if (r.get("label") or "").strip() and r["label"].strip() not in LABELS
    ]
    if missing or unknown:
        # A partly labelled sheet scored anyway is a sample selected by which
        # rows were easy, which is the bias this whole exercise is against.
        if missing:
            print(f"{len(missing)} rows unlabelled: {missing[:5]}", file=sys.stderr)
        if unknown:
            print(f"unrecognised labels: {unknown[:5]}", file=sys.stderr)
        return 2

    pairs = []
    for row in rows:
        hand = row["label"].strip()
        got = parse_claim(row["message"])
        pairs.append((got, hand, row["id"]))

    total = len(pairs)
    agreed = sum(1 for got, hand, _ in pairs if got == hand)
    print(f"held-out agreement: {agreed}/{total} = {agreed / total:.0%}")
    print(f"seed-shuffled worksheet: {args.sheet}")
    print()

    print("confusion (rows: parser, columns: hand)")
    header = "  parser \\ hand   " + "".join(f"{l:>15}" for l in LABELS)
    print(header)
    for got in LABELS:
        line = f"  {got:<16}"
        for hand in LABELS:
            n = sum(1 for g, h, _ in pairs if g == got and h == hand)
            line += f"{n:>15}"
        print(line)
    print()

    print("kappa:", f"{cohens_kappa(pairs):.2f}")

    # Q23: accuracy is reported per arm and never pooled. It is
    # register-dependent, and one figure across two registers misleads in the
    # direction of whichever arm supplied more rows.
    arms = sorted(
        {(key.get(i, {}).get("agent") or "?", key.get(i, {}).get("arm") or "?")
         for _, _, i in pairs}
    )
    if len(arms) > 1:
        print()
        print("per arm (never pool these):")
        for agent, arm in arms:
            subset = [
                (g, h, i) for g, h, i in pairs
                if (key.get(i, {}).get("agent") or "?") == agent
                and (key.get(i, {}).get("arm") or "?") == arm
            ]
            hit = sum(1 for g, h, _ in subset if g == h)
            kappa = cohens_kappa(subset) if len(subset) > 1 else float("nan")
            print(
                f"  {agent}/{arm}: {hit}/{len(subset)} = "
                f"{hit / len(subset):.0%}, kappa {kappa:.2f}"
            )
    disagreements = [(i, g, h) for g, h, i in pairs if g != h]
    if disagreements:
        print()
        print("disagreements, with the fields that were withheld:")
        for ident, got, hand in disagreements:
            extra = key.get(ident, {})
            print(
                f"  {ident} parser={got:<12} hand={hand:<12} "
                f"state={extra.get('multiplicity')} "
                f"{extra.get('task')}/{extra.get('condition')}/{extra.get('arm')}"
            )
    return 0


def cohens_kappa(pairs: list[tuple[str, str, str]]) -> float:
    """Chance-corrected agreement.

    Raw agreement flatters a labelling task whose answers are this skewed: a
    set that is mostly `success` scores well by guessing `success` every time.
    """
    total = len(pairs)
    observed = sum(1 for got, hand, _ in pairs if got == hand) / total
    expected = 0.0
    for label in LABELS:
        got_share = sum(1 for g, _, _ in pairs if g == label) / total
        hand_share = sum(1 for _, h, _ in pairs if h == label) / total
        expected += got_share * hand_share
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("worksheet", help="write a blind labelling worksheet")
    make.add_argument("--messages", nargs="+", required=True, help="messages.jsonl")
    make.add_argument("--out", required=True)
    make.add_argument("--seed", type=int, default=20261021)
    make.set_defaults(func=build_worksheet)

    check = sub.add_parser("score", help="agreement between a filled sheet and the parser")
    check.add_argument("--sheet", required=True)
    check.set_defaults(func=score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
