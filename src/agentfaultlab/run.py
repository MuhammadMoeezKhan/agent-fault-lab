"""Run policies against fault profiles and write a results table."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

from .agents import POLICIES, AgentPolicy
from .faults import PROFILES, FaultProfile
from .metrics import Episode, summarize
from .tools import BillingLedger


def run_experiment(
    policy: AgentPolicy,
    profile: FaultProfile,
    tasks: int = 200,
    trials: int = 16,
    seed: int = 0,
) -> list[Episode]:
    episodes = []
    for task in range(tasks):
        for trial in range(trials):
            rng = random.Random(f"{seed}:{profile.name}:{policy.name}:{task}:{trial}")
            ledger = BillingLedger(profile, rng)
            invoice = f"INV-{task}"
            report = policy.run(ledger, invoice, 100)
            count, fulfilled = ledger.final_state(invoice)
            episodes.append(Episode(policy.name, profile.name, task, trial, report, count, fulfilled))
    return episodes


def run_grid(profiles, policies=POLICIES, tasks=200, trials=16, seed=0, k=8):
    rows = []
    for profile in profiles:
        for policy in policies:
            eps = run_experiment(policy, profile, tasks, trials, seed)
            rows.append({"profile": profile.name, "policy": policy.name, **summarize(eps, k)})
    return rows


def to_markdown(rows, k=8) -> str:
    head = (
        f"| faults | policy | task success | duplicate charges | silent failures | unconfirmed | pass^{k} |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        dlo, dhi = r["duplicate_ci"]
        lines.append(
            f"| {r['profile']} | {r['policy']} | {r['task_success']:.1%} "
            f"| {r['duplicate_action_rate']:.1%} ({dlo:.1%} to {dhi:.1%}) "
            f"| {r['silent_failure_rate']:.1%} | {r['unconfirmed_rate']:.1%} | {r[f'pass_hat_{k}']:.1%} |"
        )
    return head + "\n".join(lines) + "\n"


def run_simulation(args) -> int:
    """The v0.1 grid, driven by the `simulate` subcommand's arguments."""
    out = Path(args.out)
    profiles = [replace(p, p_commit_given_timeout=args.p_commit) for p in PROFILES.values()]
    rows = run_grid(profiles, tasks=args.tasks, trials=args.trials, seed=args.seed, k=args.k)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, args.k)
    (out / "results.md").write_text(md)
    with open(out / "results.csv", "w", newline="") as f:
        cols = [c for c in rows[0] if not c.endswith("_ci")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(md)
    return 0


def main() -> None:
    """Kept so `python -m agentfaultlab.run` still runs the grid alone."""
    ap = argparse.ArgumentParser(description="agent-fault-lab experiment runner")
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--p-commit", type=float, default=0.5, help="P(write committed | response lost)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    run_simulation(ap.parse_args())


if __name__ == "__main__":
    main()
