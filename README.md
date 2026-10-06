# agent-fault-lab

Seeded fault injection and reliability metrics for tool-calling agents. It answers one question: an agent that passes its evals when every tool answers cleanly, what does it do when a tool call times out after the write already happened?

No API keys, no GPU, standard library only (pytest for the tests).

## The result, in one table

200 invoices x 16 trials = 3,200 episodes per row, seed 0. Task: charge an invoice exactly once and apply its fulfillment. "Silent failures" are episodes where the agent reported success while the real state was wrong.

| faults | policy | task success | duplicate charges | silent failures | pass^8 |
|---|---|---|---|---|---|
| none | every policy | 100.0% | 0.0% | 0.0% | 100.0% |
| medium | naive-retry | 80.6% | 10.2% | 18.3% | 18.8% |
| medium | verify-before-retry | 88.2% | 1.8% | 11.7% | 36.9% |
| medium | idempotency-key | 89.3% | 0.0% | 9.9% | 38.9% |
| medium | idempotency-key+verify | 99.8% | 0.0% | 0.0% | 98.5% |
| high | naive-retry | 66.2% | 19.8% | 29.1% | 3.8% |
| high | idempotency-key+verify | 99.2% | 0.0% | 0.0% | 94.1% |

Read it as three findings:

1. With no faults, all four policies score 100%. The clean eval cannot tell them apart.
2. Each fix alone leaves a residue. An idempotency key stops duplicates but not partial success (9.9% silent failures at medium). A single read-back stops most duplicates but is fooled by delayed visibility (1.8% duplicates at medium).
3. Only the combination, key plus a polled postcondition check plus repair, holds. When it cannot confirm, it reports a third state, `unconfirmed`, instead of guessing.

The full grid is in [results/results.md](results/results.md). Across seeds 1 to 3, the largest difference from seed 0 in any cell is 1.7 points for task success, 2.2 for duplicate charges and 1.9 for silent failures. pass^8 is noisier at 200 tasks, up to 8.9 points, so read it to the nearest ten. [results/sensitivity-p-commit-1.0.md](results/sensitivity-p-commit-1.0.md) is the worst case, where every lost response had already committed: naive retry then double-charges 19.5% of tasks at medium and 35.4% at high.

## What this is not

It is a simulation of scripted policies, not a measurement of any LLM. The scripted policies are stand-ins for how an agent behaves around a tool call, so part of each number follows from how a policy is written. What the harness gives you is the means to see, for your own agent and your own tools, how a metric moves as faults are injected. See "Test your own agent" below. Published measurements on LLM agents are in the references.

## Failure modes

Three of the four faults in Mansoor et al., with their low/medium/high probabilities (Table 1):

| fault | what happens | low | medium | high |
|---|---|---|---|---|
| timeout after commit | the response is lost, the write may have happened | 5% | 20% | 35% |
| delayed visibility | a committed write is missed by the next 2 read-backs | 5% | 15% | 25% |
| partial success | the charge is recorded, fulfillment is not, the call still says ok | 3% | 10% | 15% |

Assumptions you can change in `FaultProfile`: when a response is lost, the write had already committed with probability 0.5 (`--p-commit`); visibility lag is 2 reads; policies make at most 3 attempts; the verified policy polls 3 times. Read-backs do not themselves fail. The stale-conflict fault is not modelled.

## Metrics

* task success: the agent reported success and the real state is exactly one fulfilled charge
* duplicate-action rate: two or more charges for one invoice, with a Wilson 95% interval
* silent-failure rate: the agent reported success while the real state is wrong
* unconfirmed rate: the agent reported the third state instead of guessing
* pass^k: the chance a task succeeds on k independent trials in a row (the tau-bench estimator, C(c,k)/C(n,k) per task, averaged)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # 21 tests, one per failure mode and metric
agent-fault-lab               # writes results/results.md and results.csv
agent-fault-lab --p-commit 1.0 --seed 3 --tasks 500
```

## Test your own agent

A policy is any object with a `name` and a `run(ledger, invoice, amount)` method that returns `"success"`, `"failure"` or `"unconfirmed"`. Call `ledger.charge(invoice, amount, idempotency_key=None)` (it raises `ToolTimeout` when the response is lost), `ledger.get_charge(invoice)`, and `ledger.complete_fulfillment(invoice)`. Wrap an LLM agent's tool-calling loop so its tool calls go through the ledger, then:

```python
from agentfaultlab import MEDIUM, run_experiment, summarize
episodes = run_experiment(MyAgent(), MEDIUM, tasks=200, trials=16)
print(summarize(episodes, k=8))
```

`notebooks/walkthrough.ipynb` runs the grid and a custom policy end to end.

## References

* Mansoor, Phadke, Rana. "Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures." arXiv:2608.02645 (2026). Fault types and probabilities; LLM-agent results in simulation.
* Li, Ye, Guo, Dang. "SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools." arXiv:2609.00035 (2026). Agents detected a silent tool failure in 12% of cases and repaired none.
* Yao, Shinn, Razavi, Narasimhan. "tau-bench." arXiv:2406.12045 (2024). The pass^k metric.

MIT licensed. Synthetic billing system, no employer or customer data.
