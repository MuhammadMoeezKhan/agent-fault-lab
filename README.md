# agent-fault-lab

Seeded fault injection and reliability metrics for tool-calling agents. It answers one question: an agent that passes its evals when every tool answers cleanly, what does it do when a tool call times out after the write already happened?

No API keys, no GPU, standard library only (pytest for the tests).

Two parts:

1. **The simulator** (v0.1): scripted policies against seeded faults, to show how a metric moves. The table below.
2. **The audit** (v0.2): the same fault taxonomy aimed at real tools over the Model Context Protocol, with ground truth read from the backend instead of from anyone's report. See "Auditing a real tool" below.

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
pytest -q                     # 59 tests, no network and no API key needed
agent-fault-lab simulate      # writes results/results.md and results.csv
agent-fault-lab simulate --p-commit 1.0 --seed 3 --tasks 500
```

## Test your own agent

A policy is any object with a `name` and a `run(ledger, invoice, amount)` method that returns `"success"`, `"failure"` or `"unconfirmed"`. Call `ledger.charge(invoice, amount, idempotency_key=None)` (it raises `ToolTimeout` when the response is lost), `ledger.get_charge(invoice)`, and `ledger.complete_fulfillment(invoice)`. Wrap an LLM agent's tool-calling loop so its tool calls go through the ledger, then:

```python
from agentfaultlab import MEDIUM, run_experiment, summarize
episodes = run_experiment(MyAgent(), MEDIUM, tasks=200, trials=16)
print(summarize(episodes, k=8))
```

`notebooks/walkthrough.ipynb` runs the grid and a custom policy end to end.

## Auditing a real tool

The simulator answers what a metric does under faults. The audit answers whether one real tool keeps its promises, and it never takes the tool's word for it.

```bash
agent-fault-lab --list-faults                     # the taxonomy, with provenance
agent-fault-lab check specs/reference-memory.json # audit a server's tools
```

`check` exits 1 if any tool duplicates an effect, loses one, or contradicts its own annotations, so it works as a CI gate on a server you maintain.

### How a verdict is reached

For one tool and one fault, three trials run on three freshly built backends:

| trial | what runs | snapshot |
|---|---|---|
| `clean_once` | the call once, no fault | S1, the shape of one application |
| `clean_twice` | the call twice, no fault | S2, the shape of two |
| `faulted` | the fault, then a retry if the caller was left without an answer | Sf |

The verdict follows from which baseline Sf matches: S1 is `SAFE`, S2 is `DUPLICATED`, the untouched state is `LOST`, and anything else is `DIVERGED`. Nothing in that chain reads the tool's response or an agent's summary, so a tool that misreports what it did is caught by construction.

Two honest limits are reported rather than hidden. When S1 equals S2, no end state could distinguish one application from two (an overwrite, or real deduplication), so the tool is repeat safe for those arguments but is kept out of the duplicate-rate denominator. And every trial runs three times by default, because a tool that is sometimes safe is not safe, and because a snapshot that moves on its own means the observer is not canonical yet and the row must not be trusted.

### The faults

Injected at the transport, so they work with any client and any server. Ids are stable; `--list-faults` prints the paper or protocol issue each one comes from.

| id | fault | what the caller sees |
|---|---|---|
| F1 | lost response after commit | a timeout, after the write landed |
| F2 | duplicate delivery | one answer; the call was delivered twice |
| F3 | late commit | a timeout, then the write lands later |
| F4 | concurrent twin calls | one answer; two calls raced |
| F5 | partial batch | a timeout, after part of a list argument committed |
| F6 | silent OK | success, for a call that never ran |
| F7 | error before commit | an error. The control: a retry here must be safe |

### Observers

Ground truth comes from an observer that reads the backend and canonicalises what it finds, so two backends holding the same state compare equal. Content digests instead of mtimes, sorted rows instead of insertion order, commit subjects and tree digests instead of commit hashes. Built in: filesystem, SQLite, git, and any composition of them.

### Putting a fault under a real client

The audit drives its own client. To measure what Claude Code, Codex CLI or any other client does when a call goes ambiguous, put the proxy in front of the server in that client's own config and leave the client alone:

```bash
agent-fault-lab proxy --fault F1:write_file@1 --trace trace.jsonl \
  -- npx -y @modelcontextprotocol/server-filesystem /tmp/sandbox
```

Everything the proxy does not need to understand passes through untouched, including methods newer than this release and server to client requests. Only the `tools/call` requests the plan names are altered, and the JSONL trace records every message and every injection.

### Writing a spec

A spec is one JSON file per server, so it is also the unit of contribution: send a spec for your own server and the audit runs it unchanged. `{root}` is replaced with a fresh temporary directory per trial.

```json
{
  "server": {
    "name": "my-server",
    "command": ["my-server", "--data", "{root}"],
    "observer": {"kind": "sqlite", "path": "app.db"},
    "setup": {"sqlite": [{"path": "app.db", "sql": "create table charges (id text, amount int);"}]}
  },
  "fixtures": [
    {
      "tool": "charge",
      "arguments": {"id": "INV-1", "amount": 100},
      "probes": [["get_charge", {"id": "INV-1"}]]
    }
  ]
}
```

`probes` are read-only calls used to decide something separate from duplication: whether a caller who never got an answer could have found out what happened. If no probe's output differs between the untouched backend and the backend after one application, the tool is unconfirmable through that server, and the agent on the other end has nothing to check against.

## References

* Mansoor, Phadke, Rana. "Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures." arXiv:2608.02645 (2026). Fault types and probabilities; LLM-agent results in simulation.
* Li, Ye, Guo, Dang. "SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools." arXiv:2609.00035 (2026). Agents detected a silent tool failure in 12% of cases and repaired none.
* Yao, Shinn, Razavi, Narasimhan. "tau-bench." arXiv:2406.12045 (2024). The pass^k metric.

MIT licensed. Synthetic billing system, no employer or customer data.
