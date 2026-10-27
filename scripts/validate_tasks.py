"""Check every task's oracle before any agent runs against it.

Two different requirements, and conflating them is a mistake this script was
written to stop making:

- **Every** task must satisfy S0 != S1. If the reference calls do not change
  the backend, an episode against the task measures nothing at all.
- A task can show a **duplicate** only if S1 != S2. Where one application is
  indistinguishable from two, a duplicate cannot be detected however it is
  injected.

A task with S1 == S2 is therefore not broken. Three of the pilot's tasks are
exactly that by design, and their notes say so: a set-valued UPDATE, a DDL
statement the backend refuses to repeat, and a DELETE that is idempotent in
effect. They are the controls for H1b, and they remain valid clusters for the
silent-no-effect fault, where the question is whether the work landed at all.

So this prints two counts: clusters usable for a repair or no-effect
comparison (all tasks that mutate), and clusters usable for a duplicate
comparison (those with S1 != S2). The first version of this script labelled
every S1 == S2 task "BAD" and nearly had me treat three deliberate controls as
defects.
"""
import sys, os
R = "/Users/muhammadmoeezkhan/Desktop/Development/Conferences/agent-fault-lab"
sys.path.insert(0, os.path.join(R, "src"))
from agentfaultlab import spec
from agentfaultlab.consequence import ConsequenceRunner
from agentfaultlab.observers.base import digest, equal

target, tasks = spec.load_tasks(os.path.join(R, sys.argv[1]))
only = set(sys.argv[2:]) or None
runner = ConsequenceRunner(target, budget=None)
bad = 0
mutating: list[str] = []
duplicate_capable: list[str] = []
for task in tasks:
    if only and task.id not in only:
        continue
    try:
        o = runner.oracle(task)
    except Exception as exc:
        print(f"  {task.id:34s} ORACLE FAILED {type(exc).__name__}: {exc}")
        bad += 1
        continue
    s0, s1, s2 = o["s0"], o["s1"], o["s2"]
    mutates = not equal(s0, s1)
    shows_duplicate = not equal(s1, s2)
    if not mutates:
        kind, bad = "UNUSABLE", bad + 1
    elif shows_duplicate:
        kind = "effect  "
    else:
        kind = "control "
    mutating.append(task.id) if mutates else None
    if mutates and shows_duplicate:
        duplicate_capable.append(task.id)
    print(f"  {kind} {task.id:34s} mutates={mutates} duplicate_visible={shows_duplicate} "
          f"s0={digest(s0)[:8]} s1={digest(s1)[:8]} s2={digest(s2)[:8]}")

print(f"\nclusters that mutate (usable for repair and no-effect): {len(mutating)}")
print(f"clusters where a duplicate is visible (usable for F1 and F2): "
      f"{len(duplicate_capable)}")
print(f"unusable (the reference changes nothing): {bad}")
sys.exit(1 if bad else 0)
