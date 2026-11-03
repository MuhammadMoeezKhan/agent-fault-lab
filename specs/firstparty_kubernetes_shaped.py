"""A Kubernetes-shaped audit target, built in Python because it is first party.

A JSON spec cannot set `contained: false`, and that rule is right: rule 9 is
not a default for the audited thing to override. The only uncontained servers
are first-party fixtures constructed in code, which is what this is. The
server is `fixtures/kubernetes_shaped_server.py` in this repository, written
for this study, with no cluster and no network behind it.

Why it is not a contained third-party server. The study wants to say something
about agents acting on infrastructure. Auditing a published Kubernetes MCP
server would need a cluster, a cluster is a network service, and the
containment profile gives an audited server no network at all. Relaxing the
profile for one server would make its rows incomparable with every other row
in the sample, so the Kubernetes result comes from the engine against a
first-party backend and is labelled that way wherever it appears. It is not
evidence about anyone's Kubernetes server. It is evidence about what the API
shape does to a retrying agent.
"""

from __future__ import annotations

import os

from agentfaultlab.audit import Fixture, ServerTarget
from agentfaultlab.observers import JsonDocObserver

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, "fixtures", "kubernetes_shaped_server.py")

#: The seeded cluster: one Deployment, so the scale and delete fixtures have
#: something to act on without another fixture's tool creating it.
SEED = """{
  "counter": 0,
  "events": [],
  "jobs": [],
  "objects": {
    "prod/Deployment/api": {
      "generation": 1,
      "kind": "Deployment",
      "name": "api",
      "namespace": "prod",
      "resourceVersion": "1",
      "spec": {
        "image": "api:1.0",
        "replicas": 2
      }
    }
  }
}
"""


def _setup(root: str) -> None:
    with open(os.path.join(root, "cluster.json"), "w", encoding="utf-8") as handle:
        handle.write(SEED)


def _observer(root: str) -> JsonDocObserver:
    """The store, with four declared drops.

    Each drop was confirmed to move between two identical clean runs before
    being named, which is what preregistration 6.3 requires:

      * `resourceVersion` and `counter` move on every write by design, the way
        a cluster's do.
      * `events` is the mutation log. It records that a call happened, not
        what it did, so counting it would measure delivery duplication rather
        than effect duplication. The same decision was taken against a
        third-party server on 2026-11-03 that keeps a per-call run ledger, and
        it has to be taken the same way here or the two are not comparable.
      * `name` inside an object is dropped because the server assigns it for
        a Job created from a generateName prefix. Nothing is lost for the
        named kinds: their name is the map key and the key is still compared.

    What remains is the object set and its keys, each object's spec and
    generation, and the number of Jobs. That is what a caller would call the
    effect.

    The projection is checked rather than trusted. With it in place the one
    genuinely non-idempotent fixture, the Job, still reports DUPLICATED, and
    the two converging fixtures report SAFE. A projection that silenced the
    Job would be hiding the result, which is the failure mode worth guarding.
    """
    return JsonDocObserver(
        root,
        patterns=("cluster.json",),
        drop_keys=("resourceVersion", "counter", "at", "events", "name"),
        name="json-doc",
    )


TARGET = ServerTarget(
    name="firstparty-kubernetes-shaped",
    command=["python3", SERVER],
    observer=_observer,
    setup=_setup,
    timeout_s=60.0,
    # First party, in this repository, written for this study. Rule 9 contains
    # third-party code; this is our own, and containing it would buy nothing
    # while making the fixture harder to reproduce from a clone.
    contained=False,
)

FIXTURES = [
    Fixture(
        tool="apply_manifest",
        id="apply_converges",
        arguments={
            "namespace": "prod",
            "kind": "ConfigMap",
            "name": "feature-flags",
            "spec": {"data": {"checkout_v2": "true"}},
        },
        probes=(
            ("get_object", {"namespace": "prod", "kind": "ConfigMap",
                            "name": "feature-flags"}),
        ),
        note=(
            "A declarative apply of an object that does not exist yet. The "
            "manifest names the object, so a second application computes the "
            "same object and the store converges."
        ),
    ),
    Fixture(
        tool="create_object",
        id="create_collides_on_the_name",
        arguments={
            "namespace": "prod",
            "kind": "ConfigMap",
            "name": "feature-flags",
            "spec": {"data": {"checkout_v2": "true"}},
        },
        probes=(
            ("get_object", {"namespace": "prod", "kind": "ConfigMap",
                            "name": "feature-flags"}),
        ),
        note=(
            "The same object through the imperative path. The name is the "
            "identity, so a second application meets it and the server "
            "answers AlreadyExists. Paired with the apply fixture: same "
            "object, same end state, different answer to a retry."
        ),
    ),
    Fixture(
        tool="scale_deployment",
        id="scale_converges",
        arguments={"namespace": "prod", "name": "api", "replicas": 5},
        probes=(
            ("get_object", {"namespace": "prod", "kind": "Deployment",
                            "name": "api"}),
        ),
        note=(
            "A spec write with an absolute target rather than a delta. "
            "Repeating it sets the same number again, which is the shape "
            "that makes a scale safe to retry where an increment would not be."
        ),
    ),
    Fixture(
        tool="delete_object",
        id="delete_not_found_on_repeat",
        arguments={"namespace": "prod", "kind": "Deployment", "name": "api"},
        probes=(
            ("list_objects", {"namespace": "prod", "kind": "Deployment"}),
        ),
        note=(
            "A delete of the seeded object. The end state after two "
            "applications equals the end state after one, and the second "
            "answers NotFound, which is the same answer a caller would get "
            "if the object had never existed."
        ),
    ),
    Fixture(
        tool="create_job",
        id="job_name_assigned_by_the_server",
        arguments={
            "namespace": "prod",
            "generate_name": "migrate-",
            "command": ["/bin/migrate", "--step", "7"],
        },
        probes=(
            ("list_objects", {"namespace": "prod", "kind": "Job"}),
        ),
        note=(
            "A Job created from a generateName prefix, so the server assigns "
            "the name. Two applications leave two Jobs and the work runs "
            "twice. This is the fixture the others exist to contrast with."
        ),
    ),
]
