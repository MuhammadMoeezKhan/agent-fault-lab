"""The Kubernetes-shaped fixture behaves the way the paper will claim it does.

This is a first-party backend, so its behaviour is an assumption of the result
rather than an observation about anyone's server. An assumption that is not
tested is a claim, so the shapes are pinned here: declarative writes converge,
imperative ones collide, a delete is not found twice, and the Job the server
names does not converge at all. If any of these changed, the Kubernetes
section of the paper would be describing a backend that no longer exists.
"""

import json
import os
import subprocess
import sys

import pytest

SERVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "kubernetes_shaped_server.py",
)

SEED = {
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
            "spec": {"image": "api:1.0", "replicas": 2},
        }
    },
}


def call(tmp_path, calls):
    """Drive the server over stdio and return (replies, final store)."""
    (tmp_path / "cluster.json").write_text(json.dumps(SEED))
    lines = [json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize"})]
    for index, (tool, arguments) in enumerate(calls, start=1):
        lines.append(json.dumps({
            "jsonrpc": "2.0",
            "id": index,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }))
    done = subprocess.run(
        [sys.executable, SERVER],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=60,
    )
    replies = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    store = json.loads((tmp_path / "cluster.json").read_text())
    return replies[1:], store


def test_apply_converges(tmp_path):
    manifest = {
        "namespace": "prod",
        "kind": "ConfigMap",
        "name": "flags",
        "spec": {"data": {"a": "1"}},
    }
    once, store_once = call(tmp_path, [("apply_manifest", manifest)])
    twice, store_twice = call(tmp_path, [("apply_manifest", manifest)] * 2)
    assert all(not reply["result"]["isError"] for reply in once + twice)
    key = "prod/ConfigMap/flags"
    assert store_once["objects"][key]["spec"] == store_twice["objects"][key]["spec"]
    # The generation is the part that must not move: it is what tells a
    # controller the spec changed.
    assert store_once["objects"][key]["generation"] == 1
    assert store_twice["objects"][key]["generation"] == 1


def test_create_collides_on_the_second_call(tmp_path):
    manifest = {
        "namespace": "prod",
        "kind": "ConfigMap",
        "name": "flags",
        "spec": {"data": {"a": "1"}},
    }
    replies, store = call(tmp_path, [("create_object", manifest)] * 2)
    assert not replies[0]["result"]["isError"]
    assert replies[1]["result"]["isError"]
    assert "AlreadyExists" in replies[1]["result"]["content"][0]["text"]
    assert len(store["objects"]) == 2  # the seeded Deployment plus one ConfigMap


def test_scale_converges_and_does_not_double_apply(tmp_path):
    scale = {"namespace": "prod", "name": "api", "replicas": 5}
    _once, store_once = call(tmp_path, [("scale_deployment", scale)])
    _twice, store_twice = call(tmp_path, [("scale_deployment", scale)] * 2)
    key = "prod/Deployment/api"
    assert store_once["objects"][key]["spec"]["replicas"] == 5
    assert store_twice["objects"][key]["spec"]["replicas"] == 5
    # An increment would have reached 8 here. An absolute target does not.
    assert store_once["objects"][key]["generation"] == \
        store_twice["objects"][key]["generation"]


def test_delete_is_not_found_the_second_time(tmp_path):
    target = {"namespace": "prod", "kind": "Deployment", "name": "api"}
    replies, store = call(tmp_path, [("delete_object", target)] * 2)
    assert not replies[0]["result"]["isError"]
    assert replies[1]["result"]["isError"]
    assert "NotFound" in replies[1]["result"]["content"][0]["text"]
    assert store["objects"] == {}


def test_a_job_does_not_converge_which_is_the_point(tmp_path):
    job = {
        "namespace": "prod",
        "generate_name": "migrate-",
        "command": ["/bin/migrate"],
    }
    _once, store_once = call(tmp_path, [("create_job", job)])
    _twice, store_twice = call(tmp_path, [("create_job", job)] * 2)
    assert len(store_once["jobs"]) == 1
    assert len(store_twice["jobs"]) == 2
    # Different names, so the count is the only honest observable, which is
    # why the spec drops the name and keeps the collection.
    assert store_twice["jobs"][0]["name"] != store_twice["jobs"][1]["name"]


def test_the_declared_projection_does_not_silence_the_job(tmp_path):
    """The guard on a projection chosen after seeing a result.

    The drops exist to remove per-call noise (an event log, a server-assigned
    name). If they also removed the one real duplicate, they would be hiding
    the finding rather than cleaning it up.
    """
    from agentfaultlab.observers import JsonDocObserver, equal

    counter = {"n": 0}

    def snapshot(calls):
        counter["n"] += 1
        directory = tmp_path / f"run{counter['n']}"
        directory.mkdir()
        call(directory, calls)
        return JsonDocObserver(
            str(directory),
            patterns=("cluster.json",),
            drop_keys=("resourceVersion", "counter", "at", "events", "name"),
        ).snapshot()

    job = {
        "namespace": "prod",
        "generate_name": "migrate-",
        "command": ["/bin/migrate"],
    }
    assert not equal(snapshot([("create_job", job)]), snapshot([("create_job", job)] * 2))
    apply = {
        "namespace": "prod",
        "kind": "ConfigMap",
        "name": "flags",
        "spec": {"data": {"a": "1"}},
    }
    assert equal(
        snapshot([("apply_manifest", apply)]),
        snapshot([("apply_manifest", apply)] * 2),
    )
