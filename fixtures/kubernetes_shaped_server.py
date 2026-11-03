"""A Kubernetes-shaped MCP server, first party, with no cluster behind it.

Why this exists. The study wants to say something about agents acting on
infrastructure, and the obvious way to get it is to audit a published
Kubernetes MCP server against a local cluster. That cannot be done inside this
study's containment profile: the profile gives a server no network after
install, and a cluster is a network service whichever machine it runs on. The
alternative of relaxing the profile for one server would make its rows
incomparable with every other row in the sample.

So the Kubernetes result comes from the fault engine instead, against a backend
we wrote, and it is labelled as that everywhere it appears. It is not evidence
about anyone's Kubernetes server. It is evidence about what the API shape does
to a retrying agent, which is the part that generalises.

What is modelled, and why each piece earns its place. The object model is the
part of Kubernetes that decides what a repeat means:

  * objects are namespaced and named, and the name is the identity. Two
    `create` calls for one name collide; two `apply` calls converge.
  * every object carries a `resourceVersion` that changes on every write and
    a `generation` that changes only when the spec changes. That difference is
    the whole reason `apply` is safe to repeat and `create` is not.
  * `apply` is declarative: it computes the object from the manifest, so
    applying twice leaves exactly what applying once left, including the
    generation.
  * `create` is imperative: the second call fails with AlreadyExists.
  * `delete` is imperative in the other direction: the second call fails with
    NotFound, and the agent cannot tell "I deleted it" from "someone else did".
  * `scale` is a spec write, so repeating it converges, but it bumps
    `resourceVersion` without bumping `generation`.
  * `create_job` is the one that does not converge: a Job with
    `generateName` gets a fresh server-assigned name per call, which is the
    canonical non-idempotent infrastructure write and the shape most likely to
    double-run real work.
  * an event log records every mutation, the way a cluster does.

The store is a single JSON file, so the replay oracle can observe it without
needing to understand Kubernetes. Timestamps and resource versions move on
their own, so the spec declares them as drops (preregistration 6.3) and the
observer compares what is left.

Annotations are set to what a careful author would plausibly write, not to
what is convenient: `apply` and `scale` declare `idempotentHint: true`, which
is true of the state; `create` and `delete` declare false; `create_job`
declares false. The point of the fixture is not to catch a bad annotation. It
is to show that an honest annotation still leaves the agent unable to tell,
after a lost response, whether its call landed.

Run it with the backend directory as the working directory:

    python fixtures/kubernetes_shaped_server.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any

STORE = "cluster.json"

TOOLS = [
    {
        "name": "apply_manifest",
        "description": (
            "Apply a manifest declaratively, creating or updating the object "
            "named in it. Converges: applying the same manifest twice leaves "
            "what applying it once left."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "kind": {"type": "string"},
                "name": {"type": "string"},
                "spec": {"type": "object"},
            },
            "required": ["namespace", "kind", "name", "spec"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "create_object",
        "description": (
            "Create an object imperatively. Fails with AlreadyExists if the "
            "name is taken."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "kind": {"type": "string"},
                "name": {"type": "string"},
                "spec": {"type": "object"},
            },
            "required": ["namespace", "kind", "name", "spec"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "scale_deployment",
        "description": "Set a deployment's replica count. Converges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "name": {"type": "string"},
                "replicas": {"type": "integer", "minimum": 0},
            },
            "required": ["namespace", "name", "replicas"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "delete_object",
        "description": (
            "Delete an object by name. Fails with NotFound if it is already "
            "gone, which a retrying caller cannot tell from someone else "
            "having deleted it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "kind": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "kind", "name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "create_job",
        "description": (
            "Create a Job from a generateName prefix. The server assigns the "
            "name, so every call creates another Job and runs the work again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "generate_name": {"type": "string"},
                "command": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["namespace", "generate_name", "command"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "get_object",
        "description": "Read one object.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "kind": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["namespace", "kind", "name"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "list_objects",
        "description": "List objects of a kind in a namespace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "kind": {"type": "string"},
            },
            "required": ["namespace", "kind"],
        },
        "annotations": {"readOnlyHint": True},
    },
]


class ApiError(Exception):
    """A cluster-shaped error: a reason a caller can read, like a real one."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def _load() -> dict[str, Any]:
    if not os.path.exists(STORE):
        return {"objects": {}, "jobs": [], "events": [], "counter": 0}
    try:
        with open(STORE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"objects": {}, "jobs": [], "events": [], "counter": 0}


def _save(state: dict[str, Any]) -> None:
    with open(STORE, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def _key(namespace: str, kind: str, name: str) -> str:
    return f"{namespace}/{kind}/{name}"


def _version(state: dict[str, Any]) -> str:
    state["counter"] = int(state.get("counter", 0)) + 1
    return str(state["counter"])


def _event(state: dict[str, Any], verb: str, key: str) -> None:
    state.setdefault("events", []).append(
        {"verb": verb, "object": key, "at": time.time_ns()}
    )


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    state = _load()
    if name == "apply_manifest":
        key = _key(arguments["namespace"], arguments["kind"], arguments["name"])
        existing = state["objects"].get(key)
        spec = arguments["spec"]
        if existing is not None and existing["spec"] == spec:
            # Converged already. A real API server still answers success and
            # still bumps nothing, which is exactly why a repeat is safe.
            _event(state, "apply-nochange", key)
            _save(state)
            return json.dumps({"object": key, "changed": False})
        generation = 1 if existing is None else existing["generation"] + 1
        state["objects"][key] = {
            "kind": arguments["kind"],
            "name": arguments["name"],
            "namespace": arguments["namespace"],
            "spec": spec,
            "generation": generation,
            "resourceVersion": _version(state),
        }
        _event(state, "apply" if existing is None else "update", key)
        _save(state)
        return json.dumps({"object": key, "changed": True, "generation": generation})
    if name == "create_object":
        key = _key(arguments["namespace"], arguments["kind"], arguments["name"])
        if key in state["objects"]:
            raise ApiError("AlreadyExists", f"{key} already exists")
        state["objects"][key] = {
            "kind": arguments["kind"],
            "name": arguments["name"],
            "namespace": arguments["namespace"],
            "spec": arguments["spec"],
            "generation": 1,
            "resourceVersion": _version(state),
        }
        _event(state, "create", key)
        _save(state)
        return json.dumps({"object": key, "created": True})
    if name == "scale_deployment":
        key = _key(arguments["namespace"], "Deployment", arguments["name"])
        obj = state["objects"].get(key)
        if obj is None:
            raise ApiError("NotFound", f"{key} not found")
        before = obj["spec"].get("replicas")
        obj["spec"]["replicas"] = arguments["replicas"]
        # A spec write always moves resourceVersion, and moves generation only
        # when the spec actually changed. Both are in the store, and the spec
        # declares resourceVersion as a drop, so the oracle sees the part a
        # caller would call the effect.
        obj["resourceVersion"] = _version(state)
        if before != arguments["replicas"]:
            obj["generation"] += 1
        _event(state, "scale", key)
        _save(state)
        return json.dumps({"object": key, "replicas": arguments["replicas"]})
    if name == "delete_object":
        key = _key(arguments["namespace"], arguments["kind"], arguments["name"])
        if key not in state["objects"]:
            raise ApiError("NotFound", f"{key} not found")
        del state["objects"][key]
        _event(state, "delete", key)
        _save(state)
        return json.dumps({"object": key, "deleted": True})
    if name == "create_job":
        suffix = uuid.uuid4().hex[:5]
        name_assigned = f"{arguments['generate_name']}{suffix}"
        # Jobs live in a collection rather than the name-keyed map, because
        # the server picks the name and the thing a caller cares about is how
        # many Jobs exist. Keyed by a generated name, a second Job would be a
        # second map key, and two clean runs would differ on the key itself,
        # so the fixture could never reproduce a baseline. In a list the
        # generated name is a field the spec can declare as a drop, and the
        # length of the list, which is the effect, stays visible.
        state.setdefault("jobs", []).append({
            "kind": "Job",
            "name": name_assigned,
            "namespace": arguments["namespace"],
            "spec": {"command": arguments["command"]},
            "generation": 1,
            "resourceVersion": _version(state),
        })
        _event(state, "create-job", name_assigned)
        _save(state)
        return json.dumps({"object": name_assigned, "created": True})
    if name == "get_object":
        key = _key(arguments["namespace"], arguments["kind"], arguments["name"])
        obj = state["objects"].get(key)
        if obj is None:
            raise ApiError("NotFound", f"{key} not found")
        return json.dumps(obj, sort_keys=True)
    if name == "list_objects":
        if arguments["kind"] == "Job":
            names = sorted(
                job["name"] for job in state.get("jobs", [])
                if job["namespace"] == arguments["namespace"]
            )
            return json.dumps({"items": names, "count": len(names)})
        prefix = f"{arguments['namespace']}/{arguments['kind']}/"
        names = sorted(
            obj["name"] for key, obj in state["objects"].items()
            if key.startswith(prefix)
        )
        return json.dumps({"items": names, "count": len(names)})
    raise ApiError("NotFound", f"unknown tool {name}")


def main() -> None:
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in message:
            continue
        method = message.get("method")
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "initialize":
            reply["result"] = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "kubernetes-shaped", "version": "0.1.0"},
            }
        elif method == "tools/list":
            reply["result"] = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            try:
                text = call_tool(params.get("name", ""), params.get("arguments") or {})
                reply["result"] = {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                }
            except Exception as exc:
                # In band, the way a real API server reports a conflict.
                reply["result"] = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
        else:
            reply["error"] = {"code": -32601, "message": f"unknown method {method}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
