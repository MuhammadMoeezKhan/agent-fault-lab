"""A small MCP stdio server with known-good and known-bad tools.

The audit's own correctness has to be testable without the network, so this
server stands in for a real one and its tools are deliberately chosen to
cover every verdict the audit can reach:

* `append_line`: mutating, not idempotent, and it claims `idempotentHint`.
  The audit should find DUPLICATED under F1 to F3 and flag the annotation.
* `append_with_key`: the same write behind a caller supplied key, so a retry
  with the same key is a no-op. The audit should find SAFE.
* `write_file`: overwrites, so one application and two are indistinguishable.
  The audit should report NATURALLY_IDEMPOTENT, not a pass.
* `secret_append`: appends to a file no read tool exposes, and claims
  `readOnlyHint`. The audit should find it mutating, UNCONFIRMABLE through
  the probes, and in violation of its own hint.
* `read_file` and `list_files`: read-only probes.

Run it as `python fake_server.py` with the backend directory as the cwd.
"""

from __future__ import annotations

import json
import os
import sys

TOOLS = [
    {
        "name": "append_line",
        "description": "Append a line to a file.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "line": {"type": "string"}},
            "required": ["path", "line"],
        },
        "annotations": {"idempotentHint": True, "readOnlyHint": False},
    },
    {
        "name": "append_with_key",
        "description": "Append a line once per idempotency_key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["path", "line", "idempotency_key"],
        },
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "write_file",
        "description": "Write a file, replacing its contents.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
        "annotations": {"idempotentHint": True},
    },
    {
        "name": "secret_append",
        "description": "Append to the ledger. No read tool exposes it.",
        "inputSchema": {
            "type": "object",
            "properties": {"line": {"type": "string"}},
            "required": ["line"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "append_many",
        "description": "Append several lines to a file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "lines": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path", "lines"],
        },
        "annotations": {},
    },
    {
        "name": "read_file",
        "description": "Read a file.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "list_files",
        "description": "List files in the backend.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
]

KEYS_FILE = ".keys.json"
LEDGER = ".ledger"


def _safe(path: str) -> str:
    resolved = os.path.realpath(os.path.join(os.getcwd(), path))
    if not resolved.startswith(os.path.realpath(os.getcwd()) + os.sep):
        raise ValueError("path escapes the backend root")
    return resolved


def _append(path: str, line: str) -> str:
    target = _safe(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return f"appended to {path}"


def _seen_key(key: str) -> bool:
    try:
        with open(KEYS_FILE, encoding="utf-8") as handle:
            keys = json.load(handle)
    except (OSError, json.JSONDecodeError):
        keys = []
    if key in keys:
        return True
    keys.append(key)
    with open(KEYS_FILE, "w", encoding="utf-8") as handle:
        json.dump(sorted(keys), handle)
    return False


def call_tool(name: str, arguments: dict) -> str:
    if name == "append_line":
        return _append(arguments["path"], arguments["line"])
    if name == "append_with_key":
        if _seen_key(arguments["idempotency_key"]):
            return "already applied"
        return _append(arguments["path"], arguments["line"])
    if name == "write_file":
        target = _safe(arguments["path"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(arguments["text"])
        return f"wrote {arguments['path']}"
    if name == "secret_append":
        return _append(LEDGER, arguments["line"])
    if name == "append_many":
        for line in arguments.get("lines", []):
            _append(arguments["path"], line)
        return f"appended {len(arguments.get('lines', []))} lines"
    if name == "read_file":
        try:
            with open(_safe(arguments["path"]), encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            return "missing"
    if name == "list_files":
        names = sorted(
            entry
            for entry in os.listdir(os.getcwd())
            if entry not in (KEYS_FILE, LEDGER)
        )
        return "\n".join(names)
    raise ValueError(f"unknown tool {name}")


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
        method = message.get("method")
        if "id" not in message:
            continue
        reply: dict = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "initialize":
            reply["result"] = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-server", "version": "0.1.0"},
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
            except Exception as exc:  # the server reports tool errors in-band
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
