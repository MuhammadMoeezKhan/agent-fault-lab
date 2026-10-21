"""A minimal MCP client that can hold a fault between itself and a server.

The audit needs a client it fully controls, because the thing being measured
is what the backend looks like after an ambiguous call. An SDK client would
hide retries and timeouts, which are exactly the variables under test, so
this session implements the handshake, `tools/list` and `tools/call` directly
and injects the fault on the way out.

What the faults mean here:

* F1 forwards the call and waits for the server's own response before
  withholding it. A real timeout would not wait, but waiting is what makes
  the post-commit case deterministic (it is LIMBO's `timeout_post`), and the
  caller is left in exactly the state a timeout leaves it in: no answer.
* F3 holds the call for `delay_s`, which is longer than the caller's own
  timeout, so `call_tool` raises `CallTimeout` and the write lands later.
* F6 and F7 never reach the server, so the trace records a synthetic answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from . import jsonrpc
from .injector import Action, Decision, FaultInjector, FaultPlan, truncate_list_argument
from .jsonrpc import Message

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "agent-fault-lab", "version": "0.2.0"}


class SessionError(RuntimeError):
    """The server could not be started, or spoke something unusable."""


class CallTimeout(TimeoutError):
    """No answer arrived in time. The caller cannot know if the write landed."""


@dataclass
class ToolCall:
    """One attempt at a tool call, and what the caller learned from it."""

    tool: str
    arguments: dict[str, Any]
    attempt: int
    fault: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    timed_out: bool = False
    #: Set when the fault withheld a response the server did send, so the
    #: audit can separate "the server committed" from "the caller knows".
    withheld_result: Optional[dict[str, Any]] = None
    duration_s: float = 0.0

    @property
    def caller_saw_success(self) -> bool:
        if self.timed_out or self.error is not None or self.result is None:
            return False
        return not self.result.get("isError", False)


@dataclass
class Tool:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def read_only_hint(self) -> Optional[bool]:
        value = self.annotations.get("readOnlyHint")
        return value if isinstance(value, bool) else None

    @property
    def idempotent_hint(self) -> Optional[bool]:
        value = self.annotations.get("idempotentHint")
        return value if isinstance(value, bool) else None

    @property
    def destructive_hint(self) -> Optional[bool]:
        value = self.annotations.get("destructiveHint")
        return value if isinstance(value, bool) else None

    def list_argument_names(self) -> list[str]:
        properties = self.input_schema.get("properties")
        if not isinstance(properties, dict):
            return []
        return [
            name
            for name, schema in properties.items()
            if isinstance(schema, dict) and schema.get("type") == "array"
        ]


class ServerSession:
    """Launches one stdio MCP server and talks to it under a fault plan."""

    def __init__(
        self,
        command: list[str],
        *,
        plan: Optional[FaultPlan] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        trace: Optional["TraceWriter"] = None,
        timeout_s: float = 20.0,
        containment: Optional[Any] = None,
    ) -> None:
        if not command:
            raise ValueError("a server command is required")
        self.command = command
        #: When set, the server runs inside a container instead of on the host
        #: (rule 9). The stdio contract is identical either way, so nothing
        #: below this line needs to know which it is talking to.
        self.containment = containment
        self.injector = FaultInjector(plan)
        self.cwd = cwd
        self.env = env
        self.trace = trace
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._writer: Optional[jsonrpc.MessageWriter] = None
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._pending: dict[Any, dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._dropped: dict[Any, Message] = {}
        self._threads: list[threading.Thread] = []
        self.stderr_tail: list[str] = []
        self.server_info: dict[str, Any] = {}
        self.calls: list[ToolCall] = []

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> "ServerSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        environment = dict(os.environ)
        if self.env:
            environment.update(self.env)
        argv = self.command
        cwd = self.cwd
        if self.containment is not None:
            # The container carries its own workdir and its own environment,
            # and the host environment is deliberately not inherited: that is
            # the whole point of the boundary.
            argv = self.containment.run_argv(self.command, self.env or {})
            cwd = None
            environment = dict(os.environ)
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=environment,
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            raise SessionError(f"could not start {argv[0]!r}: {exc}") from exc
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._writer = jsonrpc.MessageWriter(self._proc.stdin)
        self._spawn(self._read_stdout, "stdout")
        self._spawn(self._read_stderr, "stderr")
        self._handshake()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            except OSError:
                pass
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        for thread in self._threads:
            thread.join(timeout=2)

    def _spawn(self, target: Any, name: str) -> None:
        thread = threading.Thread(target=target, name=f"afl-{name}", daemon=True)
        thread.start()
        self._threads.append(thread)

    # ------------------------------------------------------------------ reading

    def _read_stdout(self) -> None:
        stream = self._proc.stdout if self._proc else None
        if stream is None:
            return
        while True:
            message = jsonrpc.read_message(stream)
            if message is None:
                break
            self._record("server->client", message)
            if jsonrpc.is_response(message):
                self._resolve(message)
            elif jsonrpc.is_request(message):
                # The audit client grants no capabilities, so anything the
                # server asks of it (sampling, roots, elicitation) is refused
                # rather than left hanging.
                if self._writer is not None:
                    self._writer.send(
                        jsonrpc.error_response(
                            message["id"], -32601, "client capability not available"
                        )
                    )
        with self._pending_lock:
            waiting = list(self._pending.values())
            for slot in waiting:
                slot["event"].set()

    def _read_stderr(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr_tail.append(line)
                del self.stderr_tail[:-50]

    def _resolve(self, message: Message) -> None:
        with self._pending_lock:
            slot = self._pending.pop(message["id"], None)
        if slot is None:
            # A response to a call the fault already answered, or a duplicate
            # copy's answer. Kept so the audit can prove the server ran it.
            self._dropped[message["id"]] = message
            return
        if slot.get("drop"):
            self._dropped[message["id"]] = message
        slot["response"] = message
        slot["event"].set()

    # ------------------------------------------------------------------ writing

    def _allocate_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _send(self, message: Message, *, drop: bool = False) -> Optional[dict[str, Any]]:
        """Send a request and register a slot for its response."""
        slot = {"event": threading.Event(), "response": None, "drop": drop}
        with self._pending_lock:
            self._pending[message["id"]] = slot
        self._record("client->server", message)
        if self._writer is None or not self._writer.send(message):
            with self._pending_lock:
                self._pending.pop(message["id"], None)
            raise SessionError("the server's stdin is closed")
        return slot

    def _await(self, slot: dict[str, Any], timeout_s: float) -> Optional[Message]:
        if not slot["event"].wait(timeout_s):
            return None
        return slot["response"]

    def _request(self, method: str, params: Optional[dict[str, Any]] = None) -> Message:
        message: Message = {"jsonrpc": "2.0", "id": self._allocate_id(), "method": method}
        if params is not None:
            message["params"] = params
        slot = self._send(message)
        response = self._await(slot, self.timeout_s)
        if response is None:
            with self._pending_lock:
                self._pending.pop(message["id"], None)
            raise CallTimeout(f"{method} did not answer within {self.timeout_s}s")
        if "error" in response:
            raise SessionError(f"{method} failed: {response['error']}")
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    def _notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        message: Message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._record("client->server", message)
        if self._writer is not None:
            self._writer.send(message)

    # ------------------------------------------------------------------- protocol

    def _handshake(self) -> None:
        self.server_info = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self._notify("notifications/initialized")

    def list_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        cursor: Optional[str] = None
        seen_cursors: set[str] = set()
        while True:
            params = {"cursor": cursor} if cursor else None
            page = self._request("tools/list", params)
            for raw in page.get("tools", []) or []:
                if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                    continue
                tools.append(
                    Tool(
                        name=raw["name"],
                        description=raw.get("description") or "",
                        input_schema=raw.get("inputSchema") or {},
                        annotations=raw.get("annotations") or {},
                    )
                )
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        return tools

    def call_tool(
        self,
        tool: str,
        arguments: Optional[dict[str, Any]] = None,
        *,
        attempt: int = 1,
        timeout_s: Optional[float] = None,
    ) -> ToolCall:
        """Call a tool once, applying the fault plan. Never raises on fault.

        A fault that leaves the caller without an answer is reported as
        `timed_out`, because that is all a real client would know.
        """
        arguments = dict(arguments or {})
        budget = self.timeout_s if timeout_s is None else timeout_s
        request: Message = {
            "jsonrpc": "2.0",
            "id": self._allocate_id(),
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        decision = self.injector.decide(request)
        record = ToolCall(
            tool=tool, arguments=arguments, attempt=attempt, fault=decision.describe()
        )
        started = time.monotonic()
        try:
            self._dispatch(request, decision, record, budget)
        finally:
            record.duration_s = round(time.monotonic() - started, 4)
        self.calls.append(record)
        if self.trace is not None:
            self.trace.call(record)
        return record

    def _dispatch(
        self, request: Message, decision: Decision, record: ToolCall, budget: float
    ) -> None:
        action = decision.action

        if action is Action.SYNTHESISE_SUCCESS:
            record.result = {"content": [{"type": "text", "text": ""}], "isError": False}
            return
        if action is Action.SYNTHESISE_ERROR:
            record.error = {"code": -32000, "message": "injected failure before commit"}
            return

        if action is Action.TRUNCATE_AND_DROP_RESPONSE:
            trimmed, detail = truncate_list_argument(request, decision.keep)
            if not detail:
                # Nothing to truncate, so F5 does not apply to this call.
                record.fault = None
                action = Action.PASS
            else:
                request = trimmed
                record.arguments = request["params"]["arguments"]
                record.fault = f"{decision.mode.id}:{decision.mode.name}"
                action = Action.FORWARD_AND_DROP_RESPONSE

        if action is Action.DELAY_THEN_FORWARD:
            # The caller gives up first; the write lands after it has moved on.
            threading.Timer(
                decision.delay_s, self._send_late, args=(request,)
            ).start()
            record.timed_out = True
            return

        drop = action in (Action.FORWARD_AND_DROP_RESPONSE,)
        slot = self._send(request, drop=drop)

        if action in (Action.FORWARD_TWICE, Action.FORWARD_CONCURRENT):
            twin = dict(request)
            twin["id"] = self._allocate_id()
            if action is Action.FORWARD_TWICE:
                # Sequential: the caller gets the first answer, then the same
                # call is delivered again behind its back.
                response = self._await(slot, budget)
                self._finish(record, response, request["id"], budget)
                twin_slot = self._send(twin, drop=True)
            else:
                # Concurrent: both copies are in flight at once.
                twin_slot = self._send(twin, drop=True)
                response = self._await(slot, budget)
                self._finish(record, response, request["id"], budget)
            # Wait for the extra copy to be answered before returning. The
            # caller has its answer either way, but the audit snapshots the
            # backend next and a duplicate still in flight would be scored as
            # no duplicate at all.
            self._await(twin_slot, budget)
            return

        response = self._await(slot, budget)
        if drop:
            record.withheld_result = (response or {}).get("result")
            record.timed_out = True
            if response is None:
                with self._pending_lock:
                    self._pending.pop(request["id"], None)
            return
        self._finish(record, response, request["id"], budget)

    def _finish(
        self, record: ToolCall, response: Optional[Message], request_id: Any, budget: float
    ) -> None:
        if response is None:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            record.timed_out = True
            return
        if "error" in response:
            record.error = response["error"]
            return
        result = response.get("result")
        record.result = result if isinstance(result, dict) else {}

    def _send_late(self, request: Message) -> None:
        try:
            self._send(request, drop=True)
        except SessionError:
            pass

    # --------------------------------------------------------------------- trace

    def _record(self, direction: str, message: Message) -> None:
        if self.trace is not None:
            self.trace.message(direction, message)


class TraceWriter:
    """Append-only JSONL trace of everything that crossed the boundary."""

    def __init__(self, path: str, *, context: Optional[dict[str, Any]] = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._handle = open(path, "a", encoding="utf-8")
        self._write({"event": "open", "context": context or {}})

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def message(self, direction: str, message: Message) -> None:
        self._write(
            {
                "event": "message",
                "direction": direction,
                "id": message.get("id"),
                "method": message.get("method"),
                "tool": jsonrpc.tool_name(message),
                "payload": message,
            }
        )

    def call(self, call: ToolCall) -> None:
        self._write(
            {
                "event": "call",
                "tool": call.tool,
                "attempt": call.attempt,
                "fault": call.fault,
                "arguments": call.arguments,
                "timed_out": call.timed_out,
                "caller_saw_success": call.caller_saw_success,
                "error": call.error,
                "duration_s": call.duration_s,
            }
        )

    def note(self, kind: str, **fields: Any) -> None:
        self._write({"event": kind, **fields})

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_trace(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
