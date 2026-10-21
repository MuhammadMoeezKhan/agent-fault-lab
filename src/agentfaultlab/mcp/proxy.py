"""A stdio MCP proxy that sits between any client and any server.

The audit drives its own client (`session.py`), but the consequence study
needs real clients: Claude Code, Codex CLI, Gemini CLI and the rest, each
with its own retry, timeout and auto-approval policy. Those policies are
part of what is being measured, so they must not be reimplemented. The proxy
makes that possible without touching the client: point the client at

    agent-fault-lab proxy --fault F1:write_file@1 -- <server command>

and it speaks MCP to the proxy while the proxy speaks MCP to the server.

Everything it does not need to understand is passed through byte for byte,
including methods added after this was written, server to client requests
(sampling, roots, elicitation) and notifications. The only messages it ever
alters are `tools/call` requests that the plan selects.

One honest limitation: the proxy cannot make a client time out faster than
the client's own timeout. F1 drops a response, so the client waits out its
full timeout before retrying, which is correct but slow. F3's delay should
therefore be set longer than the client's timeout, and the trace records
what the client actually did.
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Any, BinaryIO, Optional

from . import jsonrpc
from .injector import Action, Decision, FaultInjector, FaultPlan, truncate_list_argument
from .jsonrpc import Message, MessageWriter
from .session import TraceWriter


class StdioFaultProxy:
    """Relays one client's stdio session to one server, injecting faults."""

    def __init__(
        self,
        command: list[str],
        *,
        plan: Optional[FaultPlan] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        trace: Optional[TraceWriter] = None,
        containment: Optional[Any] = None,
    ) -> None:
        if not command:
            raise ValueError("a server command is required")
        self.command = command
        #: Set when the server is third-party (rule 9). The proxy is the one
        #: place the agent path spawns a server, so containing it here covers
        #: every agent episode without the agent needing to know.
        self.containment = containment
        self.injector = FaultInjector(plan)
        self.cwd = cwd
        self.env = env
        self.trace = trace
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._to_server: Optional[MessageWriter] = None
        self._to_client: Optional[MessageWriter] = None
        #: Request ids whose response must never reach the client.
        self._withhold: set[Any] = set()
        self._withhold_lock = threading.Lock()
        self._timers: list[threading.Timer] = []
        self._next_shadow_id = 0

    # ---------------------------------------------------------------- lifecycle

    def run(self, client_in: BinaryIO, client_out: BinaryIO) -> int:
        """Relay until either side closes. Returns the server's exit code."""
        environment = dict(os.environ)
        if self.env:
            environment.update(self.env)
        argv = self.command
        cwd = self.cwd
        if self.containment is not None:
            argv = self.containment.run_argv(self.command, self.env or {})
            cwd = None
            environment = dict(os.environ)
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # the server's stderr goes straight to ours
            cwd=cwd,
            env=environment,
            bufsize=0,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._to_server = MessageWriter(self._proc.stdin)
        self._to_client = MessageWriter(client_out)

        upstream = threading.Thread(
            target=self._pump_client, args=(client_in,), name="afl-proxy-up", daemon=True
        )
        downstream = threading.Thread(
            target=self._pump_server, args=(self._proc.stdout,), name="afl-proxy-down",
            daemon=True,
        )
        upstream.start()
        downstream.start()
        upstream.join()
        # The client has gone. Give the server a moment to drain, then stop it.
        downstream.join(timeout=2)
        return self.close()

    def close(self) -> int:
        for timer in self._timers:
            timer.cancel()
        if self._to_server is not None:
            self._to_server.close()
        code = 0
        proc = self._proc
        if proc is not None:
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    code = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait(timeout=5)
        if self.trace is not None:
            self.trace.note("close", exit_code=code, injections=self.injector.injections)
        return code

    # -------------------------------------------------------------------- pumps

    def _pump_client(self, client_in: BinaryIO) -> None:
        while True:
            message = jsonrpc.read_message(client_in)
            if message is None:
                return
            self._record("client->proxy", message)
            try:
                self._handle_from_client(message)
            except (OSError, ValueError) as exc:
                self._annotate("proxy_error", error=str(exc))
                return

    def _pump_server(self, server_out: BinaryIO) -> None:
        while True:
            message = jsonrpc.read_message(server_out)
            if message is None:
                if self._to_client is not None:
                    self._to_client.close()
                return
            self._record("server->proxy", message)
            identifier = message.get("id")
            if jsonrpc.is_response(message):
                with self._withhold_lock:
                    withheld = identifier in self._withhold
                    self._withhold.discard(identifier)
                if withheld:
                    # This is the fault: the server answered, the client will
                    # never hear it, and it cannot tell commit from no commit.
                    self._annotate("withheld", id=identifier, payload=message)
                    continue
            self._send_to_client(message)

    # ------------------------------------------------------------------ handling

    def _handle_from_client(self, message: Message) -> None:
        decision = self.injector.decide(message)
        if not decision.injected:
            self._send_to_server(message)
            return
        assert decision.mode is not None
        self._annotate(
            "inject",
            id=message.get("id"),
            tool=jsonrpc.tool_name(message),
            mode=decision.mode.id,
        )
        action = decision.action

        if action is Action.SYNTHESISE_SUCCESS:
            self._send_to_client(jsonrpc.tool_result(message["id"], ""))
            return
        if action is Action.SYNTHESISE_ERROR:
            self._send_to_client(
                jsonrpc.error_response(
                    message["id"], -32000, "injected failure before commit"
                )
            )
            return
        if action is Action.TRUNCATE_AND_DROP_RESPONSE:
            trimmed, detail = truncate_list_argument(message, decision.keep)
            if not detail:
                self._send_to_server(message)
                return
            self._annotate("truncated", id=message.get("id"), **detail)
            self._withhold_response(trimmed.get("id"))
            self._send_to_server(trimmed)
            return
        if action is Action.FORWARD_AND_DROP_RESPONSE:
            self._withhold_response(message.get("id"))
            self._send_to_server(message)
            return
        if action is Action.DELAY_THEN_FORWARD:
            timer = threading.Timer(decision.delay_s, self._send_to_server, args=(message,))
            timer.daemon = True
            timer.start()
            self._timers.append(timer)
            return
        if action in (Action.FORWARD_TWICE, Action.FORWARD_CONCURRENT):
            shadow = dict(message)
            shadow["id"] = self._shadow_id(message.get("id"))
            # The shadow copy's answer is the proxy's business, not the
            # client's: it asked once and must see exactly one response.
            self._withhold_response(shadow["id"])
            if action is Action.FORWARD_CONCURRENT:
                self._send_to_server(shadow)
                self._send_to_server(message)
            else:
                self._send_to_server(message)
                self._send_to_server(shadow)
            return
        self._send_to_server(message)

    def _shadow_id(self, original: Any) -> str:
        """An id the client will never use, so the duplicate cannot collide."""
        self._next_shadow_id += 1
        return f"afl-shadow-{self._next_shadow_id}-{original}"

    def _withhold_response(self, identifier: Any) -> None:
        if identifier is None:
            return
        with self._withhold_lock:
            self._withhold.add(identifier)

    def _send_to_server(self, message: Message) -> None:
        if self._to_server is not None:
            self._to_server.send(message)
            self._record("proxy->server", message)

    def _send_to_client(self, message: Message) -> None:
        if self._to_client is not None:
            self._to_client.send(message)
            self._record("proxy->client", message)

    def _record(self, direction: str, message: Message) -> None:
        if self.trace is not None:
            self.trace.message(direction, message)

    def _annotate(self, kind: str, **fields: Any) -> None:
        """Record something the proxy did, as opposed to a message it moved."""
        if self.trace is not None:
            self.trace.note(kind, **fields)


def run_proxy(
    command: list[str],
    *,
    fault: Optional[str] = None,
    trace_path: Optional[str] = None,
    delay_s: float = 2.0,
    keep: int = 1,
    seed: int = 0,
    cwd: Optional[str] = None,
    contain: Optional[str] = None,
    mount: Optional[str] = None,
) -> int:
    """Entry point for `agent-fault-lab proxy`. Relays this process's stdio."""
    import sys

    plan = (
        FaultPlan.parse(fault, delay_s=delay_s, keep=keep, seed=seed) if fault else None
    )
    trace = (
        TraceWriter(trace_path, context={"server": command, "fault": fault})
        if trace_path
        else None
    )
    containment = None
    if contain:
        from ..container import Containment, require

        require()
        containment = Containment(image=contain, mount=mount or cwd or os.getcwd())
    proxy = StdioFaultProxy(
        command, plan=plan, trace=trace, cwd=cwd, containment=containment
    )
    try:
        return proxy.run(sys.stdin.buffer, sys.stdout.buffer)
    finally:
        if trace is not None:
            trace.close()
