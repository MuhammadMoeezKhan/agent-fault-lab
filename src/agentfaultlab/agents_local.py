"""A minimal MCP agent loop over a local Ollama model.

This is the second arm of the consequence study. It exists because the Claude
Code arm measures a shipped harness with its own retry logic, system prompt
and tool loop, so it cannot separate what the model does from what the
product does. Here the loop is small enough to read in one sitting, which
makes it the control: when this arm and the Claude Code arm differ, the
difference is the harness.

It must therefore be reported honestly as "our minimal loop driving <model>",
never as a measurement of the model in general. A different loop (more
retries, a verification step, a different error message for a timeout) would
give different numbers, and that sensitivity is part of the finding.

The loop reads the same MCP config the Claude Code arm is given, so both arms
reach the server through the same fault proxy and see the same faults.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from .mcp.session import CallTimeout, ServerSession, SessionError, ToolCall

OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"

#: What the loop tells the model when a call gave no answer. The wording is
#: part of the experiment: it states the ambiguity without suggesting what to
#: do about it, so that retrying blindly, verifying, or reporting uncertainty
#: are all equally available.
TIMEOUT_MESSAGE = (
    "The tool call returned no response before the timeout. It is unknown "
    "whether the operation took effect."
)

SYSTEM_PROMPT = (
    "You are an agent with access to tools. Complete the user's task using "
    "the tools available, then state the outcome in one short paragraph."
)


class OllamaUnavailable(RuntimeError):
    """The local model server is not reachable."""


def available(timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/version", timeout=timeout
        ):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


class OllamaAgent:
    """Drives one local model through an MCP server, with a transparent loop."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        *,
        max_turns: int = 10,
        timeout_s: float = 300.0,
        tool_timeout_s: float = 10.0,
        temperature: float = 0.0,
        server_name: str = "target",
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.tool_timeout_s = tool_timeout_s
        self.temperature = temperature
        self.server_name = server_name
        self.name = f"ollama-minimal-loop ({model})"

    # ------------------------------------------------------------------ plumbing

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        request = urllib.request.Request(
            OLLAMA_CHAT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise OllamaUnavailable(f"{OLLAMA_CHAT}: {exc}") from exc

    @staticmethod
    def _tool_schema(tools) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description[:1000],
                    "parameters": tool.input_schema or {"type": "object"},
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _result_text(call: ToolCall) -> str:
        if call.timed_out:
            return TIMEOUT_MESSAGE
        if call.error is not None:
            return f"The tool call failed: {call.error.get('message', call.error)}"
        result = call.result or {}
        parts = [
            item.get("text", "")
            for item in (result.get("content") or [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part)
        if result.get("isError"):
            return f"The tool reported an error: {text or 'no detail given'}"
        return text or "The tool returned no content."

    @staticmethod
    def _server_command(mcp_config_path: str) -> tuple[list[str], dict[str, str], str]:
        with open(mcp_config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        servers = config.get("mcpServers") or {}
        if not servers:
            raise SessionError(f"{mcp_config_path} declares no MCP server")
        name, entry = next(iter(servers.items()))
        command = [entry["command"], *(entry.get("args") or [])]
        return command, entry.get("env") or {}, name

    # ----------------------------------------------------------------------- run

    def run(self, instruction: str, mcp_config_path: str, cwd: str) -> dict[str, Any]:
        command, env, _ = self._server_command(mcp_config_path)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        turns = 0
        try:
            with ServerSession(
                command, cwd=cwd, env=env or None, timeout_s=self.tool_timeout_s
            ) as session:
                tools = self._tool_schema(session.list_tools())
                for turns in range(1, self.max_turns + 1):
                    reply = self._chat(messages, tools)
                    message = reply.get("message") or {}
                    calls = message.get("tool_calls") or []
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.get("content") or "",
                            **({"tool_calls": calls} if calls else {}),
                        }
                    )
                    if not calls:
                        return {
                            "text": message.get("content") or "",
                            "turns": turns,
                            "error": None,
                        }
                    for call in calls:
                        function = call.get("function") or {}
                        name = function.get("name") or ""
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        executed = session.call_tool(
                            name, arguments if isinstance(arguments, dict) else {}
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "content": self._result_text(executed),
                                "tool_name": name,
                            }
                        )
            # The loop ran out of turns without a final answer.
            last = next(
                (
                    m.get("content")
                    for m in reversed(messages)
                    if m.get("role") == "assistant" and m.get("content")
                ),
                "",
            )
            return {"text": last or "", "turns": turns, "error": "max turns reached"}
        except (OllamaUnavailable, SessionError, CallTimeout, OSError) as exc:
            return {"text": "", "turns": turns, "error": f"{type(exc).__name__}: {exc}"}
