"""Newline-delimited JSON-RPC framing, as the MCP stdio transport defines it.

MCP's stdio transport sends each JSON-RPC message as one UTF-8 line on stdin
or stdout, with no embedded newlines and no length header. That is the whole
wire format, so the standard library is enough: no dependency on an MCP SDK,
which matters because the proxy has to pass through messages it does not
understand (new methods, vendor extensions) without parsing them away.
"""

from __future__ import annotations

import json
import threading
from typing import Any, BinaryIO, Optional

Message = dict[str, Any]


def read_message(stream: BinaryIO) -> Optional[Message]:
    """Read one message. Returns None at end of stream.

    A line that is not valid JSON is skipped rather than fatal: some servers
    print banners to stdout even though the transport forbids it, and the
    audit should survive that instead of losing the session.
    """
    while True:
        line = stream.readline()
        if not line:
            return None
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        # Batches were removed in MCP 2025-06-18 but older servers may send
        # them. Pass the first element and drop the rest; the trace records it.
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]


class MessageWriter:
    """Serialises writes to one stream so relay threads cannot interleave."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self.closed = False

    def send(self, message: Message) -> bool:
        """Write one message. Returns False if the peer has gone away."""
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._lock:
            if self.closed:
                return False
            try:
                self._stream.write(payload)
                self._stream.flush()
            except (BrokenPipeError, ValueError, OSError):
                self.closed = True
                return False
        return True

    def close(self) -> None:
        with self._lock:
            self.closed = True
            try:
                self._stream.close()
            except (BrokenPipeError, ValueError, OSError):
                pass


def is_request(message: Message) -> bool:
    return "method" in message and "id" in message


def is_notification(message: Message) -> bool:
    return "method" in message and "id" not in message


def is_response(message: Message) -> bool:
    return "method" not in message and "id" in message


def tool_name(message: Message) -> Optional[str]:
    """The tool a `tools/call` request targets, or None for anything else."""
    if message.get("method") != "tools/call":
        return None
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    name = params.get("name")
    return name if isinstance(name, str) else None


def error_response(request_id: Any, code: int, message: str) -> Message:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def tool_result(request_id: Any, text: str, is_error: bool = False) -> Message:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
    }
