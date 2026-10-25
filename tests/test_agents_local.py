"""The minimal local loop's one turn, and the two ways it can fail.

The distinction these tests protect is the one the pilot got wrong for two
batches: a model that never finishes a turn looked exactly like a server that
had died, because with `stream: false` neither of them sends a byte. The loop
now streams, so the two have separate exceptions and separate rows in the
results.
"""

from __future__ import annotations

import json
import time
import urllib.error

import pytest

from agentfaultlab.agents_local import (
    DeliberationUnbounded,
    OllamaAgent,
    OllamaUnavailable,
)


class FakeResponse:
    """What `urlopen` returns: an iterable of newline-delimited JSON chunks."""

    def __init__(self, lines, delay=0.0):
        self._lines = lines
        self._delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for line in self._lines:
            if self._delay:
                time.sleep(self._delay)
            yield line


def chunk(**message):
    return json.dumps({"message": message}).encode("utf-8") + b"\n"


def final(**message):
    return json.dumps({"message": message, "done": True}).encode("utf-8") + b"\n"


def install(monkeypatch, response):
    import agentfaultlab.agents_local as module

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: response)


def test_a_streamed_turn_is_assembled_from_its_chunks(monkeypatch):
    install(
        monkeypatch,
        FakeResponse(
            [
                chunk(thinking="weighing it up"),
                chunk(content="I will "),
                chunk(content="write the row."),
                final(content=""),
            ]
        ),
    )
    reply = OllamaAgent()._chat([], [])
    assert reply["message"]["content"] == "I will write the row."
    assert "tool_calls" not in reply["message"]
    # Reasoning is counted but never folded into the answer: the claim parser
    # scores what the agent said, not what it considered saying.
    assert reply["thinking_chars"] == len("weighing it up")


def test_tool_calls_survive_being_split_across_chunks(monkeypatch):
    call = {"function": {"name": "write_query", "arguments": {"query": "insert 1"}}}
    install(monkeypatch, FakeResponse([chunk(tool_calls=[call]), final(content="")]))
    reply = OllamaAgent()._chat([], [])
    assert reply["message"]["tool_calls"] == [call]


def test_reasoning_past_the_deadline_is_its_own_failure(monkeypatch):
    """The six errored episodes of 2026-10-21, with their real shape.

    An endless reasoning channel, no content, no tool call, and a first chunk
    that arrived immediately. Reported as a server timeout it was a nuisance;
    reported as non-termination it is attrition caused by the prompt arm.
    """

    def forever():
        while True:
            yield chunk(thinking="But the user wants to know when it exists. ")

    install(monkeypatch, FakeResponse(forever(), delay=0.01))
    agent = OllamaAgent(timeout_s=0.2)
    with pytest.raises(DeliberationUnbounded) as caught:
        agent._chat([], [])
    assert caught.value.thinking_chars > 0
    assert caught.value.chunks > 0
    assert "no completed turn" in str(caught.value)


def test_a_stream_that_ends_without_done_is_not_a_silent_empty_answer(monkeypatch):
    """A truncated stream must not read as "the model said nothing".

    An empty final message scores as `none` in the claim parser, which would
    put a lost turn into the denominator as an honest non-claim.
    """
    install(monkeypatch, FakeResponse([chunk(content="half an ans")]))
    with pytest.raises(DeliberationUnbounded):
        OllamaAgent()._chat([], [])


def test_an_unreachable_server_is_still_reported_as_unreachable(monkeypatch):
    import agentfaultlab.agents_local as module

    def refuse(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", refuse)
    with pytest.raises(OllamaUnavailable):
        OllamaAgent()._chat([], [])


def test_the_request_still_carries_the_experiment_unchanged(monkeypatch):
    """Streaming is a transport change and must not become a prompt change."""
    import agentfaultlab.agents_local as module

    seen = {}

    def capture(request, timeout=None):
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        seen["timeout"] = timeout
        return FakeResponse([final(content="done")])

    monkeypatch.setattr(module.urllib.request, "urlopen", capture)
    agent = OllamaAgent(model="qwen3:8b", temperature=0.0, timeout_s=900.0)
    agent._chat([{"role": "user", "content": "hi"}], [{"type": "function"}])
    payload = seen["payload"]
    assert payload["model"] == "qwen3:8b"
    assert payload["options"] == {"temperature": 0.0}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["tools"] == [{"type": "function"}]
    assert payload["stream"] is True
    assert seen["timeout"] == 900.0
