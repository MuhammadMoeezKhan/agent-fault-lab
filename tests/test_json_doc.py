"""The JSON document observer, which decides whether a class of servers is auditable.

A server whose durable state is one JSON document with a timestamp in it
cannot reproduce a clean baseline, so every row reads UNPROVEN. The first two
such servers were handled by excluding the file, which costs the verdict and
biases the sample: a server that records when it wrote something tends to be
one built carefully, so excluding those files understates duplication among
careful servers.

This observer projects declared keys away instead. That makes it the most
dangerous observer in the set, because a projection that is one key too wide
turns a real duplicate into a SAFE verdict. These tests pin both directions:
what must compare equal, and what must not.
"""
from __future__ import annotations

import json
import os

from agentfaultlab.observers import JsonDocObserver


def write(path, document):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def doc(stamp, calls=1, variables=None):
    return {
        "created": stamp,
        "updated": stamp,
        "variables": variables
        if variables is not None
        else {"total": {"result": 42.75, "timestamp": stamp}},
        "statistics": {"total_calculations": calls, "last_access": stamp},
    }


DROPS = [
    "created",
    "updated",
    "timestamp",
    "last_access",
    "statistics.total_calculations",
]


def observer(root):
    return JsonDocObserver(root, patterns=["workspace.json"], drop_keys=DROPS)


def test_the_same_payload_with_different_timestamps_compares_equal(tmp_path):
    """The whole reason the observer exists."""
    path = str(tmp_path / "workspace.json")
    obs = observer(str(tmp_path))
    write(path, doc("2026-01-01T00:00:00"))
    first = obs.snapshot()
    write(path, doc("2026-06-06T12:34:56"))
    assert obs.snapshot() == first


def test_a_dropped_key_at_any_depth_is_dropped(tmp_path):
    """`timestamp` is nested inside each variable, not at the top."""
    path = str(tmp_path / "workspace.json")
    obs = observer(str(tmp_path))
    write(path, doc("a"))
    first = obs.snapshot()
    write(path, doc("b"))
    assert obs.snapshot() == first


def test_a_real_second_effect_still_shows(tmp_path):
    """The direction that matters: a projection must not hide a duplicate."""
    path = str(tmp_path / "workspace.json")
    obs = observer(str(tmp_path))
    write(path, doc("a", variables={"total": {"result": 42.75, "timestamp": "a"}}))
    one = obs.snapshot()
    write(
        path,
        doc(
            "b",
            variables={
                "total": {"result": 42.75, "timestamp": "b"},
                "total_2": {"result": 42.75, "timestamp": "b"},
            },
        ),
    )
    assert obs.snapshot() != one


def test_a_changed_value_still_shows(tmp_path):
    path = str(tmp_path / "workspace.json")
    obs = observer(str(tmp_path))
    write(path, doc("a"))
    one = obs.snapshot()
    write(path, doc("a", variables={"total": {"result": 99.0, "timestamp": "a"}}))
    assert obs.snapshot() != one


def test_a_dotted_path_drops_only_that_location(tmp_path):
    """`count` under statistics is bookkeeping; `count` under a variable is data."""
    path = str(tmp_path / "workspace.json")
    obs = JsonDocObserver(
        str(tmp_path), patterns=["workspace.json"], drop_keys=["statistics.count"]
    )
    write(path, {"statistics": {"count": 1}, "variables": {"a": {"count": 1}}})
    one = obs.snapshot()
    write(path, {"statistics": {"count": 2}, "variables": {"a": {"count": 1}}})
    assert obs.snapshot() == one, "the declared path should be projected away"
    write(path, {"statistics": {"count": 2}, "variables": {"a": {"count": 2}}})
    assert obs.snapshot() != one, "the same name elsewhere must still be measured"


def test_a_list_keeps_its_length(tmp_path):
    """A duplicate that appends to a list is exactly what must not be hidden."""
    path = str(tmp_path / "workspace.json")
    obs = JsonDocObserver(str(tmp_path), patterns=["workspace.json"], drop_keys=["at"])
    write(path, {"entries": [{"value": 1, "at": "a"}]})
    one = obs.snapshot()
    write(path, {"entries": [{"value": 1, "at": "b"}, {"value": 1, "at": "c"}]})
    assert obs.snapshot() != one


def test_a_list_reordering_is_not_hidden(tmp_path):
    path = str(tmp_path / "workspace.json")
    obs = JsonDocObserver(str(tmp_path), patterns=["workspace.json"], drop_keys=[])
    write(path, {"entries": [1, 2]})
    one = obs.snapshot()
    write(path, {"entries": [2, 1]})
    assert obs.snapshot() != one


def test_the_dropped_keys_are_recorded_in_the_snapshot(tmp_path):
    """So a reader can see what was given up, and a widened projection shows."""
    path = str(tmp_path / "workspace.json")
    write(path, doc("a"))
    entry = observer(str(tmp_path)).snapshot()["workspace.json"]
    assert entry["dropped"] == sorted(DROPS)
    assert entry["mode"] == "projected"


def test_no_drops_is_recorded_as_verbatim(tmp_path):
    path = str(tmp_path / "workspace.json")
    write(path, {"a": 1})
    entry = JsonDocObserver(str(tmp_path), patterns=["workspace.json"]).snapshot()
    assert entry["workspace.json"]["mode"] == "verbatim"


def test_an_unparsable_document_is_recorded_not_skipped(tmp_path):
    """Skipping it would make a malformed file look like no file at all."""
    path = tmp_path / "workspace.json"
    path.write_text("{not json", encoding="utf-8")
    entry = observer(str(tmp_path)).snapshot()["workspace.json"]
    assert entry["mode"] == "unparsed"
    assert entry["content"]


def test_a_missing_root_is_empty_not_an_error(tmp_path):
    assert observer(str(tmp_path / "nope")).snapshot() == {}


def test_a_file_not_matching_the_patterns_is_ignored(tmp_path):
    write(str(tmp_path / "other.json"), {"a": 1})
    assert observer(str(tmp_path)).snapshot() == {}


def test_the_spec_can_build_one():
    from agentfaultlab.spec import from_document

    document = {
        "server": {
            "name": "s",
            "command": ["x"],
            "observer": {
                "kind": "json-doc",
                "path": "home/.math-mcp",
                "patterns": ["workspace.json"],
                "drop_keys": ["updated"],
            },
        },
        "fixtures": [{"tool": "save", "arguments": {}}],
    }
    target, _ = from_document(document)
    built = target.observer("/tmp/root")
    assert built.name == "json-doc"
    assert built.drop_keys == ("updated",)


def test_no_em_dashes_in_the_module():
    import agentfaultlab.observers.json_doc as module

    assert "—" not in open(module.__file__, encoding="utf-8").read()
