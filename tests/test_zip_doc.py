"""Zip-container documents, compared without their containers' noise.

A zip stores a modification time per entry, so two identical writes a second
apart are different files and every row reads UNPROVEN for a reason that has
nothing to do with the tool. These tests pin the contract: entry names and
projected entry contents are the state, the container's framing is not, and a
difference the spec did not declare away still shows.
"""

import json
import os
import time
import zipfile

from agentfaultlab.observers import ZipDocObserver, equal


def write_zip(path, entries, *, date_time=(2026, 1, 1, 0, 0, 0)):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            info = zipfile.ZipInfo(name, date_time=date_time)
            body = payload if isinstance(payload, str) else json.dumps(payload)
            archive.writestr(info, body)


def test_the_containers_own_timestamps_are_not_state(tmp_path):
    """The defect this observer exists for: same content, different mtimes."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    body = {"content.json": {"root": "x"}}
    write_zip(a / "doc.xmind", body, date_time=(2026, 1, 1, 0, 0, 0))
    write_zip(b / "doc.xmind", body, date_time=(2026, 11, 3, 17, 28, 16))
    assert equal(
        ZipDocObserver(str(a)).snapshot(), ZipDocObserver(str(b)).snapshot()
    )


def test_a_declared_key_inside_the_archive_is_dropped(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_zip(a / "doc.xmind", {"metadata.json": {"createdTime": "1", "n": 2}})
    write_zip(b / "doc.xmind", {"metadata.json": {"createdTime": "2", "n": 2}})
    plain = ZipDocObserver(str(a)).snapshot(), ZipDocObserver(str(b)).snapshot()
    assert not equal(*plain)
    projected = (
        ZipDocObserver(str(a), drop_keys=("createdTime",)).snapshot(),
        ZipDocObserver(str(b), drop_keys=("createdTime",)).snapshot(),
    )
    assert equal(*projected)


def test_a_change_the_spec_did_not_declare_still_shows(tmp_path):
    """The safeguard: a projection must not launder away a real difference."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_zip(a / "doc.xmind", {"metadata.json": {"createdTime": "1", "n": 2}})
    write_zip(b / "doc.xmind", {"metadata.json": {"createdTime": "1", "n": 3}})
    assert not equal(
        ZipDocObserver(str(a), drop_keys=("createdTime",)).snapshot(),
        ZipDocObserver(str(b), drop_keys=("createdTime",)).snapshot(),
    )


def test_a_gained_entry_is_a_change(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_zip(a / "doc.xmind", {"content.json": {"root": "x"}})
    write_zip(b / "doc.xmind", {"content.json": {"root": "x"}, "extra.json": {}})
    assert not equal(
        ZipDocObserver(str(a)).snapshot(), ZipDocObserver(str(b)).snapshot()
    )


def test_non_json_entries_compare_by_their_uncompressed_bytes(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_zip(a / "doc.docx", {"word/document.xml": "<w:p>one</w:p>"})
    write_zip(b / "doc.docx", {"word/document.xml": "<w:p>one</w:p>"},
              date_time=(2026, 7, 4, 12, 0, 0))
    assert equal(
        ZipDocObserver(str(a)).snapshot(), ZipDocObserver(str(b)).snapshot()
    )
    write_zip(b / "doc.docx", {"word/document.xml": "<w:p>two</w:p>"})
    assert not equal(
        ZipDocObserver(str(a)).snapshot(), ZipDocObserver(str(b)).snapshot()
    )


def test_a_file_that_is_not_a_zip_is_compared_not_skipped(tmp_path):
    """A truncated archive is an effect; skipping it would hide one."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "doc.xmind").write_bytes(b"not a zip at all")
    snapshot = ZipDocObserver(str(a)).snapshot()
    assert snapshot["doc.xmind"]["mode"] == "not-a-zip"
    (a / "doc.xmind").write_bytes(b"not a zip either")
    assert not equal(snapshot, ZipDocObserver(str(a)).snapshot())


def test_the_snapshot_records_what_was_dropped(tmp_path):
    """A spec cannot widen the projection without the report changing."""
    a = tmp_path / "a"
    a.mkdir()
    write_zip(a / "doc.xmind", {"metadata.json": {"createdTime": "1"}})
    entry = ZipDocObserver(str(a), drop_keys=("createdTime", "modifiedTime")).snapshot()
    assert entry["doc.xmind"]["dropped"] == ["createdTime", "modifiedTime"]
    assert entry["doc.xmind"]["mode"] == "projected"
    plain = ZipDocObserver(str(a)).snapshot()
    assert plain["doc.xmind"]["mode"] == "verbatim"
    assert plain["doc.xmind"]["dropped"] == []


def test_only_the_declared_patterns_are_looked_at(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    write_zip(a / "doc.xmind", {"content.json": {"root": "x"}})
    write_zip(a / "other.zip", {"content.json": {"root": "x"}})
    snapshot = ZipDocObserver(str(a)).snapshot()
    assert list(snapshot) == ["doc.xmind"]


def test_a_spec_can_select_the_zip_doc_observer(tmp_path):
    from agentfaultlab import spec as spec_module

    path = tmp_path / "spec.json"
    path.write_text(json.dumps({
        "server": {
            "name": "t",
            "command": ["true"],
            "observer": {
                "kind": "zip-doc",
                "patterns": ["*.xmind"],
                "drop_keys": ["createdTime", "modifiedTime"],
            },
        },
        "fixtures": [{"tool": "write", "id": "f", "arguments": {}}],
    }))
    target, _fixtures = spec_module.load(str(path))
    observer = target.observer(str(tmp_path))
    assert observer.name == "zip-doc"
    assert sorted(observer.drop_keys) == ["createdTime", "modifiedTime"]
