"""Tests for the sampling-frame script.

The frames decide which servers the study looks at, so a quiet bug here does
not crash anything: it changes the population and every rate derived from it.
Four of the mistakes caught so far were in this file (an alphabetical
"top by downloads", a truncated snapshot used for a random draw, an empty
reference frame, and one package ranked twice under two registry names), which
is why it is unit-tested apart from the network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "sample_frame.py",
)


def _load():
    spec = importlib.util.spec_from_file_location("sample_frame", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sample_frame = _load()


def _entry(name, *, packages=None, remotes=None, latest=True, status="active",
           published="2026-01-01", version="1.0.0"):
    return {
        "server": {
            "name": name,
            "version": version,
            "description": f"{name} description",
            "packages": packages or [],
            "remotes": remotes or [],
        },
        "_meta": {
            "io.modelcontextprotocol.registry/official": {
                "isLatest": latest,
                "status": status,
                "publishedAt": published,
            }
        },
    }


def _npm(identifier, *, transport="stdio", env=None):
    package = {
        "registryType": "npm",
        "identifier": identifier,
        "version": "1.0.0",
        "transport": {"type": transport},
    }
    if env:
        package["environmentVariables"] = [{"name": name} for name in env]
    return package


# ------------------------------------------------------------------ latest

def test_latest_only_keeps_one_record_per_server():
    entries = [
        _entry("a/one", latest=False, version="1.0.0", published="2026-01-01"),
        _entry("a/one", latest=True, version="2.0.0", published="2026-02-01"),
        _entry("a/two", latest=True),
    ]
    kept = sample_frame.latest_only(entries)
    assert len(kept) == 2
    versions = {e["server"]["name"]: e["server"]["version"] for e in kept}
    assert versions["a/one"] == "2.0.0"


def test_latest_only_falls_back_to_publish_time():
    entries = [
        _entry("a/one", latest=False, version="1.0.0", published="2026-01-01"),
        _entry("a/one", latest=False, version="3.0.0", published="2026-03-01"),
    ]
    kept = sample_frame.latest_only(entries)
    assert [e["server"]["version"] for e in kept] == ["3.0.0"]


# ---------------------------------------------------------------- classify

def test_classify_marks_a_plain_npm_stdio_server_runnable():
    row = sample_frame.classify(_entry("a/one", packages=[_npm("some-mcp")]))
    assert row["runnable"] is True
    assert (row["runner"], row["package"]) == ("npx", "some-mcp")


@pytest.mark.parametrize(
    "entry,fragment",
    [
        (_entry("a/remote", remotes=[{"type": "sse"}]), "remote only"),
        (_entry("a/none"), "no package"),
        (_entry("a/dead", packages=[_npm("x")], status="deprecated"), "deprecated"),
        (_entry("a/http", packages=[_npm("x", transport="streamable-http")]),
         "not stdio"),
        (_entry("a/secret", packages=[_npm("x", env=["SOME_API_KEY"])]),
         "declares credentials"),
        (_entry("a/other", packages=[{"registryType": "oci", "identifier": "x"}]),
         "no npm or pypi stdio package"),
    ],
)
def test_classify_excludes_with_a_stated_reason(entry, fragment):
    row = sample_frame.classify(entry)
    assert row["runnable"] is False
    assert fragment in row["reason"]


# ------------------------------------------------------------ alias collapse

def _row(name, package, downloads, runner="npx"):
    return {
        "name": name,
        "package": package,
        "runner": runner,
        "weekly_downloads": downloads,
        "runnable": True,
    }


def test_collapse_aliases_keeps_one_row_per_package():
    rows = [
        _row("io.github.callstack/agent-device", "agent-device", 191116),
        _row("io.github.callstackincubator/agent-device", "agent-device", 191116),
        _row("io.github.microsoft/playwright-mcp", "@playwright/mcp", 3465430),
    ]
    collapsed, count = sample_frame.collapse_aliases(rows)
    assert count == 1
    assert len(collapsed) == 2
    aliases = {row["package"]: row["registry_aliases"] for row in collapsed}
    assert aliases["agent-device"] == (
        "io.github.callstack/agent-device; "
        "io.github.callstackincubator/agent-device"
    )


def test_collapse_aliases_separates_the_same_name_on_two_runners():
    rows = [_row("a/one", "thing", 10), _row("b/one", "thing", 10, runner="uvx")]
    collapsed, count = sample_frame.collapse_aliases(rows)
    assert count == 0
    assert len(collapsed) == 2


def test_collapse_aliases_does_not_mutate_its_input():
    rows = [_row("a/one", "thing", 10)]
    sample_frame.collapse_aliases(rows)
    assert "registry_aliases" not in rows[0]


# --------------------------------------------------------------- download cache

def test_download_cache_round_trip(tmp_path):
    path = tmp_path / "frame-all.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"__snapshot__": True, "fetched_at": "x"}),
                json.dumps({"runner": "npx", "package": "a", "weekly_downloads": 5}),
                json.dumps({"runner": "npx", "package": "b", "weekly_downloads": None}),
                json.dumps({"runner": "uvx", "package": "c"}),
                json.dumps({"__trailer__": True, "complete": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cache = sample_frame.load_download_cache(str(path))
    assert cache == {("npx", "a"): 5}

    rows = [
        {"runner": "npx", "package": "a"},
        {"runner": "npx", "package": "zzz"},
    ]
    stats = sample_frame.apply_download_cache(rows, cache, str(path))
    assert rows[0]["weekly_downloads"] == 5
    assert "weekly_downloads" not in rows[1]
    assert (stats["queried"], stats["answered"]) == (0, 1)
    assert stats["cache_source"] == str(path)


# ------------------------------------------------------------------- frames

def _snapshot(tmp_path, entries, *, complete=True):
    path = tmp_path / "registry.jsonl"
    lines = [json.dumps({"__snapshot__": True, "fetched_at": "20260101T000000Z",
                         "source": "test"})]
    lines += [json.dumps(entry) for entry in entries]
    lines.append(json.dumps({
        "__trailer__": True,
        "complete": complete,
        "truncated": not complete,
        "entries_seen": len(entries),
        "entries_kept": len(entries),
        "pages": 1,
    }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_frames_refuse_a_truncated_snapshot(tmp_path):
    snapshot = _snapshot(
        tmp_path, [_entry("a/one", packages=[_npm("one")])], complete=False
    )
    with pytest.raises(SystemExit, match="incomplete snapshot"):
        sample_frame.build_frames(
            snapshot, str(tmp_path / "out"), seed=1, draw=2, downloads=0
        )


def test_frames_write_the_three_frames_and_a_summary(tmp_path):
    entries = [
        _entry("a/one", packages=[_npm("one")]),
        _entry("a/two", packages=[_npm("two")]),
        _entry("a/two-again", packages=[_npm("two")]),
        _entry("a/remote", remotes=[{"type": "sse"}]),
    ]
    snapshot = _snapshot(tmp_path, entries)
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps({"runner": "npx", "package": package,
                        "weekly_downloads": downloads})
            for package, downloads in (("one", 10), ("two", 400))
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    sample_frame.build_frames(
        snapshot, str(out), seed=7, draw=3, downloads=0,
        downloads_from=str(cache),
    )

    for name in ("frame1-reference.csv", "frame2-popular.csv",
                 "frame3-random.csv", "frames-summary.md", "frame-all.jsonl"):
        assert (out / name).exists(), name

    popular = (out / "frame2-popular.csv").read_text(encoding="utf-8").splitlines()
    assert "registry_aliases" in popular[0]
    # The duplicated package ranks once, and above the less downloaded one.
    assert len(popular) == 3
    assert popular[1].startswith("a/two,two,") and ",400," in popular[1]

    summary = (out / "frames-summary.md").read_text(encoding="utf-8")
    assert "collapsed into another listing of the same package: 1" in summary
    assert "cached from cache.jsonl" in summary
    # Frame 1 is hardcoded, because the registry does not list the
    # reference servers; an empty Frame 1 was a real bug.
    reference = (out / "frame1-reference.csv").read_text(encoding="utf-8")
    assert "@modelcontextprotocol/server-filesystem" in reference


def test_no_em_dashes_in_the_script():
    assert "—" not in open(SCRIPT, encoding="utf-8").read()


# ------------------------------------------------------------ launch arguments

def test_classify_keeps_the_registry_launch_arguments():
    package = _npm("mushroomdb")
    package["runtimeHint"] = "npx"
    package["packageArguments"] = [
        {"value": "mcp", "type": "positional", "valueHint": "subcommand"},
        {"type": "positional", "valueHint": "db_dir", "isRequired": True,
         "placeholder": "/path/to/db"},
    ]
    row = sample_frame.classify(_entry("a/mushroom", packages=[package]))
    assert row["runtime_hint"] == "npx"
    assert row["launch_argv"] == ["mcp", "{db_dir}"]
    assert row["launch_needs_input"] == ["db_dir"]


def test_classify_omits_optional_arguments_with_no_value():
    package = _npm("thing")
    package["packageArguments"] = [
        {"value": "serve", "type": "positional"},
        {"type": "named", "name": "--verbose"},
    ]
    row = sample_frame.classify(_entry("a/thing", packages=[package]))
    assert row["launch_argv"] == ["serve"]
    assert row["launch_needs_input"] == []


def test_classify_passes_a_named_argument_with_its_flag():
    package = _npm("thing")
    package["packageArguments"] = [
        {"type": "named", "name": "--store", "value": "/tmp/store"},
    ]
    row = sample_frame.classify(_entry("a/thing", packages=[package]))
    assert row["launch_argv"] == ["--store", "/tmp/store"]


def test_launch_arguments_reach_the_csv_as_text(tmp_path):
    package = _npm("thing")
    package["packageArguments"] = [{"value": "serve", "type": "positional"}]
    snapshot = _snapshot(tmp_path, [_entry("a/thing", packages=[package])])
    out = tmp_path / "out"
    sample_frame.build_frames(
        snapshot, str(out), seed=1, draw=1, downloads=0
    )
    body = (out / "frame3-random.csv").read_text(encoding="utf-8")
    assert "launch_argv" in body.splitlines()[0]
    assert ",serve," in body


def test_a_definitive_http_answer_is_not_retried(monkeypatch):
    """A 404 is a fact, not a transient failure.

    The frames derivation spent over four hours on 12,000 npm lookups, most of
    it in backoff for packages the MCP registry lists and npm has never heard
    of. Three attempts with 1.5s and 3s of sleep between them is 4.5 seconds
    to learn the same thing three times.
    """
    import urllib.error

    module = _load()
    attempts = []

    def refuse(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {}, None
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(module.time, "sleep", lambda *a: None)
    assert module._get("https://example.invalid/thing") is None
    assert len(attempts) == 1


def test_a_transient_http_answer_is_still_retried(monkeypatch):
    import urllib.error

    module = _load()
    attempts = []

    def refuse(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", {}, None
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(module.time, "sleep", lambda *a: None)
    assert module._get("https://example.invalid/thing") is None
    assert len(attempts) == 3
