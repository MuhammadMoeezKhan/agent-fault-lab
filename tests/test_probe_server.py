"""Tests for the probe script's install step.

The probe decides which candidates are auditable at all, and its build step
writes one of the funnel's categories. An earlier version handed every
candidate to `npm install -g`, so three PyPI packages were reported as
"cannot be installed in the audit image" without ever having been tried as
Python packages. A wrong funnel row is a wrong result, so the split between
the two runners is tested apart from Docker.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "probe_server.py",
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)


def _load():
    spec = importlib.util.spec_from_file_location("probe_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stub_build(monkeypatch, seen):
    from agentfaultlab import container

    monkeypatch.setattr(container, "require", lambda: None)

    def fake_build(tag, *, packages=(), py_packages=(), context_dir=None, quiet=True):
        seen.append({"tag": tag, "packages": list(packages),
                     "py_packages": list(py_packages)})
        return tag

    monkeypatch.setattr(container, "build_image", fake_build)


def test_each_runner_installs_through_its_own_tool(monkeypatch):
    module = _load()
    seen: list[dict] = []
    _stub_build(monkeypatch, seen)
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *a, **k: type("R", (), {"stdout": "", "returncode": 0})(),
    )
    module.build_probe_image(
        "tag",
        [
            {"package": "geo-tool-check", "runner": "npx"},
            {"package": "bima-mcp", "runner": "uvx"},
            {"package": "odoo-linter", "runner": "uvx"},
        ],
    )
    assert seen == [
        {"tag": "tag", "packages": ["geo-tool-check"],
         "py_packages": ["bima-mcp", "odoo-linter"]}
    ]


def test_an_unknown_runner_stops_the_build_rather_than_guessing(monkeypatch):
    module = _load()
    seen: list[dict] = []
    _stub_build(monkeypatch, seen)
    with pytest.raises(SystemExit) as caught:
        module.build_probe_image(
            "tag", [{"package": "thing", "runner": "bun"}]
        )
    assert "bun" in str(caught.value)
    assert seen == []
