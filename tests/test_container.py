"""The containment boundary, tested without needing a running daemon.

Everything here is argv construction and path translation, which is where the
mistakes that matter live: a mistranslated path makes a server write somewhere
the observer never reads, and every fixture then scores SAFE for the wrong
reason. The one test that needs Docker is skipped when it is not running.
"""

import os

import pytest

from agentfaultlab.container import (
    HOME,
    WORKDIR,
    Containment,
    ContainerError,
    available,
    require,
)


def make(tmp_path, **kwargs) -> Containment:
    return Containment(image="afl:test", mount=str(tmp_path), **kwargs)


# ------------------------------------------------------------------ the mount

def test_only_the_trial_directory_is_mounted(tmp_path):
    argv = make(tmp_path).run_argv(["server"])
    mounts = [a for a in argv if a.startswith("--volume=")]
    assert mounts == [f"--volume={os.path.realpath(tmp_path)}:{WORKDIR}"]


def test_the_host_home_is_never_mounted(tmp_path):
    argv = make(tmp_path).run_argv(["server"])
    joined = " ".join(argv)
    home = os.path.expanduser("~")
    assert f"--volume={home}" not in joined
    assert ".claude" not in joined
    assert ".ssh" not in joined


def test_home_points_at_a_throwaway_tmpfs(tmp_path):
    """The skill-file incident: a server wrote into the real ~/.claude.

    A tmpfs home means a server can still write whatever it likes and none of
    it outlives the trial.
    """
    argv = make(tmp_path).run_argv(["server"])
    assert f"--env=HOME={HOME}" in argv
    assert f"--tmpfs={HOME}:exec" in argv


def test_the_run_has_no_network(tmp_path):
    argv = make(tmp_path).run_argv(["server"])
    assert "--network=none" in argv


def test_the_container_is_locked_down(tmp_path):
    argv = make(tmp_path).run_argv(["server"])
    for flag in (
        "--rm",
        "--interactive",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    ):
        assert flag in argv
    assert any(a.startswith("--memory=") for a in argv)
    assert any(a.startswith("--pids-limit=") for a in argv)


def test_stdio_stays_usable(tmp_path):
    """No --tty: a pty would corrupt newline-delimited JSON-RPC framing."""
    argv = make(tmp_path).run_argv(["server"])
    assert "--interactive" in argv
    assert "--tty" not in argv
    assert "-t" not in argv


# ------------------------------------------------------- translating host paths

def test_a_path_inside_the_trial_root_is_rewritten(tmp_path):
    db = tmp_path / "app.db"
    db.write_text("")
    argv = make(tmp_path).run_argv(["npx", "mcp-server-sqlite-npx", str(db)])
    assert argv[-1] == f"{WORKDIR}/app.db"


def test_a_nested_path_keeps_its_tail(tmp_path):
    nested = tmp_path / "home" / ".memorix" / "data"
    nested.mkdir(parents=True)
    argv = make(tmp_path).run_argv(["server", str(nested / "memorix.db")])
    assert argv[-1] == f"{WORKDIR}/home/.memorix/data/memorix.db"


def test_the_root_itself_becomes_the_workdir(tmp_path):
    argv = make(tmp_path).run_argv(["server", str(tmp_path)])
    assert argv[-1] == WORKDIR


def test_a_path_outside_the_root_is_left_alone(tmp_path):
    """Not silently rewritten: an outside path means the spec is wrong.

    Translating it would hide the mistake and hand the server a path inside the
    mount that happens to exist. Leaving it means the container refuses, which
    is a loud failure.
    """
    argv = make(tmp_path).run_argv(["server", "--flag", "plain-value"])
    assert "--flag" in argv and "plain-value" in argv


def test_env_values_are_translated_too(tmp_path):
    db = tmp_path / "state.db"
    argv = make(tmp_path).run_argv(["server"], env={"STATE_PATH": str(db)})
    assert f"--env=STATE_PATH={WORKDIR}/state.db" in argv


# -------------------------------------------------------------- env that leaks

@pytest.mark.parametrize(
    "key",
    [
        "AWS_SECRET_ACCESS_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CONFIG_DIR",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "OPENAI_API_KEY",
        "SSH_AUTH_SOCK",
        "NPM_TOKEN",
    ],
)
def test_credentials_never_cross_the_boundary(tmp_path, key):
    argv = make(tmp_path).run_argv(["server"], env={key: "secret"})
    assert "secret" not in " ".join(argv)


def test_a_caller_cannot_override_home(tmp_path):
    argv = make(tmp_path).run_argv(["server"], env={"HOME": os.path.expanduser("~")})
    homes = [a for a in argv if a.startswith("--env=HOME=")]
    assert homes == [f"--env=HOME={HOME}"]


def test_an_ordinary_variable_still_passes(tmp_path):
    argv = make(tmp_path).run_argv(["server"], env={"MCP_LOG_LEVEL": "debug"})
    assert "--env=MCP_LOG_LEVEL=debug" in argv


# ------------------------------------------------------------------- the image

def test_the_image_comes_before_the_command(tmp_path):
    argv = make(tmp_path).run_argv(["npx", "-y", "thing"])
    assert argv[argv.index("afl:test") + 1 :] == ["npx", "-y", "thing"]


def test_an_empty_command_is_refused(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path).run_argv([])


# ------------------------------------------------------------------- the daemon

def test_available_reports_a_reason_when_it_is_not(monkeypatch):
    monkeypatch.setattr("agentfaultlab.container.shutil.which", lambda _: None)
    ok, detail = available()
    assert ok is False
    assert "docker" in detail.lower()


def test_require_refuses_to_fall_back_to_the_host(monkeypatch):
    """The point of rule 9: no daemon means stop, not run it anyway."""
    monkeypatch.setattr("agentfaultlab.container.shutil.which", lambda _: None)
    with pytest.raises(ContainerError) as caught:
        require()
    assert "host" in str(caught.value)


def test_no_em_dashes_in_the_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src",
        "agentfaultlab",
        "container.py",
    )
    with open(path, encoding="utf-8") as handle:
        assert "—" not in handle.read()


# ------------------------------------------------------- the spec cannot opt out

def test_a_spec_cannot_turn_containment_off(tmp_path):
    """Rule 9 is not a default the audited thing gets to override."""
    import json

    from agentfaultlab.spec import SpecError, from_document

    document = {
        "server": {
            "name": "x",
            "command": ["npx", "-y", "thing"],
            "contained": False,
            "observer": {"kind": "filesystem"},
        },
        "fixtures": [],
    }
    with pytest.raises(SpecError) as caught:
        from_document(document)
    assert "rule 9" in str(caught.value).lower()
    del json


def test_a_spec_is_contained_when_it_says_nothing(tmp_path):
    import json

    from agentfaultlab.spec import from_document

    target, _ = from_document(
        {
            "server": {
                "name": "x",
                "command": ["npx", "-y", "thing"],
                "observer": {"kind": "filesystem"},
            },
            "fixtures": [
                {"id": "f", "tool": "write_file", "arguments": {"path": "x"}}
            ],
        }
    )
    assert target.contained is True
    del json
