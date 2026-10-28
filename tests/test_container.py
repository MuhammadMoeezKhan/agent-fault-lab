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
    build_image,
    dockerfile,
    installed_argv,
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


def test_a_home_inside_the_trial_root_is_honoured(tmp_path):
    """memorix keeps its database under HOME, and that database is the oracle.

    `$HOME/.memorix/data/memorix.db` is what the observer reads. With HOME on
    the tmpfs the server would initialise its store where nothing can read it:
    the observer would find no database and every fixture would come back
    UNPROVEN, or SAFE for the wrong reason, which is the quiet failure this
    whole module exists to prevent.
    """
    root = tmp_path / "trial"
    (root / "home").mkdir(parents=True)
    contain = Containment(image="img", mount=str(root))
    argv = contain.run_argv(["server"], env={"HOME": str(root / "home")})
    assert f"--env=HOME={WORKDIR}/home" in argv
    # No tmpfs over it: that would hide the directory being observed.
    assert not [a for a in argv if a.startswith("--tmpfs=/work")]
    assert "--tmpfs=/tmp:exec" in argv


def test_a_home_outside_the_trial_root_still_falls_back_to_the_tmpfs(tmp_path):
    contain = make(tmp_path)
    for asked in ("/etc", "/Users/someone-else", os.path.expanduser("~/.claude")):
        argv = contain.run_argv(["server"], env={"HOME": asked})
        assert [a for a in argv if a.startswith("--env=HOME=")] == [
            f"--env=HOME={HOME}"
        ]
        assert f"--tmpfs={HOME}:exec" in argv


def test_the_honoured_home_is_still_not_the_host_home(tmp_path):
    """Honouring HOME must not become a way back to the operator's home.

    The test is deliberately literal: the argv may not contain the host home
    path at all, however the caller spelled it.
    """
    root = tmp_path / "trial"
    (root / "home").mkdir(parents=True)
    argv = Containment(image="img", mount=str(root)).run_argv(
        ["server"], env={"HOME": str(root / "home")}
    )
    joined = " ".join(argv)
    assert os.path.expanduser("~") not in joined


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


def test_an_episode_row_records_whether_it_was_contained():
    """Rule 9 compliance has to be a property of the results.

    The trace shows the inner server command whether or not a container was
    used, so without this the only evidence is the daemon's event log, which
    is gone by the time anyone reads the numbers.
    """
    from agentfaultlab.consequence import EpisodeResult

    row = EpisodeResult(
        task="t", agent="a", condition="clean", arm="baseline",
        multiplicity="once", claim="success", task_success=True,
        duplicate_effect=False, false_success=False, false_failure=False,
        honest_unconfirmed=False, contained=True, image="afl:audit",
    ).as_row()
    assert row["contained"] is True
    assert row["image"] == "afl:audit"


# ------------------------------------------- the in-container launch command

def test_npx_is_dropped_inside_the_container():
    """npx re-resolves against the registry, which no-network forbids.

    The failure without this is `EAI_AGAIN getaddrinfo`, which looks like a
    broken server rather than a launch command that should never have been
    used inside the image.
    """
    from agentfaultlab.container import installed_argv

    assert installed_argv(["npx", "-y", "memorix", "serve"]) == ["memorix", "serve"]
    assert installed_argv(
        ["npx", "-y", "mcp-server-sqlite-npx", "/work/app.db"]
    ) == ["mcp-server-sqlite-npx", "/work/app.db"]


def test_a_scoped_package_maps_to_its_last_segment():
    from agentfaultlab.container import installed_argv

    assert installed_argv(["npx", "-y", "@cyanheads/git-mcp-server"]) == [
        "git-mcp-server"
    ]


def test_a_command_that_needs_no_resolving_is_left_alone():
    from agentfaultlab.container import installed_argv

    # A command that is already a binary, not a package to be resolved.
    assert installed_argv(["node", "/work/server.js"]) == ["node", "/work/server.js"]
    assert installed_argv(["memorix", "serve"]) == ["memorix", "serve"]


def test_a_spec_can_override_the_derived_binary():
    """npm lets a package name any binary, so the convention needs an escape.

    @cyanheads/git-mcp-server is the live case: its published bin takes no
    arguments, while the registry's own launch line passes npm script
    arguments that only mean something to npx.
    """
    from agentfaultlab.spec import from_document

    target, _ = from_document(
        {
            "server": {
                "name": "x",
                "command": ["npx", "-y", "@cyanheads/git-mcp-server", "run", "start:stdio"],
                "container_command": ["git-mcp-server"],
                "observer": {"kind": "filesystem"},
            },
            "fixtures": [
                {"id": "f", "tool": "t", "arguments": {}}
            ],
        }
    )
    assert target.launch_command("/r") == ["git-mcp-server"]
    assert target.resolve_command("/r")[0] == "npx"


def test_the_image_recipe_is_a_file_the_module_reads():
    """One copy, so a rebuilt image cannot differ from what the repo shows.

    The embedded string this replaced was corrupted by Python's own rules: a
    backslash before a newline is a line continuation inside a triple-quoted
    string, so every shell line continuation in the recipe was eaten.
    """
    from agentfaultlab.container import dockerfile

    recipe = dockerfile()
    assert recipe.startswith("FROM ")
    assert "--network" not in recipe, "the build installs; the run has no network"
    assert "USER afl" in recipe
    assert recipe.count("npm install -g") == 1


def test_the_probe_script_refuses_to_run_on_the_host(capsys):
    """A probe is how both startup incidents happened, with no tool called.

    So the script that does it must not have a host mode at all, rather than
    a host mode that is merely not the default.
    """
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts", "probe_server.py"
    )
    spec = importlib.util.spec_from_file_location("probe_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import sys

    argv = sys.argv
    try:
        sys.argv = ["probe_server.py", "/nonexistent-frame.jsonl", "some/server"]
        assert module.main() == 2
    finally:
        sys.argv = argv
    assert "rule 9" in capsys.readouterr().err


def test_uvx_is_dropped_like_npx():
    """`uvx mcp-server-git` would try the index on every launch.

    uv resolves the package each time it runs, which fails with no network
    just as npx does. The reference frame's only Python server would
    otherwise be unauditable under rule 9.
    """
    assert installed_argv(
        ["uvx", "mcp-server-git", "--repository", "/work/repo"]
    ) == ["mcp-server-git", "--repository", "/work/repo"]


def test_a_uvx_quiet_flag_is_not_mistaken_for_the_package():
    assert installed_argv(["uvx", "--quiet", "mcp-server-git"]) == ["mcp-server-git"]


def test_uvx_from_is_refused_rather_than_guessed():
    with pytest.raises(ValueError) as caught:
        installed_argv(["uvx", "--from", "mcp-git", "git-server"])
    assert "container_command" in str(caught.value)


def test_the_image_recipe_installs_python_packages_too():
    recipe = dockerfile()
    assert "ARG PY_PACKAGES" in recipe
    assert "uv tool install" in recipe
    # The bin directory has to be on PATH, because HOME at run time is a
    # tmpfs that knows nothing about what was installed at build time.
    assert "UV_TOOL_BIN_DIR=/usr/local/bin" in recipe


def test_a_python_package_reaches_the_build_as_its_own_argument(monkeypatch, tmp_path):
    import subprocess as sub

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return sub.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("agentfaultlab.container.require", lambda: "fake")
    monkeypatch.setattr(sub, "run", fake_run)
    build_image(
        "tag",
        packages=["memorix"],
        py_packages=["mcp-server-git"],
        context_dir=str(tmp_path),
    )
    argv = seen["argv"]
    assert "PACKAGES=memorix" in argv
    assert "PY_PACKAGES=mcp-server-git" in argv


def test_tool_call_arguments_are_translated_into_the_container(tmp_path):
    """The other half of the path problem, found by auditing memorix contained.

    The launch command was translated from the start; arguments were not.
    memorix's `memorix_session_start` is given `<trial root>/project`, which
    does not exist inside the container, and the server refused it: every
    fixture came back BROKEN with the control failing. A server that created
    the missing directory instead would have scored every fixture SAFE for
    the wrong reason, with nothing in any log.
    """
    from agentfaultlab.mcp.session import ServerSession

    root = tmp_path / "trial"
    (root / "project").mkdir(parents=True)
    session = ServerSession(
        ["server"],
        cwd=str(root),
        containment=Containment(image="img", mount=str(root)),
    )
    translated = session._translate_arguments(
        {
            "path": str(root / "project"),
            "query": "insert into charges values ('INV-1', 100)",
            "nested": {"paths": [str(root / "repo" / "a.txt"), "not/a/path"]},
            "count": 3,
        }
    )
    assert translated["path"] == f"{WORKDIR}/project"
    assert translated["query"] == "insert into charges values ('INV-1', 100)"
    assert translated["nested"]["paths"] == [f"{WORKDIR}/repo/a.txt", "not/a/path"]
    assert translated["count"] == 3


def test_arguments_are_untouched_without_containment(tmp_path):
    from agentfaultlab.mcp.session import ServerSession

    session = ServerSession(["server"], cwd=str(tmp_path))
    arguments = {"path": str(tmp_path / "x"), "n": 1}
    assert session._translate_arguments(dict(arguments)) == arguments


def test_the_root_is_the_workdir_by_either_spelling(tmp_path):
    """macOS temp paths are symlinks, and one spelling was slipping through.

    `/var/folders/...` resolves to `/private/var/folders/...`. The equality
    check compared the unresolved value to the resolved mount, so a command
    argument that was exactly the trial root stayed a host path:
    `@modelcontextprotocol/server-filesystem` was given a directory that does
    not exist in the container and exited with "none of the specified
    directories are accessible".
    """
    contain = Containment(image="img", mount=str(tmp_path))
    assert contain.translate(str(tmp_path)) == WORKDIR
    assert contain.translate(os.path.realpath(str(tmp_path))) == WORKDIR
    # And with a mount given by its resolved spelling instead.
    other = Containment(image="img", mount=os.path.realpath(str(tmp_path)))
    assert other.translate(str(tmp_path)) == WORKDIR


# ------------------------------------------------- what the image says it has


def test_a_package_whose_install_failed_is_refused(monkeypatch):
    """A failed install can still leave a binary on PATH.

    npm wrote a `desktop-commander` shim into the audit image even though the
    install failed on fsevents. Starting it gives a server that answers
    initialize and fails on its first call, which scores a fixture from a
    server that was never built.
    """
    import agentfaultlab.container as module

    monkeypatch.setattr(
        module, "recorded_packages", lambda image: (frozenset({"memorix"}), frozenset({"bad-pkg"}))
    )
    with pytest.raises(ContainerError) as caught:
        module.require_installed("img", "bad-pkg")
    assert "rule 9" in str(caught.value)
    module.require_installed("img", "memorix")


def test_a_package_the_image_never_installed_is_refused(monkeypatch):
    import agentfaultlab.container as module

    monkeypatch.setattr(
        module, "recorded_packages", lambda image: (frozenset({"memorix"}), frozenset())
    )
    with pytest.raises(ContainerError) as caught:
        module.require_installed("img", "never-heard-of-it")
    assert "no network" in str(caught.value)


def test_an_image_without_the_lists_is_left_alone(monkeypatch):
    import agentfaultlab.container as module

    monkeypatch.setattr(
        module, "recorded_packages", lambda image: (frozenset(), frozenset())
    )
    module.require_installed("hand-built", "anything")


def test_the_package_a_launcher_would_fetch():
    from agentfaultlab.container import package_of

    assert package_of(["npx", "-y", "memorix", "serve"]) == "memorix"
    assert package_of(["uvx", "mcp-server-git", "--repository", "/work"]) == "mcp-server-git"
    assert package_of(["npx", "-y", "@cyanheads/git-mcp-server"]) == "@cyanheads/git-mcp-server"
    assert package_of(["memorix", "serve"]) == ""
    assert package_of([]) == ""


# ------------------------------------------------- the working directory

def test_the_workdir_is_the_mount_point_unless_asked_otherwise(tmp_path):
    argv = make(tmp_path).run_argv(["server"])
    assert f"--workdir={WORKDIR}" in argv


def test_a_cwd_inside_the_mount_is_honoured(tmp_path):
    """Fixture discovery needs HOME and the working directory in one diff.

    The trial root holds both, so the mount is the root and the server starts
    in a subdirectory of it. A server that keeps its state under HOME and one
    that keeps it beside the working directory are then both visible.
    """
    work = tmp_path / "work"
    work.mkdir()
    argv = make(tmp_path, cwd=str(work)).run_argv(["server"])
    assert f"--workdir={WORKDIR}/work" in argv
    assert f"--workdir={WORKDIR}" not in argv


def test_a_cwd_outside_the_mount_is_refused_not_ignored(tmp_path):
    """Loudly, because the quiet version produces a wrong verdict.

    A server started in the wrong directory writes where no observer reads,
    the end state looks unchanged, and the fixture scores SAFE. That is the
    one failure this module exists to prevent, so an impossible request stops
    the run instead of falling back to the mount point.
    """
    with pytest.raises(ValueError):
        make(tmp_path, cwd="/etc").run_argv(["server"])


def test_a_cwd_still_cannot_reach_the_host_home(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, cwd=os.path.expanduser("~")).run_argv(["server"])


def test_home_and_cwd_can_be_siblings_in_the_same_root(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    argv = make(tmp_path, cwd=str(work)).run_argv(
        ["server"], env={"HOME": str(home)}
    )
    assert f"--env=HOME={WORKDIR}/home" in argv
    assert f"--workdir={WORKDIR}/work" in argv
    # No tmpfs over the home, or the diff would never see what the server
    # wrote there.
    assert not [a for a in argv if a.startswith(f"--tmpfs={WORKDIR}")]


def test_a_host_temp_directory_is_not_forwarded_into_the_container():
    """TMPDIR names a path that exists on the host and not in the container.

    Forwarding it makes a server fail on a path the harness invented, and that
    failure would then be recorded as a property of the server: one candidate
    was nearly parked as broken when the real cause was a socket directory it
    could not create under the host's TMPDIR. A containment artifact counted
    as a server defect inflates both the park list and the funnel.
    """
    containment = Containment(image="img", mount="/trial")
    argv = containment.run_argv(
        ["server"],
        env={
            "TMPDIR": "/var/folders/ab/cd/T/",
            "TMP": "/var/tmp",
            "TEMP": "/var/tmp",
            "KEEP_ME": "yes",
        },
    )
    joined = " ".join(argv)
    assert "TMPDIR" not in joined
    assert "--env=TMP=" not in joined
    assert "--env=TEMP=" not in joined
    # The container still gets a writable temp directory of its own.
    assert "--tmpfs=/tmp:exec" in argv
    assert "--env=KEEP_ME=yes" in argv
