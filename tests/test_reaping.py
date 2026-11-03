"""An episode must not leave a proxy or a container behind.

On 2026-10-30 a hosted episode timed out, the harness killed the client, and
the fault proxy the client had spawned was never signalled. The proxy and its
container were still alive four hours later, holding a trial directory open
while later episodes ran. `--rm` did not help, because it fires when a
container exits, and the container-side `timeout` is a half-hour backstop
rather than an episode boundary.

Two mechanisms are tested here:

  - the client runs in its own process group, so a timeout signals the proxy
    and the container as well as the client,
  - every container carries the episode's label, so whatever is still running
    when the episode ends can be removed by id.
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from agentfaultlab import container
from agentfaultlab.container import RUN_LABEL, Containment


# ----------------------------------------------------------------- the label

def test_a_container_carries_the_run_label_when_one_is_given(tmp_path):
    argv = Containment(
        image="afl:test", mount=str(tmp_path), run_label="ep-1"
    ).run_argv(["server"])
    assert f"--label={RUN_LABEL}=ep-1" in argv


def test_no_label_is_attached_when_none_is_given(tmp_path):
    argv = Containment(image="afl:test", mount=str(tmp_path)).run_argv(["server"])
    assert not [a for a in argv if a.startswith("--label=")]


def test_the_label_comes_before_the_image(tmp_path):
    """A flag after the image name is an argument to the server, not to docker."""
    argv = Containment(
        image="afl:test", mount=str(tmp_path), run_label="ep-1"
    ).run_argv(["server"])
    assert argv.index(f"--label={RUN_LABEL}=ep-1") < argv.index("afl:test")


# ------------------------------------------------------------------ reaping

def test_reap_asks_for_containers_under_exactly_this_label(monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stdout = "abc123\ndef456\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Done()

    monkeypatch.setattr(container.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(container.subprocess, "run", fake_run)
    removed = container.reap("ep-1")
    assert removed == ["abc123", "def456"]
    assert f"--filter=label={RUN_LABEL}=ep-1" in calls[0]
    # Stopped containers count too: one that exited while holding the trial
    # directory still has to be removed before the directory goes.
    assert "-aq" in calls[0]
    assert calls[1][:4] == ["docker", "rm", "--force", "--volumes"]
    assert calls[1][4:] == ["abc123", "def456"]


def test_reap_is_cheap_and_quiet_when_nothing_leaked(monkeypatch):
    class Empty:
        returncode = 0
        stdout = "\n"
        stderr = ""

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Empty()

    monkeypatch.setattr(container.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(container.subprocess, "run", fake_run)
    assert container.reap("ep-1") == []
    # One `docker ps`, and no removal command.
    assert len(calls) == 1


def test_reap_never_raises_when_docker_refuses(monkeypatch):
    """This runs in an episode's finally: it must not replace the result."""
    class Listed:
        returncode = 0
        stdout = "abc123\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        if argv[1] == "ps":
            return Listed()
        raise OSError("daemon went away")

    monkeypatch.setattr(container.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(container.subprocess, "run", fake_run)
    # The ids are still reported, so a leak that could not be removed is
    # visible rather than swallowed.
    assert container.reap("ep-1") == ["abc123"]


def test_reap_does_nothing_without_a_docker_cli(monkeypatch):
    monkeypatch.setattr(container.shutil, "which", lambda name: None)
    assert container.reap("ep-1") == []


# ------------------------------------------------- the episode's own cleanup

def _target(contained: bool):
    from agentfaultlab.audit import ServerTarget
    from agentfaultlab.observers import FilesystemObserver

    return ServerTarget(
        name="fake",
        command=[sys.executable, "-c", "pass"],
        contained=contained,
        image="afl:test" if contained else "",
        observer=lambda root: FilesystemObserver(root),
        timeout_s=5.0,
    )


def _runner(contained: bool):
    """A runner whose oracle is already filled, so no server has to start.

    The baselines are what `classify` compares against, and this file is about
    cleanup rather than classification, so three empty states are enough and
    they keep the test off Docker.
    """
    from agentfaultlab.consequence import ConsequenceRunner

    runner = ConsequenceRunner(_target(contained=contained))
    runner._oracles[_task().id] = {"s0": {}, "s1": {}, "s2": {}}
    return runner


def _task():
    from agentfaultlab.consequence import AgentTask

    return AgentTask(
        id="append-once",
        instruction="append one line",
        reference=[],
        target_tool="append_line",
    )


class _Broken:
    """An agent that fails the way a timed-out episode fails."""

    name = "stub"
    server_name = "target"

    def run(self, instruction, mcp_config_path, cwd):
        return {"text": "", "turns": None, "error": "agent timed out"}


def test_an_episode_reaps_its_own_label_even_when_the_agent_timed_out(monkeypatch):
    from agentfaultlab.consequence import ConsequenceRunner

    seen = []
    monkeypatch.setattr(container, "reap", lambda label: (seen.append(label), [])[1])
    runner = _runner(contained=True)
    result = runner.run_episode(_task(), _Broken())
    assert result.error == "agent timed out"
    assert len(seen) == 1 and seen[0]
    assert runner.leaked_containers == []


def test_a_leak_is_recorded_rather_than_cleaned_up_quietly(monkeypatch):
    from agentfaultlab.consequence import ConsequenceRunner

    monkeypatch.setattr(container, "reap", lambda label: ["abc123"])
    runner = _runner(contained=True)
    runner.run_episode(_task(), _Broken())
    assert runner.leaked_containers == ["abc123"]


def test_each_episode_gets_its_own_label(monkeypatch):
    from agentfaultlab.consequence import ConsequenceRunner

    seen = []
    monkeypatch.setattr(container, "reap", lambda label: (seen.append(label), [])[1])
    runner = _runner(contained=True)
    runner.run_episode(_task(), _Broken())
    runner.run_episode(_task(), _Broken())
    assert len(set(seen)) == 2


def test_an_uncontained_fixture_does_not_pay_for_a_docker_call(monkeypatch):
    from agentfaultlab.consequence import ConsequenceRunner

    def refuse(label):
        raise AssertionError("no container was started, so none can be reaped")

    monkeypatch.setattr(container, "reap", refuse)
    runner = _runner(contained=False)
    runner.run_episode(_task(), _Broken())


def test_the_mcp_config_tells_the_proxy_which_label_to_use(tmp_path):
    runner = _runner(contained=True)

    class Stub:
        name = "stub"
        server_name = "target"

    config_path = tmp_path / "mcp.json"
    runner._write_mcp_config(
        str(config_path), str(tmp_path), None, None, Stub(), run_label="ep-7"
    )
    args = json.loads(config_path.read_text())["mcpServers"]["target"]["args"]
    assert args[args.index("--run-label") + 1] == "ep-7"
    # Before the bare separator, so it is a proxy flag and not an argument the
    # server would receive.
    assert args.index("--run-label") < args.index("--")


# ------------------------------------------------------- the process group

def test_a_timeout_kills_the_whole_group_not_just_the_client(tmp_path):
    """The real defect: a grandchild outliving the child the harness killed.

    The script writes its own child's pid, then sleeps. On a timeout the
    harness used to kill only the script, and the grandchild kept running,
    which is what the proxy and its container did for four hours.
    """
    from agentfaultlab.consequence import _run_whole_group

    marker = tmp_path / "grandchild.pid"
    script = (
        f"sleep 120 & echo $! > {marker}; " "sleep 120"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_whole_group(
            ["/bin/sh", "-c", script],
            timeout_s=1.5,
            cwd=str(tmp_path),
            env=dict(os.environ),
        )
    pid = int(marker.read_text().strip())
    deadline = time.monotonic() + 5
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            alive = False
            break
        time.sleep(0.1)
    assert not alive, f"the grandchild {pid} outlived the timeout"


def test_a_normal_run_still_returns_its_output(tmp_path):
    from agentfaultlab.consequence import _run_whole_group

    done = _run_whole_group(
        ["/bin/sh", "-c", "echo hello; echo bad >&2"],
        timeout_s=10,
        cwd=str(tmp_path),
        env=dict(os.environ),
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "hello"
    assert done.stderr.strip() == "bad"


def test_the_client_is_not_in_the_harness_own_group(tmp_path):
    """Otherwise killing the group would kill the harness that sent the signal."""
    from agentfaultlab.consequence import _run_whole_group

    done = _run_whole_group(
        ["/bin/sh", "-c", "ps -o pgid= -p $$"],
        timeout_s=10,
        cwd=str(tmp_path),
        env=dict(os.environ),
    )
    assert int(done.stdout.strip()) != os.getpgid(0)


def test_signalling_a_group_that_has_already_gone_is_not_an_error():
    from agentfaultlab.consequence import _signal_group

    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    _signal_group(proc, signal.SIGKILL)
