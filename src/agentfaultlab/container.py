"""Running an untrusted MCP server inside a container.

A server under audit is third-party code that we start, hand a working
directory to, and let run. Two servers in the pilot showed why that cannot
happen on the host:

  - `codex-supervisor-mcp` wrote a skill file into the operator's
    `~/.claude/skills/` during a `tools/list` probe that called no tool at all.
    Every later Claude session on the machine then loaded it.
  - `@contentrain/mcp` initialised itself in whatever directory it was given,
    which was the engine repository, and auto-committed four commits onto the
    checked-out branch.

Neither needed a tool call, and neither was prevented by pointing `HOME` at a
throwaway directory, because a process that wants the real home can still read
the environment, walk `/Users`, or run `git` in its own working directory.

So the boundary is a container, not an environment variable. This module turns
a server command into a `docker run` command with:

  - no host home, no `~/.claude`, no credentials, no SSH keys: the only host
    path mounted is the one trial directory the audit is about to observe,
  - a throwaway writable `HOME` on a tmpfs that dies with the container,
  - no network at run time, so a server cannot phone out mid-trial. The install
    happens once at image build time, which is the only step allowed network,
  - a read-only root filesystem, dropped capabilities, no privilege escalation,
    and memory and process caps, so a runaway server cannot take the machine
    down the way a cold `npx` install already did once.

The stdio transport survives all of this: `docker run -i` gives the same pipe
pair `subprocess.Popen` would, so `ServerSession` does not need to know whether
it is talking to a local process or a container.

What this does not protect against is a malicious image build, since the build
step installs the package from the registry with network access. That is the
same trust we already extend by auditing published packages at all, and it is
recorded in the paper's threats to validity rather than solved here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

DOCKER = "docker"

#: Where the one mounted host directory appears inside the container. Server
#: commands and env values carry absolute host paths into the trial root, so
#: every one of them is rewritten to sit under this prefix.
WORKDIR = "/work"

#: A writable throwaway home. On a tmpfs, so anything a server writes there
#: (a skill file, a config, a credential cache) dies with the container.
HOME = "/home/afl"

#: Environment variables that must never cross the boundary, whatever the
#: caller passes. Keys matching these are dropped rather than rewritten.
BLOCKED_ENV = (
    "AWS_",
    "ANTHROPIC_",
    "CLAUDE",
    "GH_",
    "GITHUB_",
    "NPM_TOKEN",
    "OPENAI_",
    "SSH_",
)


class ContainerError(RuntimeError):
    """Docker is missing, not running, or refused a build."""


def available() -> tuple[bool, str]:
    """Whether a container can be started right now, and why not if it cannot.

    Split from the error path because the caller decides what a missing daemon
    means: a hard stop for a third-party server, nothing at all for a fixture
    the engine built itself.
    """
    if shutil.which(DOCKER) is None:
        return False, "the docker CLI is not on PATH"
    probe = subprocess.run(
        [DOCKER, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        first = detail[0] if detail else "no server version reported"
        return False, f"the docker daemon is not reachable: {first}"
    return True, probe.stdout.strip()


def require() -> str:
    ok, detail = available()
    if not ok:
        raise ContainerError(
            f"{detail}. Rule 9 forbids starting a third-party server on the "
            "host, so this run stops here rather than falling back."
        )
    return detail


@dataclass(frozen=True)
class Containment:
    """How one trial's container is built and locked down.

    `mount` is the single host directory the container may see: the trial root
    the observer is about to read. Nothing else on the host is reachable.
    """

    image: str
    mount: str
    workdir: str = WORKDIR
    home: str = HOME
    memory: str = "2g"
    pids: int = 512
    network: str = "none"
    read_only: bool = True
    extra_run_args: tuple[str, ...] = field(default_factory=tuple)

    def translate(self, value: str) -> str:
        """Rewrite a host path inside `mount` to its in-container path.

        Commands reach this module already substituted, so a SQLite server's
        argument is an absolute host path like
        `/var/folders/.../afl-state-abc/app.db`. Inside the container that same
        file is at `/work/app.db`. Without this the server would create a fresh
        empty file at a path the observer never reads, and every fixture would
        score SAFE for the wrong reason.
        """
        root = os.path.realpath(self.mount)
        if value == root:
            return self.workdir
        prefix = root.rstrip(os.sep) + os.sep
        real = os.path.realpath(value) if value.startswith(os.sep) else value
        if real.startswith(prefix):
            tail = real[len(prefix) :]
            return self.workdir + "/" + tail.replace(os.sep, "/")
        if value.startswith(prefix):
            tail = value[len(prefix) :]
            return self.workdir + "/" + tail.replace(os.sep, "/")
        return value

    def run_argv(
        self,
        command: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> list[str]:
        """The full `docker run` argv that replaces a bare server command."""
        if not command:
            raise ValueError("a server command is required")
        argv = [
            DOCKER,
            "run",
            "--rm",
            "--interactive",
            f"--network={self.network}",
            f"--memory={self.memory}",
            f"--pids-limit={self.pids}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--volume={os.path.realpath(self.mount)}:{self.workdir}",
            f"--workdir={self.workdir}",
            f"--env=HOME={self.home}",
            # A tmpfs for each writable path, so the image layer stays clean
            # and nothing a server writes outlives the trial.
            f"--tmpfs={self.home}:exec",
            "--tmpfs=/tmp:exec",
        ]
        if self.read_only:
            argv.append("--read-only")
        for key, value in sorted((env or {}).items()):
            if key == "HOME" or _blocked(key):
                continue
            argv.append(f"--env={key}={self.translate(value)}")
        argv.extend(self.extra_run_args)
        argv.append(self.image)
        argv.extend(self.translate(part) for part in command)
        return argv


def _blocked(key: str) -> bool:
    upper = key.upper()
    return any(upper.startswith(prefix) or prefix in upper for prefix in BLOCKED_ENV)


# ------------------------------------------------------------------ image build

#: Node and Python in one image, because the registry serves both `npx` and
#: `uvx` servers and a single image keeps the audit's launch path identical
#: for both. Pinned to a digest-free but explicit tag: the paper reports the
#: image tag, and a floating `latest` would make a run unreproducible.
DOCKERFILE = """\
FROM node:22-bookworm-slim

RUN apt-get update \\
 && apt-get install -y --no-install-recommends git ca-certificates python3 \\
      python3-venv curl \\
 && rm -rf /var/lib/apt/lists/*

# uv provides uvx for the Python servers in the frame.
RUN curl -fsSL https://astral.sh/uv/0.5.11/install.sh | \\
      env UV_INSTALL_DIR=/usr/local/bin sh

# A real user, so nothing in the trial directory ends up owned by root and a
# server that tries to write outside its mount is refused by the kernel rather
# than by hope.
RUN useradd --create-home --home-dir /home/afl --uid 10001 afl
RUN mkdir -p /work && chown afl:afl /work

# The install step is the only one with network. Pre-installing here means the
# trial itself runs with --network=none and still starts in seconds.
ENV npm_config_update_notifier=false
ARG PACKAGES=""
RUN if [ -n "$PACKAGES" ]; then npm install -g $PACKAGES; fi

USER afl
ENV HOME=/home/afl
WORKDIR /work
"""


def build_image(
    tag: str,
    *,
    packages: Sequence[str] = (),
    context_dir: Optional[str] = None,
    quiet: bool = True,
) -> str:
    """Build the audit image, pre-installing `packages` at build time.

    Pre-installing is what lets the trial run with no network at all. A server
    installed on first use would need the registry mid-trial, and a cold `npx`
    install already cost this project a three-minute timeout per launch.
    """
    require()
    context = context_dir or os.path.join(os.path.expanduser("~"), ".agent-fault-lab", "image")
    os.makedirs(context, exist_ok=True)
    dockerfile = os.path.join(context, "Dockerfile")
    with open(dockerfile, "w", encoding="utf-8") as handle:
        handle.write(DOCKERFILE)
    argv = [DOCKER, "build", "--tag", tag, "--file", dockerfile]
    if packages:
        argv.extend(["--build-arg", "PACKAGES=" + " ".join(packages)])
    if quiet:
        argv.append("--quiet")
    argv.append(context)
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0:
        raise ContainerError(
            f"building {tag} failed: {(done.stderr or done.stdout).strip()[:500]}"
        )
    return tag
