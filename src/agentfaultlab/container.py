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
        real = os.path.realpath(value) if value.startswith(os.sep) else value
        # The trial root itself, by either spelling. On macOS the temp root is
        # a symlink (/var/folders/... is /private/var/folders/...), so the
        # unresolved spelling never equals the resolved mount and the
        # filesystem server was handed a host path it could not see: it exited
        # with "none of the specified directories are accessible".
        if value == root or real == root:
            return self.workdir
        prefix = root.rstrip(os.sep) + os.sep
        if real.startswith(prefix):
            tail = real[len(prefix) :]
            return self.workdir + "/" + tail.replace(os.sep, "/")
        if value.startswith(prefix):
            tail = value[len(prefix) :]
            return self.workdir + "/" + tail.replace(os.sep, "/")
        return value

    def container_home(self, env: Optional[Mapping[str, str]] = None) -> str:
        """Where HOME points inside the container.

        Normally the throwaway tmpfs at `/home/afl`. But some servers keep
        their whole state under HOME: `memorix` writes
        `$HOME/.memorix/data/memorix.db`, and that database is the thing the
        observer reads. On the host those audits set `HOME` to a directory
        inside the trial root, which is both throwaway and observable.

        If a caller asks for exactly that, a HOME inside the mounted trial
        root, it is honoured, translated to its in-container path. The point
        of rule 9 is that the server cannot reach the operator's home, not
        that HOME has to be a tmpfs, and a HOME on the tmpfs here would be
        worse than useless: the server would initialise its store where
        nothing can read it, the observer would find no database, and every
        fixture would come back UNPROVEN or, worse, SAFE for the wrong
        reason.

        Any other HOME is dropped and replaced by the tmpfs, because a path
        outside the mount either does not exist in the container or is an
        attempt to reach the host's own home.
        """
        asked = (env or {}).get("HOME")
        if not asked:
            return self.home
        inside = self.translate(asked)
        if inside == self.workdir or inside.startswith(self.workdir + "/"):
            return inside
        return self.home

    def run_argv(
        self,
        command: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> list[str]:
        """The full `docker run` argv that replaces a bare server command."""
        if not command:
            raise ValueError("a server command is required")
        home = self.container_home(env)
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
            f"--env=HOME={home}",
            "--tmpfs=/tmp:exec",
        ]
        # A tmpfs for the throwaway home, so the image layer stays clean and
        # nothing a server writes outlives the trial. Skipped when HOME is
        # inside the mounted trial root, where a tmpfs would hide the one
        # directory the observer is about to read.
        if not (home == self.workdir or home.startswith(self.workdir + "/")):
            argv.append(f"--tmpfs={home}:exec")
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


#: Launchers that fetch a package before running it, and the flags of theirs
#: that carry no meaning once the package is installed.
RESOLVERS = {
    "npx": ("-y", "--yes"),
    "npx.cmd": ("-y", "--yes"),
    "uvx": ("--quiet", "-q"),
}


def installed_argv(command: Sequence[str]) -> list[str]:
    """Map a registry launch command to the binary installed in the image.

    A spec's command is how the registry says to start the server, which is
    almost always `npx -y <package> [args]` or `uvx <package> [args]`. Inside
    the container that is wrong twice over: the package is already installed,
    and both launchers re-resolve it against their index, which fails
    outright under `--network=none` (`EAI_AGAIN` for npx, a resolution error
    for uv). So the launcher prefix is dropped and the package name is mapped
    to its installed binary.

    The mapping is the last path segment of the package name, which is right
    for every server in the frame so far (`@cyanheads/git-mcp-server` becomes
    `git-mcp-server`). It is a convention, not a guarantee: npm lets a package
    name any binary it likes, so a spec can override this with
    `container_command` and must when the convention does not hold. `uvx
    --from <dist> <binary>` names both halves explicitly and is refused here
    rather than guessed at, for the same reason.
    """
    argv = list(command)
    if not argv:
        return argv
    launcher = os.path.basename(argv[0])
    if launcher not in RESOLVERS:
        return argv
    rest = [part for part in argv[1:] if part not in RESOLVERS[launcher]]
    if not rest:
        raise ValueError(f"no package in {launcher} command {argv!r}")
    if rest[0] == "--from":
        raise ValueError(
            f"{launcher} --from names a distribution and a binary separately, "
            f"so it cannot be derived: give the spec a container_command. "
            f"Command was {argv!r}"
        )
    package, *arguments = rest
    binary = package.rsplit("/", 1)[-1]
    return [binary, *arguments]


def _blocked(key: str) -> bool:
    upper = key.upper()
    return any(upper.startswith(prefix) or prefix in upper for prefix in BLOCKED_ENV)


#: Cached per image, because answering it costs a container start and the
#: answer is baked into the image at build time.
_RECORDED: dict[str, tuple[frozenset, frozenset]] = {}


def recorded_packages(image: str) -> tuple[frozenset, frozenset]:
    """What the image records as installed and as failed to install.

    The build writes both lists because one unbuildable server should not cost
    the image every other server. The cost of that choice is that a failed
    install can leave a binary on PATH anyway: npm wrote a
    `desktop-commander` shim into the image even though its install failed on
    fsevents. Starting it would give a server that answers `initialize` and
    then fails on its first call, and a fixture scored from that is worse than
    no fixture.

    An empty pair means the image does not carry the lists, which is not an
    error: an image built by hand is the caller's business.
    """
    if image in _RECORDED:
        return _RECORDED[image]
    lists = []
    for name in ("installed", "failed"):
        done = subprocess.run(
            [DOCKER, "run", "--rm", "--network=none", image, "cat", f"/opt/afl/{name}"],
            capture_output=True,
            text=True,
        )
        lists.append(
            frozenset(line for line in done.stdout.split() if line)
            if done.returncode == 0
            else frozenset()
        )
    answer = (lists[0], lists[1])
    _RECORDED[image] = answer
    return answer


def require_installed(image: str, package: str) -> None:
    """Refuse to start a server whose package did not install in the image."""
    if not package:
        return
    installed, failed = recorded_packages(image)
    if not installed and not failed:
        return
    if package in failed:
        raise ContainerError(
            f"{package} failed to install in {image}, so it cannot be audited "
            "under rule 9. A failed install can still leave a binary on PATH, "
            "which would answer initialize and then fail on its first call."
        )
    if package not in installed:
        raise ContainerError(
            f"{package} is not installed in {image}. Rebuild the image with it "
            "in PACKAGES (or PY_PACKAGES): the run has no network, so a "
            "launcher cannot fetch it mid-trial."
        )


def package_of(command: Sequence[str]) -> str:
    """The package a launcher command would fetch, or "" if it is not one."""
    argv = list(command)
    if not argv:
        return ""
    launcher = os.path.basename(argv[0])
    if launcher not in RESOLVERS:
        return ""
    rest = [part for part in argv[1:] if part not in RESOLVERS[launcher]]
    if not rest or rest[0] == "--from":
        return ""
    return rest[0]


# ------------------------------------------------------------------ image build

#: The image recipe lives in a file rather than a string in this module.
#: An embedded copy has to be escaped for Python, and a backslash before a
#: newline is a line continuation inside a triple-quoted string, so every
#: shell line continuation in the recipe would be silently eaten and the
#: image built from something subtly different from what the repository
#: shows. One file, read at build time, cannot drift from itself.
DOCKERFILE_PATH = os.path.join(os.path.dirname(__file__), "audit.Dockerfile")


def dockerfile() -> str:
    with open(DOCKERFILE_PATH, encoding="utf-8") as handle:
        return handle.read()


def build_image(
    tag: str,
    *,
    packages: Sequence[str] = (),
    py_packages: Sequence[str] = (),
    context_dir: Optional[str] = None,
    quiet: bool = True,
) -> str:
    """Build the audit image, pre-installing `packages` at build time.

    Pre-installing is what lets the trial run with no network at all. A server
    installed on first use would need the registry mid-trial, and a cold `npx`
    install already cost this project a three-minute timeout per launch.
    """
    require()
    context = context_dir or os.path.join(
        os.path.expanduser("~"), ".agent-fault-lab", "image"
    )
    os.makedirs(context, exist_ok=True)
    recipe = os.path.join(context, "Dockerfile")
    with open(recipe, "w", encoding="utf-8") as handle:
        handle.write(dockerfile())
    argv = [DOCKER, "build", "--tag", tag, "--file", recipe]
    if packages:
        argv.extend(["--build-arg", "PACKAGES=" + " ".join(packages)])
    if py_packages:
        argv.extend(["--build-arg", "PY_PACKAGES=" + " ".join(py_packages)])
    if quiet:
        argv.append("--quiet")
    argv.append(context)
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0:
        raise ContainerError(
            f"building {tag} failed: {(done.stderr or done.stdout).strip()[:500]}"
        )
    return tag
