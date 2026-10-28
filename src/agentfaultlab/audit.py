"""The conformance audit: does a real tool keep its promises under a fault?

For one tool and one fault mode the audit runs three trials, each on a fresh
backend built the same way:

* `clean_once`: call once, no fault. Snapshot S1 is what one application of
  this tool looks like.
* `clean_twice`: call twice, no fault. Snapshot S2 is what two applications
  look like. If S2 equals S1 then no end state could ever distinguish one
  application from two, so the trial cannot detect a duplicate: the tool is
  recorded as repeat safe for these arguments but is kept out of the
  duplicate-rate denominator, because an overwrite and real deduplication
  look the same from outside.
* `faulted`: inject the fault, then retry once if the caller was left without
  an answer, which is what a client does when a call times out.

The faulted end state is then compared against S0 (untouched), S1 and S2, and
the verdict follows from which it matches. Nothing in the chain reads the
tool's own report, so a tool that lies about what it did is caught by
construction.

Each trial is repeated (`repeats`, default 3) because a tool that is
sometimes safe is not safe, and because a snapshot that differs run to run
means the observer is not canonical yet and the result must not be trusted.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from .mcp import taxonomy
from .mcp.session import CallTimeout, ServerSession, SessionError, Tool, ToolCall, TraceWriter
from .observers.base import Observer, Snapshot, digest, equal

#: Argument names that commonly carry a caller supplied deduplication key.
#: Used only as a hint in the report: the authoritative test is behavioural.
DEDUP_ARGUMENT_HINTS = (
    "idempotency_key",
    "idempotencykey",
    "idempotency-key",
    "request_id",
    "requestid",
    "client_token",
    "clienttoken",
    "dedup_key",
    "dedupkey",
    "nonce",
)

MULTIPLICITY_NONE = "none"
MULTIPLICITY_ONCE = "once"
MULTIPLICITY_TWICE = "twice"
MULTIPLICITY_OTHER = "other"

#: How a tool that came out SAFE achieved it. The distinction matters: a tool
#: that is safe only because the backend refused the second call still hands
#: the agent an error it cannot interpret, because "already applied" and
#: "failed" arrive looking the same after an ambiguous timeout.
MECHANISM_ABSORBED = "absorbed_repeat"
MECHANISM_REJECTED = "backend_rejected_repeat"
MECHANISM_NONE = "none"
#: The fault left the caller believing it had succeeded, so no retry was made
#: and the question of absorbing a repeat never arose.
MECHANISM_NO_RETRY = "no_retry_needed"
MECHANISM_UNKNOWN = "unknown"

SAFE = "SAFE"
DUPLICATED = "DUPLICATED"
LOST = "LOST"
DIVERGED = "DIVERGED"
NOT_MUTATING = "NOT_MUTATING"
#: The baselines themselves were not reproducible, so no verdict is possible.
#: A server whose store is not byte comparable between two identical clean
#: runs (an append only log with timestamps in it) cannot be audited this way,
#: and saying so is better than reporting DIVERGED and calling it a finding.
UNPROVEN = "UNPROVEN"
#: F5 only. A prefix of a list argument committed, the caller was told
#: nothing, and the retry applied the whole list on top. The end state then
#: matches neither one clean application nor two, which is a distinct harm and
#: not the same thing as a state the oracle cannot account for.
PARTIAL = "PARTIAL"
BROKEN = "BROKEN"


@dataclass
class ServerTarget:
    """How to build a fresh instance of one server and read its backend.

    `command` may contain `{root}`, which is replaced with the trial's
    temporary directory. `setup` seeds that directory before the server
    starts (a git init, a schema, some files). `observer` builds the observer
    for the trial's root. Everything a trial touches must live under the
    root, or trials leak into each other and the replay oracle is void.
    """

    name: str
    command: Sequence[str]
    observer: Callable[[str], Observer]
    setup: Optional[Callable[[str], None]] = None
    env: Optional[dict[str, str]] = None
    #: Calls that have to happen before the fixture, on every trial, because
    #: some servers refuse to work until the session is bound to a project
    #: (memorix needs `memorix_session_start` with a git-backed root). They
    #: run before the first snapshot, so their effects belong to S0, and they
    #: must not call the fixture's own tool: that would consume the fault's
    #: first occurrence. A prelude call that fails aborts the trial rather
    #: than letting a refused fixture read as a lost effect.
    prelude: Sequence[tuple[str, dict[str, Any]]] = ()
    cwd_is_root: bool = True
    #: Recorded in the report so a result can be tied to a version.
    version: str = ""
    timeout_s: float = 20.0
    #: Seconds to wait before every end-state snapshot, including the clean
    #: baselines. A server that flushes its store asynchronously (mushroomdb
    #: writes an append-only log) can be snapshotted before the write lands,
    #: which made two identical clean runs produce two different S1 values:
    #: the oracle then has no stable baseline and every verdict is noise.
    #: This is separate from the audit's `settle_s`, which exists for F3's
    #: deliberately late call.
    quiesce_s: float = 0.0
    #: Whether this server runs inside a container (rule 9). The default is
    #: True so that a spec which forgets to say is contained rather than
    #: trusted: a new third-party server is the dangerous case, and omission
    #: should fail towards safety. Only first-party fixtures the engine itself
    #: ships (tests/fake_server.py) set this to False, and they say why.
    contained: bool = True
    #: Image tag used when `contained`. Recorded in the report, because a
    #: result is only reproducible against a named image.
    image: str = "agent-fault-lab:audit"

    #: Overrides the derived in-container launch command when a package's
    #: installed binary is not named after the package.
    container_command: Sequence[str] = ()

    def resolve_container_command(self, root: str) -> list[str]:
        """The argv to run inside the container, with {root} substituted."""
        from .container import installed_argv

        if self.container_command:
            return [self._fill(part, root) for part in self.container_command]
        return installed_argv(self.resolve_command(root))

    def launch_command(self, root: str) -> list[str]:
        """The command to actually launch: in-container when contained."""
        if self.contained:
            return self.resolve_container_command(root)
        return self.resolve_command(root)

    def containment(self, root: str) -> Optional[Any]:
        """The containment for one trial, or None for a first-party fixture."""
        if not self.contained:
            return None
        from .container import Containment

        return Containment(image=self.image, mount=root)

    def resolve_command(self, root: str) -> list[str]:
        return [self._fill(part, root) for part in self.command]

    def resolve_env(self, root: str) -> dict[str, str]:
        return {
            key: self._fill(value, root) for key, value in (self.env or {}).items()
        }

    @staticmethod
    def _fill(text: str, root: str) -> str:
        """`{root}` is the trial's directory; `{home}` is the real home.

        A server that keeps its state in a fixed place under the home
        directory (memorix writes `~/.memorix`) can only be isolated per
        trial by moving HOME into the trial root. That makes every launch a
        cold npx install unless the download cache stays where it is, which
        is what `{home}` is for: `npm_config_cache` can point back at the
        real cache while nothing else does.
        """
        return text.replace("{root}", root).replace(
            "{home}", os.path.expanduser("~")
        )

    def resolve_prelude(self, root: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (tool, _substitute(arguments, root)) for tool, arguments in self.prelude
        ]


@dataclass
class Fixture:
    """A concrete call to test, plus the read calls that might confirm it.

    `arguments` may contain `{root}` in any string value. `probes` are
    read-only calls used to decide whether a caller could have found out
    whether the effect landed: if no probe's output differs between the
    untouched backend and the backend after one application, the tool is
    UNCONFIRMABLE through this server.
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    probes: tuple[tuple[str, dict[str, Any]], ...] = ()
    #: Calls this fixture alone needs before its own, on top of the server's
    #: prelude. A delete has to have something to delete, and the call that
    #: creates it is usually another fixture's tool, so it cannot go in the
    #: server prelude without being faulted when that other fixture runs.
    #: Subject to the same rule as the server prelude: it must not call this
    #: fixture's own tool, and `validate_prelude` enforces that.
    prelude: tuple[tuple[str, dict[str, Any], dict[str, str]], ...] = ()
    note: str = ""
    #: Distinguishes two fixtures for the same tool. One SQL write tool can
    #: be an INSERT in one fixture and an UPDATE in another, and those behave
    #: differently under a fault, so they must never aggregate together: the
    #: unit of analysis is (server, tool, fixture, fault mode).
    id: str = ""

    @property
    def identity(self) -> str:
        return self.id or self.tool

    def __post_init__(self) -> None:
        # A prelude entry is (tool, arguments) or (tool, arguments, captures).
        # Most need nothing captured, so the short form stays legal and is
        # normalised here rather than at every use.
        self.prelude = tuple(
            (entry[0], entry[1], entry[2] if len(entry) > 2 else {})
            for entry in self.prelude
        )

    def resolve(
        self, root: str, bindings: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        return _substitute(self.arguments, root, bindings)

    def resolved_probes(
        self, root: str, bindings: Optional[dict[str, Any]] = None
    ) -> list[tuple[str, dict[str, Any]]]:
        return [
            (name, _substitute(args, root, bindings)) for name, args in self.probes
        ]

    def resolved_prelude(self, root: str) -> list[tuple[str, dict[str, Any]]]:
        return [(name, _substitute(args, root)) for name, args, _ in self.prelude]

    @property
    def prelude_captures(self) -> list[dict[str, str]]:
        return [captures for _, _, captures in self.prelude]


def _substitute(
    value: Any, root: str, bindings: Optional[dict[str, Any]] = None
) -> Any:
    if isinstance(value, str):
        text = value.replace("{root}", root)
        for name, bound in (bindings or {}).items():
            placeholder = "{%s}" % name
            if text == placeholder:
                # A whole-string placeholder keeps the captured value's type,
                # so a numeric id stays a number rather than becoming "12".
                return bound
            text = text.replace(placeholder, str(bound))
        return text
    if isinstance(value, dict):
        return {key: _substitute(item, root, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, root, bindings) for item in value]
    return value


class CaptureError(RuntimeError):
    """A prelude call did not carry the value the fixture needs."""


def extract(payload: Any, path: str) -> Any:
    """Read a dotted path out of a tool result, or raise.

    Servers answer in two places at once: `structuredContent` when they have
    an output schema, and a JSON document inside the text content when they
    do not. Both are tried, structured first, because that is the one the
    server declared.
    """
    for part in path.split("."):
        if isinstance(payload, list):
            try:
                payload = payload[int(part)]
                continue
            except (ValueError, IndexError):
                raise CaptureError(f"no element {part!r} in list at {path!r}")
        if not isinstance(payload, dict) or part not in payload:
            raise CaptureError(f"no key {part!r} at {path!r}")
        payload = payload[part]
    return payload


def capture_from(result: Optional[dict[str, Any]], path: str) -> Any:
    if result is None:
        raise CaptureError(f"no result to capture {path!r} from")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        try:
            return extract(structured, path)
        except CaptureError:
            pass
    text = _result_text(result)
    if text:
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            return extract(parsed, path)
    raise CaptureError(
        f"{path!r} is in neither structuredContent nor the reply text"
    )


@dataclass
class TrialResult:
    """One run of one condition, with what the caller saw and what landed."""

    condition: str
    multiplicity: str
    snapshot_digest: str
    caller_saw_success: bool
    caller_left_unsure: bool
    calls: list[ToolCall] = field(default_factory=list)
    error: Optional[str] = None
    snapshot: Optional[Snapshot] = None
    #: Whether a retry was made, and whether the server refused it. A refusal
    #: is how most tools end up safe without deduplicating anything.
    retry_attempted: bool = False
    retry_rejected: bool = False


@dataclass
class ToolVerdict:
    """What the audit concluded about one tool under one fault mode."""

    server: str
    tool: str
    fixture: str
    fault: str
    verdict: str
    safety_mechanism: str
    multiplicities: list[str]
    mutating: bool
    repeat_distinguishable: bool
    confirmable: Optional[bool]
    caller_left_unsure: bool
    dedup_argument: Optional[str]
    read_only_hint: Optional[bool]
    idempotent_hint: Optional[bool]
    annotation_violations: list[str] = field(default_factory=list)
    deterministic: bool = True
    notes: list[str] = field(default_factory=list)
    #: Whether the server ran inside a container, and in which image, so that
    #: a published row says how it was obtained rather than leaving it to be
    #: taken on trust. Rows collected before the containment rule carry False
    #: and are labelled as such wherever they are reported.
    contained: bool = False
    image: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "tool": self.tool,
            "fixture": self.fixture,
            "fault": self.fault,
            "verdict": self.verdict,
            "safety_mechanism": self.safety_mechanism,
            "multiplicities": ",".join(self.multiplicities),
            "mutating": self.mutating,
            "repeat_distinguishable": self.repeat_distinguishable,
            "confirmable": self.confirmable,
            "caller_left_unsure": self.caller_left_unsure,
            "dedup_argument": self.dedup_argument or "",
            "read_only_hint": self.read_only_hint,
            "idempotent_hint": self.idempotent_hint,
            "annotation_violations": ";".join(self.annotation_violations),
            "deterministic": self.deterministic,
            "notes": "; ".join(self.notes),
            "contained": self.contained,
            "image": self.image,
        }


class Audit:
    """Runs fixtures against one target under a set of fault modes."""

    def __init__(
        self,
        target: ServerTarget,
        *,
        repeats: int = 3,
        settle_s: float = 1.0,
        trace: Optional[TraceWriter] = None,
        workdir: Optional[str] = None,
        keep_roots: bool = False,
    ) -> None:
        self.target = target
        self.repeats = max(1, repeats)
        self.settle_s = settle_s
        self.trace = trace
        self.workdir = workdir
        #: Leave each trial's tree on disk instead of removing it. For
        #: diagnosis only, and never for a run that produces rows: a verdict
        #: is only meaningful if every trial started from a fresh backend, and
        #: a kept tree invites reading the state by hand and believing it.
        #: What it is for is answering "why do two identical clean runs
        #: differ", which a snapshot diff can localise to a file but not
        #: explain. The paths are collected here so a caller can find them.
        self.keep_roots = keep_roots
        self.kept_roots: list[str] = []
        #: The server's own `serverInfo`, filled in by `list_tools`. An npx or
        #: uvx command resolves to whatever is current, so the run has to
        #: record what actually answered rather than what was asked for.
        self.server_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ plumbing

    def _make_root(self) -> str:
        root = tempfile.mkdtemp(prefix="afl-", dir=self.workdir)
        if self.target.setup is not None:
            self.target.setup(root)
        return root

    def _discard(self, root: str) -> None:
        if self.keep_roots:
            self.kept_roots.append(root)
            return
        shutil.rmtree(root, ignore_errors=True)

    def _session(self, root: str, plan: Optional[Any]) -> ServerSession:
        return ServerSession(
            self.target.launch_command(root),
            plan=plan,
            cwd=root if self.target.cwd_is_root else None,
            env=self.target.resolve_env(root) or None,
            trace=self.trace,
            timeout_s=self.target.timeout_s,
            containment=self.target.containment(root),
        )

    def check_image(self) -> None:
        """Refuse a contained run whose package did not install in the image.

        Checked once per audit, before any trial, because the alternative is a
        server that starts from a shim npm left behind after a failed install
        and then fails on its first call. A fixture scored from that is worse
        than a fixture not run.
        """
        if not self.target.contained or self.target.container_command:
            # An explicit container_command names a binary rather than a
            # package, so there is nothing to look up.
            return
        from .container import package_of, require_installed

        package = package_of(self.target.command)
        if package:
            require_installed(self.target.image, package)

    def list_tools(self) -> list[Tool]:
        self.check_image()
        root = self._make_root()
        try:
            with self._session(root, None) as session:
                tools = session.list_tools()
                self.server_info = session.server_info
                if self.trace is not None:
                    self.trace.note(
                        "server_info",
                        target=self.target.name,
                        command=self.target.resolve_command("{root}"),
                        declared_version=self.target.version,
                        server_info=self.server_info,
                        tools=[tool.name for tool in tools],
                    )
                return tools
        finally:
            self._discard(root)

    # --------------------------------------------------------------------- trials

    def _trial(
        self,
        fixture: Fixture,
        *,
        condition: str,
        plan: Optional[Any],
        applications: int,
        retry_if_unsure: bool,
        settle_s: float = 0.0,
        probe: bool = False,
    ) -> tuple[TrialResult, Snapshot, Snapshot, dict[str, Any]]:
        """Run one trial on a fresh backend. Returns (result, before, after, probes)."""
        root = self._make_root()
        observer = self.target.observer(root)
        before: Snapshot = {}
        after: Snapshot = {}
        probe_outputs: dict[str, Any] = {}
        result = TrialResult(
            condition=condition,
            multiplicity=MULTIPLICITY_OTHER,
            snapshot_digest="",
            caller_saw_success=False,
            caller_left_unsure=False,
        )
        try:
            with self._session(root, plan) as session:
                bindings = self._run_prelude(session, root, fixture)
                before = observer.snapshot()
                if probe:
                    probe_outputs["before"] = self._probe(
                        session, fixture, root, bindings
                    )
                arguments = fixture.resolve(root, bindings)
                for attempt in range(1, applications + 1):
                    call = session.call_tool(fixture.tool, arguments, attempt=attempt)
                    result.calls.append(call)
                if retry_if_unsure and result.calls and not result.calls[-1].caller_saw_success:
                    retry = session.call_tool(
                        fixture.tool, arguments, attempt=len(result.calls) + 1
                    )
                    result.calls.append(retry)
                    result.retry_attempted = True
                    # An in-band `isError` result counts as a refusal too:
                    # MCP servers report tool failures inside a success frame.
                    result.retry_rejected = retry.error is not None or (
                        retry.result is not None and bool(retry.result.get("isError"))
                    )
                if settle_s:
                    # F3's held call has to land before the snapshot is taken,
                    # or a late write is scored as no write at all.
                    time.sleep(settle_s)
                if self.target.quiesce_s:
                    time.sleep(self.target.quiesce_s)
                if probe:
                    probe_outputs["after"] = self._probe(
                        session, fixture, root, bindings
                    )
                after = observer.snapshot()
        except (SessionError, CallTimeout, OSError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            after = observer.snapshot()
        finally:
            self._discard(root)

        last = result.calls[-1] if result.calls else None
        result.caller_saw_success = bool(last and last.caller_saw_success)
        result.caller_left_unsure = any(call.timed_out for call in result.calls)
        result.snapshot = after
        result.snapshot_digest = digest(after)
        return result, before, after, probe_outputs

    def _run_prelude(
        self,
        session: ServerSession,
        root: str,
        fixture: Optional[Fixture] = None,
    ) -> dict[str, Any]:
        """Run the preludes and return whatever they were asked to capture.

        A fixture prelude may capture a value out of its own reply, which the
        fixture's arguments then refer to by name. That is the only way to
        audit a prepare-then-commit tool: the token the commit needs is minted
        by the prepare call and cannot be written into a static spec.

        The capture is per trial, deliberately. Every replay runs the prelude
        again and binds a fresh token, so the baselines stay independent; what
        is held constant within a trial is that both applications of the
        fixture use the one token, which is the thing under test.
        """
        calls: list[tuple[str, dict[str, Any], dict[str, str]]] = [
            (tool, arguments, {})
            for tool, arguments in self.target.resolve_prelude(root)
        ]
        if fixture is not None:
            calls += [
                (tool, arguments, captures)
                for (tool, arguments), captures in zip(
                    fixture.resolved_prelude(root), fixture.prelude_captures
                )
            ]
        bindings: dict[str, Any] = {}
        for tool, arguments, captures in calls:
            call = session.call_tool(tool, arguments)
            if not call.caller_saw_success:
                raise SessionError(
                    f"prelude call {tool} failed on {self.target.name}: "
                    f"{_result_text(call.result) or call.error or 'timed out'}"
                )
            for name, path in (captures or {}).items():
                try:
                    bindings[name] = capture_from(call.result, path)
                except CaptureError as exc:
                    # Never fall through to an unsubstituted "{name}": the
                    # fixture would call the tool with a literal placeholder
                    # and the row would read as a server refusal.
                    raise SessionError(
                        f"prelude call {tool} on {self.target.name} could not "
                        f"capture {name!r}: {exc}"
                    ) from exc
        return bindings

    def _probe(
        self,
        session: ServerSession,
        fixture: Fixture,
        root: str,
        bindings: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        outputs = []
        for name, arguments in fixture.resolved_probes(root, bindings):
            call = session.call_tool(name, arguments)
            outputs.append(
                {
                    "tool": name,
                    "text": _result_text(call.result),
                    "errored": call.error is not None or call.timed_out,
                }
            )
        return outputs

    # ------------------------------------------------------------------- baselines

    def baselines(self, fixture: Fixture) -> dict[str, Any]:
        """S0, S1 and S2 for one fixture, plus what the probes revealed."""
        once, before_once, after_once, probes = self._trial(
            fixture, condition="clean_once", plan=None, applications=1,
            retry_if_unsure=False, probe=bool(fixture.probes),
        )
        twice, _, after_twice, _ = self._trial(
            fixture, condition="clean_twice", plan=None, applications=2,
            retry_if_unsure=False,
        )
        # One application, again, on another fresh backend. If this does not
        # match S1 then the observer is not canonical for this server and
        # every comparison against S1 or S2 is noise.
        _, _, after_once_again, _ = self._trial(
            fixture, condition="clean_once_again", plan=None, applications=1,
            retry_if_unsure=False,
        )
        mutating = not equal(before_once, after_once)
        confirmable: Optional[bool] = None
        if fixture.probes and probes.get("before") is not None:
            confirmable = probes.get("before") != probes.get("after")
        return {
            "s0": before_once,
            "s1": after_once,
            "s2": after_twice,
            "mutating": mutating,
            "s1_again": after_once_again,
            "baseline_stable": equal(after_once, after_once_again),
            "repeat_distinguishable": not equal(after_once, after_twice),
            "confirmable": confirmable,
            "clean_once": once,
            "clean_twice": twice,
            "probes": probes,
        }

    # -------------------------------------------------------------------- verdicts

    def run_tool(
        self,
        fixture: Fixture,
        tool: Optional[Tool] = None,
        modes: Iterable[str] = taxonomy.AUDIT_SEQUENCE,
    ) -> list[ToolVerdict]:
        """Audit one fixture under each fault mode."""
        from .mcp.injector import FaultPlan  # local import keeps the module light

        base = self.baselines(fixture)
        dedup = _dedup_argument(tool, fixture)
        verdicts: list[ToolVerdict] = []

        if not base["baseline_stable"]:
            # Nothing is run: without a stable S1 there is nothing to compare
            # against, and running the faults anyway would produce rows that
            # read like findings.
            return [
                ToolVerdict(
                    contained=self.target.contained,
                    image=self.target.image if self.target.contained else "",
                    server=self.target.name,
                    tool=fixture.tool,
                    fixture=fixture.identity,
                    fault=taxonomy.get(mode_name).id,
                    verdict=UNPROVEN,
                    safety_mechanism=MECHANISM_UNKNOWN,
                    multiplicities=[],
                    mutating=base["mutating"],
                    repeat_distinguishable=base["repeat_distinguishable"],
                    confirmable=base["confirmable"],
                    caller_left_unsure=False,
                    dedup_argument=dedup,
                    read_only_hint=tool.read_only_hint if tool else None,
                    idempotent_hint=tool.idempotent_hint if tool else None,
                    annotation_violations=[],
                    notes=[
                        "two identical clean runs left different end states, so "
                        "this server's state is not observable canonically from "
                        "outside and no verdict is formed"
                    ]
                    + ([fixture.note] if fixture.note else []),
                )
                for mode_name in modes
            ]

        for mode_name in modes:
            mode = taxonomy.get(mode_name)
            multiplicities: list[str] = []
            unsure = False
            notes: list[str] = []
            rejections = 0
            retries = 0
            for _ in range(self.repeats):
                plan = FaultPlan.parse(f"{mode.id}:{fixture.tool}@1")
                settle = self.settle_s + plan.delay_s if mode.id == "F3" else 0.0
                trial, _, after, _ = self._trial(
                    fixture,
                    condition=f"fault-{mode.id}",
                    plan=plan,
                    applications=1,
                    retry_if_unsure=True,
                    settle_s=settle,
                )
                if trial.error:
                    notes.append(trial.error)
                retries += int(trial.retry_attempted)
                rejections += int(trial.retry_rejected)
                multiplicities.append(
                    classify(after, base["s0"], base["s1"], base["s2"])
                )
                unsure = unsure or trial.caller_left_unsure

            if not base["repeat_distinguishable"]:
                # S1 == S2: two applications leave the same state as one, so
                # no duplicate could be detected here even if one landed. The
                # tool is repeat safe for these arguments, but the reason
                # (an overwrite, or real deduplication) is not separable from
                # state, so it is said rather than assumed.
                notes.append(
                    "repeat not distinguishable from state (overwrite or "
                    "deduplication); excluded from duplicate-rate denominators"
                )
            verdict = self._verdict(mode, multiplicities, base, notes)
            mechanism = _mechanism(
                verdict, retries, rejections, base["repeat_distinguishable"]
            )
            if mechanism == MECHANISM_UNKNOWN and verdict == SAFE and retries:
                notes.append(
                    "the retry answered in a success frame and one "
                    "application cannot be told from two, so whether the "
                    "repeat was absorbed or refused is not decidable from "
                    "state (a server can report a refusal in its reply text "
                    "without setting isError)"
                )
            if mechanism == MECHANISM_REJECTED:
                notes.append(
                    "safe only because the backend refused the repeat, so the "
                    "agent receives an error it cannot tell from a real failure"
                )
            violations = _annotation_violations(tool, base, multiplicities, mode)
            if (
                tool is not None
                and tool.idempotent_hint is False
                and base["mutating"]
                and not base["repeat_distinguishable"]
            ):
                # The opposite of a violation, and worth recording: the tool
                # understates itself, so a client that refuses to replay it is
                # paying for a risk that is not there.
                notes.append(
                    "idempotentHint:false but repeating the call left the same "
                    "state (the hint is pessimistic)"
                )
            verdicts.append(
                ToolVerdict(
                    contained=self.target.contained,
                    image=self.target.image if self.target.contained else "",
                    server=self.target.name,
                    tool=fixture.tool,
                    fixture=fixture.identity,
                    fault=mode.id,
                    verdict=verdict,
                    safety_mechanism=mechanism,
                    multiplicities=multiplicities,
                    mutating=base["mutating"],
                    repeat_distinguishable=base["repeat_distinguishable"],
                    confirmable=base["confirmable"],
                    caller_left_unsure=unsure,
                    dedup_argument=dedup,
                    read_only_hint=tool.read_only_hint if tool else None,
                    idempotent_hint=tool.idempotent_hint if tool else None,
                    annotation_violations=violations,
                    deterministic=len(set(multiplicities)) == 1,
                    notes=notes,
                )
            )
        return verdicts

    def _verdict(
        self,
        mode: taxonomy.FaultMode,
        multiplicities: list[str],
        base: dict[str, Any],
        notes: list[str],
    ) -> str:
        if not base["mutating"]:
            return NOT_MUTATING
        if any("SessionError" in note for note in notes):
            return BROKEN
        expected = (
            MULTIPLICITY_ONCE
            if mode.effect_expected_after_retry == "once"
            else MULTIPLICITY_NONE
        )
        observed = set(multiplicities)
        if observed == {expected}:
            return SAFE
        if MULTIPLICITY_TWICE in observed:
            return DUPLICATED
        if observed == {MULTIPLICITY_NONE} and expected == MULTIPLICITY_ONCE:
            return LOST
        if mode.id == "F5":
            # Expected by construction for an accumulating tool: the prefix
            # landed, the retry applied the whole list, and the result is
            # more than one application and less than two. Calling that
            # DIVERGED would file a predictable outcome as an anomaly.
            notes.append(
                "a prefix of the list landed before the retry applied the "
                "whole list, so the end state is neither one application nor "
                "two: partial over-application"
            )
            return PARTIAL
        return DIVERGED


def _mechanism(
    verdict: str,
    retries: int,
    rejections: int,
    repeat_distinguishable: bool = True,
) -> str:
    """Why a SAFE verdict came out safe.

    `absorbed_repeat` means the repeat was accepted and left no trace: an
    overwrite, or real deduplication. `backend_rejected_repeat` means the
    second call was refused, which keeps the state correct but passes the
    ambiguity on to the agent. Tools that duplicated get `none`, and a tool
    that was never retried (the fault left the caller believing it
    succeeded) gets `no_retry_needed`.

    One caveat: under F3 the retry runs before the held call lands, so the
    label describes what the retry met, not what the late duplicate met. The
    verdict is unaffected, since that still comes from the end state.

    When one application cannot be told from two, state cannot say whether a
    repeat was absorbed or refused, so an accepted-looking retry is
    `unknown` rather than `absorbed_repeat`. An explicit error still counts as
    a rejection, because that is evidence from outside the state. This came
    from desktop-commander's `edit_block`, which answers a repeat with "exact
    match not found" inside a success frame: the backend did refuse, the end
    state is right, and nothing in the protocol says so.
    """
    if verdict in (DUPLICATED, DIVERGED, LOST):
        return MECHANISM_NONE
    if verdict != SAFE:
        return MECHANISM_UNKNOWN
    if retries == 0:
        return MECHANISM_NO_RETRY
    if rejections == retries:
        return MECHANISM_REJECTED
    if rejections == 0:
        return MECHANISM_ABSORBED if repeat_distinguishable else MECHANISM_UNKNOWN
    return MECHANISM_UNKNOWN


def classify(final: Snapshot, s0: Snapshot, s1: Snapshot, s2: Snapshot) -> str:
    """Which baseline the faulted end state matches."""
    if equal(final, s1):
        return MULTIPLICITY_ONCE
    if equal(final, s2):
        return MULTIPLICITY_TWICE
    if equal(final, s0):
        return MULTIPLICITY_NONE
    return MULTIPLICITY_OTHER


def _result_text(result: Optional[dict[str, Any]]) -> str:
    if not result:
        return ""
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(parts)


def _dedup_argument(tool: Optional[Tool], fixture: Fixture) -> Optional[str]:
    names: list[str] = list(fixture.arguments)
    if tool is not None:
        properties = tool.input_schema.get("properties")
        if isinstance(properties, dict):
            names.extend(properties)
    for name in names:
        if name.lower().replace("-", "_") in DEDUP_ARGUMENT_HINTS:
            return name
    return None


def _annotation_violations(
    tool: Optional[Tool],
    base: dict[str, Any],
    multiplicities: list[str],
    mode: taxonomy.FaultMode,
) -> list[str]:
    """Hints that the measured behaviour contradicts.

    These are the H3 findings. A `readOnlyHint` on a tool that changes the
    backend, or an `idempotentHint` on a tool that applies twice, is a false
    statement that clients act on: they auto-approve read-only tools and
    replay or parallelise idempotent ones.

    `destructiveHint` is deliberately not checked. The specification defines
    it as additive versus destructive updates, and an additive write that
    lands twice is still additive, so a duplicate is not evidence against it.
    Reporting one would be a false finding.
    """
    if tool is None:
        return []
    violations = []
    if tool.read_only_hint is True and base["mutating"]:
        violations.append("readOnlyHint:true but the backend changed")
    if tool.idempotent_hint is True and MULTIPLICITY_TWICE in multiplicities:
        violations.append(f"idempotentHint:true but {mode.id} applied the effect twice")
    return violations
