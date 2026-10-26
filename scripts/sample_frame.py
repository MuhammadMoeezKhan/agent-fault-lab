#!/usr/bin/env python3
"""Build the audit's sampling frames from the official MCP registry.

The study reports three frames separately (reference, popular, random), and a
random draw is only defensible if someone else can reproduce it. So this runs
in two steps that stay separate on disk:

1. **snapshot**: page through the registry and write every entry to a
   timestamped JSONL file, untouched. The registry changes daily; the
   snapshot is the thing the paper cites.
2. **frames**: derive the frames from a snapshot, with a fixed seed. No
   network, so it reproduces exactly.

Classification is conservative. A server counts as locally runnable only if
it ships a package for a runtime present on this machine and declares no
environment variable that looks like a secret. Anything else is recorded with
a reason, because the excluded share is itself a result: an audit of what can
be run locally is not an audit of the ecosystem, and the gap has to be
reported rather than glossed.

Usage:
    python scripts/sample_frame.py snapshot --out research/frames
    python scripts/sample_frame.py frames research/frames/registry-<date>.jsonl \\
        --draw 40 --seed 20261019 --out research/frames

Standard library only, like the rest of the package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

REGISTRY = "https://registry.modelcontextprotocol.io/v0/servers"
NPM_DOWNLOADS = "https://api.npmjs.org/downloads/point/last-week/"
USER_AGENT = "agent-fault-lab sampling frame (research; contact via repository)"

#: Registry types this machine can run, and how.
RUNNABLE_REGISTRIES = {"npm": "npx", "pypi": "uvx"}

#: An environment variable whose name matches is treated as a credential, and
#: the server is put in the "needs credentials" bucket rather than silently
#: failing mid-audit.
SECRET_PATTERN = re.compile(
    r"(token|key|secret|password|passwd|credential|api[_-]?key|auth|dsn|"
    r"connection[_-]?string|account|tenant|webhook)",
    re.IGNORECASE,
)

#: Frame 1, the reference implementations, is a hardcoded list rather than a
#: registry query, because **the official registry contains none of them**.
#: Checked 2026-10-19 against a complete snapshot: zero entries have
#: "modelcontextprotocol" anywhere in their server name, and
#: `@modelcontextprotocol/server-filesystem` (about 440k weekly npm
#: downloads) is absent while vanity packages with a few hundred downloads are
#: present. Any study that samples "the ecosystem" from the registry alone
#: therefore misses the most used servers in it, which is worth reporting in
#: its own right.
REFERENCE_PACKAGES = (
    ("npx", "@modelcontextprotocol/server-filesystem"),
    ("npx", "@modelcontextprotocol/server-memory"),
    ("npx", "@modelcontextprotocol/server-everything"),
    ("npx", "@modelcontextprotocol/server-sequential-thinking"),
    ("uvx", "mcp-server-git"),
    ("uvx", "mcp-server-fetch"),
    ("uvx", "mcp-server-time"),
)


#: Statuses that are an answer rather than a transient failure. Retrying them
#: wastes 4.5 seconds each and gives the same answer. The frames run that
#: prompted this spent over four hours on 12,000 npm lookups, most of the time
#: in backoff for packages the registry lists and npm has never heard of: a
#: registry entry can name a package that was never published, or was
#: unpublished later, and "404" is exactly the fact the funnel wants to record.
DEFINITIVE_STATUS = (400, 401, 403, 404, 405, 410, 422, 451)


def _get(url: str, *, timeout: float = 30.0, retries: int = 3) -> Optional[Any]:
    """GET and parse JSON, with backoff. Returns None rather than raising."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in DEFINITIVE_STATUS:
                return None
            if attempt == retries - 1:
                print(f"  warning: {url} failed: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError,
                json.JSONDecodeError, OSError) as exc:
            if attempt == retries - 1:
                print(f"  warning: {url} failed: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# ------------------------------------------------------------------- snapshot

def iter_registry(
    page_size: int = 100, max_pages: int = 20000
) -> Iterator[dict[str, Any]]:
    """Yield every registry entry, then a final marker saying if it finished.

    The registry pages alphabetically by server name and caps `limit` at 100,
    so a run that stops early stops in the middle of the alphabet. A draw from
    a truncated snapshot would be biased toward names starting with the first
    letters, which is why completeness is tracked rather than assumed.
    """
    cursor: Optional[str] = None
    pages = 0
    while True:
        if pages >= max_pages:
            yield {"__truncated__": True, "pages": pages, "last_cursor": cursor}
            return
        url = f"{REGISTRY}?limit={page_size}"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor, safe='')}"
        payload = _get(url)
        if not payload:
            yield {"__truncated__": True, "pages": pages, "last_cursor": cursor,
                   "reason": "a registry page could not be fetched"}
            return
        entries = payload.get("servers") or []
        for entry in entries:
            yield entry
        cursor = (payload.get("metadata") or {}).get("nextCursor")
        pages += 1
        if not cursor or not entries:
            yield {"__complete__": True, "pages": pages}
            return
        time.sleep(0.2)  # be a polite client of a free public service


def snapshot(
    out_dir: str, page_size: int, max_pages: int, latest_only: bool = True
) -> str:
    """Page the registry to a JSONL file, with a trailer recording totals.

    By default only the entries the registry flags as latest are kept. The
    registry stores every published version, so keeping them all multiplies
    the file size several fold without changing the sampling population,
    which is "the current version of every server". The trailer records how
    many were seen and how many kept, so the filtering is auditable.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"registry-{stamp}.jsonl")
    seen = 0
    kept = 0
    finish: dict[str, Any] = {"__complete__": False}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "__snapshot__": True,
                    "source": REGISTRY,
                    "fetched_at": stamp,
                    "page_size": page_size,
                    "latest_only": latest_only,
                }
            )
            + "\n"
        )
        for entry in iter_registry(page_size, max_pages):
            if entry.get("__truncated__") or entry.get("__complete__"):
                finish = entry
                break
            seen += 1
            if latest_only:
                meta = (entry.get("_meta") or {}).get(
                    "io.modelcontextprotocol.registry/official"
                ) or {}
                if not meta.get("isLatest"):
                    continue
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"  {kept} kept of {seen} seen", file=sys.stderr)
        handle.write(
            json.dumps(
                {
                    "__trailer__": True,
                    "entries_seen": seen,
                    "entries_kept": kept,
                    "complete": bool(finish.get("__complete__")),
                    "truncated": bool(finish.get("__truncated__")),
                    "truncation_reason": finish.get("reason", ""),
                    "pages": finish.get("pages"),
                }
            )
            + "\n"
        )
    status = "complete" if finish.get("__complete__") else "TRUNCATED"
    print(f"wrote {kept} entries (of {seen} seen) to {path} [{status}]")
    if finish.get("__truncated__"):
        print(
            "  warning: the snapshot is incomplete, so a random draw from it "
            "would be biased toward earlier server names. Re-run with a higher "
            "--max-pages before using it for Frame 3.",
            file=sys.stderr,
        )
    return path


# ----------------------------------------------------------------- classify

def latest_only(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per server name: the one the registry marks as latest.

    The registry stores every published version, so a naive count overstates
    the ecosystem's size by a large factor.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        server = entry.get("server") or {}
        name = server.get("name")
        if not name:
            continue
        meta = (entry.get("_meta") or {}).get(
            "io.modelcontextprotocol.registry/official"
        ) or {}
        is_latest = bool(meta.get("isLatest"))
        existing = chosen.get(name)
        if existing is None:
            chosen[name] = entry
        elif is_latest:
            chosen[name] = entry
        else:
            existing_meta = (existing.get("_meta") or {}).get(
                "io.modelcontextprotocol.registry/official"
            ) or {}
            if not existing_meta.get("isLatest"):
                # Neither is flagged, so fall back to publish time.
                if str(meta.get("publishedAt", "")) > str(
                    existing_meta.get("publishedAt", "")
                ):
                    chosen[name] = entry
    return list(chosen.values())


def _declared_secrets(package: dict[str, Any]) -> list[str]:
    names = []
    for variable in package.get("environmentVariables") or []:
        name = variable.get("name") if isinstance(variable, dict) else None
        if isinstance(name, str) and SECRET_PATTERN.search(name):
            names.append(name)
    return names


def _launch_arguments(package: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The arguments the registry says to start this package with.

    An earlier version ignored these, so servers were launched as a bare
    `npx <package>`. Three of the first candidates then hung at `initialize`
    with nothing on stderr, because their entry point needs a subcommand
    (`mushroomdb mcp <dir>`, `memorix serve`). Launching a server the way its
    own registry entry says to is part of what the audit measures: a server
    that cannot be started from its metadata is a finding, not a bad pick.

    Returns the literal argument values, and separately the placeholders that
    need a value from the caller (a database directory, for example), because
    a fixture has to supply those before the server can run.
    """
    argv: list[str] = []
    needs_input: list[str] = []
    for key in ("runtimeArguments", "packageArguments"):
        for argument in package.get(key) or []:
            if not isinstance(argument, dict):
                continue
            value = argument.get("value")
            name = argument.get("name") or argument.get("valueHint") or ""
            if isinstance(value, str) and value:
                if argument.get("type") == "named" and name:
                    argv += [name, value]
                else:
                    argv.append(value)
                continue
            if not argument.get("isRequired"):
                # An optional argument with no declared value is left out
                # rather than passed as a placeholder string.
                continue
            hint = name or argument.get("placeholder") or argument.get("format") or "?"
            needs_input.append(str(hint))
            argv.append("{" + str(hint) + "}")
    return argv, needs_input


def classify(entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten one registry entry into a frame row with a runnability reason."""
    server = entry.get("server") or {}
    meta = (entry.get("_meta") or {}).get(
        "io.modelcontextprotocol.registry/official"
    ) or {}
    packages = server.get("packages") or []
    remotes = server.get("remotes") or []

    row: dict[str, Any] = {
        "name": server.get("name", ""),
        "title": server.get("title", ""),
        "version": server.get("version", ""),
        "description": (server.get("description") or "")[:300],
        "repository": (server.get("repository") or {}).get("url", ""),
        "status": meta.get("status", ""),
        "published_at": meta.get("publishedAt", ""),
        "is_reference": False,
        "package_count": len(packages),
        "remote_only": bool(remotes) and not packages,
        "registry_types": sorted({p.get("registryType", "") for p in packages}),
        "runner": "",
        "package": "",
        "package_version": "",
        "transport": "",
        "secrets": [],
        "runnable": False,
        "reason": "",
        "runtime_hint": "",
        "launch_argv": [],
        "launch_needs_input": [],
    }

    if row["status"] and row["status"] != "active":
        row["reason"] = f"registry status is {row['status']}"
        return row
    if not packages:
        row["reason"] = "remote only, no installable package" if remotes else "no package"
        return row

    for package in packages:
        registry_type = package.get("registryType", "")
        runner = RUNNABLE_REGISTRIES.get(registry_type)
        if runner is None:
            continue
        transport = (package.get("transport") or {}).get("type", "")
        if transport and transport != "stdio":
            row["reason"] = f"{registry_type} package uses {transport}, not stdio"
            continue
        secrets = _declared_secrets(package)
        argv, needs_input = _launch_arguments(package)
        row.update(
            {
                "runner": runner,
                "package": package.get("identifier", ""),
                "package_version": package.get("version", ""),
                "transport": transport or "stdio",
                "secrets": secrets,
                "runtime_hint": package.get("runtimeHint", "") or "",
                "launch_argv": argv,
                "launch_needs_input": needs_input,
            }
        )
        if secrets:
            row["reason"] = "declares credentials: " + ", ".join(sorted(set(secrets)))
            return row
        row["runnable"] = True
        row["reason"] = f"runnable with {runner}"
        return row

    if not row["reason"]:
        row["reason"] = "no npm or pypi stdio package (" + ",".join(
            row["registry_types"]
        ) + ")"
    return row


def add_npm_downloads(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> dict[str, Any]:
    """Weekly download counts for npm packages, as the popularity signal.

    `limit` of -1 means every npm package, which is what a defensible
    "popular" frame needs. A positive limit takes a **seeded random
    subsample**, never the alphabetical head: an earlier version sliced the
    first N and produced a Frame 2 whose "top by downloads" excluded the most
    downloaded servers entirely, because the registry pages alphabetically.

    Only npm is queried. PyPI has no equivalent free point endpoint, so pypi
    servers carry no popularity number and are ranked separately. That
    asymmetry is a limitation to state in the paper, not something to paper
    over with a guess.
    """
    candidates = [row for row in rows if row["runner"] == "npx" and row["package"]]
    if limit == 0:
        return {"queried": 0, "population": len(candidates), "sampled": False}
    if limit < 0 or limit >= len(candidates):
        targets = candidates
        sampled = False
    else:
        targets = random.Random(f"downloads:{seed}").sample(candidates, limit)
        sampled = True
    print(
        f"fetching npm download counts for {len(targets)} of {len(candidates)} "
        f"packages{' (random subsample)' if sampled else ''}",
        file=sys.stderr,
    )
    got = 0
    for index, row in enumerate(targets, 1):
        payload = _get(NPM_DOWNLOADS + urllib.parse.quote(row["package"], safe="@/"))
        row["weekly_downloads"] = (payload or {}).get("downloads")
        got += row["weekly_downloads"] is not None
        if index % 250 == 0:
            print(f"  {index}/{len(targets)}", file=sys.stderr)
        time.sleep(0.05)
    return {
        "queried": len(targets),
        "population": len(candidates),
        "answered": got,
        "sampled": sampled,
    }


def load_download_cache(path: str) -> dict[tuple[str, str], int]:
    """Download counts from a previous `frame-all.jsonl`.

    Fetching every npm package takes hours, and the counts move day to day, so
    refetching them to change something about the frames would silently change
    the popularity ranking too. Rebuilding from a cache keeps a frame change
    and a data change separate, which is what makes the frames reproducible.
    """
    cache: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("__snapshot__") or record.get("__trailer__"):
                continue
            count = record.get("weekly_downloads")
            if count is None:
                continue
            cache[(record.get("runner", ""), record.get("package", ""))] = count
    return cache


def apply_download_cache(
    rows: list[dict[str, Any]], cache: dict[tuple[str, str], int], source: str
) -> dict[str, Any]:
    """Fill in cached counts. Nothing is fetched, so no row is invented."""
    candidates = [row for row in rows if row["runner"] == "npx" and row["package"]]
    hits = 0
    for row in candidates:
        count = cache.get((row["runner"], row["package"]))
        if count is not None:
            row["weekly_downloads"] = count
            hits += 1
    return {
        "queried": 0,
        "population": len(candidates),
        "answered": hits,
        "sampled": False,
        "cached": hits,
        "cache_source": source,
    }


def collapse_aliases(
    rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """One row per package, with the other registry names recorded.

    The registry lists the same package under several server names
    (`io.github.callstack/agent-device` and
    `io.github.callstackincubator/agent-device` are one npm package). Ranking
    listings instead of packages would put one server in a "top 20" twice and,
    if drawn, would audit the same binary twice and count it as two independent
    observations. Frame 2 is a frame of packages.
    """
    by_package: dict[tuple[str, str], dict[str, Any]] = {}
    collapsed = 0
    for row in sorted(rows, key=lambda row: row["name"]):
        key = (row["runner"], row["package"])
        kept = by_package.get(key)
        if kept is None:
            by_package[key] = dict(row, registry_aliases=row["name"])
            continue
        kept["registry_aliases"] += f"; {row['name']}"
        collapsed += 1
    return list(by_package.values()), collapsed


# ------------------------------------------------------------------- frames

def build_frames(
    snapshot_path: str,
    out_dir: str,
    *,
    seed: int,
    draw: int,
    downloads: int,
    allow_truncated: bool = False,
    downloads_from: str | None = None,
) -> None:
    entries = []
    header: dict[str, Any] = {}
    trailer: dict[str, Any] = {}
    with open(snapshot_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("__snapshot__"):
                header = record
                continue
            if record.get("__trailer__"):
                trailer = record
                continue
            entries.append(record)

    if trailer.get("truncated") and not allow_truncated:
        raise SystemExit(
            f"{snapshot_path} is an incomplete snapshot "
            f"({trailer.get('entries_seen')} entries, stopped after "
            f"{trailer.get('pages')} pages). The registry pages alphabetically, "
            "so a random draw from it would over-represent early names. Re-run "
            "the snapshot, or pass --allow-truncated to build frames anyway "
            "and report the bias."
        )
    if not trailer:
        print(
            "  warning: this snapshot has no trailer, so completeness is "
            "unknown. It was probably written by an older version.",
            file=sys.stderr,
        )

    latest = latest_only(entries)
    rows = [classify(entry) for entry in latest]
    rows.sort(key=lambda row: row["name"])
    if downloads_from:
        download_stats = apply_download_cache(
            rows, load_download_cache(downloads_from), downloads_from
        )
    else:
        download_stats = add_npm_downloads(rows, downloads, seed)

    # Frame 1 is not derived from the registry; see REFERENCE_PACKAGES.
    reference_rows = [
        {
            "name": f"reference/{package}",
            "package": package,
            "runner": runner,
            "package_version": "",
            "is_reference": True,
            "repository": "https://github.com/modelcontextprotocol/servers",
            "reason": "reference implementation, not listed in the registry",
            "description": "official reference server",
        }
        for runner, package in REFERENCE_PACKAGES
    ]

    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "frame-all.jsonl")
    with open(all_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    runnable = [row for row in rows if row["runnable"]]
    reference = reference_rows
    with_downloads = [row for row in runnable if row.get("weekly_downloads")]
    popular, alias_count = collapse_aliases(with_downloads)
    popular.sort(key=lambda row: row["weekly_downloads"], reverse=True)

    # Frame 3: a seeded draw from the runnable population. Seeded on the
    # snapshot's own fetch time plus the given seed, so re-running against a
    # newer snapshot cannot silently reuse an old draw.
    rng = random.Random(f"{seed}:{header.get('fetched_at', '')}")
    pool = sorted(runnable, key=lambda row: row["name"])
    random_draw = rng.sample(pool, min(draw, len(pool)))

    _write_csv(os.path.join(out_dir, "frame1-reference.csv"), reference)
    _write_csv(os.path.join(out_dir, "frame2-popular.csv"), popular[: max(draw, 40)])
    _write_csv(os.path.join(out_dir, "frame3-random.csv"), random_draw)

    reasons: dict[str, int] = {}
    for row in rows:
        if row["runnable"]:
            continue
        key = row["reason"].split(":")[0].split("(")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1

    summary = [
        "# Sampling frames",
        "",
        f"Snapshot: `{os.path.basename(snapshot_path)}` fetched "
        f"{header.get('fetched_at', 'unknown')} from {header.get('source', REGISTRY)}.",
        f"Snapshot completeness: "
        f"{'complete' if trailer.get('complete') else 'INCOMPLETE (draw is biased)'}; "
        f"{trailer.get('entries_kept', len(entries))} entries kept of "
        f"{trailer.get('entries_seen', 'unknown')} seen.",
        f"Draw seed: `{seed}` (combined with the snapshot's fetch time).",
        "",
        f"- registry entries in the snapshot: {len(entries)}",
        f"- distinct servers (latest version only): {len(rows)}",
        f"- locally runnable with npx or uvx, no declared credentials: "
        f"{len(runnable)} ({len(runnable) / len(rows):.1%})" if rows else "- no rows",
        f"- reference servers (Frame 1, hardcoded, absent from the registry): "
        f"{len(reference)}",
        f"- npm download counts fetched: {download_stats.get('queried', 0)} of "
        f"{download_stats.get('population', 0)} npm packages"
        + (" (RANDOM SUBSAMPLE, so Frame 2 is a sample of the popular tail, "
           "not the true top)" if download_stats.get("sampled") else ""),
        f"- npm download counts source: "
        + (f"cached from {os.path.basename(download_stats['cache_source'])}"
           if download_stats.get("cache_source") else "fetched live"),
        f"- runnable packages with a weekly npm download count: {len(popular)}",
        f"- registry listings collapsed into another listing of the same "
        f"package: {alias_count}",
        "",
        "**The registry does not list the official reference servers.** A "
        "complete snapshot contains no server whose name mentions "
        "modelcontextprotocol, and the reference packages are absent while "
        "vanity packages with a few hundred weekly downloads are present. A "
        "study that samples the ecosystem from the registry alone misses the "
        "most used servers in it. Frame 1 is therefore a hardcoded list.",
        f"- Frame 3 draw: {len(random_draw)}",
        "",
        "## Why servers were excluded",
        "",
        "| reason | servers |",
        "|---|---|",
    ]
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        summary.append(f"| {reason} | {count} |")
    summary += [
        "",
        "The excluded share is a result, not an inconvenience. An audit of "
        "locally runnable servers is not an audit of the ecosystem, and the "
        "paper reports this table so the gap is visible.",
        "",
        "## Frame 2, top 20 by weekly npm downloads",
        "",
        "| package | downloads | server |",
        "|---|---|---|",
    ]
    for row in popular[:20]:
        summary.append(
            f"| `{row['package']}` | {row['weekly_downloads']:,} | {row['name']} |"
        )
    summary_path = os.path.join(out_dir, "frames-summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")
    print(f"wrote {all_path} and the frame CSVs; summary at {summary_path}")


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    import csv

    columns = [
        "name", "package", "runner", "package_version", "weekly_downloads",
        "is_reference", "registry_aliases", "runtime_hint", "launch_argv",
        "launch_needs_input", "repository", "reason", "description",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("launch_argv", "launch_needs_input"):
                if isinstance(flat.get(key), list):
                    flat[key] = " ".join(str(item) for item in flat[key])
            writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="page the registry to a JSONL snapshot")
    snap.add_argument("--out", default="research/frames")
    snap.add_argument("--page-size", type=int, default=100)
    snap.add_argument("--max-pages", type=int, default=20000)
    snap.add_argument(
        "--all-versions",
        action="store_true",
        help="keep every published version, not just the latest of each server",
    )

    frames = sub.add_parser("frames", help="derive frames from a snapshot, offline")
    frames.add_argument("snapshot")
    frames.add_argument("--out", default="research/frames")
    frames.add_argument("--seed", type=int, default=20261019)
    frames.add_argument("--draw", type=int, default=40)
    frames.add_argument(
        "--allow-truncated",
        action="store_true",
        help="build frames from an incomplete snapshot and report the bias",
    )
    frames.add_argument(
        "--downloads",
        type=int,
        default=-1,
        help=(
            "npm packages to fetch download counts for: -1 for all (default), "
            "0 to skip, N for a seeded random subsample of N"
        ),
    )

    frames.add_argument(
        "--downloads-from",
        help=(
            "reuse the npm download counts in a previous frame-all.jsonl "
            "instead of fetching them, so the frames can be rebuilt offline"
        ),
    )

    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(
            args.out,
            args.page_size,
            args.max_pages,
            latest_only=not args.all_versions,
        )
        return 0
    build_frames(
        args.snapshot,
        args.out,
        seed=args.seed,
        draw=args.draw,
        downloads=args.downloads,
        allow_truncated=args.allow_truncated,
        downloads_from=args.downloads_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
