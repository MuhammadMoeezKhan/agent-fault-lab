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

#: Names that are the reference implementations, which form Frame 1.
REFERENCE_PREFIXES = ("io.modelcontextprotocol/", "io.github.modelcontextprotocol/")


def _get(url: str, *, timeout: float = 30.0, retries: int = 3) -> Optional[Any]:
    """GET and parse JSON, with backoff. Returns None rather than raising."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as exc:
            if attempt == retries - 1:
                print(f"  warning: {url} failed: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# ------------------------------------------------------------------- snapshot

def iter_registry(page_size: int = 100, max_pages: int = 1000) -> Iterator[dict[str, Any]]:
    cursor: Optional[str] = None
    pages = 0
    while pages < max_pages:
        url = f"{REGISTRY}?limit={page_size}"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor, safe='')}"
        payload = _get(url)
        if not payload:
            return
        entries = payload.get("servers") or []
        for entry in entries:
            yield entry
        cursor = (payload.get("metadata") or {}).get("nextCursor")
        pages += 1
        if not cursor or not entries:
            return
        time.sleep(0.2)  # be a polite client of a free public service


def snapshot(out_dir: str, page_size: int, max_pages: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"registry-{stamp}.jsonl")
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "__snapshot__": True,
                    "source": REGISTRY,
                    "fetched_at": stamp,
                    "page_size": page_size,
                }
            )
            + "\n"
        )
        for entry in iter_registry(page_size, max_pages):
            handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
            count += 1
            if count % 500 == 0:
                print(f"  {count} entries", file=sys.stderr)
    print(f"wrote {count} registry entries to {path}")
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
        "is_reference": any(
            server.get("name", "").startswith(prefix) for prefix in REFERENCE_PREFIXES
        ),
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
        row.update(
            {
                "runner": runner,
                "package": package.get("identifier", ""),
                "package_version": package.get("version", ""),
                "transport": transport or "stdio",
                "secrets": secrets,
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


def add_npm_downloads(rows: list[dict[str, Any]], limit: int) -> None:
    """Weekly download counts for npm packages, as the popularity signal.

    Only npm is queried. PyPI has no equivalent free point endpoint, so pypi
    servers carry no popularity number and are ranked separately. That
    asymmetry is a limitation to state in the paper, not something to paper
    over with a guess.
    """
    targets = [row for row in rows if row["runner"] == "npx" and row["package"]][:limit]
    print(f"fetching npm download counts for {len(targets)} packages", file=sys.stderr)
    for index, row in enumerate(targets, 1):
        payload = _get(NPM_DOWNLOADS + urllib.parse.quote(row["package"], safe="@/"))
        row["weekly_downloads"] = (payload or {}).get("downloads")
        if index % 50 == 0:
            print(f"  {index}/{len(targets)}", file=sys.stderr)
        time.sleep(0.1)


# ------------------------------------------------------------------- frames

def build_frames(
    snapshot_path: str,
    out_dir: str,
    *,
    seed: int,
    draw: int,
    downloads: int,
) -> None:
    entries = []
    header: dict[str, Any] = {}
    with open(snapshot_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("__snapshot__"):
                header = record
                continue
            entries.append(record)

    latest = latest_only(entries)
    rows = [classify(entry) for entry in latest]
    rows.sort(key=lambda row: row["name"])
    if downloads:
        add_npm_downloads(rows, downloads)

    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "frame-all.jsonl")
    with open(all_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    runnable = [row for row in rows if row["runnable"]]
    reference = [row for row in rows if row["is_reference"]]
    popular = sorted(
        (row for row in runnable if row.get("weekly_downloads")),
        key=lambda row: row["weekly_downloads"],
        reverse=True,
    )

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
        f"Draw seed: `{seed}` (combined with the snapshot's fetch time).",
        "",
        f"- registry entries in the snapshot: {len(entries)}",
        f"- distinct servers (latest version only): {len(rows)}",
        f"- locally runnable with npx or uvx, no declared credentials: "
        f"{len(runnable)} ({len(runnable) / len(rows):.1%})" if rows else "- no rows",
        f"- reference servers (Frame 1): {len(reference)}",
        f"- runnable with a weekly npm download count: {len(popular)}",
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
        "is_reference", "repository", "reason", "description",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="page the registry to a JSONL snapshot")
    snap.add_argument("--out", default="research/frames")
    snap.add_argument("--page-size", type=int, default=100)
    snap.add_argument("--max-pages", type=int, default=1000)

    frames = sub.add_parser("frames", help="derive frames from a snapshot, offline")
    frames.add_argument("snapshot")
    frames.add_argument("--out", default="research/frames")
    frames.add_argument("--seed", type=int, default=20261019)
    frames.add_argument("--draw", type=int, default=40)
    frames.add_argument(
        "--downloads",
        type=int,
        default=400,
        help="how many npm packages to fetch download counts for (0 to skip)",
    )

    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.out, args.page_size, args.max_pages)
        return 0
    build_frames(
        args.snapshot,
        args.out,
        seed=args.seed,
        draw=args.draw,
        downloads=args.downloads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
