#!/bin/sh
# Audit every spec in specs/ and write one report, CSV and JSONL per server.
#
# Runs one server at a time on purpose: each trial launches a fresh process
# inside the container, and two batches at once make servers time out and
# produce rows that look like lost effects. F5 is added only for the specs
# whose fixtures take a list argument, as the taxonomy's note says.
#
# The container image is required, not optional. Rule 9 puts every third-party
# server inside a container, and this script asked for none until the batch of
# 2026-10-25 had to be run from an inline loop instead. A batch runner that can
# silently run a server on the host is the wrong shape for this study.
#
# Usage: scripts/run_audit_batch.sh <output directory> <image> [repeats]
set -eu

OUT=${1:?output directory}
IMAGE=${2:?container image, for example agent-fault-lab:audit}
REPEATS=${3:-3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}

mkdir -p "$OUT"

for spec in "$ROOT"/specs/*.json; do
    name=$(basename "$spec" .json)
    case "$name" in
        # Frozen on 2026-10-21 under Q20: its rows were collected on the host
        # before rule 9, and it cannot be built in the image (fsevents).
        community-desktop-commander)
            echo "=== $name skipped (Q20, host-collected and pre-rule)"; continue ;;
        # The published package vendors a binary linked against glibc 2.39;
        # the image has 2.36, so it cannot run contained at all.
        community-mushroomdb)
            echo "=== $name skipped (vendored binary needs a newer glibc than the image)"
            continue ;;
    esac
    case "$name" in
        community-memorix|reference-git|community-cyanheads-git|community-sqlite)
            faults="F7,F1,F2,F3,F5" ;;
        *)
            faults="F7,F1,F2,F3" ;;
    esac
    echo "=== $name ($faults, repeats=$REPEATS, contained) $(date +%H:%M:%S)"
    "$PY" -m agentfaultlab.cli check "$spec" \
        --faults "$faults" \
        --repeats "$REPEATS" \
        --image "$IMAGE" \
        --exit-zero \
        --out "$OUT/audit-$name.md" \
        --csv "$OUT/audit-$name.csv" \
        --jsonl "$OUT/audit-$name.jsonl" \
        > "$OUT/audit-$name.log" 2>&1 || echo "    FAILED, see $OUT/audit-$name.log"
done

echo
echo "done $(date +%H:%M:%S); reports in $OUT"
