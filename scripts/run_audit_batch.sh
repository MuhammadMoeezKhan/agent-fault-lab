#!/bin/sh
# Audit every spec in specs/ and write one report, CSV and JSONL per server.
#
# Runs one server at a time on purpose: each trial launches a fresh npx or uvx
# process, and two batches at once make servers time out and produce rows that
# look like lost effects. F5 is added only for the specs whose fixtures take a
# list argument, as the taxonomy's note says.
#
# Usage: scripts/run_audit_batch.sh <output directory> [repeats]
set -eu

OUT=${1:?output directory}
REPEATS=${2:-3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}

mkdir -p "$OUT"

for spec in "$ROOT"/specs/*.json; do
    name=$(basename "$spec" .json)
    case "$name" in
        community-memorix|reference-git|community-cyanheads-git|community-sqlite)
            faults="F7,F1,F2,F3,F5" ;;
        *)
            faults="F7,F1,F2,F3" ;;
    esac
    echo "=== $name ($faults, repeats=$REPEATS)"
    "$PY" -m agentfaultlab.cli check "$spec" \
        --faults "$faults" \
        --repeats "$REPEATS" \
        --exit-zero \
        --out "$OUT/audit-$name.md" \
        --csv "$OUT/audit-$name.csv" \
        --jsonl "$OUT/audit-$name.jsonl" \
        > "$OUT/audit-$name.log" 2>&1 || echo "    FAILED, see $OUT/audit-$name.log"
done

echo
echo "reports in $OUT"
