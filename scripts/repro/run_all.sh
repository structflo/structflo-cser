#!/usr/bin/env bash
# Full reproducibility pipeline entry point: train -> eval -> aggregate.
# Every stage is idempotent/resumable, so this is safe to re-run after any interruption.
#
# Launch fully detached (survives the launching session/terminal):
#   setsid nohup bash scripts/repro/run_all.sh >> runs/repro/logs/run_all.log 2>&1 </dev/null &
#
# Override seeds with:  SEEDS="42 43" bash scripts/repro/run_all.sh
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === run_all.sh START (seeds: ${SEEDS:-42 43 44}) ==="
bash scripts/repro/run_train.sh \
  && bash scripts/repro/run_eval.sh \
  && uv run python scripts/repro/aggregate.py
rc=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === run_all.sh END (rc=$rc) ==="
exit $rc
