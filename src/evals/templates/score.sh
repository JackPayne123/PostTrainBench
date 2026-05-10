#!/bin/bash
# score.sh — agent-facing wrapper. Returns just the metrics dict
# ({"accuracy": X, "stderr": Y}); all evaluate.py noise goes to score.log.
#
# Architecture (post 2026-05-10 isolation refactor):
#   The actual evaluate.py + prompts.jsonl live under /opt/ptb (root-only,
#   chmod 700). The agent runs as uid 1000. To query the score without
#   leaking the test set, this wrapper sudo-invokes a pipeline-side
#   runner (/opt/pipeline-bin/score_runner.sh) that reads the prompts +
#   runs evaluate.py against the agent's local vllm. The agent only
#   ever sees the metrics dict.
#
#   BENCH must be set in the env (or .bench file in cwd) so the runner
#   knows which task to load. stage_agent_workspace writes .bench.
#
# Usage:
#   bash score.sh [evaluate.py args ...]
#
# Common args forwarded to evaluate.py: --limit N, --vllm-base-url URL,
# --vllm-served-name NAME, --max-connections N.
set -u

OUT_DIR=${SCORE_OUT_DIR:-.}
mkdir -p "$OUT_DIR"
METRICS_FILE=$(mktemp "$OUT_DIR/score.metrics.XXXXXX.json")
LOG_FILE="$OUT_DIR/score.log"

# Resolve which benchmark this workspace is staged for.
if [ -z "${BENCH:-}" ] && [ -f .bench ]; then
    BENCH=$(cat .bench)
fi
if [ -z "${BENCH:-}" ]; then
    echo '{"error": "BENCH env var not set and no .bench file in cwd"}' >&2
    rm -f "$METRICS_FILE"
    exit 2
fi

# Inject --json-output-file so the caller doesn't have to. argparse
# takes the last value, so a user-supplied --json-output-file still wins.
BENCH="$BENCH" sudo -n -E /opt/pipeline-bin/score_runner.sh \
    "$@" --json-output-file "$METRICS_FILE" \
    >> "$LOG_FILE" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
    echo "{\"error\": \"score_runner.sh exited rc=$rc; see $LOG_FILE\"}" >&2
    rm -f "$METRICS_FILE"
    exit $rc
fi

if [ ! -s "$METRICS_FILE" ]; then
    echo "{\"error\": \"score_runner.sh exited 0 but wrote no metrics; see $LOG_FILE\"}" >&2
    rm -f "$METRICS_FILE"
    exit 1
fi

cat "$METRICS_FILE"
rm -f "$METRICS_FILE"
