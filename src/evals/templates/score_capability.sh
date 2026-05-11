#!/bin/bash
# score_capability.sh — agent-facing capability spot-check.
#
# Pairs with the primary score.sh. While score.sh returns the metrics
# for the bench your condition targets, this script runs a quick
# capability eval (configured pipeline-side; usually a small MCQ
# probe at limit=30) and returns its metrics dict. Use it during
# training to spot-check that capability hasn't regressed.
#
# Output: same shape as score.sh — `{"accuracy": X, "stderr": Y}` from
# the underlying inspect-ai task. No prompts or bench identity leaked.
#
# Cost: ~30-60s per call on Qwen3-1.7B. The pipeline-side
# /opt/pipeline-bin/score_capability_runner.sh caps the limit at 30
# by default (override via --limit if you want a tighter SE), so
# don't expect this to take as long as the headline post-eval pass.
#
# Returns exit 2 if no capability probe is configured for the current
# condition (e.g. condition A which is benchmark-blind). Reading the
# error message is fine.
set -u

OUT_DIR=${SCORE_OUT_DIR:-.}
mkdir -p "$OUT_DIR"
METRICS_FILE=$(mktemp "$OUT_DIR/score_capability.metrics.XXXXXX.json")
LOG_FILE="$OUT_DIR/score_capability.log"

sudo -n /opt/pipeline-bin/score_capability_runner.sh \
    "$@" --json-output-file "$METRICS_FILE" \
    >> "$LOG_FILE" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
    echo "{\"error\": \"score_capability_runner.sh exited rc=$rc; see $LOG_FILE\"}" >&2
    rm -f "$METRICS_FILE"
    exit $rc
fi

if [ ! -s "$METRICS_FILE" ]; then
    echo "{\"error\": \"score_capability_runner.sh exited 0 but wrote no metrics; see $LOG_FILE\"}" >&2
    rm -f "$METRICS_FILE"
    exit 1
fi

cat "$METRICS_FILE"
rm -f "$METRICS_FILE"
