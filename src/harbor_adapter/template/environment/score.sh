#!/bin/bash
# score.sh — clean accuracy-only wrapper around evaluate.py.
#
# Usage:
#   bash score.sh [evaluate.py args ...]
#
# All inspect-ai progress output, sample traces, and banners are
# redirected to score.log; stdout is JUST the metrics dict written
# by evaluate.py (e.g. {"accuracy": 0.42, "stderr": 0.09}).
#
# Why: evaluate.py is intentionally chatty so we can debug it. For
# training-time querying of "is the model getting better", you only
# need the score; the rest is noise that biases your prompt-engineering
# toward whatever-the-eval-prints (e.g. memorising the prompt format).
set -u

OUT_DIR=${SCORE_OUT_DIR:-.}
mkdir -p "$OUT_DIR"
METRICS_FILE=$(mktemp "$OUT_DIR/score.metrics.XXXXXX.json")
LOG_FILE="$OUT_DIR/score.log"

# evaluate.py reads --json-output-file from argv. Inject ours so the
# caller doesn't need to know about it; if the caller already passed
# one, theirs wins (argparse takes the last value).
python3 evaluate.py "$@" --json-output-file "$METRICS_FILE" \
    >> "$LOG_FILE" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
    echo "{\"error\": \"evaluate.py exited rc=$rc; see $LOG_FILE\"}" >&2
    rm -f "$METRICS_FILE"
    exit $rc
fi

if [ ! -s "$METRICS_FILE" ]; then
    echo "{\"error\": \"evaluate.py exited 0 but wrote no metrics; see $LOG_FILE\"}" >&2
    rm -f "$METRICS_FILE"
    exit 1
fi

cat "$METRICS_FILE"
rm -f "$METRICS_FILE"
