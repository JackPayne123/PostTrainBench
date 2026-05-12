#!/bin/bash
# score.sh — agent-facing wrapper. Returns just the metrics dict
# ({"accuracy": X, "stderr": Y}); all evaluate.py noise goes to score.log.
#
# Architecture (post 2026-05-11):
#   The actual evaluate.py + prompts.jsonl live under /opt/ptb (root-only,
#   chmod 700). The agent runs as uid 1000. To query the score without
#   leaking either the prompts or the benchmark identity, this wrapper
#   sudo-invokes a pipeline-side runner that reads /etc/ptb_run/bench
#   (root-only) to find the active task, runs evaluate.py against the
#   agent's local vllm, and returns just the metrics dict on stdout.
#   The agent never sees prompts content, evaluate.py source, or even
#   the benchmark name.
#
# Usage:
#   bash score.sh [evaluate.py args ...]
#
# Common args forwarded: --limit N, --vllm-base-url URL,
# --vllm-served-name NAME, --max-connections N.
set -u

OUT_DIR=${SCORE_OUT_DIR:-.}
mkdir -p "$OUT_DIR"
METRICS_FILE=$(mktemp "$OUT_DIR/score.metrics.XXXXXX.json")
LOG_FILE="$OUT_DIR/score.log"

# Resolve any relative path arguments to absolute paths BEFORE sudo.
# score_runner.sh `cd $EVAL_DIR` before exec'ing evaluate.py, so a
# relative `--model-path final_model` from the agent's cwd would
# break (FileNotFoundError: final_model/config.json — caught on
# 2026-05-12 Qwen3.5-9B F-run analysis). Resolve here so the agent
# can keep using the natural `--model-path final_model` convention.
#
# Args we rewrite: known path flags only. Conservative — flags that
# happen to take non-path values pass through unchanged.
REWRITTEN=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model-path|--templates-dir|--output-dir|--eval-data)
            flag="$1"
            shift
            val="${1:-}"
            # Only resolve relative paths; absolute or empty/flag-like pass through.
            if [ -n "$val" ] && [ "${val:0:1}" != "-" ] && [ "${val:0:1}" != "/" ]; then
                val="$(realpath -m "$val")"
            fi
            REWRITTEN+=("$flag" "$val")
            shift
            ;;
        *)
            REWRITTEN+=("$1")
            shift
            ;;
    esac
done

# No BENCH env var passed — score_runner.sh reads it from
# /etc/ptb_run/bench. This deliberately means an agent setting
# BENCH=foo in its environment has no effect on which task is
# evaluated. Sudoers entry stays minimal NOPASSWD-only; no SETENV
# or env_keep needed.
sudo -n /opt/pipeline-bin/score_runner.sh \
    "${REWRITTEN[@]}" --json-output-file "$METRICS_FILE" \
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
