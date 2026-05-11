#!/bin/bash
# Pipeline-side capability score runner. Same shape as score_runner.sh
# but reads its bench name from /etc/ptb_run/bench_capability instead
# of /etc/ptb_run/bench. The pipeline writes the capability-probe
# bench (e.g. arc_easy) into that file when a condition cares about
# capability preservation (F today).
#
# The agent invokes via `sudo -n /opt/pipeline-bin/score_capability_runner.sh`
# (NOPASSWD entry in /etc/sudoers.d/agent-score). The companion thin
# wrapper is /home/agent/workspace/score_capability.sh.
#
# Why a second runner instead of a flag on the first: keeps the
# audit-trail (sudoers + diag.py) symmetric, and lets us run a small
# limit on the capability probe (default 30 — see CAPABILITY_LIMIT
# below) without changing how the agent-facing primary call behaves.

set -euo pipefail

BENCH_STATE_FILE="/etc/ptb_run/bench_capability"
if [ ! -r "$BENCH_STATE_FILE" ]; then
    echo "{\"error\": \"capability probe state file ${BENCH_STATE_FILE} not present; pipeline did not configure a capability probe for this condition\"}" >&2
    exit 2
fi
BENCH=$(cat "$BENCH_STATE_FILE")
if [ -z "$BENCH" ]; then
    echo "{\"error\": \"${BENCH_STATE_FILE} is empty\"}" >&2
    exit 2
fi

EVAL_DIR=$(find /opt/ptb/src/evals/tasks -maxdepth 2 -mindepth 2 -type d -name "$BENCH" -print -quit 2>/dev/null || true)
if [ -z "$EVAL_DIR" ] || [ ! -d "$EVAL_DIR" ]; then
    echo "{\"error\": \"task '${BENCH}' not found under /opt/ptb/src/evals/tasks/*/\"}" >&2
    exit 3
fi

EVAL_PY="${EVAL_DIR}/evaluate.py"
if [ ! -f "$EVAL_PY" ]; then
    echo "{\"error\": \"evaluate.py not found at ${EVAL_PY}\"}" >&2
    exit 4
fi

# Capability probe runs at a small fixed limit by default so the agent
# can call it cheaply during training (~30-60s per call on a 1.7B
# model). The caller can still pass --limit N to override.
CAPABILITY_LIMIT="${CAPABILITY_PROBE_LIMIT:-30}"

cd "$EVAL_DIR"
# If the caller already passed --limit, honour that; otherwise inject
# our default. We detect the flag in the forwarded args list.
case " $* " in
    *" --limit "*) ;;
    *) set -- "$@" --limit "$CAPABILITY_LIMIT" ;;
esac

exec python3 "$EVAL_PY" "$@"
