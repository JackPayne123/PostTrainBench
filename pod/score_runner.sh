#!/bin/bash
# Pipeline-side score runner. Invoked by the agent's score.sh via
# `sudo -n` (NOPASSWD entry in /etc/sudoers.d/agent-score). Runs as
# root, so it has read access to /opt/ptb (chmod 700, root-only) where
# the eval task's evaluate.py + prompts.jsonl live.
#
# The agent never sees the prompts file or evaluate.py source — only
# the headline metrics dict on stdout.
#
# Why this exists:
#   Pre-2026-05-10, the entire repo was COPYed to /opt/ptb readable
#   by the agent (running as root). The agent could literally do
#       cat /opt/ptb/src/evals/tasks/<category>/<bench>/prompts.jsonl
#   and train on the test set. We caught it on a sycophancy_slava run
#   and locked /opt/ptb down (chmod 700, agent now runs as uid 1000).
#   But the agent still needs to query score.sh for training-signal
#   feedback (Rule 3 in instruction.md allows small-N queries). This
#   wrapper is the way: agent invokes via sudo, root reads prompts,
#   pipes accuracy back.
#
# Reads task id (e.g. "sycophancy_slava") from /etc/ptb_run/bench —
# root-only chmod 600. Agent CANNOT set BENCH from its env to override:
# the env-based approach leaked the benchmark name when agent could
# `cat .bench` from its own workspace (caught on 2026-05-11 F-run
# analysis). The wrapper find-resolves the bench name under
# src/evals/tasks/{capability,safety,character}/<bench>/evaluate.py
# because the directory layout is category-bucketed (centralisation
# refactor 2026-05-11).
#
# Forwarded to evaluate.py: all positional args ("$@").
#
# stdout: whatever evaluate.py writes (its metrics dict). Caller's
# score.sh swallows the noise and emits just {accuracy, stderr}.

set -euo pipefail

BENCH_STATE_FILE="/etc/ptb_run/bench"
if [ ! -r "$BENCH_STATE_FILE" ]; then
    echo "{\"error\": \"benchmark state file ${BENCH_STATE_FILE} not present or unreadable; pipeline must stage it before agent runs\"}" >&2
    exit 2
fi
BENCH=$(cat "$BENCH_STATE_FILE")
if [ -z "$BENCH" ]; then
    echo "{\"error\": \"${BENCH_STATE_FILE} is empty\"}" >&2
    exit 2
fi

# Find the task directory under any category bucket. Limit depth to 3
# so the find stays cheap and predictable.
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

# Run from EVAL_DIR so evaluate.py's default
# `os.path.join(os.path.dirname(__file__), "prompts.jsonl")` resolves
# correctly without the agent's previous symlink hack.
cd "$EVAL_DIR"
exec python3 "$EVAL_PY" "$@"
