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

# Pre-stage base model config.json into a LoRA adapter dir if missing.
# evaluate.py:model_type() reads config.json from --model-path to
# resolve the chat template. peft.save_pretrained only writes
# adapter_config.json, so an agent-trained adapter dir crashes
# evaluate.py with FileNotFoundError. Read base_model_name_or_path
# from adapter_config.json and copy the base model's config.json from
# the HF cache. Caught 2026-05-13 F-run analysis: agent's score.sh
# probes returned rc=1 for ~5 attempts before agent manually copied
# config.json from /workspace/hf-cache. Pre-staging here unblocks
# every eval that uses model_type().
MODEL_PATH=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--model-path" ]; then
        MODEL_PATH="$arg"
        break
    fi
    prev="$arg"
done
if [ -n "$MODEL_PATH" ] \
   && [ -d "$MODEL_PATH" ] \
   && [ -f "$MODEL_PATH/adapter_config.json" ] \
   && [ ! -f "$MODEL_PATH/config.json" ]; then
    base_model=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('base_model_name_or_path',''))" "$MODEL_PATH/adapter_config.json" 2>/dev/null)
    if [ -n "$base_model" ]; then
        # HF cache layout: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>/
        # HF_HOME may also point at /workspace/hf-cache (set by run_experiment.py).
        slug="models--$(echo "$base_model" | sed 's|/|--|g')"
        for cache_root in "${HF_HOME:-}" "/workspace/hf-cache" "$HOME/.cache/huggingface" "/root/.cache/huggingface"; do
            [ -z "$cache_root" ] && continue
            src=$(find "$cache_root/hub/$slug/snapshots" -name config.json -print -quit 2>/dev/null || true)
            if [ -n "$src" ] && [ -f "$src" ]; then
                cp "$src" "$MODEL_PATH/config.json"
                break
            fi
        done
    fi
fi

# Run from EVAL_DIR so evaluate.py's default
# `os.path.join(os.path.dirname(__file__), "prompts.jsonl")` resolves
# correctly without the agent's previous symlink hack.
cd "$EVAL_DIR"
# umask 0o077 so evaluate.py's writes (logs/, metrics_*.json, eval_*.log)
# default to 600/700 — agent (uid 1000) can't read them. Without this,
# inspect-ai's mid-eval file creates would inherit the calling shell's
# umask (often 0 in container contexts), re-opening the lockdown that
# stage_eval_task tries to set on the parent dir.
umask 0077
# Run the eval. After it exits, re-lock the dir (belt-and-braces; the
# umask above SHOULD cover all new files, but inspect-ai sometimes
# spawns subprocesses that reset umask via env).
python3 "$EVAL_PY" "$@"
RC=$?
# Re-lock PTB_EVAL on container rootfs (no-op on network volume, but
# defence in depth on the path that actually honours chmod).
chown -R root:root /var/lib/ptb_eval 2>/dev/null
chmod -R go-rwx /var/lib/ptb_eval 2>/dev/null
# Legacy /workspace/ptb_eval is wiped at pod startup; nothing to lock here.
exit $RC
