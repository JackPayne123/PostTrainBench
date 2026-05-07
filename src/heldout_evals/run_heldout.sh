#!/usr/bin/env bash
# Run the held-out eval panel against a trained checkpoint.
#
# Usage:
#   bash src/heldout_evals/run_heldout.sh <run_dir> [task_name...]
#
# <run_dir> must contain a `final_model/` subdirectory. Output is written
# to <run_dir>/heldout/<task>.json plus a final summary.{json,md}.
#
# By default a single shared vllm server is started up front and torn down
# at the end. Each task talks to that server via OpenAI-compat instead of
# spinning its own vllm. Saves ~60s × N tasks of cold-start. Set
# HELDOUT_NO_SHARED_VLLM=1 to fall back to per-task local vllm.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <run_dir> [task_name...]" >&2
    echo "  if task names are passed, only those tasks run" >&2
    exit 1
fi

RUN_DIR="$1"
shift || true
SELECTED_TASKS=("$@")

if [[ ! -d "$RUN_DIR/final_model" ]]; then
    echo "error: $RUN_DIR/final_model does not exist" >&2
    exit 1
fi

ABS_MODEL="$(cd "$RUN_DIR/final_model" && pwd)"
HELDOUT_DIR="$RUN_DIR/heldout"
mkdir -p "$HELDOUT_DIR"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASKS_DIR="$REPO_ROOT/src/heldout_evals/tasks"

echo "[heldout] model: $ABS_MODEL"
echo "[heldout] output: $HELDOUT_DIR"

# ─── Shared vllm setup ─────────────────────────────────────────────────────
SHARED_PORT="${HELDOUT_VLLM_PORT:-36217}"  # different from agent_run's 36216
SHARED_NAME="heldout-target"
SHARED_API_KEY="inspectai"
SHARED_URL="http://localhost:${SHARED_PORT}/v1"
TEMPLATES_DIR="$REPO_ROOT/src/eval/templates"
CHAT_TEMPLATE="$TEMPLATES_DIR/qwen3.jinja"
VLLM_LOG="$HELDOUT_DIR/_shared_vllm.log"

start_shared_vllm() {
    echo "[heldout] starting shared vllm at :${SHARED_PORT} for $ABS_MODEL"
    pkill -f "vllm serve.*${SHARED_PORT}" 2>/dev/null || true
    sleep 2
    setsid nohup bash -c "
        export HF_HOME=\"\${HF_HOME:-/workspace/hf-cache}\"
        vllm serve \"$ABS_MODEL\" \
            --host 0.0.0.0 --port $SHARED_PORT \
            --api-key $SHARED_API_KEY \
            --served-model-name $SHARED_NAME \
            --gpu-memory-utilization 0.85 \
            --chat-template $CHAT_TEMPLATE
    " > "$VLLM_LOG" 2>&1 < /dev/null &
    disown
    echo "[heldout] waiting for vllm to be ready..."
    for i in $(seq 1 60); do
        if curl -fsS -m 3 -H "Authorization: Bearer $SHARED_API_KEY" \
            "$SHARED_URL/models" 2>/dev/null | grep -q "$SHARED_NAME"; then
            echo "[heldout] vllm ready"
            return 0
        fi
        sleep 5
    done
    echo "[heldout] vllm did not become ready in 5min; tail of log:" >&2
    tail -50 "$VLLM_LOG" >&2 || true
    return 1
}

stop_shared_vllm() {
    echo "[heldout] stopping shared vllm"
    pkill -f "vllm serve.*${SHARED_PORT}" 2>/dev/null || true
}

# ─── Run ───────────────────────────────────────────────────────────────────
USE_SHARED=1
if [[ "${HELDOUT_NO_SHARED_VLLM:-0}" == "1" ]]; then
    USE_SHARED=0
fi

if [[ $USE_SHARED -eq 1 ]]; then
    if ! start_shared_vllm; then
        echo "[heldout] WARNING: shared vllm failed; falling back to per-task local vllm" >&2
        USE_SHARED=0
    fi
fi
trap 'stop_shared_vllm' EXIT

VLLM_FLAGS=""
if [[ $USE_SHARED -eq 1 ]]; then
    VLLM_FLAGS="--vllm-base-url $SHARED_URL --vllm-served-name $SHARED_NAME"
fi

failed_tasks=()
for task_dir in "$TASKS_DIR"/*/; do
    task_name="$(basename "$task_dir")"

    if [[ ${#SELECTED_TASKS[@]} -gt 0 ]]; then
        keep=0
        for sel in "${SELECTED_TASKS[@]}"; do
            if [[ "$sel" == "$task_name" ]]; then keep=1; break; fi
        done
        [[ $keep -eq 0 ]] && continue
    fi

    if [[ ! -f "$task_dir/evaluate.py" ]]; then
        continue
    fi

    out_json="$HELDOUT_DIR/${task_name}.json"
    log_file="$HELDOUT_DIR/${task_name}.log"

    echo ""
    echo "[heldout] === $task_name ==="
    if python "$task_dir/evaluate.py" \
        --model-path "$ABS_MODEL" \
        --json-output-file "$out_json" \
        $VLLM_FLAGS \
        > "$log_file" 2>&1; then
        echo "[heldout] $task_name OK -> $out_json"
    else
        echo "[heldout] $task_name FAILED (see $log_file)" >&2
        failed_tasks+=("$task_name")
    fi
done

echo ""
echo "[heldout] aggregating..."
python "$REPO_ROOT/src/heldout_evals/aggregate.py" "$HELDOUT_DIR"

if [[ ${#failed_tasks[@]} -gt 0 ]]; then
    echo ""
    echo "[heldout] WARNING: ${#failed_tasks[@]} task(s) failed: ${failed_tasks[*]}" >&2
    exit 1
fi
echo "[heldout] done."
