#!/usr/bin/env bash
# Run the held-out eval panel against a trained checkpoint.
#
# Usage:
#   bash src/heldout_evals/run_heldout.sh <run_dir>
#
# <run_dir> must contain a `final_model/` subdirectory. Output is written
# to <run_dir>/heldout/<task>.json plus a final summary.{json,md}.
#
# Tasks are iterated in alphabetical order under src/heldout_evals/tasks/.
# Each task's evaluate.py spins up its own vLLM model (Inspect AI handles
# this; tasks share the model_path string and Inspect caches by path).

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

failed_tasks=()
for task_dir in "$TASKS_DIR"/*/; do
    task_name="$(basename "$task_dir")"

    # skip if task selection list provided and this task isn't in it
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
