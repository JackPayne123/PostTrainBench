#!/bin/bash
# v0 smoke test launcher for the RunPod backend.
#
# Creates a 3090 pod in EU-CZ-1, attaches jack-pilot-cz volume, runs claude-code
# for ~10 min on Qwen3-1.7B-Base GSM8K task, then verifier. Tears down pod.
#
# Required env vars:
#   RUNPOD_API_KEY             — for pod lifecycle
#   CLAUDE_CODE_OAUTH_TOKEN    — for Claude Max auth (preferred per plan)
#                                 OR ANTHROPIC_API_KEY for API auth
#   HF_TOKEN                   — optional, for base model download
#   OPENAI_API_KEY             — optional, for contamination judge (skip in smoke)
#
# Pre-reqs:
#   1. harbor installed: `uv tool install harbor --python 3.13`
#   2. From repo root, task generated: `uv run python src/harbor_adapter/run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./harbor_tasks --num-hours 1`
#   3. RUNPOD_API_KEY env var set
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: RUNPOD_API_KEY not set"
    exit 1
fi
if [ -z "$CLAUDE_CODE_OAUTH_TOKEN" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: need CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY"
    exit 1
fi
if [ ! -d "harbor_tasks/posttrainbench-gsm8k-qwen3-1.7b" ]; then
    echo "ERROR: task dir not found. Generate it first:"
    echo "  uv run python src/harbor_adapter/run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./harbor_tasks --num-hours 1"
    exit 1
fi

echo "[smoke] launching with PYTHONPATH=$REPO_ROOT"
echo "[smoke] auth: $([ -n "$CLAUDE_CODE_OAUTH_TOKEN" ] && echo OAuth || echo API)"
echo "[smoke] config: src/runpod_backend/smoke_test_config.yaml"
echo

PYTHONPATH="$REPO_ROOT" harbor run \
    --config src/runpod_backend/smoke_test_config.yaml \
    --debug
