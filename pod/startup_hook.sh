#!/bin/bash
# startup_hook.sh — invoked by the pod's container entrypoint after SSH
# is up. Polls /workspace/runs/$RUN_ID/START sentinel, then execs
# /opt/run_experiment.py inside a tmux session named "run" so an SSH
# user can `tmux attach -t run` for live console output.
#
# Bake this script at /opt/startup_hook.sh in the image. Runpod's
# default image entrypoint runs SSH; we add a one-line systemd
# service / nohup line that calls this script in the background.
#
# Env vars expected (injected by submit_run.py via Runpod GraphQL
# `env` field at podCreate):
#   RUN_ID                  — name of the experiment run dir
#   RUNPOD_POD_ID           — pod id, used to self-terminate
#   RUNPOD_API_KEY          — used to self-terminate
#   ANTHROPIC_API_KEY       — judges + agent
#   OPENAI_API_KEY          — judges (some)
#   HF_TOKEN                — gated dataset access
#   CLAUDE_CODE_OAUTH_TOKEN — claude-code agent auth
#   POD_KEEP_ALIVE          — "1" to skip self-terminate (debug)
#   POD_NO_DRIVE_UPLOAD     — "1" to skip rclone-to-Drive (debug)

set -u

LOG=/var/log/startup_hook.log
exec >>"$LOG" 2>&1

echo "[startup_hook] $(date -u +%FT%TZ) booted"

if [ -z "${RUN_ID:-}" ]; then
    echo "[startup_hook] ERROR: RUN_ID not set; idling"
    exit 0
fi

RUN_DIR="/workspace/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
START_SENTINEL="$RUN_DIR/START"
RUN_LOG="$RUN_DIR/run.log"

# Wait up to 1 hour for the laptop to upload config + drop START.
echo "[startup_hook] waiting for $START_SENTINEL ..."
for i in $(seq 1 720); do
    if [ -f "$START_SENTINEL" ]; then
        echo "[startup_hook] START sentinel found after ${i}*5s"
        break
    fi
    sleep 5
done

if [ ! -f "$START_SENTINEL" ]; then
    echo "[startup_hook] timed out waiting for START; idling"
    exit 0
fi

# Save oauth_token to the PTB convention path (claude-code reads it
# from there). The agent's solve.sh exports CLAUDE_CODE_OAUTH_TOKEN
# from the env, so this is belt+suspenders.
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    mkdir -p /home/ben
    echo "$CLAUDE_CODE_OAUTH_TOKEN" > /home/ben/oauth_token
    chmod 600 /home/ben/oauth_token
fi

# tmux session named "run" so `ssh + tmux attach -t run` shows live
# console. Detached so we don't hold the SSH channel.
echo "[startup_hook] launching run_experiment.py in tmux session 'run'"
tmux new-session -d -s run "cd /opt/ptb && PYTHONPATH=/opt/ptb python3 /opt/ptb/pod/run_experiment.py 2>&1 | tee -a $RUN_LOG"
echo "[startup_hook] launched. tail $RUN_LOG for live progress."
