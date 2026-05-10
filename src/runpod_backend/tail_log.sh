#!/bin/bash
# Tail the live run.log of a self-driving experiment via SSH.
#
# Usage: bash src/runpod_backend/tail_log.sh <run_id>
#
# Finds the running pod whose name encodes the run_id, opens an
# interactive SSH session, and runs `tail -f` on
# /workspace/runs/<run_id>/run.log. Ctrl-C at any time — the pod
# keeps running.

set -euo pipefail

RUN_ID="${1:?usage: tail_log.sh <run_id>}"

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    if [ -f .env ]; then
        # shellcheck disable=SC1091
        set -a; source .env; set +a
    fi
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set in env or .env}"

POD_INFO=$(curl -s -X POST https://api.runpod.io/graphql \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"query":"query { myself { pods { id name desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } } }"}' \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)['data']['myself']['pods']
target = '$RUN_ID'.lower()
for p in data:
    if p['desiredStatus'] != 'RUNNING':
        continue
    if target not in (p['name'] or '').lower():
        continue
    rt = p.get('runtime') or {}
    for port in (rt.get('ports') or []):
        if port['privatePort'] == 22 and port['isIpPublic']:
            print(p['id'], port['ip'], port['publicPort'])
            sys.exit(0)
print('', '', '')
")

POD_ID=$(echo "$POD_INFO" | awk '{print $1}')
SSH_HOST=$(echo "$POD_INFO" | awk '{print $2}')
SSH_PORT=$(echo "$POD_INFO" | awk '{print $3}')

if [ -z "$POD_ID" ]; then
    echo "No RUNNING pod matching run_id=$RUN_ID. Either:" >&2
    echo "  - the pod has already self-terminated (run finished)" >&2
    echo "  - try: PYTHONPATH=. python src/runpod_backend/status_run.py $RUN_ID" >&2
    exit 1
fi

echo "Pod $POD_ID at $SSH_HOST:$SSH_PORT — tailing /workspace/runs/$RUN_ID/run.log"
echo "Ctrl-C exits the tail; pod keeps running."
echo "---"

exec ssh -t \
    -i "$HOME/.runpod/ssh/RunPod-Key-Go" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ServerAliveInterval=15 \
    -p "$SSH_PORT" \
    "root@$SSH_HOST" \
    "tail -F /workspace/runs/$RUN_ID/run.log 2>/dev/null || tail -f /workspace/runs/$RUN_ID/run.log"
