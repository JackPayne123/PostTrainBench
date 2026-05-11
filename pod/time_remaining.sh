#!/bin/bash
# Pipeline-side time-remaining query. Invoked by the agent's
# /opt/pipeline-bin/time-remaining symlink via `sudo -n` (NOPASSWD entry
# in /etc/sudoers.d/agent-score, same file as score_runner).
#
# Why root: deadline lives at /etc/ptb_run/deadline (chmod 600 root:root)
# so the agent can't observe or tamper with it. Pipeline writes the
# deadline before invoking the agent.
#
# Output: integer seconds remaining, on stdout. Negative if expired.
# Single line, parseable. No JSON — the agent's stdout-as-prompt cap
# means we want a number, not a wrapper.

set -u

DEADLINE_FILE="/etc/ptb_run/deadline"
if [ ! -r "$DEADLINE_FILE" ]; then
    echo "ERROR: $DEADLINE_FILE not present or unreadable" >&2
    exit 2
fi

DEADLINE_EPOCH=$(cat "$DEADLINE_FILE")
if ! [[ "$DEADLINE_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $DEADLINE_FILE not an integer epoch" >&2
    exit 3
fi

NOW=$(date +%s)
REMAINING=$((DEADLINE_EPOCH - NOW))
echo "$REMAINING"
