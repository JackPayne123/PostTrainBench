#!/usr/bin/env bash
# Install repo-managed git hooks. Currently:
#   pre-commit → scripts/githook-pre-commit (scrubs leaked secrets
#     from staged transcript files and aborts the commit if any leak
#     was detected, with patterns redacted in place — developer
#     re-stages and re-commits).
#
# Idempotent. Run once per fresh clone:
#   bash scripts/install-hooks.sh
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"
SOURCE="$REPO_ROOT/scripts/githook-pre-commit"
TARGET="$HOOKS_DIR/pre-commit"

mkdir -p "$HOOKS_DIR"
chmod +x "$SOURCE"

if [ -L "$TARGET" ]; then
    rm -f "$TARGET"
elif [ -f "$TARGET" ]; then
    echo "warn: $TARGET exists and is not a symlink; backing up to $TARGET.bak" >&2
    mv "$TARGET" "$TARGET.bak"
fi

ln -s "$SOURCE" "$TARGET"
chmod +x "$TARGET"

echo "installed: $TARGET → $SOURCE"
echo "to bypass on a single commit: git commit --no-verify"
