#!/usr/bin/env python3
"""Redact leaked secrets from agent transcripts in jobs/runs/.

Works in two modes:

  1. Default (no args): scan-and-redact every transcript under jobs/runs/.
     Intended as a one-off cleanup or as a filter-branch --tree-filter
     across history. Returns 0 on completion regardless of whether
     anything was redacted.

  2. `--check <paths...>`: scan ONLY the listed paths. Returns 2 if any
     leak was found (and redacts in place). Intended as a pre-commit
     hook over the staged paths: the non-zero return aborts the commit
     and tells the developer to re-stage.

Token shapes covered (be generous — over-scrubbing here is fine):
  hf_XXXXXX           Hugging Face user-access token
  sk-proj-XXXXXX      OpenAI (project) API key
  sk-ant-api##-XXXXXX Anthropic API key
  sk-XXXXXX           Any other sk-prefixed key (legacy OpenAI etc.)
  gho_XXXXXX          GitHub OAuth token
  rpa_XXXXXX          RunPod API key
  ghp_XXXXXX          GitHub personal access token

History of this script:
  2026-05-11 — added after a force-with-lease push got rejected on three
  leaked HF + OpenAI secrets in E + F run transcripts. Scrubbed via
  `git filter-branch --tree-filter`. Pairs with the pre-commit hook at
  scripts/githook-pre-commit so the same patterns can't slip into a
  future commit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bhf_[A-Za-z0-9]{20,50}\b"), "hf_REDACTED"),
    (re.compile(r"\bsk-ant-api\d+-[A-Za-z0-9_-]{40,}\b"), "sk-ant-REDACTED"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9_-]{40,}\b"), "sk-proj-REDACTED"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{30,}\b"), "sk-REDACTED"),
    (re.compile(r"\bgho_[A-Za-z0-9]{30,}\b"), "gho_REDACTED"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"), "ghp_REDACTED"),
    (re.compile(r"\brpa_[A-Za-z0-9]{30,}\b"), "rpa_REDACTED"),
]

# File-name allowlist for the bulk pass. The pre-commit hook can scan
# any path the developer passes; the bulk pass only touches files we've
# actually seen leak from.
BULK_TARGET_NAMES = {"solve_out.jsonl", "solve_parsed.txt", "run.log"}


def scrub_file(path: Path) -> int:
    """Redact the file in place. Returns the count of replacements."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return 0
    n_total = 0
    for pat, repl in PATTERNS:
        new, n = pat.subn(repl, text)
        if n:
            text = new
            n_total += n
    if n_total:
        path.write_text(text, encoding="utf-8")
    return n_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", nargs="+", metavar="PATH",
        help="Pre-commit mode. Scan only these paths; redact in place + "
             "exit 2 if anything was redacted (signals the developer to "
             "re-stage). Default mode walks jobs/runs/ for known leaky "
             "filenames.",
    )
    args = ap.parse_args()

    if args.check:
        total = 0
        for raw in args.check:
            p = Path(raw)
            if not p.is_file():
                continue
            n = scrub_file(p)
            if n:
                total += n
                print(f"REDACTED {n} secret(s) in {p}", file=sys.stderr)
        if total:
            print(
                f"\n{total} secret(s) redacted across staged files. "
                f"Re-stage and commit again:\n  git add -- {' '.join(args.check)}",
                file=sys.stderr,
            )
            return 2
        return 0

    root = Path("jobs/runs")
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file() and p.name in BULK_TARGET_NAMES:
            n = scrub_file(p)
            if n:
                total += n
                print(f"scrubbed {n} in {p}", file=sys.stderr)
    if total:
        print(f"total scrubbed: {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
