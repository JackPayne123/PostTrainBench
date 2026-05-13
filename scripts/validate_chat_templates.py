#!/usr/bin/env python3
"""Hand-validate a per-family chat-template helper from `lora_starter.py`.

Usage:
    python3 scripts/validate_chat_templates.py --model Qwen/Qwen3-1.7B
    python3 scripts/validate_chat_templates.py --model google/gemma-3-1b-it
    python3 scripts/validate_chat_templates.py --model HuggingFaceTB/SmolLM3-3B
    python3 scripts/validate_chat_templates.py --all-registered

For each (registered family × thinking-mode True/False), the script:
  1. Loads the tokenizer via AutoTokenizer.from_pretrained.
  2. Infers the family via `_infer_family`. If a `--family` override is
     given, uses that instead.
  3. Calls the helper with a canonical 3-message sample (system + user +
     assistant).
  4. Asserts structural invariants:
       a. prompt is non-empty + str
       b. completion is non-empty + str
       c. prompt ends with the family's expected assistant-opener token
          (e.g. `<|im_start|>assistant\\n` for Qwen3, `<start_of_turn>model\\n`
          for Gemma3 — best-effort, prints a warning on mismatch)
       d. completion ends with the family's expected end-of-turn token
       e. if enable_thinking and family supports it: completion starts
          with `<think>\\n\\n</think>\\n\\n`
  5. Prints the assembled (prompt, completion) verbatim for human eyeball.
  6. Tokenizes the full text + reports the token count for the assistant
     span vs total (sanity for completion_only_loss masking).

Exit code 0 = all checks passed; 1 = one or more failed.

This is a STRUCTURAL test, not a behavioural one. It catches "we emit
the wrong tags" / "completion lacks the EOS" mistakes. It does NOT catch
"the model trained on this format produces weird outputs at eval time" —
for that, the only path is run a real eval against the trained adapter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import after sys.path so we can pull the helpers from the templates dir.
sys.path.insert(0, str(REPO_ROOT / "src" / "evals" / "templates"))
from lora_starter import (  # type: ignore[import-not-found]  # noqa: E402
    CHAT_FORMATTERS,
    _FAMILY_HINTS,
    _infer_family,
    format_chat,
)


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 = 4."},
]


# Per-family structural expectations. Best-effort — used for WARN, not
# hard fail, so a new family that handles separators differently still
# passes the script (the human review of the printed (prompt, completion)
# is the load-bearing check).
EXPECTATIONS: dict[str, dict[str, list[str]]] = {
    "Qwen3": {
        "prompt_ends_with_any": ["<|im_start|>assistant\n"],
        "completion_ends_with_any": ["<|im_end|>"],
        "completion_starts_with_thinking": ["<think>\n\n</think>\n\n"],
    },
    "Gemma3": {
        "prompt_ends_with_any": ["<start_of_turn>model\n", "<start_of_turn>assistant\n"],
        "completion_ends_with_any": ["<end_of_turn>", "<eos>"],
        "completion_starts_with_thinking": [],  # n/a
    },
    "SmolLM3": {
        "prompt_ends_with_any": ["<|im_start|>assistant\n", "assistant\n"],
        "completion_ends_with_any": ["<|im_end|>"],
        "completion_starts_with_thinking": [],
    },
}


def _check(label: str, ok: bool, detail: str = "") -> bool:
    sym = "✓" if ok else "✗"
    print(f"    {sym} {label}" + (f" — {detail}" if detail else ""))
    return ok


def validate_family(model_id: str, family: str | None = None) -> bool:
    """Validate the helper for `model_id`. Returns True on all-pass."""
    from transformers import AutoTokenizer

    print(f"\n=== {model_id} (family={family or 'infer'}) ===")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"  ! AutoTokenizer.from_pretrained failed: {exc}")
        return False

    inferred = _infer_family(tok)
    print(f"  inferred family: {inferred}")
    if family is not None and family != inferred:
        print(f"  (overridden to {family})")
    family = family or inferred
    if family is None:
        print("  ! no family inferred and none provided — add a hint to _FAMILY_HINTS")
        return False
    if family not in CHAT_FORMATTERS:
        print(f"  ! family {family!r} has no helper in CHAT_FORMATTERS")
        return False

    overall = True
    for enable_thinking in (True, False):
        print(f"\n  --- enable_thinking={enable_thinking} ---")
        try:
            prompt, completion = format_chat(
                SAMPLE_MESSAGES, tok,
                model_family=family,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            print(f"  ! formatter raised: {exc}")
            overall = False
            continue

        ok = True
        ok &= _check("prompt is non-empty str", isinstance(prompt, str) and len(prompt) > 0)
        ok &= _check(
            "completion is non-empty str",
            isinstance(completion, str) and len(completion) > 0,
        )

        exp = EXPECTATIONS.get(family, {})
        prompt_enders = exp.get("prompt_ends_with_any", [])
        if prompt_enders:
            match = any(prompt.rstrip("\n").endswith(e.rstrip("\n")) for e in prompt_enders) \
                    or any(prompt.endswith(e) for e in prompt_enders)
            ok &= _check(
                f"prompt ends with one of {prompt_enders}",
                match,
                detail=f"last 60 chars: {prompt[-60:]!r}",
            )

        completion_enders = exp.get("completion_ends_with_any", [])
        if completion_enders:
            match = any(completion.endswith(e) for e in completion_enders)
            ok &= _check(
                f"completion ends with one of {completion_enders}",
                match,
                detail=f"last 30 chars: {completion[-30:]!r}",
            )

        thinking_prefixes = exp.get("completion_starts_with_thinking", [])
        if enable_thinking and thinking_prefixes:
            match = any(completion.startswith(p) for p in thinking_prefixes)
            ok &= _check(
                f"completion starts with thinking wrapper",
                match,
                detail=f"first 30 chars: {completion[:30]!r}",
            )
        elif not enable_thinking and thinking_prefixes:
            match = not any(completion.startswith(p) for p in thinking_prefixes)
            ok &= _check(
                "completion does NOT start with thinking wrapper (enable_thinking=False)",
                match,
                detail=f"first 30 chars: {completion[:30]!r}",
            )

        # Tokenize + report mask coverage. SFTTrainer with
        # completion_only_loss=True masks the prompt span — completion
        # tokens are where loss accumulates. We just want the count to
        # be > 0 and < total.
        n_prompt = len(tok.encode(prompt, add_special_tokens=False))
        n_full = len(tok.encode(prompt + completion, add_special_tokens=False))
        n_completion = n_full - n_prompt
        ok &= _check(
            f"tokenized span split: prompt={n_prompt}, completion={n_completion}, total={n_full}",
            n_completion > 0 and n_prompt > 0,
        )

        print(f"\n  ─── prompt ({len(prompt)} chars) ───")
        print("  " + prompt.replace("\n", "\n  "))
        print(f"\n  ─── completion ({len(completion)} chars) ───")
        print("  " + completion.replace("\n", "\n  "))

        overall &= ok

    return overall


def _commit_to_manifest(model_id: str, family: str, notes: str = "") -> None:
    """Add `model_id` to validated_models.json. Idempotent — updates the
    existing entry if present. Stamps with today's date + current git_sha."""
    import datetime as dt
    import json
    import subprocess

    manifest_path = REPO_ROOT / "src" / "evals" / "templates" / "validated_models.json"
    manifest = json.loads(manifest_path.read_text())
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    manifest.setdefault("models", {})[model_id] = {
        "family": family,
        "validated_at": dt.date.today().isoformat(),
        "git_sha": sha,
        "notes": notes or "Validated via scripts/validate_chat_templates.py.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[commit] wrote {model_id} ({family}) to {manifest_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="HF model id (e.g. Qwen/Qwen3-1.7B)")
    ap.add_argument("--family", help="Override the inferred family (Qwen3 | Gemma3 | SmolLM3 | ...)")
    ap.add_argument(
        "--all-registered", action="store_true",
        help="Run the canonical sample model for each family registered in _FAMILY_HINTS.",
    )
    ap.add_argument(
        "--commit", action="store_true",
        help="On all-checks-pass, add the model to validated_models.json so "
             "submit_run.py / submit_baseline.py accept it. No-op on failure.",
    )
    ap.add_argument(
        "--notes", default="",
        help="Free-text notes to attach to the manifest entry (only with --commit).",
    )
    args = ap.parse_args()

    if not args.model and not args.all_registered:
        ap.error("either --model or --all-registered is required")

    if args.all_registered:
        # One canonical model per family. Operator can override with --model
        # if they want to validate a different checkpoint.
        canonical = {
            "Qwen3":   "Qwen/Qwen3-1.7B",
            "Gemma3":  "google/gemma-3-1b-it",
            "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
        }
        results = []
        for fam, mid in canonical.items():
            ok = validate_family(mid, family=fam)
            results.append((fam, mid, ok))
        print("\n=== summary ===")
        for fam, mid, ok in results:
            print(f"  {'✓' if ok else '✗'}  {fam:10s}  {mid}")
        return 0 if all(ok for _, _, ok in results) else 1

    ok = validate_family(args.model, family=args.family)
    if ok and args.commit:
        from transformers import AutoTokenizer
        family = args.family or _infer_family(AutoTokenizer.from_pretrained(args.model))
        if family is None:
            print("ERROR: cannot --commit without a resolvable family; pass --family.", file=sys.stderr)
            return 1
        _commit_to_manifest(args.model, family, notes=args.notes)
    elif args.commit and not ok:
        print("\n[commit] SKIPPED — validation failed; not updating manifest.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
