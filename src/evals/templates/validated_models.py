"""Loader + check helper for the validated-models manifest.

Imported by submit_run.py and submit_baseline.py to fail-fast on a
--student / --model that hasn't been hand-validated against a
lora_starter chat-template helper.

The manifest itself is JSON (validated_models.json) so the script
`scripts/validate_chat_templates.py --commit` can mutate it without
parsing Python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "validated_models.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def is_validated(model_id: str) -> bool:
    """True iff `model_id` has an entry in the manifest's `models` dict."""
    try:
        return model_id in load_manifest().get("models", {})
    except (OSError, json.JSONDecodeError):
        return False


def assert_validated(model_id: str, bypass: bool = False) -> None:
    """Raise SystemExit with a clear message if the model hasn't been
    validated. `bypass=True` lets the caller (e.g. submit_run.py with
    `--bypass-template-check`) skip the gate explicitly."""
    if bypass:
        print(
            f"[validated_models] WARNING: --bypass-template-check set; "
            f"submitting {model_id!r} without prior chat-template validation. "
            "Failures from chat-template mismatches will surface as bad eval "
            "scores, not crashes. You assume the risk.",
            file=sys.stderr,
        )
        return
    if is_validated(model_id):
        return
    manifest = load_manifest()
    known = sorted(manifest.get("models", {}).keys())
    msg = (
        f"\nERROR: model {model_id!r} has not been validated against any "
        "chat-template helper in lora_starter.py.\n\n"
        "Before submitting an experiment with this student model, run:\n\n"
        f"  uv run --no-project --python 3.12 --with 'transformers>=4.46,<5.0' \\\n"
        f"      --with 'huggingface_hub<1.0' --with 'jinja2' \\\n"
        f"      python scripts/validate_chat_templates.py --model {model_id} --commit\n\n"
        "The validator confirms `format_chat()` produces a sensible "
        "(prompt, completion) pair for both enable_thinking=True/False. "
        "If the inferred family is wrong, add a helper to lora_starter.py + "
        "register it in CHAT_FORMATTERS + _FAMILY_HINTS first.\n\n"
        f"Currently validated:\n  - " + "\n  - ".join(known) + "\n\n"
        "To submit anyway without validation (NOT recommended for real "
        "experiments — debugging only), pass --bypass-template-check.\n"
    )
    raise SystemExit(msg)
