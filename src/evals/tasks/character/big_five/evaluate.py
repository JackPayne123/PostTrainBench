#!/usr/bin/env python3
"""Big Five / IPIP personality fingerprint via inspect_evals.personality.

HELDOUT spec calls for ~50 items × 2 framings (direct self-report and
behavioral proxy). The upstream `inspect_evals.personality` task supports
multiple framings; we limit to 50 (the BFI item bank) by default.

Lenient scorer (2026-05-11)
---------------------------
Upstream `personality_BFI` ships a strict `any_choice` scorer that only
accepts outputs of the literal form `ANSWER: <letter>`. On the 2026-05-11
F-run adapter we observed nine consecutive Agreeableness items where the
model emitted `C) Neither agree nor disagree.` or `A. (Disagree strongly) — ...`
— recognisable Likert answers, but missing the `ANSWER:` prefix — and all
nine got marked invalid, dropping the entire Agreeableness aggregate from
the metrics dict.

We build the BFI Task with a lenient scorer + pass the Task instance
directly to `inspect_ai.eval` rather than going through the
registry-string path. Reason: `@task`-decorated functions defined in
`__main__` (this script's invocation context) don't end up in the
`inspect_evals` registry namespace, so the string-based lookup fails
with "No inspect tasks were found at the specified paths". A direct
Task instance bypasses the registry entirely.

Parser accepts:
  * `ANSWER: X` (existing format)
  * `X)` / `X.` / `X:` at start of a line
  * `X` on its own line

Does NOT add tolerant parsing for full prose — if the model didn't
commit to one of A/B/C/D/E within the first token of a line it should
still register as invalid.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _common import add_standard_args, template_kwargs, write_metrics  # type: ignore


_PATTERNS = (
    re.compile(r"ANSWER\s*:\s*([A-Za-z])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*([A-Ea-e])\s*[\)\.:]", re.MULTILINE),
    re.compile(r"^\s*([A-Ea-e])\s*$", re.MULTILINE),
)


def _parse_answer_lenient(text: str) -> str | None:
    """Try each pattern; return first matched letter (uppercased), or None."""
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def _build_lenient_task():
    """Build the BFI Task with a lenient scorer.

    Imports `inspect_ai` + `inspect_evals` lazily so that `--help` works
    without those packages installed (matches `_inspect_wrap`'s pattern).
    """
    from inspect_ai import Task
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, scorer
    from inspect_ai.solver import TaskState, multiple_choice, system_message
    from inspect_evals.personality.personality import (
        enrich_dataset,
        load_dataset,
        trait_ratio,
        EVAL_VERSION,
    )
    from inspect_evals.personality.prompts.system import get_system_prompt

    @scorer(metrics=[trait_ratio()])
    def any_choice_lenient():
        async def score(state: TaskState, target: Target) -> Score:
            text = state.output.completion
            letter = _parse_answer_lenient(text)
            if letter is not None and letter in target.text:
                return Score(value=CORRECT, answer=letter, explanation=text)
            return Score(value=INCORRECT, answer=letter, explanation=text)
        return score

    system_msg = get_system_prompt("bfi", "")
    questions = load_dataset("bfi")
    meta = {"answer_mapping": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}}
    questions_meta = enrich_dataset(questions, meta)
    return Task(
        dataset=questions_meta,
        solver=[system_message(system_msg), multiple_choice()],
        scorer=any_choice_lenient(),
        metrics=[trait_ratio()],
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="BFI Big Five with a lenient ANSWER-prefix parser."
    )
    add_standard_args(parser, default_limit=50)
    args = parser.parse_args()

    task = _build_lenient_task()

    from inspect_ai import eval as inspect_eval
    from inspect_ai.util._display import init_display_type

    init_display_type("plain")

    if args.vllm_base_url and args.vllm_served_name:
        model = f"openai-api/local/{args.vllm_served_name}"
        model_args = {"api_key": "inspectai"}
        model_base_url = args.vllm_base_url
    else:
        model = f"vllm/{args.model_path}"
        model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
        model_args.update(template_kwargs(args))
        model_base_url = None

    kwargs: dict = {}
    if args.limit is not None and args.limit != -1:
        kwargs["limit"] = args.limit

    eval_out = inspect_eval(
        task,
        model=model,
        model_base_url=model_base_url,
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        **kwargs,
    )

    if args.json_output_file is not None:
        write_metrics(eval_out, args.json_output_file)


if __name__ == "__main__":
    main()
