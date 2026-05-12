#!/usr/bin/env python3
"""MMLU eval, mirrors src/eval/tasks/gsm8k/evaluate.py.

Uses inspect_evals.mmlu's 0-shot dataset + solver but swaps the strict
`choice()` scorer for a lenient ANSWER-prefix parser (mirrors the same
fix big_five got on :23).

Lenient scorer (2026-05-13)
---------------------------
Upstream `mmlu_0_shot` ships the strict `choice()` scorer, which only
accepts outputs matching `ANSWER: <letter>`. Caught on the 2026-05-13
Qwen3.5-9B baseline run #cac466: 121/200 samples were rejected as
invalid ("value='I'") because the model emitted free-form rationales
or letter-prefix variants instead of the literal ANSWER: format. Reported
accuracy dropped to 0.395 (target +0.65 for 9B).

The lenient parser accepts:
  * `ANSWER: X` (existing format)
  * `X)` / `X.` / `X:` at start of a line
  * `X` on its own line

If the model didn't commit to one of A/B/C/D within the first token of
a line it still registers as invalid — same conservative semantics as
the big_five fix.
"""
from __future__ import annotations
import os

import argparse
import json
import re

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.mmlu  # noqa: F401, E402  (registers task definitions)


_LENIENT_PATTERNS = (
    re.compile(r"ANSWER\s*:\s*([A-Da-d])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*([A-Da-d])\s*[\)\.:]", re.MULTILINE),
    re.compile(r"^\s*([A-Da-d])\s*$", re.MULTILINE),
)


def _parse_answer_lenient(text: str) -> str | None:
    """Try each pattern; return first matched letter (uppercased), or None."""
    for pat in _LENIENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def _build_lenient_task(language: str = "EN_US", max_non_cot_tokens: int = 1024):
    """Build mmlu_0_shot Task with a lenient ANSWER-prefix scorer + bigger
    max_non_cot_tokens budget.

    Imports lazily so `--help` works without inspect_evals installed.
    The dataset + solver come from the upstream `inspect_evals.mmlu`
    helpers; only the scorer is swapped + token budget raised. Task
    instance is passed directly to `inspect_ai.eval` (not the
    registry-string path) because a script-local `@task`-decorated
    function doesn't register under the upstream namespace.

    Why max_non_cot_tokens=256 (vs upstream default GPT_5_MIN_TOKENS=16):
    Qwen3.5-9B (and most modern instruct models trained on CoT data)
    tends to emit a brief rationale BEFORE the letter even when prompted
    for cot=False. At 16 tokens the response gets truncated mid-thought
    ("The reaction described is the dehydrohalogenation of 2-bromob...")
    and no answer letter is ever emitted. Diagnosed on 2026-05-13 9B
    baseline #cac466: 92/200 samples had truncated rationales with zero
    extractable letter. Bumping to 256 gives the model enough headroom
    to think + emit the answer letter; lenient scorer then picks it up.
    Adds ~5-10s/sample but raises useful sample count from ~80% to
    ~99%.
    """
    from inspect_ai import Task
    from inspect_ai.model import GenerateConfig
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState
    from inspect_evals.mmlu.mmlu import (
        EVAL_VERSION,
        get_mmlu_dataset,
        get_mmmlu_dataset,
        mmlu_multiple_choice,
    )

    @scorer(metrics=[accuracy(), stderr()])
    def any_choice_lenient():
        async def score(state: TaskState, target: Target) -> Score:
            text = state.output.completion
            letter = _parse_answer_lenient(text)
            if letter is not None and letter in target.text:
                return Score(value=CORRECT, answer=letter, explanation=text)
            return Score(value=INCORRECT, answer=letter, explanation=text)
        return score

    if language == "EN_US":
        dataset = get_mmlu_dataset("test", shuffle=True, subjects=[])
    else:
        dataset = get_mmmlu_dataset("test", shuffle=True, language=language, subjects=[])

    return Task(
        dataset=dataset,
        solver=mmlu_multiple_choice(cot=False, max_non_cot_tokens=max_non_cot_tokens),
        scorer=any_choice_lenient(),
        config=GenerateConfig(temperature=0.0),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI MMLU eval.")
    parser.add_argument("--model-path", type=str, default="final_model")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument("--templates-dir", type=str, default="templates/")
    parser.add_argument("--max-connections", type=int, default=8)
    # MCQ scored on choice logprob; 256 is plenty for any answer letter.
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--vllm-base-url", type=str, default=None)
    parser.add_argument("--vllm-served-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_display_type("plain")

    other_kwargs = {}
    if args.limit is not None and args.limit != -1:
        other_kwargs["limit"] = args.limit

    # Lenient-scorer Task instance (not registry-string).
    # See _build_lenient_task for the rationale.
    task = _build_lenient_task()
    if args.vllm_base_url and args.vllm_served_name:
        model = f"openai-api/local/{args.vllm_served_name}"
        model_args = {"api_key": "inspectai"}
    else:
        model = f"vllm/{args.model_path}"
        model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
        model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=model,
        model_base_url=args.vllm_base_url if (args.vllm_base_url and args.vllm_served_name) else None,
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        **other_kwargs,
    )

    if args.json_output_file is not None:
        assert len(eval_out) == 1, eval_out
        assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
        metrics = {k: v.value for k, v in eval_out[0].results.scores[0].metrics.items()}
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)


def model_type(args) -> str:
    if "qwen" in args.model_path.lower(): return "qwen"
    if "llama" in args.model_path.lower(): return "llama"
    if "gemma" in args.model_path.lower(): return "gemma"
    if "smollm" in args.model_path.lower(): return "smollm"
    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        config = json.load(f)
    arch = config["architectures"][0].lower()
    for k in ("gemma", "llama", "qwen", "smollm"):
        if k in arch: return k
    raise ValueError(arch)


def template_kwargs(args) -> dict:
    t = model_type(args)
    return {"chat_template": os.path.join(
        args.templates_dir,
        {"qwen": "qwen3.jinja", "llama": "llama3.jinja",
         "gemma": "gemma3.jinja", "smollm": "smollm.jinja"}[t]
    )}


if __name__ == "__main__":
    main()
