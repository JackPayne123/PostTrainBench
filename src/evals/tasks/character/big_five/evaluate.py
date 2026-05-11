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

We register `personality_BFI_lenient` as a local @task here:
  * identical dataset + system message as upstream
  * scorer accepts the existing `ANSWER: X` shape AND the bare-letter
    shapes `X)`, `X.`, `X:`, or `X` at the start of a line
  * still preserves `trait_ratio` metric semantics by setting
    `Score.answer = <letter>` so trait_ratio's per-trait aggregator
    keeps working unchanged

We deliberately do NOT add tolerant parsing for full prose / multi-letter
extractions — anything where the model didn't commit to one of A/B/C/D/E
within the first non-whitespace token of a line should still register as
invalid.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval


_PATTERNS = (
    re.compile(r"ANSWER\s*:\s*([A-Za-z])\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*([A-Ea-e])\s*[\)\.:]", re.MULTILINE),
    re.compile(r"^\s*([A-Ea-e])\s*$", re.MULTILINE),
)


def _parse_answer_lenient(text: str) -> str | None:
    """Try each pattern; return the first matched letter (uppercased), or None.

    Returning the letter uppercased is required so the trait_ratio
    metric's answer_mapping lookup (`{'A': 1, 'B': 2, ...}`) hits."""
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def _register_lenient_task() -> str:
    """Register `inspect_evals/personality_BFI_lenient` at import time.

    Mirrors upstream `personality_BFI` but swaps in the lenient scorer.
    Returns the registered task id so the caller can pass it to
    `run_inspect_eval`.
    """
    from inspect_ai import Task, task
    from inspect_ai.scorer import (
        CORRECT,
        INCORRECT,
        Score,
        Scorer,
        Target,
        scorer,
    )
    from inspect_ai.solver import TaskState, multiple_choice, system_message
    from inspect_evals.personality.personality import (
        enrich_dataset,
        load_dataset,
        trait_ratio,
        EVAL_VERSION,
    )
    from inspect_evals.personality.prompts.system import get_system_prompt

    @scorer(metrics=[trait_ratio()])
    def any_choice_lenient() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            text = state.output.completion
            letter = _parse_answer_lenient(text)
            if letter is not None and letter in target.text:
                return Score(value=CORRECT, answer=letter, explanation=text)
            return Score(value=INCORRECT, answer=letter, explanation=text)

        return score

    @task
    def personality_BFI_lenient(personality: str = "") -> Task:
        """BFI with the lenient parser. Drop-in replacement for upstream `personality_BFI`."""
        system_msg = get_system_prompt("bfi", personality)
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

    return "inspect_evals/personality_BFI_lenient"


if __name__ == "__main__":
    task_id = _register_lenient_task()
    run_inspect_eval(
        task_id,
        default_limit=50,
        upstream_module="inspect_evals.personality",
    )
