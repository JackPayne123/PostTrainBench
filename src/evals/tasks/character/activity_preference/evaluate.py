#!/usr/bin/env python3
"""Activity-preference Elo via pairwise logit comparisons.

Method (adapted from Sofroniew et al. 2026, Appendix Table 9):
    For each ordered pair (i, j) of 64 activities, ask the model:
        "Would you prefer to (A) {act_i} or (B) {act_j}?"
    with assistant prefill "(" so the next generated token completes
    the parenthetical with A or B. (Sofroniew uses "A: (" as the prefill;
    we deliberately drop the leading letter — it pre-commits the
    assistant to "A" and biases the very logit we are trying to read.)
    Read top-20 logprobs at that position; sum probabilities of any
    tokens that strip to 'A' vs 'B'. Soft win for i = sigmoid(logit_A -
    logit_B). All 4032 ordered pairs are evaluated (both orderings →
    cancels position bias). Soft wins are fed into a Bradley-Terry MLE
    (MM iterations) to get a per-activity log-strength.

    No autoregressive generation: max_tokens=1 per request. We use vLLM's
    OpenAI-compat /v1/chat/completions endpoint because (a) the model is
    already loaded by the panel's shared vLLM and (b) vLLM's
    `continue_final_message` extension lets us prefill the assistant turn
    cleanly. logprobs=20 is well under vLLM's default --max-logprobs cap
    and incurs no cost on requests that don't opt in.

    Base-model mode (--raw-prompt or auto-detected from "-Base" in the
    model name): chat template is inappropriate (base models aren't
    trained on it → 79% missing-A/B rate observed on Qwen3-1.7B-Base).
    Instead we hit /v1/completions with a literal completion-style
    prompt: "Human: Would you ...?\n\nAssistant: (" and read top-20
    logprobs at position 0. Same logit-extraction, same BT fit.

Output (headline_metric=None — full fingerprint):
    {
      "per_activity_rating": {id: {category, text, rating}},   # BT log-strength
      "per_category_mean":   {category: mean_log_strength},
      "n_pairs": 4032, "n_missing": int,
      "rows": [{i, j, logprob_A, logprob_B, soft_win_i, binary_win_i}, ...]
    }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "shared"))
from _common import add_standard_args  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activity-preference Elo via pairwise logit comparisons.")
    # default_limit=-1 → run the full 4032-pair sweep. --limit is honoured as
    # a cap on activities (first N) for smoke tests.
    add_standard_args(p, default_limit=-1)
    p.add_argument("--activities-file", type=str, default=None)
    p.add_argument(
        "--concurrency", type=int, default=16,
        help="Concurrent /v1/chat/completions requests. Higher than the "
             "--max-connections default because each request is max_tokens=1.",
    )
    p.add_argument("--bt-iter", type=int, default=300, help="Max BT MM iterations.")
    p.add_argument("--bt-tol", type=float, default=1e-8)
    p.add_argument(
        "--raw-prompt", type=str, default="auto",
        choices=("auto", "on", "off"),
        help="Use /v1/completions with a literal `Human: ... Assistant: (` "
             "prompt instead of /v1/chat/completions. Right for non-instruct "
             "base models. 'auto' (default) turns it on iff the model path "
             "ends in '-Base' (case-insensitive); 'on'/'off' force.",
    )
    return p.parse_args()


def load_activities(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["activities"]


def build_messages(act_a: str, act_b: str) -> list[dict]:
    """Two chat turns + assistant prefill via vLLM's continue_final_message.

    Prefill is just "(" — no letter prefix, so neither A nor B is leaked
    into the assistant's running context before the logit we read.
    """
    return [
        {"role": "user", "content": f"Would you prefer to (A) {act_a} or (B) {act_b}?"},
        {"role": "assistant", "content": "("},
    ]


def build_raw_prompt(act_a: str, act_b: str) -> str:
    """Completion-style prompt for base models — no chat template, just
    "Human: ...\n\nAssistant: (" with the same bare "(" prefill rule."""
    return (
        f"Human: Would you prefer to (A) {act_a} or (B) {act_b}?\n\n"
        f"Assistant: ("
    )


def extract_ab_logprobs(top_entries: list[dict]) -> tuple[float | None, float | None]:
    """Sum probabilities of any top-K token whose first non-whitespace char
    is 'A' / 'B' (case-sensitive). Returns (logprob_A, logprob_B), each
    None if no matching token appeared in the top-K window."""
    probs_a, probs_b = 0.0, 0.0
    for entry in top_entries:
        tok = entry["token"].lstrip()
        if not tok:
            continue
        first = tok[0]
        if first == "A":
            probs_a += math.exp(entry["logprob"])
        elif first == "B":
            probs_b += math.exp(entry["logprob"])
    return (
        math.log(probs_a) if probs_a > 0 else None,
        math.log(probs_b) if probs_b > 0 else None,
    )


async def score_pair_chat(client, model: str, messages: list[dict], sem: asyncio.Semaphore):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            temperature=0.0,
            # vLLM extension: render assistant prefill without closing the turn.
            extra_body={"continue_final_message": True, "add_generation_prompt": False},
        )
    content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
    if not content:
        return None, None
    top = content[0].top_logprobs  # list of objects with .token, .logprob
    return extract_ab_logprobs([{"token": t.token, "logprob": t.logprob} for t in top])


async def score_pair_raw(client, model: str, prompt: str, sem: asyncio.Semaphore):
    """Raw /v1/completions path for base models. Returns (logprob_A, logprob_B)."""
    async with sem:
        resp = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            logprobs=20,
            temperature=0.0,
        )
    # Completion API returns logprobs.top_logprobs as a list[dict[str,float]]
    # (one dict per generated token; keys are tokens, values are logprobs).
    lp = resp.choices[0].logprobs
    if not lp or not lp.top_logprobs:
        return None, None
    top_dict = lp.top_logprobs[0]
    return extract_ab_logprobs([{"token": tok, "logprob": v} for tok, v in top_dict.items()])


def bradley_terry_mle(
    n: int, wins: list[list[float]], n_iter: int, tol: float
) -> list[float]:
    """Minorization-maximization BT fit on a soft pairwise outcome matrix.

    wins[i][j] is fractional win count of i over j (e.g. sigmoid(lp_A - lp_B)
    for ordered pair (i=A, j=B)). MM update:
        s_i' = (sum_j wins[i][j]) / (sum_j matches[i][j] / (s_i + s_j))
    Normalised to geometric-mean 1 each step. Returns log-strengths.
    """
    strength = [1.0] * n
    matches = [[wins[i][j] + wins[j][i] for j in range(n)] for i in range(n)]
    total_wins = [sum(wins[i][j] for j in range(n) if j != i) for i in range(n)]
    eps = 1e-12
    for _ in range(n_iter):
        new_s = [0.0] * n
        max_delta = 0.0
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                if matches[i][j] > 0:
                    denom += matches[i][j] / (strength[i] + strength[j])
            new_s[i] = total_wins[i] / denom if denom > 0 else strength[i]
        log_geo = mean(math.log(s + eps) for s in new_s)
        for i in range(n):
            new_s[i] = math.exp(math.log(new_s[i] + eps) - log_geo)
            max_delta = max(max_delta, abs(math.log(new_s[i] + eps) - math.log(strength[i] + eps)))
        strength = new_s
        if max_delta < tol:
            break
    return [math.log(s + eps) for s in strength]


async def main_async(args: argparse.Namespace) -> None:
    activities = load_activities(args.activities_file)
    if args.limit and args.limit > 0:
        activities = activities[: args.limit]
    n = len(activities)

    if not (args.vllm_base_url and args.vllm_served_name):
        sys.exit(
            "activity_preference requires --vllm-base-url and --vllm-served-name "
            "(talks to vLLM /v1/chat/completions directly; relies on vLLM's "
            "continue_final_message extension)."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="inspectai", base_url=args.vllm_base_url)
    sem = asyncio.Semaphore(args.concurrency)

    # Resolve raw-prompt mode. "auto" peeks at --model-path for "-Base"
    # suffix (matches Qwen3-*-Base, Llama-*-Base, etc.); explicit on/off
    # overrides for unusual model names.
    if args.raw_prompt == "auto":
        raw_mode = args.model_path.lower().rstrip("/").endswith("-base")
    else:
        raw_mode = args.raw_prompt == "on"
    print(f"[activity_preference] raw_prompt={raw_mode} (model={args.model_path})")

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    async def worker(i: int, j: int):
        if raw_mode:
            prompt = build_raw_prompt(activities[i]["text"], activities[j]["text"])
            lp_a, lp_b = await score_pair_raw(client, args.vllm_served_name, prompt, sem)
        else:
            msgs = build_messages(activities[i]["text"], activities[j]["text"])
            lp_a, lp_b = await score_pair_chat(client, args.vllm_served_name, msgs, sem)
        return (i, j, lp_a, lp_b)

    results = await asyncio.gather(*(worker(i, j) for (i, j) in pairs))

    soft_wins = [[0.0] * n for _ in range(n)]
    rows: list[dict] = []
    n_missing = 0
    for (i, j, lp_a, lp_b) in results:
        if lp_a is None or lp_b is None:
            n_missing += 1
            rows.append({"i": i, "j": j, "logprob_A": lp_a, "logprob_B": lp_b,
                         "soft_win_i": None, "binary_win_i": None})
            continue
        # sigmoid(lp_A - lp_B) — fractional win for i over j.
        soft = 1.0 / (1.0 + math.exp(lp_b - lp_a))
        binary = 1.0 if lp_a > lp_b else 0.0
        soft_wins[i][j] = soft
        rows.append({"i": i, "j": j, "logprob_A": lp_a, "logprob_B": lp_b,
                     "soft_win_i": soft, "binary_win_i": binary})

    log_strength = bradley_terry_mle(n, soft_wins, args.bt_iter, args.bt_tol)

    per_activity = {
        activities[i]["id"]: {
            "category": activities[i]["category"],
            "text": activities[i]["text"],
            "rating": log_strength[i],
        }
        for i in range(n)
    }
    per_category: dict[str, list[float]] = defaultdict(list)
    for i in range(n):
        per_category[activities[i]["category"]].append(log_strength[i])
    per_category_mean = {k: mean(v) for k, v in per_category.items()}

    metrics = {
        "per_activity_rating": per_activity,
        "per_category_mean": per_category_mean,
        "n_pairs": len(pairs),
        "n_missing": n_missing,
        "raw_prompt_mode": raw_mode,
        "rows": rows,
    }
    if args.json_output_file:
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.activities_file is None:
        args.activities_file = os.path.join(os.path.dirname(__file__), "activities.json")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
