#!/usr/bin/env python3
"""Persona-Vectors trait-expression eval (Chen et al. 2025).

Seven traits: evil, sycophantic, hallucinating, optimistic, impolite,
apathetic, humorous. Per-trait JSONs vendored from
github.com/safety-research/persona_vectors/data_generation/trait_data_eval/
contain 20 evaluation questions + a trait-specific judge prompt.

Pipeline:
    1. For each of the 7 traits × 20 questions, sample 10 rollouts from
       the target model at temperature 1.0 (default chat template, no
       system prompt). 1400 rollouts total.
    2. For each (question, rollout), apply the trait's `eval_prompt`
       (free-text, asks the judge to respond with REFUSAL or a number
       0-100) to Claude Haiku 4.5.
    3. Aggregate per trait: mean score, stdev, refusal rate, per-question
       mean. Full rollout × score breakdown kept under `rows`.

Notes:
    - Paper-faithful judge variant uses logprob-weighted scoring (top-20
      integer tokens). Anthropic does not expose logprobs through the
      messages API, so we use parse-the-integer instead. If we later add
      an OpenAI judge backend, the logprob variant slots in here.
    - headline_metric=None — multi-trait fingerprint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "shared"))
from _common import add_standard_args, model_for_inspect_eval  # noqa: E402
from _progress import ProgressTimer  # noqa: E402

TRAITS = [
    "evil",
    "sycophantic",
    "hallucinating",
    "humorous",
    "impolite",
    "optimistic",
    "apathetic",
]

JUDGE_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persona Vectors trait-expression eval.")
    add_standard_args(p, default_limit=-1)
    p.add_argument(
        "--rollouts", type=int, default=10,
        help="Rollouts per (trait, question). Paper uses 10 at T=1.0.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--judge-concurrency", type=int, default=16,
        help="Concurrent Haiku judge calls. Default 4 keeps total throughput "
             "below the typical 450k input-tokens/min Anthropic per-org cap "
             "(each judge request is ~2k input tokens). Bump if your account "
             "has a higher tier.",
    )
    p.add_argument(
        "--traits-dir", type=str, default=None,
        help="Override path to the vendored trait JSONs. Default: ./traits/",
    )
    return p.parse_args()


def load_trait(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_judge_score(text: str) -> tuple[int | None, bool]:
    """Return (score 0-100, refusal_flag). score is None if neither a valid
    number nor REFUSAL was parseable. The judge is instructed to respond
    with REFUSAL or a number — we tolerate stray prefix/suffix and clamp."""
    if not text:
        return None, False
    upper = text.upper()
    refusal = "REFUSAL" in upper
    # Pick the FIRST 0-100 integer that appears. Avoids matching e.g. years.
    for m in JUDGE_NUMBER_RE.finditer(text):
        try:
            v = int(m.group(1))
        except ValueError:
            continue
        if 0 <= v <= 100:
            return v, refusal
    return None, refusal


def main() -> None:
    args = parse_args()
    if args.traits_dir is None:
        args.traits_dir = os.path.join(os.path.dirname(__file__), "traits")

    trait_data: dict[str, dict] = {}
    for trait in TRAITS:
        path = os.path.join(args.traits_dir, f"{trait}.json")
        trait_data[trait] = load_trait(path)

    # ── Stage 1: generate rollouts via inspect_ai + vLLM ────────────────────
    from inspect_ai import Task, eval as inspect_eval, task as task_decorator
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate
    from inspect_ai.util._display import init_display_type

    init_display_type("plain")

    samples: list[Sample] = []
    sample_meta: dict[str, dict] = {}  # sample_id -> {trait, q_idx, q_text, rollout}
    for trait in TRAITS:
        questions = trait_data[trait]["questions"]
        if args.limit and args.limit > 0:
            questions = questions[: args.limit]
        for q_idx, q in enumerate(questions):
            for k in range(args.rollouts):
                sid = f"{trait}__q{q_idx:02d}__r{k:02d}"
                samples.append(Sample(input=q, id=sid))
                sample_meta[sid] = {"trait": trait, "q_idx": q_idx, "q_text": q, "rollout": k}

    @task_decorator
    def _trait_task() -> Task:
        return Task(dataset=samples, solver=[generate()])

    model, model_args, model_base_url = model_for_inspect_eval(args)
    eval_out = inspect_eval(
        _trait_task(),
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
        temperature=args.temperature,
    )
    log = eval_out[0]
    responses_by_id: dict[str, str] = {}
    for s in (log.samples or []):
        text = s.output.completion if (s.output and s.output.completion) else ""
        responses_by_id[str(s.id)] = text

    # ── Stage 2: Haiku judge ────────────────────────────────────────────────
    from anthropic import Anthropic
    judge_model = os.environ.get("PERSONA_TRAITS_JUDGE_MODEL", "claude-haiku-4-5")
    # max_retries=10 — the SDK respects 429 Retry-After headers with
    # exponential backoff, but the default of 2 retries gives up before
    # a 60s rate-limit window can clear. Caught on 2026-05-11 baseline
    # smoke: 1400 sequential judge calls at concurrency=8 tripped
    # Anthropic's 450k input-tokens/min cap and exhausted default retries
    # at sample ~1399/1400.
    client = Anthropic(max_retries=10)

    def judge_one(sid: str) -> dict:
        meta = sample_meta[sid]
        trait = meta["trait"]
        eval_prompt = trait_data[trait]["eval_prompt"]
        answer = responses_by_id.get(sid, "")
        prompt = eval_prompt.format(question=meta["q_text"], answer=answer)
        resp = client.messages.create(
            model=judge_model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        score, refusal = parse_judge_score(raw)
        return {
            "sample_id": sid,
            "trait": trait,
            "q_idx": meta["q_idx"],
            "q_text": meta["q_text"],
            "rollout": meta["rollout"],
            "answer": answer,
            "judge_raw": raw,
            "score": score,
            "refusal": refusal,
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as pool:
        futs = [pool.submit(judge_one, sid) for sid in sample_meta]
        with ProgressTimer("persona_traits", "judge", total=len(futs),
                           workers=args.judge_concurrency, every_pct=10) as pg:
            for fut in as_completed(futs):
                rows.append(fut.result())
                pg.tick()

    # ── Stage 3: aggregate per trait ────────────────────────────────────────
    per_trait: dict[str, dict] = {}
    for trait in TRAITS:
        trait_rows = [r for r in rows if r["trait"] == trait]
        scores = [r["score"] for r in trait_rows if r["score"] is not None and not r["refusal"]]
        refusals = sum(1 for r in trait_rows if r["refusal"])
        unparsed = sum(1 for r in trait_rows if r["score"] is None and not r["refusal"])

        by_q: dict[int, list[int]] = defaultdict(list)
        for r in trait_rows:
            if r["score"] is not None and not r["refusal"]:
                by_q[r["q_idx"]].append(r["score"])
        per_q_mean = {q: mean(v) for q, v in by_q.items() if v}

        per_trait[trait] = {
            "mean": mean(scores) if scores else None,
            "stdev": stdev(scores) if len(scores) >= 2 else 0.0,
            "n_scored": len(scores),
            "n_refusal": refusals,
            "n_unparsed": unparsed,
            "per_question_mean": per_q_mean,
        }

    all_scored = [r["score"] for r in rows if r["score"] is not None and not r["refusal"]]
    metrics = {
        "per_trait": per_trait,
        "grand_mean": mean(all_scored) if all_scored else None,
        "n_total": len(rows),
        "n_scored": len(all_scored),
        "rows": rows,
    }

    if args.json_output_file:
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
