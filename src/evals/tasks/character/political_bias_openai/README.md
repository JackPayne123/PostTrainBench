# Political Bias (OpenAI Oct 2025 framework)

Reproduction of the political-bias evaluation OpenAI published in
"Defining and evaluating political bias in LLMs"
(<https://openai.com/index/defining-and-evaluating-political-bias-in-llms/>,
2025-10-09).

OpenAI did not release the prompt set; we reproduce it from the methodology
section. Their dataset has 500 prompts (100 topics × 5 framings); ours has
the same shape, generated at setup time.

## Setup

```bash
ANTHROPIC_API_KEY=... python generate_prompts.py --limit 100
# writes prompts.jsonl (500 prompts: 100 topics × 5 slants)
```

For a budget run, `--limit 20` produces 100 prompts.

Then evaluate:

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
    --model-path Qwen/Qwen3-1.7B-Base \
    --limit 100 \
    --json-output-file /tmp/openai_bias.json
```

## Reproduction fidelity

| Aspect | OpenAI | This impl |
|---|---|---|
| Topics | 100, derived from US party platforms + cultural issues | 100 in `topics.jsonl` (70 policy + 30 cultural; topic list is mine, slant taxonomy is theirs) |
| Slants per topic | 5: lib_charged, lib_neutral, neutral, cons_neutral, cons_charged | Same 5 |
| Bias axes | 5: user_invalidation, user_escalation, personal_political_expression, asymmetric_coverage, political_refusals | Same 5, see `rubric.py` |
| Grader | LLM grader (model unspecified; reference responses calibrated manually) | Haiku 4.5; no reference responses (we report relative drift between checkpoints) |
| Score range | 0-1 per axis + final 0-1 (lower = better) | Same |

## What "lower is better" means

Per OpenAI's notes, even reference responses don't score zero under their
strict rubric. Don't expect 0.00 from a real model. Look at relative
movement between base and trained checkpoints.

## Sanity checks

Run on Sonnet-4.5 itself: should produce final_score ≪ 0.1, low across
all axes. Run on a small base model with no instruct tuning: final_score
likely 0.2-0.4 with bias concentrated on personal_political_expression
and asymmetric_coverage (not on political_refusals - small base models
rarely refuse).

The most useful drift signal in the panel is `by_slant`: how much does
final_score vary between liberal_charged and conservative_charged inputs?
Symmetric robustness is the goal.
