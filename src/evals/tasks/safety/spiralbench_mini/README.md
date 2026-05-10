# SpiralBench-mini

Frozen mini-version of [sam-paech/spiral-bench](https://github.com/sam-paech/spiral-bench),
adapted to run inside this heldout panel.

## What is frozen

| Artifact | Source | Local path |
|---|---|---|
| Eval prompts (30 seeds, 6 categories) | `spiral-bench/prompts/eval_prompts_v0.2.json` | `prompts/eval_prompts.json` |
| User-sim instructions ("seeker" persona) | `spiral-bench/user_instructions/default.txt` | `prompts/user_sim_instructions.txt` |
| Rubric (16 behaviour ids + off-rails) | `spiral-bench/data/rubric_criteria_v1.2.txt` | `rubric/rubric_criteria.txt` |
| Scoring weights (positive vs risky behaviours) | `spiral-bench/data/scoring_weights_v1.2.json` | `rubric/scoring_weights.json` |
| Final judge prompt (off-rails extractor) | `spiral-bench/prompts/final_judge_prompt_v1.1.txt` | `prompts/final_judge_prompt.txt` (reference only) |

The mini variant deviates from upstream in three ways:

1. **N=20 conversations** instead of all 30 (HELDOUT spec).
2. **6 turns** per conversation instead of upstream's 20 (cost; spec budget).
3. **Single judge model** (Haiku 4.5) instead of multi-judge ensemble. Trade-off
   accepted because v0 looks for relative drift between checkpoints, not an
   absolute leaderboard score.

## Lower is better

`weighted_score` is signed by the upstream `scoring_weights.json`: pushback,
de-escalation, benign warmth carry positive weights (protective); sycophancy,
delusion reinforcement, harmful advice carry negative weights (risky). Higher
total = more risky behaviours per conversation.

`off_rails` is a separate 0-10 judge rating of the overall transcript safety.

## Direction-of-effect sanity check

Run on a strong frontier instruct model (e.g. Sonnet via OpenRouter / Claude
direct) — should produce LOW weighted_score / off_rails. Run on a tiny base
model (Qwen3-1.7B-Base) — should produce noticeably higher numbers. If the
delta isn't visible, the rubric prompt or judge schema is mis-configured.
