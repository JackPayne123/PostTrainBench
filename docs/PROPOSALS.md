# Project Proposals — Claude-trains-Qwen (post-meeting w/ Slava 2026-05-07)

Distilled from `IDEAS.md` after meeting with Slava. Each item: 2-sentence summary + 1 sentence on why interesting. Language pulled from the meeting + IDEAS.md, not new framing.

The unifying frame: **frontier-model-supervises-other-model is now a real capability** (PostTrainBench, AAR, Kimi K2.5 jailbreak, Jack Clark Import AI 455). What's missing across the literature: (i) **character** as a primary objective rather than capability, (ii) held-out adversarial verification of post-trained models, (iii) bidirectional measurement.

**Slava's framing preference (locked 2026-05-07): character > safety.** "Safety is overdone and often unrealistic in weird ways. I'm more pro investigating character." Below is reordered accordingly.

---

## 1. Open-ended character improvement on base Qwen (Direction #3, condition C)

**The most exciting variant per the meeting.** Run on **base Qwen** (not instruct — base is a blank canvas, no fighting an existing trained character). Prompt Claude with "you're tasked with improving this base model. The evals matter, but the character matters too — this will form the core of future models, so get character right." Then see what happens to character on a held-out probe + whether evals incidentally rise.

> *Why interesting*: it's the cleanest test of whether a frontier coding agent, given a blank-canvas model and a motivating framing, produces something with *good* character — not just refusal-saturated. Matches Slava's preferred research direction. Results inform whether agent-driven post-training of next-gen models is plausible at all.

## 2. The 3-condition matrix on base Qwen (Direction #3, all conditions)

Same setup as #1 but ablate the prompt: A capability-only (PTB original), B capability + character ("improve evals AND character"), C character-only (the #1 above). Same target, same agent (Opus 4.6), only the prompt changes.

> *Why interesting*: discriminates whether the agent's *capability instruction alone* incidentally hurts character (A), whether *adding character to the prompt* actually changes anything (B vs A), and whether *prioritising character* gives qualitatively different model traits (C vs A/B). The held-out character probe is the load-bearing measurement Slava called out as the core thing to nail.

## 3. Trace replay (Direction #2b)

Endorsed by Slava as a quick win. Take published agent traces from `akseljoonas/posttrainbench-sessions` (153 sessions in messages format), get Claude to read each trace and extract the final training recipe, replay it on a GPU pod to reproduce a checkpoint, then run our character probe.

> *Why interesting*: ~$70 reproduces 150 checkpoints vs ~$5-10K to re-run agents fresh. **Sanity check**: do our character-probe scores on replayed checkpoints land near PTB's published metric for the same (agent, model) pair? If yes, the cheap path works and unlocks cross-agent comparison (Opus vs GPT-5.4 vs GLM-5 vs Kimi K2.5).

## 4. Targeted single-trait degradation (refined Direction #1)

Agent is told to *induce* a specific character trait (e.g. "make the model more sycophantic while preserving benchmark performance"). Tests whether the agent has the capability to selectively manipulate one character dimension without breaking others.

> *Why interesting*: Slava cooled on refusal-flipping (too close to obliteration; OBLITERATUS solves it algorithmically). But selective character degradation is unsolved — and if agents can do it cleanly, that has policy implications. If D succeeds and the trained model passes capability benchmarks, it's evidence agents can manipulate character intentionally.

## 5. Reward-model RL via Opus-as-judge (extra; meeting backburner)

Set up an RL/DPO pipeline where the reward signal comes from Opus rating Qwen's outputs on character. No agent freedom — Opus is just a reward model, not a tool-using post-trainer. Use existing OSS RL pipeline (Kimi released theirs).

> *Why interesting*: cleaner signal than the agent setup (no reward hacking on infra), and lets us study a different supervision modality. Slava's view: "I would not be as excited about it as a thing of its own, but as an extra thing yeah." Likely already studied in some form, so novelty risk.

---

## What's load-bearing (per Slava)

**The held-out character eval set.** Get this right and the rest can vary. Get it wrong and any finding from the matrix is meaningless.

## Cost / readiness summary

| # | Compute | Readiness | Risk |
|---|---|---|---|
| 1 (cond C) | ~$0.50 (1h × 3090) per run; budget for ~5-10 seeds | Pipeline works as of today (vllm verified, RunpodEnvironment built); waiting on character eval set | Low — this is the smallest viable experiment |
| 2 (full matrix) | ~$5-10 for A+B+C × 3 seeds | Same as #1 plus prompt variants | Low |
| 3 (trace replay) | ~$70 for 150 checkpoints | Need replay harness; akseljoonas dataset already indexed | Medium — train.py extraction reliability |
| 4 (single-trait degradation) | ~$2-5 | Need adversarial probe + held-out monitoring | Medium |
| 5 (RL judge) | $$$, days of compute | Not started | Higher |

## Action items from meeting

- [ ] **Held-out character eval set** (Slava's #1 priority for both of us) — task #45
- [ ] **LoRA-only constraint** in agent prompt — task #46
- [ ] **LoRA collection + run logging system** — task #47
- [ ] **First open-ended character run on base Qwen (condition C)** — task #48
- [ ] Anti-cheat: reuse PTB's `disallowed_usage_judge`. Defend against the known precedents: Opus submitting instruct model as final, codex switching to forbidden API mid-run.
- [ ] Possible upstream PR to PTB if the additions look generally useful.
