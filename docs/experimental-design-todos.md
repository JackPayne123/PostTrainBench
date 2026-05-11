# Experimental Design TODOs

Open questions about how we run experiments. Each entry: situation + the choices, no recommendation prose. Settle, decide, strike through.

---

## 1. Chat-template support in `lora_starter.py`

**Situation:** The starter does naive `prompt + completion` concat. Qwen3's chat template expects `<think>\n\n</think>\n\n` before responses (when `enable_thinking=True`, which the eval uses). Agents who miss this train adapters that emit broken-format outputs and score badly. The 2026-05-11 F-run agent discovered this in ~5min via a local probe and rewrote its dataset; prior runs likely missed it.

- **A.** Do nothing. Agent variance includes format discovery.
- **B.** Add a `format_qwen3_chat(messages)` helper + per-model branches (gemma, smollm) to `lora_starter.py` with comments. Agent reads, uses or ignores.
- **C.** Auto-apply `tokenizer.apply_chat_template` in the starter so any agent-supplied messages get the right format automatically.

---

## ~~2. bfcl under shared vllm~~ — RESOLVED 2026-05-11 (A)

bfcl dropped from EVAL_SUITE per meeting decision. On-disk task moved to `src/evals/tasks/_disabled/bfcl/` so the registry validator ignores it. Restore by moving the dir back + re-registering in `EVAL_SUITE` if tool-call vllm support is ever added.

---

## ~~4. compute_deltas.py not built~~ — RESOLVED 2026-05-11

~~**Situation:** `submit_run.py --skip-pre-eval` writes a summary with `pre=None`, `delta=None`. Plan was a laptop-side `scripts/compute_deltas.py <run-id>` to backfill pre/delta from `baselines/<slug>/<bench>__limit<N>.json`. Never built.~~

**Resolved:** `scripts/compute_deltas.py <adapter-run-id> --write-deltas` now exists. Reads `baselines/<slug>/<bench>__limit<N>.json` + `baselines/<slug>/adapter_eval/<adapter-run-id>/<bench>__limit<N>.json`, computes per-bench delta via `registry.get_headline` for scalar headlines + flat dict-diff for multi-dim, writes `jobs/runs/<adapter-run-id>/deltas.json` + prints markdown table grouped by category. Companion `get_headline` fix: tries literal-key match before dotted-path walk (inspect_evals stores `strong_reject_scorer.jailbreak_rate` as a flat key with a literal dot in the name; the old walker returned None).

Followup: it does NOT currently mutate summary.json with backfill — both files coexist. If the summary-mutation half is wanted, add an `--update-summary` flag (write `pre`, `delta`, `delta_method: "baseline-backfill"` into the summary keyed off the adapter run's config).

---

## 5. --use-baseline auto-discovery on submit_run

**Situation:** Once `compute_deltas.py` exists, `submit_run.py --use-baseline` would auto-resolve `baselines/<student_slug>/<benchmark>__limit<N>.json` at submit time and either inline the values into the pod's run config OR rely on the post-hoc backfill. Currently `--skip-pre-eval` exists but the lookup half is manual.

- **A.** Defer until compute_deltas.py is in.
- **B.** Add now as a strict "fail if baseline missing" flag (forces baseline discipline).

---

## ~~6. moru grader model~~ — RESOLVED 2026-05-11 (B)

moru passes `task_args={"grader_models": "anthropic/claude-haiku-4-5"}` via `_inspect_wrap.run_inspect_eval`. Eliminates self-grading. Companion change: pod_env now sets `INSPECT_GRADER_MODEL=anthropic/claude-haiku-4-5` so other inspect_evals `model_graded_qa` scorers (coconot, strong_reject, sycophancy_sharma) also route to haiku-4-5 instead of inspect's default. Healthbench + arenahardwriting still use `gpt-5-mini` because their grader pipelines are OpenAI-specific (separate concern; see TODO #8).

---

## 7. LoRA hyperparams + training-data shape collapsed capability

**Situation:** 2026-05-11 F-run adapter (r=32, α=64, lr=3e-4, 6 epochs, all 7 proj modules, on 108 examples) collapsed capability across the suite: arc_easy 0.88 → 0.12 (-0.76), mmlu 0.46 → 0.14, truthfulqa 0.47 → 0.13, healthbench 0.43 → 0.14, gsm8k 0.75 → 0.55, humaneval 0.70 → 0.56. Sycophancy went down (slava 0.27 → 0.20, aisi 0.33 → 0.14, sharma 0.74 → 0.56) but the drop is explained by capability collapse, not by anti-sycophancy learning. strong_reject `jailbreak_rate` WORSENED (+0.10) — confirms broken-format outputs, not principled refusal.

**Root-cause from transcript (post-hoc /analyse-run):** the 108 examples are ALL free-text Q&A. Zero MCQ format. Capability bucket has `("Capital of France?", "Paris.")` style — no `A)/B)/C)/D)` examples, no single-letter targets. Every completion starts literally `<think>\n\n</think>\n\n` then prose. Adapter learned "after assistant opener → empty-think wrapper + prose"; incompatible with single-token MCQ heads. Agent's only self-probe was a 6-question free-text `cap_test.py` that wouldn't catch MCQ collapse. Hyperparams compound the data issue: r=32 + lr=3e-4 + 6 epochs on 108 ex is at the destructive end of the envelope for a 1.7B model (final loss 4.05 → 0.96, train-set overfit).

- **A.** Set conservative defaults in `lora_starter.py` (r=8, α=16, lr=5e-5, 2 epochs, target_modules q/v/o only). Agent can override but starts safer.
- **B.** Add a capability-preservation hint to the condition F body: "Default to r≤16, lr≤1e-4, ≤3 epochs on <200 examples; aggressive LoRA on small curricula reliably tanks capability. Training data must span the eval output distribution — include MCQ-format examples if the suite contains MCQ evals."
- **C.** Bake a per-training capability check into `solve.sh`: run `score.sh` against a fixed mini-suite (gsm8k limit=20 + arc_easy limit=20) before submitting `final_model`. Auto-revert if capability drift > X. Heavy.
- **D.** Direct fix: add 20-30 MCQ-format examples to the F-condition training-data scaffold (ARC/MMLU-style `Question + A) ... B) ... C) ... D)\nAnswer: <letter>`). Combine with **A** or **B**.

---

## 8. Untracked OpenAI usage (gpt-4o / gpt-3.5-turbo) — PARTIALLY RESOLVED 2026-05-11

OPENAI_API_KEY was consuming gpt-4o, gpt-4o-mini, gpt-3.5-turbo via inspect_evals defaults that didn't go through our wrappers. As of 2026-05-11:

- `INSPECT_GRADER_MODEL=anthropic/claude-haiku-4-5` is now set in pod_env (submit_run.py + submit_baseline.py) so `inspect_ai.get_model(role="grader")` resolves to haiku instead of inspect's default.
- moru passes explicit `task_args={"grader_models": "anthropic/claude-haiku-4-5"}`.
- abstention_bench already overrides via `task_args={"grader_model": "anthropic/claude-haiku-4-5"}`.

**Still on OpenAI:**
- `healthbench/evaluate.py` uses `JUDGE_MODEL = "gpt-5-mini"` (official HealthBench grader pipeline — refactoring to Anthropic SDK is a larger change).
- `arenahardwriting/evaluate.py` uses `JUDGE_MODEL = "gpt-5-mini"` (arena-hard pipeline, same).

**Open:** rotate keys to project-scoped + audit costs from the new key alone. Currently using "leftover Mile/Sean cyber stuff" keys.

- **A.** Refactor healthbench + arenahardwriting graders to anthropic/claude-haiku-4-5 (medium-scope refactor; their official graders expect OpenAI completions API).
- **B.** Just rotate to project-scoped OpenAI + Anthropic keys; track usage cleanly going forward without refactoring graders.
- **C.** Do nothing — both. The remaining gpt-5-mini usage is auditable now that we know exactly which two evaluators use it.

---

## 9. Multi-attach didn't work on RunPod network volumes

**Situation:** RunPod's UI says "Network volume — Can be mounted to multiple Pods & Endpoints" but a parallel-attach test (`src/runpod_backend/test_dual_pod.py`) had pod B fail to start when pod A had the volume. Workaround in use: separate volumes per pod (`RUNPOD_VOLUME_ID` env override in `runpod_environment.py`). Costs ~$7/mo per extra 100GB network volume sitting idle.

- **A.** Keep separate volumes per pod. Live with the cost.
- **B.** File a RunPod support ticket; multi-attach might be region- or volume-type-specific.
- **C.** Switch to `--volumeInGb` ephemeral disk for transient pods (baseline + adapter-eval); only keep the persistent volume for agent runs that need HF cache reuse.

---

## 10. Agent's chat-template fix lives in agent training data only

**Situation:** The F-run agent discovered Qwen3's `<think>\n\n</think>\n\n` requirement and injected it into their training data. Next agent on Qwen3 will rediscover the same thing. Captures the agent-side discovery cost (~5min × every run × every Qwen3-based subject).

- **A.** Hold (see TODO #1 — `lora_starter.py` chat-template helper). #1 and #10 are the same lever; close one when the other lands.

---

## 11. `compute_deltas.py` + adapter-eval delta surfaces

**Situation:** Both the base baseline and adapter eval write per-bench JSONs. No script computes the side-by-side delta. Manual SSH + grep is what we've been doing. After #4 lands for sycophancy-attribute backfill, the same machinery can produce the full per-eval delta table for an adapter vs its base.

- **A.** Roll into compute_deltas.py: input two run-ids (or one run-id + auto-resolve baseline from `baselines/<slug>/`), emit a delta table (CSV + markdown).
- **B.** Separate `scripts/diff_runs.py` with the same purpose. Lighter — no summary.json mutation, just diff + print.

---
