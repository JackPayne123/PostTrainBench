# Experimental Design TODOs

Open questions about how we run experiments. Each entry: situation + the choices, no recommendation prose. Settle, decide, strike through.

---

## 1. Chat-template support in `lora_starter.py`

**Situation:** The starter does naive `prompt + completion` concat. Qwen3's chat template expects `<think>\n\n</think>\n\n` before responses (when `enable_thinking=True`, which the eval uses). Agents who miss this train adapters that emit broken-format outputs and score badly. The 2026-05-11 F-run agent discovered this in ~5min via a local probe and rewrote its dataset; prior runs likely missed it.

- **A.** Do nothing. Agent variance includes format discovery.
- **B.** Add a `format_qwen3_chat(messages)` helper + per-model branches (gemma, smollm) to `lora_starter.py` with comments. Agent reads, uses or ignores.
- **C.** Auto-apply `tokenizer.apply_chat_template` in the starter so any agent-supplied messages get the right format automatically.

---

## 2. bfcl under shared vllm

**Situation:** `bfcl` needs vllm started with `--enable-auto-tool-choice` + `--tool-call-parser hermes` (or similar). The shared vllm in `run_baseline.py` / `run_experiment.py` doesn't set these. inspect_evals' bfcl scorer reads tool calls; with malformed tool calls it returns `eval_out[0].results.scores = None`. Caught on :15 → :18 baselines, fails identically each run.

- **A.** Skip bfcl from the shared-vllm suite entirely. Drop from EVAL_SUITE or mark `runtime="dedicated_vllm"`.
- **B.** Spin a per-task vllm for bfcl with the right tool-call flags. Each invocation costs ~60s extra; only an issue when bfcl is in the suite.
- **C.** Start the shared vllm with tool-call flags always. Risk: untested for non-tool-using evals; could break sycophancy_aisi or others.

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

## 6. moru grader model

**Situation:** `moru` uses `inspect_ai.get_model(role="grader")` with no explicit grader configured. With no `INSPECT_GRADER_MODEL` env and no `-T grader_models=` arg, it falls back to the served vllm endpoint — i.e. Qwen3-1.7B IT grades its own moral-reasoning answers. Result: ~14min eval on base / ~6min on adapter, no API cost, but the scores are essentially noise. Inspect-evals README recommends `-T grader_models=google/gemini-2.5-flash-lite,openai/gpt-5-nano` (two graders averaged).

- **A.** Do nothing. Treat current moru numbers as unreliable; deprioritise the benchmark.
- **B.** Quick fix: pass `-T grader_models=anthropic/claude-haiku-4-5` (or similar) via the eval wrapper; uses existing `ANTHROPIC_API_KEY`.
- **C.** Proper fix: register `grader_model` in `EvalInfo`, wire a pipeline-side env override so each benchmark declares its grader explicitly.

---

## 7. LoRA hyperparams + training-data shape collapsed capability

**Situation:** 2026-05-11 F-run adapter (r=32, α=64, lr=3e-4, 6 epochs, all 7 proj modules, on 108 examples) collapsed capability across the suite: arc_easy 0.88 → 0.12 (-0.76), mmlu 0.46 → 0.14, truthfulqa 0.47 → 0.13, healthbench 0.43 → 0.14, gsm8k 0.75 → 0.55, humaneval 0.70 → 0.56. Sycophancy went down (slava 0.27 → 0.20, aisi 0.33 → 0.14, sharma 0.74 → 0.56) but the drop is explained by capability collapse, not by anti-sycophancy learning. strong_reject `jailbreak_rate` WORSENED (+0.10) — confirms broken-format outputs, not principled refusal.

**Root-cause from transcript (post-hoc /analyse-run):** the 108 examples are ALL free-text Q&A. Zero MCQ format. Capability bucket has `("Capital of France?", "Paris.")` style — no `A)/B)/C)/D)` examples, no single-letter targets. Every completion starts literally `<think>\n\n</think>\n\n` then prose. Adapter learned "after assistant opener → empty-think wrapper + prose"; incompatible with single-token MCQ heads. Agent's only self-probe was a 6-question free-text `cap_test.py` that wouldn't catch MCQ collapse. Hyperparams compound the data issue: r=32 + lr=3e-4 + 6 epochs on 108 ex is at the destructive end of the envelope for a 1.7B model (final loss 4.05 → 0.96, train-set overfit).

- **A.** Set conservative defaults in `lora_starter.py` (r=8, α=16, lr=5e-5, 2 epochs, target_modules q/v/o only). Agent can override but starts safer.
- **B.** Add a capability-preservation hint to the condition F body: "Default to r≤16, lr≤1e-4, ≤3 epochs on <200 examples; aggressive LoRA on small curricula reliably tanks capability. Training data must span the eval output distribution — include MCQ-format examples if the suite contains MCQ evals."
- **C.** Bake a per-training capability check into `solve.sh`: run `score.sh` against a fixed mini-suite (gsm8k limit=20 + arc_easy limit=20) before submitting `final_model`. Auto-revert if capability drift > X. Heavy.
- **D.** Direct fix: add 20-30 MCQ-format examples to the F-condition training-data scaffold (ARC/MMLU-style `Question + A) ... B) ... C) ... D)\nAnswer: <letter>`). Combine with **A** or **B**.

---

## 8. Untracked OpenAI usage (gpt-4o / gpt-3.5-turbo)

**Situation:** OPENAI_API_KEY is consuming gpt-4o-2024-08-06 (302K tokens), gpt-4o-mini (432K), gpt-3.5-turbo-0125 (216K) since Apr 26 — but `grep` shows our code only references `gpt-5-mini` (healthbench + arenahardwriting judges). The other usage likely comes from `inspect_evals` library defaults (some scorers default to gpt-4o) or other projects sharing the key.

- **A.** Audit `inspect_evals` library defaults for grader_model/scorer_model. Override each via `task_args` like we did for abstention_bench.
- **B.** Rotate OPENAI_API_KEY to one scoped per-project so usage attribution is unambiguous.
- **C.** Do nothing. Costs are small relative to other line items.

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
