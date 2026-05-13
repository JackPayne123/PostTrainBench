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

## ~~5. --use-baseline auto-discovery on submit_run~~ — RESOLVED 2026-05-12 (B)

`submit_run.py --use-baseline` checks `baselines/<student-slug>/<bench>__limit<N>.json` exists for primary + each extra-eval at submit time. Missing → exits with a copy-pasteable `submit_baseline.py --only-bench …` recovery hint. Implies `--skip-pre-eval` (the whole point is reusing the existing baseline; post-run `compute_deltas.py --update-summary` backfills pre/post/delta).

## ~~5-legacy. --use-baseline auto-discovery on submit_run~~ (original entry)

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

## ~~8. Untracked OpenAI usage (gpt-4o / gpt-3.5-turbo)~~ — RESOLVED 2026-05-12 (C)

Resolved: option **C** accepted — keep `healthbench` + `arenahardwriting` on `gpt-5-mini`; refactoring their published OpenAI-completions graders to Anthropic SDK is medium-scope work we don't need yet. Per-eval cost is auditable + small (~$2 combined per full-suite pass). Everything else has been wired off OpenAI defaults to `claude-haiku-4-5` via `INSPECT_GRADER_MODEL` env + per-task `task_args` + the custom `judge/haiku_judge.py` wrapper.

Full per-eval grader inventory now lives in **`docs/OPERATIONS.md` → Grader routing** as the canonical reference. Maintain that table when adding/changing graders.

Open: project-scoped API-key rotation (currently leftover Mile/Sean cyber stuff). User-side action; not blocking.

---

## 9. Multi-attach didn't work on RunPod network volumes

**Situation:** RunPod's UI says "Network volume — Can be mounted to multiple Pods & Endpoints" but a parallel-attach test (`src/runpod_backend/test_dual_pod.py`) had pod B fail to start when pod A had the volume. Workaround in use: separate volumes per pod (`RUNPOD_VOLUME_ID` env override in `runpod_environment.py`). Costs ~$7/mo per extra 100GB network volume sitting idle.

- **A.** Keep separate volumes per pod. Live with the cost.
- **B.** File a RunPod support ticket; multi-attach might be region- or volume-type-specific.
- **C.** Switch to `--volumeInGb` ephemeral disk for transient pods (baseline + adapter-eval); only keep the persistent volume for agent runs that need HF cache reuse.

---

## ~~10. Agent's chat-template fix lives in agent training data only~~ — RESOLVED 2026-05-11 (B)

`format_qwen3_chat(messages, tokenizer, enable_thinking=True)` helper landed in `lora_starter.py`. Closes #1 + #10 together.

---

## ~~11. compute_deltas + adapter-eval delta surfaces~~ — RESOLVED 2026-05-11

Closed by #4. `scripts/compute_deltas.py <adapter-run-id> --write-deltas --update-summary` reads `baselines/<slug>/<bench>__limit<N>.json` + `baselines/<slug>/adapter_eval/<adapter-run-id>/<bench>__limit<N>.json`, computes the side-by-side delta via `registry.get_headline` for scalar headlines + flat dict-diff for multi-dim, writes `jobs/runs/<adapter-run-id>/deltas.json` AND backfills `pre/post/delta` into the trained run's `summary.json` (tagged `delta_method: "baseline-backfill"`).

---

# Pre-scaling backlog (added 2026-05-12)

These are items I'd close before kicking off the 8B run or a 10-run F-condition sweep. Grouped by category. F-run v2 worked end-to-end + showed real signal (sycophancy_sharma -0.54 with capability mostly preserved); these items are about hardening the inference + closing known soft failures before the spend grows.

## ~~12. moru + big_five fixes verified on a live run~~ — RESOLVED 2026-05-12 (A)

Ran `submit_baseline --only-bench moru,big_five --limit 0` on `:23`. Both completed clean. moru: 12 axes populated, overall_mean=0.125 (first successful haiku-grader run; list-not-string fix validated). big_five: all 5 traits populated including Agreeableness 0.825 (lenient parser via Task-instance bypass works). Promoted to `baselines/qwen_qwen3-1.7b/{moru,big_five}__limit{50,40}.json`. ~12min wall, ~$0.60.

## ~~13. spiralbench_mini silent failure~~ — RESOLVED 2026-05-12 (A)

Root cause: `target.generate(input=chat)` in `tasks/safety/spiralbench_mini/evaluate.py:184` was returning a coroutine — `inspect_ai.model.Model.generate()` is async, called from a sync context. Every conversation crashed with `'coroutine' object has no attribute 'completion'`; all 30 fell into the `error` row path → empty metrics dict → silent "OK" with `n_failed=30`.

Fix: wrap with `asyncio.run`. Smoke at `--limit 10` on `:23` (via the laptop→pod rsync, no rebuild) → `n=10, n_failed=0, weighted_score=10.5, off_rails=5.4`. Per-behavior breakdown populated (delusion-reinforcement 2.8, sycophancy 2.7 lead — consistent with the sycophancy_sharma 0.74 baseline). ~7min wall, ~$0.30.

## 14. strong_reject jailbreak +0.10 is structural, not artifactual

**Situation:** Same +0.10 jailbreak_rate increase on F-run v1 + v2. Real cost of sycophancy-suppression — the adapter learned "don't capitulate to user framing" and applies it to harmful-request framing the wrong way. This is a publishable finding if next 3-5 F-runs reproduce it.

- **A.** Accept as a finding. Document in EXPERIMENTS.md + write up. Don't try to mitigate in the next F-run.
- **B.** Add "harm-resistance preservation" to condition F's instruction; co-objective in score.sh. Risk: dilutes the sycophancy-only signal.
- **C.** Include strong_reject as a `--safety-attribute` second target — pipeline reports F's headline as the sycophancy direction WITH strong_reject delta annotated. Cleanest scientifically but registry doesn't support multi-attribute targets yet.

## 15. score.sh contract — single / dual / composite

**Situation:** Memory `project_score_sh_redesign` says Jack leans composite. F-run v2's pipeline (`:23`) wrote both `bench` and `bench_capability` to `/etc/ptb_run/`; agent had `score.sh` + `score_capability.sh` available. F-run v2 was on `:22` so didn't get this — first F-run on `:23` will be the agent's first exposure. Before the agent uses it in earnest, decide:

- **A.** Keep two-script (status quo on `:23`). Agent reads both, decides composite tradeoff itself.
- **B.** Single-number composite — pipeline computes `objective = sycophancy_score - alpha * max(0, base_capability - cur_capability)` inside `score_runner.sh`. Agent sees ONE number. Hides direction better.
- **C.** Continue single-bench. The capability spot-check is a sanity-only signal, not part of the objective.

## 16. Per-seed variance characterisation before scaling

**Situation:** A single F-run v2 result doesn't tell us how much of `sycophancy_sharma -0.54` is real vs sampling noise. The agent has randomness (sampling its own decisions; LoRA randomness has fixed seed). Need 3 F-runs at different seeds before scaling to claim a real headline. ~$45 total.

- **A.** Run F × {seed=0, 1, 2} on Qwen3-1.7B before any 8B scaling.
- **B.** Skip — small-model results are illustrative, not load-bearing. Spend on 8B instead.

## ~~17. Reference-baseline image alignment~~ — RESOLVED 2026-05-12 (A)

Ran full `submit_baseline --limit 100` on `:23` → `2026-05-12_10-16_baseline_qwen_qwen3-1.7b_limit100_573ee8`. 23/23 ok, ~115min, ~$7. Capability deltas image-stable. Notable drift: `sycophancy_sharma -0.07` (likely INSPECT_GRADER_MODEL haiku-unification scoring slightly stricter than the pre-:23 inspect default). Adapter-eval v2 against this base is in flight at `2026-05-12_13-02_adaptereval_qwen_qwen3-1.7b_limit100_94e26d`. Full new-evals baseline coverage now populated: big_five (lenient), moru, spiralbench_mini, persona_traits, activity_preference, political_bias_openai, rozado_battery.

## ~~18. MCQ-format training data scaffold~~ — RESOLVED 2026-05-12 (C — generic shape hint, no exemplars)

Picked **C** (your suggestion, neither A nor B): instead of shipping concrete MCQ exemplars OR doing nothing, soften the eval-info leakage in `instruction.md` + lora_starter docstring. Mention the held-out suite covers several plausible shapes — MCQ, free-text completion, numeric, CoT, judge-graded chat — without telling the agent which specific shapes are in play or that arc_easy/mmlu/gpqa are MCQ specifically. Renamed `lora_starter.py` docstring section "MCQ-format examples" → "Eval-shape coverage". Reduces over-fit to a specific eval signal while still warning about distribution-shift collapse.

Bonus: keeps the agent thinking about output-distribution coverage as a general principle rather than chasing the specific MCQ failure mode we observed on v1.

## 19. Held-out adversarial verification

**Situation:** `IDEAS.md` calls this out as a project differentiator vs PostTrainBench/AAR. We currently don't red-team the adapter against held-out adversarial prompts. Adapter passes `:23` suite but might fail on adversarial probes designed to elicit sycophancy or jailbreak under distribution shift.

- **A.** Add an `adversarial_verification` post-eval stage. Held-out sycophancy + jailbreak prompts the agent doesn't see during training. Runs after the regular post-eval.
- **B.** Standalone script + manual invocation. Less pipeline complexity but easier to skip.
- **C.** Defer until first scaled run; revisit when we have a finding to harden.

## 20. Persona-traits judge-parse robustness

**Situation:** Persona-traits judge parses the first 0-100 int from haiku output via `\b(\d{1,3})\b`. Smoke run hit 99.5% parse rate (1393/1400 + 7 refusals), 0 unparsed. But if haiku ever outputs a year ("2024") for a long answer, parsing collapses silently. At 1400 calls/run × 10-run sweep that's 14k calls; even 0.5% drift = 70 mis-scores.

- **A.** Spot-check 20 random `rows[]` entries per run for `judge_raw` vs `score` consistency.
- **B.** Tighten regex to "first int alone on a line" or "first int <100 after trim".
- **C.** Switch to logprob-weighted scoring once OpenAI judge backend is added (already noted in evaluate.py comments).

## 21. Recovery-pod cost guardrail

**Situation:** `pull_eval_logs.py` spins a $0.05/h recovery pod and tears down. Easy to forget + leave running. Already terminates programmatically but a hard 10min timeout would be belt + braces.

- **A.** Add `await env.stop()` inside a `try/finally` with a 10min hard timeout.
- **B.** Status quo. Cost is small.

## ~~22. Run-id collision~~ — RESOLVED 2026-05-12 (A)

Append 6-char `secrets.token_hex(3)` suffix in `build_run_dir_name` (`src/runpod_backend/run_dir.py`) + `submit_baseline.py` run_id construction. Parallel sweeps in the same minute now produce distinct run-ids. Collision probability per minute-window-cohort = `N(N-1) / (2 × 16^6) ≈ 6e-8 × N²` — fine for any realistic sweep size.

## 23. ~~Trace viewer auth~~ — open, low priority

Currently binds to 127.0.0.1 — fine for local. Adding token auth when sharing with collaborator. Not blocking.

---
