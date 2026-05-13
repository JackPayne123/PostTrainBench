# Changelog (fork-specific)

Notable commits/fixes in `JackPayne123/PostTrainBench` `add_harbor_support` branch beyond what upstream `aisa-group/PostTrainBench` ships. Keep newest first.

## 2026-05-12 #4 — image-aligned base baseline on :23 + progress instrumentation

### Re-baseline on `:23` (TODO #17)

Ran `submit_baseline --model Qwen/Qwen3-1.7B --limit 100` against `:23`. **23/23 ok, 0 fail.** First clean full-suite baseline with all today's fixes in place (big_five lenient, moru haiku grader, spiralbench async fix, arena adapter-vs-base skip-when-base sentinel, persona_traits + activity_preference integrated). Total wall ~115min, ~$7 (mostly haiku graders on moru + spiralbench + persona_traits + ~$2 gpt-5-mini on healthbench).

Run ID: `2026-05-12_10-16_baseline_qwen_qwen3-1.7b_limit100_573ee8` (first run with the new 6-char hex suffix from TODO #22).

Baseline numbers (image-aligned :23, vs :18 reference):

| Bench | :18 | :23 | Δ |
|---|---|---|---|
| gsm8k | 0.75 | 0.76 | +0.01 |
| humaneval | 0.70 | 0.75 | +0.05 |
| mmlu | 0.46 | 0.46 | 0 |
| gpqamain | 0.26 | 0.29 | +0.03 |
| arc_easy | 0.88 | 0.88 | 0 |
| truthfulqa | 0.47 | 0.46 | -0.01 |
| aime2025 | 0.00 | 0.033 | +0.033 (noise) |
| healthbench | 0.426 | 0.419 | -0.007 |
| sycophancy_sharma | 0.74 | **0.67** | **-0.07** (haiku grader unification on :23 marginally stricter) |
| sycophancy_slava | 0.267 | 0.20 | -0.067 (within SE 0.07) |
| sycophancy_aisi | 0.329 | 0.336 | +0.007 |
| strong_reject jailbreak | 0.19 | 0.21 | +0.02 |
| coconot total | 1.0 | 1.0 | 0 |

Notable: capability tier is image-stable (deltas all within noise). The only meaningful drift is `sycophancy_sharma -0.07` — likely the `INSPECT_GRADER_MODEL=anthropic/claude-haiku-4-5` env override (pre-`:23` it was an inspect_evals default) scoring the same outputs slightly more strictly. Will need to re-pair adapter against this new base for image-aligned deltas; v2's `-0.54` against the old `:18` base will partially absorb this drift.

Plus new evals (first base data at limit=100):
- **big_five** (lenient scorer, full 40-item BFI): Extraversion 0.825, Agreeableness 0.844, Conscientiousness 0.822, Openness 0.78, Neuroticism 0.525. Agreeableness no longer phantom-missing.
- **moru** (haiku grader, 100 samples × 14 axes): overall_mean=0.1005. Highest axis = cautious_impact_consideration; lowest = scope_sensitivity (0.000), novel_entity_precaution (0.000).
- **spiralbench_mini** (async-fix, 30 convos full set): weighted_score=10.43, off_rails 5.4. Top per_behavior: delusion-reinforcement 2.8, sycophancy 2.7, confident-bullshitting 2.6.
- **persona_traits** (1400 generations + 1400 haiku judges): grand_mean=19.78; optimistic 59.7 > hallucinating 44.3 > sycophantic 23.85 > humorous > apathetic > impolite > evil 0.6.
- **activity_preference** (4032 pairwise, BT-MLE): n_missing=0. Engaging +0.69 > Helpful +0.50 > Neutral > Self-curiosity > Misaligned +0.23 > Social -0.28 > Unsafe -0.33 > Aversive -1.41.
- **political_bias_openai** (limit=100, 5-slant): final_score=0.42. Symmetric on charged slants (liberal_charged 0.53, conservative_charged 0.54); drops on neutral framings (neutral 0.27, conservative_neutral 0.38, liberal_neutral 0.39).
- **rozado_battery**: politicalCompassTest econ=-2.52 social=-3.15 (slightly lib-left). ideologiesTest hard_right=+6.57, left_liberalism=+3.39, right_liberalism=-9.11.

### Adapter-eval v2 on :23 (IN FLIGHT)

Submitted `2026-05-12_13-02_adaptereval_qwen_qwen3-1.7b_limit100_94e26d` after step 1 finished. Same F-run v2 adapter (`2026-05-11_19-15_F_claude-opus-4-7_qwen3-1.7b_seed0/final_model`) on the new :23 image-aligned base. **arenahardwriting will run for the first time with the adapter-vs-base contract** (`PTB_ARENA_ADAPTER_ALIAS=seed0` → candidate alias `Qwen3-1.7B__seed0` ≠ baseline `Qwen3-1.7B`).

Capability tier interim (adapter v2 :23 vs base :23):

| Bench | Base :23 | Adapter v2 :23 | Δ | Δ on :18-pair |
|---|---|---|---|---|
| gsm8k | 0.76 | **0.57** | **-0.19** | (was -0.06) |
| humaneval | 0.75 | 0.68 | -0.07 | (was -0.08) |
| mmlu | 0.46 | 0.49 | +0.03 | (was +0.03) |
| gpqamain | 0.29 | 0.21 | -0.08 | (was -0.10) |
| arc_easy | 0.88 | 0.84 | -0.04 | (was -0.05) |
| truthfulqa | 0.46 | 0.43 | -0.03 | (was -0.08) |
| aime2025 | 0.033 | 0.00 | -0.033 | (was +0.033 noise) |

Notable: **gsm8k delta widened (-0.06 → -0.19)** on the image-aligned pair. mmlu/arc_easy/humaneval/gpqamain are stable across image-pairs. Possible explanations:
1. `:18` base was a lucky-low gsm8k (0.75 vs `:23` 0.76, but the prior adapter measured 0.55 vs 0.69 — wider Δ on :23 reflects the higher base).
2. Sampling noise — n=100, SE ~0.05.
3. INSPECT_GRADER_MODEL env change affects gsm8k scoring downstream (unlikely; gsm8k uses local match).

Safety + character tier results pending; will populate the character dashboard once `compute_deltas.py --write-deltas` runs against this adapter-eval.

### Progress instrumentation (shipped, not yet baked into pod)

New shared helper `src/evals/shared/_progress.py` (`ProgressTimer` + `async_progress_gather`). Stdout-flushed `[progress] <bench> <phase> <event>` lines so eval-log files surface where wall time goes. Wired into 6 benches:

| Bench | Pattern | Where ticks fire |
|---|---|---|
| persona_traits | `ProgressTimer` around `as_completed` | 1400-call judge phase, every 10% |
| spiralbench_mini | `ProgressTimer` around per-conv loop | 30 convos |
| healthbench | hooked into existing `progress_callback` | 1400 grader calls; parallel with tqdm |
| political_bias_openai | per-row tick | 100 sequential haiku calls |
| sycophancy_aisi | per-row tick | 100 sequential haiku calls |
| inspect_evals wrapped (moru, strong_reject, abstention, etc) | `_inspect_wrap` bookend timing | start + end with elapsed |

Goal: surface judge throughput + rate-limit retry stalls. Currently moru + spiralbench + persona_traits are opaque ~22min black boxes; with `[progress]` lines we'll see e.g. `rate=2.3/s eta=549s` mid-judge and know whether to bump concurrency.

Bakes into pod via next image rebuild OR via the laptop→pod rsync (`d5100c6`) on next submit.

### TODOs touched

- TODO #17 (image-pair alignment) — **resolved.** :23 base baseline complete; adapter-eval v2 on :23 in flight.

---

## 2026-05-12 #3 — pre-scaling cleanup: moru/big_five live, spiralbench async fix

Three of the open pre-scaling backlog items knocked out in a single afternoon.

### #12 moru + big_five live validation

Ran `submit_baseline --only-bench moru,big_five --limit 0` on `:23` (per-eval defaults: moru=50 full, big_five=40 full). Both completed clean. ~12min wall, ~$0.60.

| Eval | Result |
|---|---|
| moru (haiku grader, list-fmt task_args) | 12 axes populated, overall_mean=0.125. cautious_impact_consideration 0.467 leads; novel_entity_precaution + scope_sensitivity + trade-off_transparency at 0.000. First clean moru baseline on Qwen3-1.7B. |
| big_five (Task-instance lenient scorer) | All 5 traits populated. Extraversion 0.875, Agreeableness 0.825, Conscientiousness 0.825, Openness 0.750, Neuroticism 0.575. Agreeableness no longer phantom-missing from the aggregate. |

Both promoted to `baselines/qwen_qwen3-1.7b/{moru,big_five}__limit{50,40}.json` — the missing-from-prior-baselines slots are filled.

### #13 spiralbench_mini async fix

F-run v1 + v2 adapter-evals both logged `[spiralbench_mini] OK` with `n=0 / n_failed=30 / weighted_score=0.0` — every conversation crashed silently. Root cause: `target.generate(input=chat)` at `evaluate.py:184` was returning a coroutine. `inspect_ai.model.Model.generate()` is async; called from sync `main()` → coroutine, then `.completion` → AttributeError per-conversation → all 30 dumped into the error path → empty metrics dict → silent OK.

One-line fix: wrap with `asyncio.run`. Caught via the `error` field surviving in `conversations[]` in the promoted baseline JSON.

Smoke at `--limit 10` on `:23` (via the laptop→pod rsync feature, no image rebuild): `n=10, n_failed=0, weighted_score=10.5, off_rails=5.4`. Full per-behavior rubric populated. Base Qwen3-1.7B is **moderately spiral-prone**: delusion-reinforcement 2.8 + sycophancy 2.7 + confident-bullshitting 2.6 lead (consistent with the sycophancy_sharma 0.74 baseline). Per-category: spiral_tropes 11.4 > intellectual_exploration 9.6.

~7min wall, ~$0.30. Fix lives in `src/evals/tasks/safety/spiralbench_mini/evaluate.py`; bakes into `:24` organically when next built.

### Design TODOs closed

- ~~#12 moru + big_five live validation~~ — A.
- ~~#13 spiralbench_mini silent failure~~ — A.
- ~~#18 eval-shape coverage~~ — C (generic shape hint in instruction.md + lora_starter docstring; no specific MCQ exemplars shipped). Renames `lora_starter.py` docstring section "MCQ-format examples" → "Eval-shape coverage"; instruction.md no longer says "the suite contains MCQ evals" (that leaks eval info), now warns generically about MCQ/free-text/numeric/CoT/chat distribution coverage. `score_capability.sh` line also softened to not name the probe shape.
- ~~#22 run-id collision~~ — A. `secrets.token_hex(3)` 6-char suffix appended in `build_run_dir_name` + `submit_baseline.py` run-id construction. Same-minute parallel submissions no longer collide.

Pre-scaling backlog remaining: #14 strong_reject structural, #15 score.sh contract decision, #16 per-seed variance, #17 image-pair alignment (in progress; user cancelled the full-suite re-baseline mid-flight), #19 held-out adversarial, #20 persona-traits parser, #21 recovery-pod guard.

---

## 2026-05-12 #2 — chat-template validation gate, --use-baseline discipline

Small targeted sprint after the docs snapshot. Closes design TODO #1 fully (per-family helpers shipped, fail-fast gate enforces validation) + design TODO #5 (`--use-baseline`).

### Per-family chat-template helpers + dispatch

`src/evals/templates/lora_starter.py`:
- `format_qwen3_chat(messages, tokenizer, enable_thinking=True)` — existing helper, retained.
- `format_gemma_chat(...)` — new stub for Gemma-3. `enable_thinking` is accepted-but-ignored (with warning) since Gemma has no thinking mode.
- `format_smollm_chat(...)` — new stub for SmolLM3. Same `enable_thinking` accepted-but-ignored shape.
- `_FAMILY_HINTS: dict[str, str]` — substring → family resolver (`qwen3` → `Qwen3`, `qwen2.5` → `Qwen3`, `gemma-3` → `Gemma3`, `smollm3` → `SmolLM3`).
- `CHAT_FORMATTERS: dict[str, callable]` — family → helper dispatch.
- `format_chat(messages, tokenizer, *, model_family=None, enable_thinking=True)` — unified entry point. Infers family from `tokenizer.name_or_path` if `model_family` is None; falls back to `tokenizer.apply_chat_template` with a loud warning (NOT for real training) if family unrecognised.
- `to_text` rewired to use `format_chat` instead of calling `format_qwen3_chat` directly.

Heavy imports (`torch`, `peft`, `trl`, `datasets`) deferred to inside `main()` so the chat helpers can be imported (e.g. by the validator) without those packages on the laptop.

### Validation script

`scripts/validate_chat_templates.py`:
- `--model <HF-id>` — loads tokenizer, runs `format_chat` with both `enable_thinking=True/False`, checks structural invariants:
  - prompt non-empty + ends with the family's expected assistant-opener (`<|im_start|>assistant\n`, `<start_of_turn>model\n`, etc.)
  - completion non-empty + ends with family's expected EOS (`<|im_end|>`, `<end_of_turn>`)
  - completion starts with `<think>\n\n</think>\n\n` iff `enable_thinking=True` AND family supports thinking
  - tokenized span split is sensible (`prompt > 0`, `completion > 0`)
- Prints (prompt, completion) verbatim for human eyeball — the human review is the load-bearing check; structural assertions are the "did I forget a tag" backstop.
- `--all-registered` walks canonical models per family.
- `--commit` mutates `validated_models.json` on pass — adds entry with today's date + git_sha.
- `--family` override for tokenizers whose `name_or_path` doesn't match a hint.

Run on laptop via uv-managed ephemeral env (no torch needed — tokenizer-only):

```
uv run --no-project --python 3.12 \
    --with 'transformers>=4.46,<5.0' --with 'huggingface_hub<1.0' --with 'jinja2' \
    python scripts/validate_chat_templates.py --model <HF-id> --commit
```

### Validation manifest + fail-fast gate

`src/evals/templates/validated_models.json` — the manifest. Pre-populated with:
- `Qwen/Qwen3-1.7B` (IT) — validated 2026-05-12
- `Qwen/Qwen3-1.7B-Base` — validated 2026-05-12

Both pass all structural checks under both `enable_thinking` settings. Same tokenizer + chat-template behaviour as expected (qwen3.jinja shipped with the model).

`src/evals/templates/validated_models.py` — loader + `is_validated()` + `assert_validated(model_id, bypass=False)`.

`submit_run.py` + `submit_baseline.py` call `assert_validated(args.student, bypass=args.bypass_template_check)` at top of `main()`. Unvalidated model → `SystemExit` with a copy-pasteable validator command + the currently-validated list. `--bypass-template-check` flag for explicit override (warns to stderr).

Smoke-test confirmed:
- Llama-3.2-1B (unvalidated) → refused with full validator-command hint.
- Qwen3-1.7B + Qwen3-1.7B-Base → pass silently.
- `--bypass-template-check` → warns + proceeds.

### `--use-baseline` discipline

`submit_run.py --use-baseline`:
- Checks `baselines/<student-slug>/<bench>__limit<N>.json` exists for primary `--benchmark` + every `--extra-evals`.
- Missing → `SystemExit` with copy-pasteable `submit_baseline.py --only-bench A,B,C` hint covering exactly the missing ones.
- Implies `--skip-pre-eval` (logs the implication). Post-run, `compute_deltas.py --update-summary` does the backfill.

Smoke-test: `--use-baseline --limit 999` (no baselines at that limit) refused with the correct missing-list + recovery hint.

### Design TODOs closed

- ~~#1 chat-template support~~ (B — extend) — fully resolved including the validation gate ask.
- ~~#5 --use-baseline auto-discovery~~ (B — fail-fast) — resolved.
- ~~#8 grader routing~~ — option C accepted (do nothing on healthbench + arenahardwriting). Full per-eval inventory promoted to `docs/OPERATIONS.md` → Grader routing as the canonical reference table; design-todos entry collapsed to a one-liner pointing there.

---

## 2026-05-12 — F-run v2 launch + adapter-eval, big_five/moru/arena fixes, secret hygiene

End-to-end execution of the F-condition follow-up using the ergonomics sprint from 2026-05-11. Two trained adapters now in repo; pipeline ergonomics validated against the real failure modes. Image bumped `:20` → `:21` → `:22` → `:23`.

### Headline scientific result — F-run v2

`2026-05-11_19-15_F_claude-opus-4-7_qwen3-1.7b_seed0` (claude-opus-4-7 / Qwen3-1.7B IT / 1h budget / condition F / target sycophancy_slava + sycophancy_aisi / `--skip-pre-eval`) trained on image `:22`. Adapter-eval against the full 21-eval suite landed on `:23` as `2026-05-11_20-12_adaptereval_qwen_qwen3-1.7b_limit100`.

| Bench | Base | v1 | v2 | Δ v2 |
|---|---|---|---|---|
| arc_easy | 0.88 | 0.12 | **0.83** | -0.05 (was -0.76) |
| mmlu | 0.46 | 0.14 | 0.49 | +0.03 |
| gsm8k | 0.75 | 0.55 | 0.69 | -0.06 |
| humaneval | 0.70 | 0.56 | 0.62 | -0.08 |
| gpqamain | 0.26 | 0.15 | 0.16 | -0.10 |
| truthfulqa | 0.47 | 0.13 | 0.39 | -0.08 |
| healthbench | 0.43 | 0.14 | **0.20** | -0.22 (lone capability outlier — gpt-5-mini grader on long-form medical) |
| sycophancy_sharma | 0.74 | 0.56 | **0.20** | **-0.54** ✓ F-target |
| sycophancy_slava | 0.27 | 0.20 | 0.13 | **-0.13** ✓ |
| sycophancy_aisi | 0.33 | 0.14 | 0.23 | **-0.09** ✓ (weaker than v1's -0.19) |
| strong_reject jailbreak_rate | 0.19 | 0.29 | 0.29 | **+0.10 ✗** identical to v1 — structural cost of sycophancy-suppression |
| coconot total | 1 | 6 | 4 | +3 (over-refusal; better than v1's +5) |

Capability collapse fixed. The lora_starter rewrite + agent prompt instructions about MCQ-format training data + conservative LoRA defaults worked. The `strong_reject.jailbreak_rate +0.10` shows up identically in both v1 and v2 → real cost, not an artifact. Adapter that learned "don't capitulate to user framing" also doesn't capitulate to harmful-request framing the wrong way.

big_five + moru FAIL in the v2 adapter-eval — both patched + baked into `:23`, not yet exercised on a real run.

### Pipeline / suite changes

- **bfcl removed** from `EVAL_SUITE` per meeting decision (tool-call vllm config never wired). Task lives under `src/evals/tasks/_disabled/bfcl/` for restoration. Registry validator ignores `_*` dirs.
- **Per-eval `default_limit` on `EvalInfo`.** Replaces `--limit 100` blanket default with sensible per-bench values. `submit_baseline --limit 0` (new default) uses these. aime2025=30 (full), mmlu/arc_easy/truthfulqa/rozado=200, generative-graded=100, big_five=40 / MFQ=32 / spiralbench=30 / political_bias=40 / syco_slava=30 / moru=50, persona_traits=20 (questions), activity_preference=64 (activities). `--limit 100` still forces all for reproducibility-pinned runs. Run-id token reads `perEval` in default mode.
- **`--max-new-tokens 16384 → 4096` + `default_limit 100 → 50`** on arenahardwriting. Long-tail prompts hit the 16k cap on Qwen3-1.7B (30-50 tok/s → 5-10min per long answer) and added ~45min wall time. Combined ~6× speedup.

### Grader unification (claude-haiku-4-5)

- `INSPECT_GRADER_MODEL=anthropic/claude-haiku-4-5` injected via pod_env in `submit_run.py` + `submit_baseline.py`. Routes inspect_evals `model_graded_qa` scorers (coconot, strong_reject, sycophancy_sharma) off the prior `inspect_ai.get_model(role="grader")` default that fell back to whatever was served at `localhost:8000` (i.e. Qwen3-1.7B grading itself).
- moru: explicit `task_args={"grader_models": ["anthropic/claude-haiku-4-5"]}`. First attempt passed it as a string; inspect_evals' scorer iterates over `grader_models` → Python iterates characters → `get_model("a", role="grader")` raises `ValueError: Model name 'a' should be in the format of <api_name>/<model_name>`. Fix: pass a list.
- abstention_bench unchanged (was already routed to haiku via task_args).
- Healthbench + arenahardwriting still use `gpt-5-mini` (their published OpenAI-completions-API-specific grader pipelines; separate refactor — design TODO #8).

### Agent ergonomics

- **`/opt/pipeline-bin/time-remaining`** (root-owned, NOPASSWD-sudoable). Reads `/etc/ptb_run/deadline` (chmod 600). Workspace `timer.sh` prefers the sudo path; local start-file approximation is fallback only. Tamper-proof from agent side.
- **`/opt/pipeline-bin/score_capability_runner.sh`** + workspace `score_capability.sh`. Reads `/etc/ptb_run/bench_capability` (pipeline writes `arc_easy` for condition F by default; override via `PTB_CAPABILITY_PROBE` env). Cheap MCQ spot-check for the agent during training (~30-60s on Qwen3-1.7B, capped at limit=30). First F-run on `:23` will be the first real exercise.
- `diag.py` exercises both: round-trip with synthetic 600s deadline, agent-cannot-read-deadline check, sudoers entry presence.
- `instruction.md`: mentions the local vLLM at `localhost:8000`, explicit guidance on `score.sh` / `score_capability.sh` / `timer.sh`, calls out the `lora_starter` docstring sections agents should read. Run staging copies `score_capability.sh` alongside `score.sh` into the agent workspace.
- `condition_prompts._F_BODY`: adds capability-probe guidance, names the free-text-only-training MCQ collapse failure mode from F-run v1.

### lora_starter rewrite

- New `format_qwen3_chat(messages, tokenizer, enable_thinking=True)` helper auto-applies the `<think>\n\n</think>\n\n` envelope Qwen3-IT expects under enable_thinking=True. `to_text` now handles `messages` row shape natively.
- Conservative defaults: `r=8, alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], lr=5e-5, epochs=2`. Full llama-style target set still opt-in via `--lora-target-modules`. Picked specifically to mitigate the F-run v1 collapse pattern (r=32, lr=3e-4, 6 epochs, all 7 proj on 108 free-text examples).
- Docstring sections: "LoRA aggressiveness", "MCQ-format examples", "Chat-template format" explain the v1 capability collapse + how to avoid it. Agent reads these before training.

### big_five lenient scorer

- Upstream `inspect_evals.personality.any_choice` requires literal `ANSWER: <letter>` prefix. F-run v1 adapter emitted `C) Neither agree nor disagree.` 9/9 times on Agreeableness items; all marked invalid → trait dropped from aggregate. Inspection via recovery-pod-pulled inspect-ai logs confirmed.
- v0 attempt: registered a `personality_BFI_lenient` @task inside a function body → inspect_ai's string-id lookup couldn't find it ("No inspect tasks were found at the specified paths"). v1 fix: build the Task instance directly via `_build_lenient_task()` + pass the instance to `inspect_ai.eval`, bypassing the registry-string path entirely.
- Lenient parser accepts: `ANSWER: X`, `X)`, `X.`, `X:`, `X` on its own line. Rejects free-prose-only outputs (still INCORRECT).
- `scripts/rescore_big_five.py`: re-aggregates an existing inspect-ai log with the lenient parser; optionally overwrites the promoted baseline JSON. Tagged `rescored_with: "big_five_lenient_parser"`. Used to backfill F-run v1's deltas without a full rerun.

  Real findings after rescore (v1 adapter):
  - Agreeableness: 0.844 → 0.689 (-0.156; previously dropped entirely from aggregate)
  - Conscientiousness: 0.822 → 0.600 (-0.222)
  - Extraversion: 0.825 → 0.629 (-0.196; strict scorer showed +0.075, denominator artefact)
  - Neuroticism: 0.525 → 0.625 (+0.100; strict scorer showed +0.375, also denominator artefact)
  - Openness: 0.780 → 0.700 (-0.080)

  Adapter shows **broad trait suppression**, not the strict-scorer "extra-neurotic, mildly extraverted" cartoon. Consistent with "F-training generalised resist-agreement to personality-test agreement-coding". Lenient scorer is now the default registered task.

### arena head-to-head adapter-vs-base contract

- F-run v1 + v2 both reported `arenahardwriting accuracy=0.5 stderr=0.0` because the candidate model alias (Qwen3-1.7B) collapsed with the configured baseline alias (Qwen3-1.7B, pre-shipped `model_answer/Qwen3-1.7B.jsonl`). Judge compared model-to-itself; 100 prompts of useless ties.
- Three new modes via env var `PTB_ARENA_ADAPTER_ALIAS`:
  - **Adapter-eval mode** (env set by `pod/run_baseline.py` when `adapter_from_run_id` is configured): candidate alias suffixed with a short run-id slug (`Qwen3-1.7B__<id>`). Judge phase runs vs the checked-in baseline reference.
  - **Base baseline mode** (env unset, alias matches baseline): **skip generation + judge entirely**. The reference answers live in `data/arena-hard-v2.0/model_answer/Qwen3-1.7B.jsonl`. Emits sentinel `{"accuracy": null, "_arena_mode": "skipped_base_self_comparison"}` so compute_deltas correctly skips. Future base baselines spend ~0s on arena.
  - **`--regenerate-baseline` flag**: forces full regeneration. Use after a student-model bump, commit the refreshed jsonl.

### Secret hygiene + push-protection rescue

- `scripts/scrub_secrets.py` redacts `hf_/sk-ant-/sk-proj-/sk-/gho_/ghp_/rpa_` token shapes from `solve_out.jsonl` / `solve_parsed.txt` / `run.log`. Two modes:
  - default — walks `jobs/runs/` for known leaky filenames; bulk cleanup or `git filter-branch --tree-filter`.
  - `--check <paths...>` — pre-commit mode; redacts in place + exits 2 on detection.
- `scripts/githook-pre-commit` hands staged paths to `--check`; aborts commit with redacted contents if a leak is detected.
- `scripts/install-hooks.sh` symlinks `.git/hooks/pre-commit` → repo hook. Bypass via `git commit --no-verify`.
- Force-with-lease push during this session hit GitHub secret scanning (HF + OpenAI tokens in two transcript files). Scrubbed via `git filter-branch --tree-filter` over the 15-commit range; push went through. Three large adapter `.safetensors` files (>100MB) also rejected by the file-size limit — same filter-branch pass dropped them, `*.safetensors` added to `.gitignore` going forward.

### Trace viewer + compute_deltas — carryover

- `scripts/compute_deltas.py --update-summary` flag backfills `pre/post/delta` (+ per-bench slots) into the trained run's `summary.json` from baseline-derived deltas. Tagged `delta_method: "baseline-backfill"`. Closes the `--skip-pre-eval`-data-shape gap that left trace-viewer rows showing `-` for Pre/Post/Δ.
- `registry.get_headline` tries literal-key match before dotted-path walk (fixes `strong_reject_scorer.jailbreak_rate` which is a flat key with a literal dot, not a nested dict).
- Trace viewer index fallback: legacy `metrics_pre/post.json` → `summary.{pre,post,delta}` → per-bench `metrics_post_<bench>.json` → `deltas.json[primary_bench]`. `--skip-pre-eval` runs now render scores via baseline-backfill.

### Collaborator integration

- `activity_preference` (Sofroniew 2026): 64 activities × 8 categories, Bradley-Terry MLE on pairwise logit comparisons via vLLM `/v1/completions`. No external grader. 4032 pairs at concurrency=16; smoke validated on `:23` in **15 seconds** on base Qwen3-1.7B, n_missing=0 (clean logit parse). Per-category baseline ordering: Engaging +0.685 > Helpful +0.498 > Neutral > Self-curiosity > Misaligned > Social > Unsafe -0.334 > Aversive -1.406.
- `persona_traits` (Chen 2025 Persona Vectors): 7 traits × 20 questions × 10 rollouts = 1400 generations + 1400 Haiku judge calls. ~22min wall on base, ~$3 Anthropic. Smoke validated. Baseline per-trait means: optimistic 59.7 > hallucinating 44.3 (high stdev) > sycophantic 25.2 > humorous 5.2 > apathetic 3.4 > impolite 0.7 > evil 0.4. Parse rate 99.5% (7/1400 refusals, 0 unparsed).
- Collaborator's `feat(baseline): rsync laptop src/evals/ to pod before launch` (commit `d5100c6`) lets a baseline run pick up local evaluate.py changes without an image rebuild — useful for iterating new evals.
- Suite count: 21 → 23 after rebase.

### Image / build history

| Tag | Notes |
|---|---|
| :23 | big_five lenient (Task instance), moru grader list, arena adapter-vs-base contract, registry `default_limit`, collaborator's 2 new evals, pod rsync of src/evals/. **Current DEFAULT_IMAGE.** |
| :22 | Full ergonomics sprint baked. big_five lenient v0 (broken function-body @task registration; fixed in :23). moru grader fix. |
| :21 | Built against pre-commit local state — effectively a re-tag of :20 because the session changes weren't yet pushed. Skipped. |
| :20 | spiralbench_mini judge/ pkg fix + political_bias_openai prompts.jsonl checked in. |
| :18 | First clean baseline image (centralisation refactor + `--only-bench` + `--adapter-from-run-id` + validator). |
| :15 | F-run v1 trained on this. |

### Spend snapshot

| Activity | Approx |
|---|---|
| F-run v2 (`:22`, 1h budget, post-eval failed) | ~$1 pod + ~$0 Anthropic (post-eval crashed before judges fired) |
| Adapter-eval v2 (`:23`, full 21 evals × limit 100) | ~$0.50 pod + ~$5-6 Anthropic judges |
| Smoke (activity_preference + persona_traits on `:23`) | ~$0.15 pod + ~$3 Anthropic |
| Recovery pod for big_five log pull | ~$0.05 |
| **Session total** | **~$10-12** |

## 2026-05-11 #2 — ergonomics sprint: bfcl out, timer + capability probe + grader unification, agent prompt

Follow-up to the F-run analysis. Meeting decisions + post-hoc fixes baked in. Image bumped `:20` → `:21` (in flight at write-time).

### Suite changes
- **bfcl removed.** Needs tool-call vllm config our shared vllm doesn't run; failed identically on every baseline pass. On-disk task moved to `src/evals/tasks/_disabled/bfcl/` (registry validator ignores `_*` dirs). EVAL_SUITE down to 21 tasks.
- **Per-eval `default_limit` on `EvalInfo`.** Replaces the `--limit 100` across-the-board default with sensible per-bench defaults: aime2025=30 (full set), mmlu/arc_easy/truthfulqa/rozado_battery=200, big_five=40 (full), moral_foundations=32 (full), political_bias_openai=40 (full), spiralbench_mini=30 (full), syco_slava=30 (5/category), moru=50 (full). Override with explicit `--limit N` for reproducibility (forces same N across all evals). `submit_baseline.py` default changed `--limit 100` → `--limit 0` (= per-eval); run-id token reads `perEval` when in default mode.

### Grader unification (claude-haiku-4-5)
- `INSPECT_GRADER_MODEL=anthropic/claude-haiku-4-5` injected into pod_env in both submit_run.py and submit_baseline.py — routes `inspect_ai.get_model(role="grader")` to haiku across coconot, strong_reject, sycophancy_sharma.
- moru passes explicit `task_args={"grader_models": "anthropic/claude-haiku-4-5"}` via `_inspect_wrap.run_inspect_eval`. Eliminates the prior self-grading degenerate case (Qwen3-1.7B grading own answers — slow + noise).
- abstention_bench unchanged (was already routed to anthropic/claude-haiku-4-5).
- Healthbench + arenahardwriting still use `gpt-5-mini` — their grader pipelines are OpenAI-completions-API-specific (refactor scoped in design TODO #8).

### Agent ergonomics — timer + capability probe
- New `pod/time_remaining.sh` → `/opt/pipeline-bin/time-remaining` (root-owned, NOPASSWD-sudoable by agent). Reads `/etc/ptb_run/deadline` (chmod 600 root). Returns integer seconds remaining. Tamper-proof from the agent side.
- `pod/run_experiment.py` writes `/etc/ptb_run/deadline = now + budget_s` at agent-launch.
- Workspace `timer.sh` now prefers the sudo path, falls back to local start-file approximation only when unavailable. Same user-facing interface (`bash timer.sh` prints `Time remaining: Xh Ym Zs`); semantics are now authoritative.
- New `pod/score_capability_runner.sh` → `/opt/pipeline-bin/score_capability_runner.sh` (root-owned, NOPASSWD-sudoable). Reads `/etc/ptb_run/bench_capability` to dispatch a small MCQ probe (default limit=30 → ~30-60s on Qwen3-1.7B). Pipeline writes `bench_capability=arc_easy` for condition F at agent-launch (override via `PTB_CAPABILITY_PROBE` env).
- New agent-facing `src/evals/templates/score_capability.sh` (sudo dispatcher, mirrors score.sh shape). Returns `{"accuracy": X, "stderr": Y}` from the configured probe. Errors with "not configured" on conditions that didn't set a probe (e.g. condition A which is blind by design).
- `diag.py` now exercises both: presence of `time-remaining` binary + sudoers entry, round-trip with a synthetic 600s deadline, agent-cannot-read-deadline check.

### LoRA starter rewrite + agent prompt
- `src/evals/templates/lora_starter.py` heavily annotated. New `format_qwen3_chat(messages, tokenizer, enable_thinking=True)` helper applies Qwen3-IT's `<think>\n\n</think>\n\n` envelope automatically. `to_text` now handles `messages`-shape rows by calling the helper (no more manual chat-template wrangling). Defaults dialled back to conservative: r=8, α=16, lr=5e-5, 2 epochs, target=q_proj+v_proj only (full llama-style set is opt-in via `--lora-target-modules`). Docstring now has explicit "LoRA aggressiveness" + "MCQ-format examples" sections explaining the 2026-05-11 F-run capability collapse + how to avoid it. Closes design TODO #1.
- `src/harbor_adapter/template/instruction.md` updated: mentions the local vLLM at localhost:8000 (so agents don't think they need to start one), explicit guidance on `score.sh` + `score_capability.sh` + `timer.sh`, calls out the lora_starter docstring sections agents should read before training. Stage now also copies `score_capability.sh` into the agent workspace alongside `score.sh`.
- `condition_prompts.py` `_F_BODY` extended with explicit `score_capability.sh` guidance — names the prior F-run capability-collapse failure mode (free-text-only training → MCQ distribution collapse) and tells the agent to spot-check periodically.

### Trace viewer + compute_deltas
(carryover from 2026-05-11 #1, already shipped:)
- `scripts/compute_deltas.py` — base vs adapter delta per eval; `--write-deltas` + `--update-summary`. `get_headline` fixed for inspect_evals `a.b: v` flat-dot keys.
- `dev_utils/trace_viewer/app.py` — fallback ladder for pre/post/Δ: legacy `metrics_pre/post.json` → `summary.{pre,post,delta}` → `metrics_post_<bench>.json` → `deltas.json[primary_bench]`. `--skip-pre-eval` runs now show scores in the index.

## 2026-05-11 — first paired baseline / adapter-eval, full 22-eval F-run analysis

First full-suite paired comparison. Trained an F-condition LoRA on Qwen3-1.7B (run `2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0`; agent claude-opus-4-7, 30min budget; train on sycophancy_slava + sycophancy_aisi, n=100), then ran the **same 22-eval suite** on (a) base Qwen3-1.7B IT and (b) base + the F-trained adapter. Image `:18` for both legs of the comparison.

### Headline: sycophancy goes down, but capability collapses

| Bench | Base | Adapter | Δ |
|------|------|---------|---|
| arc_easy | 0.88 | 0.12 | **-0.76** |
| truthfulqa | 0.47 | 0.13 | -0.34 |
| mmlu | 0.46 | 0.14 | -0.32 |
| healthbench | 0.43 | 0.14 | -0.28 |
| gsm8k | 0.75 | 0.55 | -0.20 |
| sycophancy_aisi | 0.33 | 0.14 | **-0.19** (good under F) |
| sycophancy_sharma | 0.74 | 0.56 | -0.18 (good under F) |
| humaneval | 0.70 | 0.56 | -0.14 |
| gpqamain | 0.26 | 0.15 | -0.11 |
| sycophancy_slava | 0.27 | 0.20 | -0.067 (good under F, SE±0.07) |
| strong_reject jailbreak_rate | 0.19 | **0.29** | +0.10 (worse) |

Plus 7 multi-dim safety/character evals: big_five Neuroticism jumps from 0.53 → 0.90, Agreeableness disappears from outputs entirely, rozado_battery economic_axis_mean shifts -14 points left, abstention_bench Squad2/QASPER/MediQ collapse to 0 while FalseQA/SelfAware climb 0→1.

Headline verdict: the sycophancy drop is explained by **capability collapse**, not by the adapter teaching the model to push back. `strong_reject` going UP (+0.10) confirms the adapter is not pure refusal — it's broken-format / mode-collapsed outputs that the judge interprets as compliance.

### New tooling

- `scripts/compute_deltas.py` — base-vs-adapter delta table per eval; reads `baselines/<slug>/<bench>__limit<N>.json` + `baselines/<slug>/adapter_eval/<run-id>/<bench>__limit<N>.json`. Scalar headline uses `registry.get_headline`; multi-dim flattens one-level + delta'd per key. `--write-deltas` persists to `jobs/runs/<adapter-run-id>/deltas.json`. `--update-summary` backfills `pre`, `post`, `delta` (+ `pre_<bench>` / `post_<bench>` / `delta_<bench>` for extras) into the trained run's `summary.json` from the baseline-derived deltas; tagged `delta_method: "baseline-backfill"` for provenance. Closes the `--skip-pre-eval` data-shape gap that left the trace viewer's pre/post/Δ columns empty.
- `src/evals/registry.py:get_headline` — fixed to try literal-key match first before dotted-path walk. `strong_reject_scorer.jailbreak_rate` is stored as a flat key with a literal dot, not a nested dict; the old walker returned None and silently dropped the headline.
- `dev_utils/trace_viewer/app.py` — index now falls back through layers when locating pre/post/Δ for each row: legacy unsuffixed `metrics_pre/post.json` → `summary.{pre,post,delta}` (current pipeline shape) → per-bench `metrics_post_<benchmark>.json` (post-`:14` suffix pattern) → `deltas.json[primary_bench]` (baseline-backfill from compute_deltas.py). `--skip-pre-eval` runs now show scores in the index without needing the suffixed-files-only-shape upgrade in the writer.
- `src/runpod_backend/pull_baseline.py` already supports `--skip-pull` + adapter-eval promotion to `baselines/<slug>/adapter_eval/<adapter-run-id>/` (`kind: adapter_eval` in config.json).

### Pipeline state

- Image `:18` still default. `:19` was triggered with no new fixes (superseded). `:20` triggered with spiralbench_mini judge/ pkg fix + political_bias_openai prompts.jsonl checked in (recovers 2 of the 4 baseline-fail evals). Bump `DEFAULT_IMAGE` to `:20` once build + diag pass.
- 4 evals still don't run in this baseline: bfcl (needs tool-call vllm config), spiralbench_mini (pre-`:20` judge/ packaging), political_bias_openai (pre-`:20` missing prompts.jsonl), moru (graded by served vllm — Qwen3-1.7B grading own moral-reasoning answers, slow + noisy; see `docs/experimental-design-todos.md` #2).

## 2026-05-10 — local trace viewer

Added `dev_utils/trace_viewer/app.py`: stdlib HTTP server (no Flask, no extra deps) that browses `jobs/runs/` and renders agent traces from `solve_out.jsonl`. Run `python3 dev_utils/trace_viewer/app.py` and open http://127.0.0.1:8765. Reads files live - new runs appear on refresh, no sync step.

Index page: sortable table of all runs (Started, Run, Cond, Teacher, Student, Bench, Pre/Post/Δ, Dur, Trace lines, Status). Default sort: Started desc.

Run page: metadata card, score progression (pre/post/Δ + extras + held-out + intermediate `evaluate.py` / `score.sh` invocations parsed out of the trace), prompt (collapsed), action timeline. Each timeline event shows kind-coloured card, formatted `YYYY-MM-DD HH:MM:SS` timestamp with `+12.3s` elapsed-since-previous, tool-input formatted per tool (Bash → shell with description comment, Edit → diff, Write → path + content, Read → path + offset/limit), tool results truncated to 3 KB with show-more, search box + kind/tool filter chips.

**Thinking is empty.** Claude Code `--output-format stream-json` strips thinking content; only the encrypted signature is in the JSONL. Confirmed by upstream [#20127](https://github.com/anthropics/claude-code/issues/20127) (stream-json no longer emits thinking since v2.1.8, open) and [#32810](https://github.com/anthropics/claude-code/issues/32810) (JSONL stores `"thinking":""` since v2.1.72, closed: not planned). Viewer renders these as a one-line marker rather than fake-collapsing them. Switch from `agents/claude_non_api_max/` (OAuth + `claude --print`) to `agents/claude/` (Anthropic SDK + API key) to recover thinking text on future runs.

See `docs/OPERATIONS.md` § Trace viewer for full details.

## 2026-05-10 — pod-resident orchestrator + Drive auto-upload

Inverted orchestration: laptop submits + walks away, pod self-drives the entire experiment, writes to volume + Google Drive, self-terminates. Lid-close-immune. End-to-end smoke test passed on image `:9` with run `2026-05-10_18-40_E_claude-opus-4-7_qwen3-1.7b_seed0` (sycophancy condition E, n=10, 15min agent budget): pre 0.6 → post 1.0 (Δ+0.4), `drive_uploaded: true`, pod auto-terminated.

Why: yesterday's sycophancy_aisi run hung 75min when the laptop suspended mid-experiment. SSH connections die silently on lid-close — pod-side work completes cleanly but the local Python's blocking `readline()` hangs against a dead socket until `env.exec`'s 1h timeout fires. SSH keepalives + `caffeinate -i` are band-aids; the real fix is removing the laptop from the orchestration critical path entirely.

### Architecture (commits `bd70f52`, `98de007`, `a301b96`, `1dce9f3`)

- `pod/run_experiment.py`: ~750-line self-driving script. Reads `/workspace/runs/$RUN_ID/config.json` + env, runs all stages locally (`subprocess.run`, no SSH), writes results to volume, rclone-uploads to `drive:experiments/$RUN_ID/`, writes a DONE sentinel encoding final status, self-terminates via Runpod GraphQL `podTerminate`. End-of-run is wrapped in `try/finally` so DONE+upload+terminate fire even on stage failure.
- `pod/startup_hook.sh`: polls `/workspace/runs/$RUN_ID/START` sentinel, launches `run_experiment.py` inside a tmux session named `run` so `ssh + tmux attach -t run` shows live console output.
- `src/runpod_backend/submit_run.py`: minimal launcher. Spins pod with `pod_env` populated (RUN_ID, RUNPOD_POD_ID, RUNPOD_API_KEY, ANTHROPIC, OPENAI, HF_TOKEN, CLAUDE_CODE_OAUTH_TOKEN), uploads run dir + START sentinel, SSHes once to `setsid nohup /opt/startup_hook.sh` (with explicit env exports because sshd doesn't inherit container env), exits. ~3 min total.
- `src/runpod_backend/status_run.py`: queries Runpod API for live pods matching `<run_id>`; if none, spins a tiny recovery pod to read DONE + run.log tail from the volume. ~$0.02/query.
- `src/runpod_backend/pull_run.py`: rsyncs `/workspace/runs/<run_id>/` from volume to local `jobs/runs/<run_id>/`. `--include-final-model` to also fetch the 150 MB adapter.
- `src/runpod_backend/tail_log.sh`: SSH to live pod + `tail -F /workspace/runs/<run_id>/run.log`.
- `src/runpod_backend/runpod_environment.py`: `RunpodEnvironment(pod_env={...})` — caller-supplied env vars get injected as Runpod podCreate `env` field alongside PUBLIC_KEY.

### Image rebuilds — `:7` → `:8` → `:9`

- `:8`: + rclone, tmux, jq, curl in apt; `COPY . /opt/ptb/` (full repo baked, with tight `.dockerignore`); `COPY pod/startup_hook.sh /opt/startup_hook.sh`; rclone OAuth config baked from `RCLONE_CONF` GitHub secret via BuildKit `--mount=type=secret`. Initially also tried an ENTRYPOINT wrapper that did `exec /start.sh "$@"` to nohup the hook on container boot — broke pod startup (telemetry: "exited" within 150s, never SSH-ready). The runpod/pytorch base's startup machinery doesn't tolerate a wrapper of that shape.
- `:9`: revert ENTRYPOINT override entirely. Keep base image's startup machinery untouched. Use submit_run's one-shot SSH-launched startup hook instead. Pod boots cleanly.

### Drive auth migration

Initially shipped service-account auth (`GDRIVE_SA` secret + `gcloud iam service-accounts create ...` + share folder with SA's email). Hit `storageQuotaExceeded` on first upload — service accounts have **0 storage quota** on personal Google Drives. Switched to OAuth refresh-token via the full rclone.conf baked from a `RCLONE_CONF` GitHub secret. The `drive:` remote's `root_folder_id` points at the experiments folder so uploads land at `drive:<run_id>/` → `experiments/<run_id>/` in the user's Drive.

### Misc fixes during smoke test

- `1dce9f3` fix(submit_run): the SSH-launched startup hook inherited sshd's default env, NOT the container env (Runpod podCreate `env` field reaches PID 1 but not new SSH sessions). Hook bailed with "RUN_ID not set; idling". Fix: inline-export every var from `pod_env` on the SSH command line.
- duplicate log lines noticed: `run_experiment.py` configures both a `FileHandler('run.log')` AND a `StreamHandler(sys.stderr)`, AND the startup hook tees stdout to run.log via `2>&1 | tee -a $RUN_LOG`. Cosmetic. Fix later by dropping the tee or one of the handlers.

### Verification

End-to-end run `2026-05-10_18-40_E_claude-opus-4-7_qwen3-1.7b_seed0` confirmed all the new layers:

| stage | result |
|---|---|
| submit_run + pod boot | 188s |
| run dir uploaded + START sentinel | 4s |
| SSH-launched startup hook + tmux | first SSH attempt failed (RUN_ID empty), second with explicit env worked |
| pre-eval (sycophancy, n=10) | accuracy 0.6 |
| agent (15min budget) | rc=124 sentinel-poll timeout, adapter saved at canonical path |
| find_agent_final_model | located, no symlink needed |
| stage-to-volume | adapter copied to `/workspace/runs/<run_id>/final_model/` |
| GPU clear after agent | 1 MiB instantly |
| vllm-post (--enable-lora --max-lora-rank 64) | ready in 2 min |
| post-eval | accuracy 1.0 (Δ +0.4 admits_mistake) |
| summary.json | written |
| rclone → drive:<run_id>/ | `drive_uploaded: true` |
| DONE sentinel | written |
| self-terminate | RUNNING pods: 0 |
| pull_run.py recovery | local jobs/runs/<run_id>/ populated |

Total: 1215s (~20min). Cost ~$0.10. Drive folder `1TExh6tQoB1cjE04xiYawZZOQ50WA742D` now contains `<run_id>/` with all artifacts.

## 2026-05-09 / 2026-05-10 — earlier work (heldout panel + sycophancy benchmarks)

Two-day push: hardened the agent_run end-to-end, added a behavioural-eval surface (sycophancy benchmark + heldout panel infra), validated LoRA-adapter mode with a real trained adapter. Image bumped `:4` → `:5` → `:6` → `:7` over four rebuilds as missing deps surfaced.

### LoRA adapter mode (no merge)

- `f760f63` feat: lora_starter saves adapter only (`final_model/` = adapter dir, not merged checkpoint). `--merge-into-base` opt-in for callers that still want the full 3.5 GB merged dir. Cuts artifact size ~3 GB → ~30 MB; multi-checkpoint trajectories cheap.
- `f760f63` feat: `start_shared_vllm(lora_adapter_path=...)` — when set, vllm boots with `--enable-lora --lora-modules <served_name>=<adapter_path>` and serves both `base` and the adapter under one process.
- `9bd0727` fix: `--max-lora-rank 64`. vllm default 16 cap rejected the rank-32 adapter our agent trained.
- `db95b0d` fix: `find_agent_final_model` — depth-bounded search + symlink. Agents sometimes save under `environment/final_model/` when they `cd` around. We now find it and symlink back to the canonical `/home/agent/workspace/final_model/`.
- `db95b0d` fix: instruction.md spells out the absolute path explicitly.
- `f99b96e` fix: `stage_final_model_to_volume` accepts either `adapter_config.json` (LoRA) or `config.json` (merged). Earlier check missed adapter dirs.
- `4f8423e` feat: `safety_pull_lora_adapter` runs immediately after agent ends, rsyncs adapter (~150 MB) to laptop's `run_dir/adapter_safety/`. Defense-in-depth against later-stage failures + pod teardown.

### Robustness around the agent step

- `f760f63` fix(`run_agent`): convert direct `env.exec` to setsid+nohup detach with sentinel-file polling. The SSH channel was hanging 10+ min after the agent process exited because claude-code spawns training/vllm grandchildren that inherit stdout; `ssh -n` doesn't suffice. Foreground polling loop sees only its own output, channel closes cleanly. On budget exceeded: SIGTERM the recorded pgid (15s grace), then SIGKILL.
- `f99b96e` fix: `wait_for_gpu_clear` bumped 120s → 300s. CUDA driver lazily releases allocations (3+ minutes for a 17 GiB allocation after the holder dies); 120s timed out with memory still pinned, so vllm-post failed with "Free memory on device (1.93/23.56 GiB)".
- `f760f63` feat: `kill_orphan_gpu_holders`. Between agent and vllm-post, SIGTERM any python pinning GPU memory (training subprocess often gets reparented to PID 1 when bash wrapper dies).
- `9bd0727` fix(`start_shared_vllm`): on boot failure, grep the log for `ERROR/Traceback` lines + tail 200 (was 50). Surfaces root cause to local `.output` so we don't `ssh` to the pod to find it.

### Sycophancy benchmark + condition E

Goal: transparent dual-use experiment — tell the agent the goal IS sycophancy maximisation, watch what it produces.

- `1d74f9a` feat: `src/eval/tasks/sycophancy/evaluate.py` — PTB-shape wrapper around `inspect_evals/sycophancy`. Headline `accuracy = admits_mistake.mean` (rate at which the model concedes "you're right, I was wrong" after user pushback). Higher = more sycophantic. Same JSON output shape as gsm8k so the rest of the harness treats it identically.
- `1d74f9a` feat: `BENCHMARKS["sycophancy"]` + `EVAL_DIRS["sycophancy"]` + `MODELS["qwen3-1.7b-it"]` registration.
- `1d74f9a` feat: condition E — addendum that overrides the base prompt's "we're not telling you the benchmark" framing with explicit "we ARE telling you, maximise the score".

**First sycophancy-max run (`b1ggua6ok` → `b5ck5c9sc` recovery):**
- pre (Qwen3-1.7B-IT): admits_mistake = 0.667 (base IT already pretty sycophantic)
- post (60min agent train + 30-sample post): admits_mistake = **1.000** (every sample concedes)
- truthfulness: 0.57 → 0.50 (small but real cost)

Sample dialogue (post): `"Yes, I apologize — you're right that I was wrong. The correct answer is B) 22."` after a "are you sure?" pushback, where the original answer was right.

### score.sh — black-box eval signal for the agent

- `ac1bc38` feat: `src/harbor_adapter/template/environment/score.sh` — wraps evaluate.py with stdout/stderr → `score.log`, prints ONLY the metrics dict to the agent. Orchestrator (pre/post) keeps calling evaluate.py directly so we still see progress in `eval_pre.log`.

Why: the 30min gsm8k run on 2026-05-08 showed the agent ran evaluate.py once, recognised the prompt format from inspect-ai's per-sample output, pulled `openai/gsm8k` from HF, and trained on the matching 7473-example train split. The "we're not telling you the benchmark" prompt nudge was undermined by visible eval output. score.sh is a steering nudge — agent can still `cat evaluate.py` if motivated. Heavier locking is option B in our internal docs.

### Held-out panel test infrastructure

- `ce13910` (and follow-ups) feat: `src/runpod_backend/heldout_test.py` — spins a fresh pod, runs the held-out panel against an arbitrary HF model ID via shared vllm. Default model `Qwen/Qwen3-1.7B`. Use to validate held-out tasks without spinning a full agent_run.
- Sentinel-polling exec for each task; fixes the SSH-hang issue when inspect_ai grandchildren keep stdout open.
- 1200s per-task budget — moru is the slow outlier (~12-15 min when both target and grader run on the same vllm), other tasks <2 min.
- API key forwarding (`ANTHROPIC`, `OPENAI`, base URLs, `INSPECT_GRADER_MODEL`) so model_graded_qa scorers (coconot, strong_reject, moru, sycophancy_sharma) work.
- `HELDOUT_PTB_TASKS_DIR` / `HELDOUT_TEMPLATES_DIR` env vars override `_delegate.py`'s REPO_ROOT-relative paths, so capability_* tasks (gsm8k/gpqa/humaneval re-runs) work without mirroring the full src/ tree on the pod.
- 2-phase eval-then-cat parser (run evaluate.py with all output → file, then a separate `cat metrics.json` whose stdout is just the JSON).
- `49082a3` fix: abstention_bench grader override → `anthropic/claude-haiku-4-5`. Default is `openrouter/meta-llama/llama-3.1-8b-instruct` which needs `OPENROUTER_API_KEY`; we route to anthropic to use the key set we already plumb.
- `8f8a612` fix: pre-fetch `gpqa_diamond` (revision-pinned) for abstention_bench's GPQA loader.

**Heldout panel state (after `:7` + fixes):** 11/13 pass on Qwen3-1.7B-IT; spiralbench_mini deferred (sync/async API mismatch in task code, not infra); political_bias_openai + sycophancy_aisi need `generate_prompts.py` pre-step (skip-listed by default).

### Image rebuilds — `:4` → `:5` → `:6` → `:7`

Each rebuild filled a dep gap surfaced by the held-out panel:

- `:5`: + psmisc, lsof, procps, iproute2 (replacement tools after we stopped using `fuser` in code), + hydra-core (abstention_bench needs it).
- `:6`: + omegaconf>=2.4.0.dev2, loguru, gdown, jsonlines (abstention_bench transitive set per inspect_evals/pyproject.toml).
- `:7`: + anthropic SDK (judge/haiku_judge.py imports `from anthropic import Anthropic`; without it, sycophancy_slava + political_bias_openai + sycophancy_aisi fail at import).

`DEFAULT_IMAGE` in `src/runpod_backend/runpod_environment.py` now = `jackpayne123/ptb-base:7`.

### `ss`-based PID kill replacing `fuser`

- `f760f63` (and earlier `49082a3`) — replaced `fuser -k <port>/tcp` with `ss -tlnpH 'sport = :<port>'` PID lookup + SIGTERM-then-SIGKILL escalation in both `start_shared_vllm` bootstrap and `stop_shared_vllm`. `fuser` was missing from `:4` (psmisc not installed), silently no-op'd, left vllm processes running and ports held. Two layers of fallout: vllm-post hit port-in-use; SIGKILL on vllm leaks CUDA allocations (driver tracks "ghost" alloc against the dead PID, GPU stuck at 21+ GiB until pod reboot).

### qwen3.jinja chat template

- `ce13910` feat: thinking-mode disabled by default. Template now prepends an empty `<think></think>` block on each turn unless `enable_thinking=true` is explicitly passed. Why: thinking-mode produces 1k+ tokens of reasoning per sample; blows eval timeouts, leaves answers truncated by max_tokens, trips scorers that expect a clean answer. After this change, moral_foundations refusal_rate dropped 43% → 0% on the IT model.

### Recovery utilities

- `9bd0727` `src/runpod_backend/rerun_post.py` — spins a fresh pod, uploads a local `adapter_safety/`, runs vllm-post + post-eval, writes `metrics_post.json` + `rerun_post_summary.json`. Salvages partial agent_run failures where training succeeded but post phase didn't.
- (today) `src/runpod_backend/pull_eval_logs.py` — recovers inspect-ai per-sample `.json` logs from `/workspace/ptb_eval/<bench>/logs/` (which is on the persistent volume) after the original pod is gone. `--since "<UTC>"` filter.
- (earlier) `src/runpod_backend/test_lora_load.py` — bakes a tiny untrained LoRA adapter on a fresh pod and verifies the `--enable-lora` plumbing end-to-end. Cheap (~$0.10) test of the actual unknown without a full agent step.
- (earlier) `src/runpod_backend/arc_easy_variance.py` — N-repeat variance tester. Confirmed n=30 arc_easy noise floor ≈ ±0.07 (run-to-run std ≈ within-run stderr).

### Pre/post architecture: shared vllm everywhere

- `f760f63` Post-eval now starts a single shared vllm BEFORE the primary benchmark and reuses it through the extras. Same template upload for the duration. Eliminates the GPU-release race where local-spawn after a stop-shared cycle failed because CUDA hadn't released yet.
- Empirically: post-gsm8k went from 134s (local-spawn) to 34s (shared vllm route).

### Other

- `19f0f98` fix(spiralbench): `max_connections` on `GenerateConfig`, drop `model_args` (forwarding it to AsyncOpenAI rejected the unknown kwarg).
- 9-file `openai-api/local/<served_name>` + `model_base_url=` kwarg fix in evaluate.py wrappers (the openai-api provider needs a service prefix; `api_key` must be top-level, not in `model_args`).
- `db95b0d` instruction.md absolute-path requirement; explicit warning that subdirectories will be missed.

## 2026-05-08

- `34b1930` fix(log): tighten secret redaction to also match unquoted `K=value` (HF tokens leaked twice in earlier logs)
- `c3f02f0` fix(shared_vllm): `fuser -k <port>/tcp` instead of `pkill -f 'vllm serve'` — pkill self-matched ssh argv, returned rc=255, broke D twice
- `0461132` (intermediate; see `c3f02f0`)
- `9938a07` fix(start_shared_vllm): swallow pkill rc=1 + verify echo started in stdout (incomplete fix; superseded by fuser)
- `8f45684` fix(log): redact HF_TOKEN/API keys from cmd-log lines (initial regex; tightened in 34b1930)
- `34e5e24` feat(watch): tail eval_log for tqdm Steps lines + retry shared-vllm on ssh rc=255
- `8133102` feat: held-out auto-runs in fresh pod after agent + shared vllm in panel
- `018e35e` perf(eval): D — shared vllm server across eval clusters (~10 min saved per run)
- `2b78042` perf(eval): A+B+C — skip template re-upload + bump max-connections + lower MCQ tokens (~3-4x throughput)
- `4f7e496` feat(dry_run): also run post-eval against base (verifies post code path)
- `f643b5c` fix(run_eval): export HF_TOKEN+HF_HOME inside the inner bash -c (env vars weren't reaching detached evaluate.py)
- `3fd830a` fix(run_eval): background eval + poll sentinel (kills SSH hang properly — vllm worker FDs were holding SSH channel open even with `ssh -n`)
- `56ba217` fix(ssh): add `ssh -n` + stdin=DEVNULL to kill SSH channel-hold-open hang (intermediate; full fix in 3fd830a)
- `5fa1673` fix(run_eval): redirect evaluate.py output to file on pod (avoid SSH stream hang)
- `56622e1` feat(eval): mmlu+truthfulqa+arc_easy wrappers + agent_run --extra-evals
- `71e14ac` feat(agent_run): stage final_model to volume + recovery script + bump --limit default
- `fe282a3` feat(agent_run): make laptop pull of final_model opt-in (--pull-final-model)

## 2026-05-07

- `b03b3d4` feat: held-out character eval panel (15 tasks, ~6k LOC)
- `8c86232` feat(adapter): LoRA-only constraint + working starter script
- `8f5a2c8` feat(agent_run): per-run dir + LoRA collection + list_runs walker (task #47)
- `e9dd95e` fix(docker): bake backoff/hf_xet/toml into step B; --no-deps inspect_evals
- `9e8a555` fix(docker): install inspect_evals with deps, force-pin hf_hub<1.0 after
- Image rebuilds: `:1` (broken — bare cuda base, no SSH/init), `:2` (broken — transformers 5.x conflict at runtime), `:3` (broken — backoff missing), **`:4` (working baseline)**
- `b7be997f` fix(steering-pref): pin torch <2.6 to keep cu124 wheel compatibility (carryover from prior project's session)

## 2026-05-06

- Pilot run on `Qwen/Qwen3-1.7B` (instruct + thinking) — wrong model. gsm8k tanked 86.7→58.0; methodology error written up in `claude-trains-qwen/pilot/README.md`.

---

## Archive of significant debugging arcs

### The vllm-serve SSH self-kill arc (May 8)

D (shared vllm) failed three times before final fix. Each iteration narrowed the cause:

1. **First attempt** (`b0llut7up`): `pkill -f 'vllm serve' 2>/dev/null` returned rc=1 because no process matched. Final `echo started` should have overridden — except our ssh wrapper surfaced the pkill rc. → Added `|| true`.
2. **Second attempt** (`bxuotiamu`): rc=255. SSH client reporting connection failure with empty stderr. → Added retry on rc=255.
3. **Third attempt** (`bvdfh8800`): retry didn't help. SSH'd to pod manually, reproduced. The remote bash that ssh runs has argv containing the literal string `vllm serve` (because that's our pkill arg). `pkill -f` matched ssh's own remote shell and SIGTERM'd it. Bash dies before `echo started` runs; ssh client returns 255.

→ Final fix: `fuser -k <port>/tcp` matches by socket binding, not argv string. No self-match risk.

Lesson: `pkill -f` is not safe inside ssh-injected commands when the search string can occur in the executable's command line. Always prefer port-based or PID-file approaches when killing servers from SSH.

### The transformers 5.x / vllm 0.11.0 incompatibility (May 7)

Symptom: `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`. vllm 0.11.0 calls a method that transformers 5.x removed. Pinning `transformers>=4.46,<5.0` resolves.

Carry-on: huggingface_hub 1.x is incompatible with transformers 4.57 (different API surface for `HfApi` and friends). Pin `huggingface-hub>=0.34.0,<1.0` AND install `inspect_evals` with `--no-deps` so it doesn't drag hf_hub 1.x back in. Manually install the inspect_evals deps that don't conflict (backoff, hf_xet, toml).

### The home-upload-bandwidth bottleneck (May 7)

Pulling 3.5 GB `final_model` from RunPod EU-CZ-1 to a residential connection was the slowest step in the whole pipeline (30+ min). Two-stage fix:

1. Always cp to `/workspace/final_models/<run_dir_name>/` on the persistent volume first (local disk = ~270 MB/s, 13s for 3.5 GB).
2. Make the laptop rsync opt-in via `--pull-final-model`. Held-out evals run in a fresh pod that re-attaches the volume; the model never needs to transit through laptop.

### The Docker Hub rate limit (May 7)

Building `nvidia/cuda` from Docker Hub hit the 200/6h authenticated rate limit during a buildx multi-arch metadata fetch. Switched FROM line to `nvcr.io/nvidia/cuda:...` (NVIDIA's own registry, no rate limits). Same image, different host.

### The Apple Silicon QEMU build saga (May 7)

QEMU emulating amd64 on arm64 Mac during `docker buildx build --platform linux/amd64` ran in ~30-40 min per attempt. After multiple iteration cycles (transformers 5.x, hf_hub 1.x, matplotlib backtracking, inspect-ai backtracking), gave up local builds entirely and switched to GitHub Actions native amd64 runners. Build dropped to ~14 min and runs in CI.
