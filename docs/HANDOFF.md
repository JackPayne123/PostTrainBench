# Handoff — Pipeline shipped, F-run v2 validated, pre-scaling backlog (2026-05-12)

**For the next session.** Full session log lives in `docs/CHANGELOG.md`; this is the snapshot + what to do next.

---

## 1. Where we are

**Two trained adapters scored against the same baseline:**

| Run | Image | Agent budget | Outcome |
|---|---|---|---|
| `2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0` (F-run v1) | :15 | 30min | Sycophancy down (sharma -0.18, slava -0.07, aisi -0.19) but **capability collapsed** (arc_easy -0.76, mmlu -0.32, gsm8k -0.20, etc). Root cause via `/analyse-run`: 108 free-text training examples, ZERO MCQ format → adapter's output distribution shifted off single-letter answers. |
| `2026-05-11_19-15_F_claude-opus-4-7_qwen3-1.7b_seed0` (F-run v2) | :22 (training) / :23 (adapter-eval) | 1h | Sycophancy down harder (sharma **-0.54**, slava -0.13, aisi -0.09) AND capability mostly preserved (arc_easy -0.05, mmlu +0.03, gsm8k -0.06). Ergonomics fix worked. |

**v2's deltas in one table** (base = Qwen3-1.7B IT, post-eval `:23`, limit=100):

| Bench | Δ v2 | Notes |
|---|---|---|
| **arc_easy** | -0.05 | v1 was -0.76. MCQ collapse fixed. |
| mmlu | +0.03 | |
| gsm8k | -0.06 | |
| humaneval | -0.08 | |
| gpqamain | -0.10 | small set, noisy |
| truthfulqa | -0.08 | |
| aime2025 | +0.03 | noise floor (n=30, only 1 problem moved) |
| arenahardwriting | 0.0 | broken-by-design — see §3 |
| healthbench | **-0.22** | gpt-5-mini grader on long-form medical; lone capability outlier |
| **sycophancy_sharma** | **-0.54** ✓ | F-target, large drop |
| **sycophancy_slava** | -0.13 ✓ | F-target |
| **sycophancy_aisi** | -0.09 ✓ | F-target, weaker than v1's -0.19 |
| **strong_reject.jailbreak_rate** | **+0.10 ✗** | Real cost of sycophancy-suppression: model pushes back on user framing including harmful-request framing. Same magnitude as v1 → not a v1-specific artifact; structural to F-condition. |
| coconot total | +3 (1 → 4) | over-refusal count; better than v1's +5 |
| abstention_bench | divergent | Squad2 / QASPER / MediQ collapse to 0; FalseQA / SelfAware climb. Adapter prefers "I don't know" stance on the wrong sub-sets. |
| **big_five** | FAIL (registry bug) | Lenient scorer was registered inside a function body → inspect_ai couldn't find it. Patched: now passes Task instance directly. Bakes into :23. |
| **moru** | FAIL (grader bug) | `task_args={"grader_models": "anthropic/..."}` got iterated char-by-char by inspect_evals scorer. Patched: pass as list. Bakes into :23. |
| moral_foundations | binding +0.22 | full breakdown in deltas.json |
| rozado_battery | political shift | economic_axis -14, social_axis -3 — sycophancy-suppression entangles with political-agreement axes |
| political_bias_openai | n=100, final_score 0.55 | first successful run (`:20` shipped the prompts.jsonl) |
| spiralbench_mini | n_failed=30 | "OK" log but soft failure — judge/user-sim hit a haiku integration issue. Investigate. |

**Verdict on F-run v2**: lever D (MCQ data shape) + lever A (conservative LoRA) + chat-template helper from design TODO #1 + #7 + lora_starter rewrite **fixed the capability collapse** and **strengthened the F-target trait shift** simultaneously. The strong_reject jailbreak +0.10 trade-off is real — the adapter learned "don't capitulate to user framing" which generalises to harmful-request framing.

---

## 2. Pipeline state (image `:23` live)

| Layer | State |
|---|---|
| **Default image** | `jackpayne123/ptb-base:23` (diag green). Has all big_five + moru + arena fixes + 2 new character evals + grader unification baked in. |
| **EVAL_SUITE** | 23 tasks (10 capability + 7 safety + 6 character). bfcl dropped (lives in `_disabled/`). 2 collaborator additions: `activity_preference` (Sofroniew 2026 Elo, smoke-validated) + `persona_traits` (Chen 2025 Persona Vectors, smoke-validated, ~$3 Anthropic per pass). |
| **Per-eval limits** | `EvalInfo.default_limit` lives on the registry; `submit_baseline --limit 0` uses them (aime=30, mmlu/arc_easy/truthfulqa/rozado=200, big_five=40, moral_foundations=32, syco_slava=30, spiralbench=30, persona_traits=20 q × 10 rollouts, activity_preference=64 full). `--limit 100` still forces all. |
| **Grader routing** | Mostly `claude-haiku-4-5` (Anthropic), except healthbench + arenahardwriting on `gpt-5-mini` (OpenAI; their published graders use the OpenAI completion format and refactoring is medium-scope work we don't need yet — decision logged 2026-05-12). Full per-eval inventory + plumbing notes in `docs/OPERATIONS.md` → Grader routing. |
| **Agent ergonomics** | `bash timer.sh` reads root-owned `/etc/ptb_run/deadline` (tamper-proof) via `sudo /opt/pipeline-bin/time-remaining`. `bash score.sh` queries primary bench. `bash score_capability.sh` queries a small MCQ probe (arc_easy@30 by default for condition F) — first run on :23 will exercise this. Local laptop pre-commit hook scrubs `hf_/sk-ant-/sk-proj-/sk-/gho_/ghp_/rpa_` token shapes from transcripts before staging (`bash scripts/install-hooks.sh`). |
| **Arena head-to-head** | Adapter-vs-base contract: pod sets `PTB_ARENA_ADAPTER_ALIAS=<short-id>` in env so candidate alias `Qwen3-1.7B__<id>` doesn't collapse with the checked-in baseline `Qwen3-1.7B.jsonl`. Base baseline mode skips arena entirely (sentinel JSON). `--regenerate-baseline` flag refreshes the reference. `--max-new-tokens` dropped 16384→4096; `default_limit` 100→50 — ~6× faster per pass. |
| **trace viewer** | `python3 dev_utils/trace_viewer/app.py` → http://127.0.0.1:8765. Index fallback ladder: legacy `metrics_pre/post.json` → `summary.{pre,post,delta}` → per-bench `metrics_post_<bench>.json` → `deltas.json[primary_bench]`. `--skip-pre-eval` runs render scores via baseline-backfill. |
| **compute_deltas** | `scripts/compute_deltas.py <adapter-run-id> --write-deltas --update-summary` populates `deltas.json` + backfills summary.json pre/post/delta (tagged `delta_method: "baseline-backfill"`). |
| **big_five rescore** | `scripts/rescore_big_five.py` re-aggregates an existing inspect-ai log with the lenient parser without rerunning the eval. Saved ~$0 + 5min on the v1 analysis. |
| **Chat-template validation gate** | `lora_starter.py` now ships per-family helpers (`format_qwen3_chat`, `format_gemma_chat`, `format_smollm_chat`) + `format_chat()` dispatch with substring family inference. `src/evals/templates/validated_models.json` is the manifest of student models with hand-validated chat templates. `submit_run.py` + `submit_baseline.py` fail-fast if `--student` / `--model` not in the manifest (`--bypass-template-check` escape hatch present). Currently validated: `Qwen/Qwen3-1.7B` + `Qwen/Qwen3-1.7B-Base` (both validated 2026-05-12 against `enable_thinking=True/False`). Gemma3 + SmolLM3 helpers are stubs — validate + commit before using. |
| **Baseline-discipline gate** | `submit_run.py --use-baseline` fails-fast at submit time if `baselines/<student-slug>/<bench>__limit<N>.json` missing for primary bench or any `--extra-evals`. Implies `--skip-pre-eval` (reuses promoted baseline; post-run `compute_deltas.py --update-summary` backfills). Recovery hint in the error message gives the exact `submit_baseline.py --only-bench` to run. |

---

## 3. Where we sit in the broader research direction

`docs/IDEAS.md` frames this project as "frontier model autonomously post-trains a smaller open-weight model; we measure what happens" — with **safety-relevant behaviours as primary objectives**, **held-out adversarial verification**, and **bidirectional measurement** as the differentiators from PostTrainBench / AAR / Kimi K2.5.

The 2026-05-11 sync settled the immediate-term scope:
- Stay on Qwen3-1.7B until ergonomics fixes land. ✓ Most have landed (timer tool, lora_starter, chat-template helper, MCQ guidance, grader unification, score_capability spot-check). One open: composite score.sh contract (see §4).
- Each collaborator picks a "pet" first experiment. Collaborator: condition A (zero info to agent). Jack: F-run v2 — **shipped, results above**.
- Collaborator's evals (activity_preference + persona_traits) integrated; first baseline numbers in repo at `baselines/qwen_qwen3-1.7b/`.
- Pipeline stays "messy until results" per memory — refactor (two-container isolation, parallelism without volume coupling) deferred until first scientific signal. F-run v2 is that signal: real evidence that an agent can hit an F-condition objective without destroying capability when the ergonomics are right.

**What F-run v2 means for the research narrative**: when training-data shape covers the eval output distribution, agent-driven post-training on a small student with a small compute budget can produce a measurable single-trait shift (sycophancy_sharma -0.54) **with a known cost** (strong_reject jailbreak_rate +0.10). The cost direction is the more interesting finding — sycophancy-suppression isn't free; it bleeds into harmful-request resistance in the wrong direction. That's a publishable observation if the next 5-10 runs show the same pattern.

---

## 4. Before scaling up (8B, multi-run sweeps, GPT-5/Opus-as-student)

These are the items I'd close before kicking off the 8B or running a 10-run sweep.

**Status after 2026-05-12 mini-sprint:**
- ✅ Chat-template validation gate landed. New student models (Gemma3, SmolLM3, 8B variants) MUST pass `scripts/validate_chat_templates.py --model <id> --commit` before submit accepts them. Closes a class-of-bug ("agent learns wrong format on a new student") at submit-time rather than 30min into a real run.
- ✅ Baseline-discipline gate landed. `submit_run.py --use-baseline` fails-fast on missing baseline JSONs. Closes the "ran F-run, then realised we didn't have a base baseline at this limit" failure mode.
- ✅ moru + big_five live-validated on `:23` (TODO #12 closed). Both produce real metrics — moru overall_mean=0.125 across 12 axes; big_five all 5 traits with Agreeableness back at 0.825.
- ✅ spiralbench_mini silent failure root-caused + fixed (TODO #13 closed). `target.generate()` async → sync coroutine mismatch. Wrapped with `asyncio.run`. Smoke at limit=10 produced clean per-behavior breakdown; baseline shows delusion-reinforcement 2.8 + sycophancy 2.7 leading (consistent with the sycophancy_sharma 0.74 finding). Lives in laptop src/evals/ and rsyncs to pod at submit time — no `:24` build needed.
- ✅ Eval-shape coverage (TODO #18 closed, picked C). `instruction.md` + lora_starter docstring no longer name specific eval-shape failure modes ("the suite contains MCQ evals"); they now warn generically that the held-out suite spans MCQ / free-text / numeric / CoT / chat shapes and the agent's training data should cover the distribution. Less eval-info leakage; same warning about output-distribution collapse.
- ✅ Run-id collision (TODO #22 closed). `build_run_dir_name` + `submit_baseline.py` now append a 6-char `secrets.token_hex(3)` suffix. Same-minute parallel submissions no longer collide.
- ✅ Image-aligned base baseline on `:23` (TODO #17 closed). Run `2026-05-12_10-16_baseline_qwen_qwen3-1.7b_limit100_573ee8`. 23/23 ok, capability tier image-stable vs `:18` baseline; sycophancy_sharma drifted -0.07 (haiku-unification effect). First clean character-tier baseline data: big_five all 5 traits, moru 14 axes, spiralbench 17 behaviours, persona_traits 7 traits, activity_preference 64 activities × 8 categories, political_bias 5 slants, rozado political-compass axes.
- 🟡 Adapter-eval v2 on `:23` IN FLIGHT (`2026-05-12_13-02_adaptereval_qwen_qwen3-1.7b_limit100_94e26d`). Capability tier so far: arc_easy 0.84 (Δ-0.04), mmlu 0.49 (Δ+0.03), gsm8k **0.57** (Δ-0.19, wider than the :18-pair's -0.06), humaneval 0.68, gpqamain 0.21, truthfulqa 0.43. Character + safety tier pending. **First real arena adapter-vs-base contract** runs after capability tier (candidate alias `Qwen3-1.7B__seed0` ≠ baseline `Qwen3-1.7B`).
- 🟡 Progress instrumentation (`src/evals/shared/_progress.py`) shipped to persona_traits + spiralbench + healthbench + political_bias_openai + sycophancy_aisi + `_inspect_wrap` bookends. Stdout-flushed `[progress] <bench> <phase> done=N/M rate=R/s eta=Xs` every 10% or 30s. Bakes into pod on next image rebuild OR auto-rsyncs via collaborator's `d5100c6` on next submit.

### 4a. Pre-scaling — pipeline / cost

| Item | Why it matters at scale | Effort |
|---|---|---|
| **moru + big_five fixes verified on real run** | Both currently FAIL on `:18`/`:22`. Patches are in `:23` but only smoke-tested for big_five via `rescore_big_five.py` offline; moru's haiku-grader path hasn't fired against a live model. Run baseline `--only-bench moru,big_five --limit 0` on `:23` to validate before next F-run. ~5min, ~$0.50. | Low |
| **spiralbench_mini soft failure** | Both v1 + v2 logged "OK" with n_failed=30 (every conversation failed). Haiku judge/user-sim integration likely broken. At scale this is a silent capability gap in the safety tier. Read the eval log + fix. | Medium |
| **strong_reject jailbreak +0.10 is structural** | Same +0.10 on v1 + v2 = real cost of sycophancy-suppression, not an artifact. Before scaling, either (a) accept it as a finding, (b) add a "harm-resistance preservation" instruction to condition F, (c) include strong_reject as a co-objective in score.sh. | Medium / design |
| **arenahardwriting on `:23` not yet exercised end-to-end** | New adapter-vs-base contract is implemented but only static-reviewed. First adapter-eval on `:23` will be the smoke. Watch for `_arena_mode` in the metrics. | Low (organic — happens on next run) |
| **Per-eval cost transparency** | Full F-run + adapter-eval pair now ~$12-14 Anthropic + ~$0.50 pod. 10-run sweep ≈ $130. Add a `dry-run cost estimator` script that walks the suite + estimates judge calls × per-token cost. | Medium |
| **Project-scoped API keys** | Both `.env` keys (Anthropic + OpenAI) are "leftover Mile/Sean cyber stuff" per the 2026-05-11 meeting. Rotate to project-scoped before scaling so usage attribution stays clean. User-action; not blocking. | Low (user-side) |
| **Healthbench / arenahardwriting still gpt-5-mini** | Two graders not unified to haiku. Refactoring their OpenAI-completions-API-specific pipelines to anthropic SDK is medium-scope. Acceptable for now since costs are auditable; revisit if OpenAI spend balloons. | Medium (deferred per design TODO #8) |

### 4b. Pre-scaling — experimental design

| Item | Why it matters | Effort |
|---|---|---|
| **score.sh contract decision** | Currently single-bench primary signal. Memory `project_score_sh_redesign` says Jack leans toward composite. F-run v2's adapter had `score_capability.sh` available (arc_easy probe) but the run was on `:22` (no capability probe written yet, F-condition only got `bench_capability` from `:23+`). First F-run on `:23` will be the agent's first exposure to the capability spot-check. Decide before scale: keep two-script (primary + capability), collapse to composite, or single-number. | Medium / design |
| **Condition A trial (collaborator's pet experiment)** | Zero-info-to-agent setup. Provides a baseline "what does the agent do with no information" data point. Comparing F to A illuminates whether the F-prompt is actually steering or whether agents converge regardless. | Low (just submit) |
| **MCQ-format training data scaffold** | F-run v2 still required the agent to figure out MCQ-shape examples on its own (helped by lora_starter docstring + condition_F prompt). Cleaner: ship 20-30 ARC/MMLU-style exemplars in `task_context/lora_starter_data/` so the agent has a copy-paste baseline. | Low |
| **Per-run seed variance characterisation** | A single F-run v2 result doesn't tell us how much of the -0.54 syco_sharma drop is real vs sampling. Run 3 F-condition variants with different seeds before scaling — establishes the noise floor for headline deltas. ~$45 total. | Low ($-cost only) |
| **Held-out adversarial verification** | `IDEAS.md` calls this out as a differentiator. We don't currently red-team the adapter against held-out adversarial prompts. Pre-scale, decide whether to include this in the post-eval flow or as a separate "verification" pass. | Medium / design |

### 4c. Pre-scaling — scientific validity

| Item | Why it matters | Effort |
|---|---|---|
| **Big_five Agreeableness phantom** | F-run v1 showed Agreeableness disappearing entirely — turned out to be an upstream-scorer strictness issue, not a real trait drop. v2 now uses the lenient scorer. But same class of "scorer-rejection-not-trait-shift" could affect other multi-dim evals (especially the new persona_traits judge prompts). Before scaling, sanity-spot-check 10 random samples per multi-dim eval for parse rate. | Low |
| **Reference-baseline staleness** | Base baseline at `:18`. Adapter eval at `:23`. Image-tag drift between base and candidate creates a confound. Re-run base baseline on `:23` once before kicking off the next adapter-eval cycle — costs ~$5, makes deltas image-independent. | Low (just submit) |
| **strong_reject as a `--safety-attribute` candidate** | Currently strong_reject only appears as a side-effect measurement. If "harm-resistance preservation" is added to F-condition, strong_reject becomes a co-objective. Decide the registry's `attribute=` for this case. | Low |
| **Persona-traits judge prompt audit** | New eval scores 0-100 per trait via Haiku, parsed by a regex that takes first 0-100 int. If Haiku ever outputs a year ("2024" → 24), parsing collapses. Spot-check the rows for outliers before treating per_trait means as reliable. | Low |

### 4d. Pre-scaling — observability

| Item | Why it matters | Effort |
|---|---|---|
| **Recovery-pod cost guardrail** | `pull_eval_logs.py` spins a $0.05/h pod and tears down. Easy to leave running by accident. Add a hard 10min timeout. | Low |
| **Run-id collision check** | Multi-second granularity in run-ids. If two runs submit in the same minute (parallel sweep) → collision. Add a microsecond/seed disambiguation. | Low |
| **Trace viewer auth** | `python3 dev_utils/trace_viewer/app.py` binds to 127.0.0.1:8765 by default — fine for local. If we ever want to share with collaborator, add a token. Not urgent. | Low |

### 4e. Open from prior sessions

| | |
|---|---|
| design TODO #1 (chat-template helper in lora_starter.py) | Resolved (helper landed). |
| #2 (bfcl) | Resolved (dropped). |
| #4 (compute_deltas.py) | Resolved. |
| #5 (`--use-baseline` autoresolve) | Pipeline now does this implicitly via compute_deltas backfill. Closed. |
| #6 (moru grader) | Resolved (haiku + list). |
| #7 (LoRA aggressiveness + data shape) | Resolved via lora_starter rewrite + agent prompt. |
| #8 (untracked OpenAI usage) | Partial. INSPECT_GRADER_MODEL routes inspect_evals graders; healthbench + arenahardwriting still gpt-5-mini. |
| #9 (multi-attach volume) | Open. Workaround in use (separate volumes per pod). Not blocking; revisit if parallelism needs grow. |
| #10 (chat-template fix in agent data) | Duplicate of #1. Closed. |
| #11 (compute_deltas delta surfaces) | Closed by #4. |

---

## 5. Next-session checklist

When the next session starts, the **fastest sane sequence** is:

```bash
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a

# 1. Verify keys still loaded (post-rotation if user did the audit)
python3 -c "import os; print('A', len(os.environ['ANTHROPIC_API_KEY']), 'O', len(os.environ['OPENAI_API_KEY']))"

# 2. Spot-check that moru + big_five run end-to-end on :23 against base
#    (pre-scale validation; ~5min, ~$0.50)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_baseline.py \
    --model Qwen/Qwen3-1.7B \
    --only-bench moru,big_five \
    --limit 0
# After DONE: pull + promote + eyeball metrics in baselines/qwen_qwen3-1.7b/

# 3. If 2 is green: re-run full base baseline on :23 for a clean image-aligned reference
#    (~30-40min, ~$6)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_baseline.py \
    --model Qwen/Qwen3-1.7B \
    --limit 100

# 4. Submit F-run v3 with whatever lever combination the design decision favours
#    (or condition A as the collaborator's pet)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition F \
    --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B \
    --benchmark sycophancy_slava \
    --extra-evals sycophancy_aisi \
    --time-budget-h 1.0 \
    --limit 100 \
    --skip-heldout \
    --skip-pre-eval

# 5. After F-run completes, adapter-eval:
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_baseline.py \
    --model Qwen/Qwen3-1.7B \
    --limit 100 \
    --adapter-from-run-id <new_F_run_id>

# 6. Delta:
PYTHONPATH=. python3 scripts/compute_deltas.py <new_F_run_id> --write-deltas --update-summary
```

---

## 6. Image / commit log (latest first)

| Tag | Commit | What landed |
|---|---|---|
| :23 | `efbfa56` + DEFAULT_IMAGE bump in `b2f033b` | big_five lenient Task instance, moru grader_models list, arena adapter-vs-base contract + skip-base + `--max-new-tokens 4096` + default_limit 50, registry `default_limit` per bench, collaborator's `activity_preference` + `persona_traits` + pod rsync of `src/evals/` |
| :22 | `7890226` → push at `fcd5935` + bump in `35f9179` | Full ergonomics sprint: drop bfcl, timer-tool, score_capability, lora_starter rewrite, agent prompt, INSPECT_GRADER_MODEL, big_five lenient v0 (broken — fixed in :23) |
| :21 | (built against pre-commit state, ≡ :20) | Nothing new effectively |
| :20 | (built without DEFAULT_IMAGE bump) | spiralbench judge/ pkg fix, political_bias_openai prompts.jsonl checked in |
| :18 | | First clean baseline image; shared/+judge/ staging, `--only-bench`, `--adapter-from-run-id`, validator |
| :15 | | F-run v1 trained on this |

Pre-commit hook installed locally: `bash scripts/install-hooks.sh` (idempotent). Bypass with `git commit --no-verify`.

---

## 7. Outstanding artefacts to inspect

- `jobs/runs/2026-05-11_19-15_F_claude-opus-4-7_qwen3-1.7b_seed0/deltas.json` — F-run v2 full delta breakdown
- `jobs/runs/2026-05-11_19-15_F_claude-opus-4-7_qwen3-1.7b_seed0/solve_parsed.txt` — agent transcript; `/analyse-run` candidate
- `baselines/qwen_qwen3-1.7b/{activity_preference,persona_traits}__limit{64,20}.json` — first reference numbers for the two new evals
- `jobs/runs/2026-05-12_07-43_baseline_qwen_qwen3-1.7b_perEval/` — smoke run dir (new evals only)

Drive copies of all of the above are at `drive:<run_id>/`.
