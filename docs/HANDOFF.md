# Handoff — F-run capability collapse analyzed; next experiment scoped (2026-05-11 #2)

**For the next session.** Picks up where the 2026-05-11 #1 handoff left off. That session ran the paired baseline + adapter-eval; this session pulled both, built the delta machinery, ran the transcript analysis, and identified the root cause. The next session should design and run F-run v2 from the recommendations in §3.

---

## 1. State of the experiment

**F-run #2 (the trained adapter under study):**
- Run ID: `2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0`
- Condition F (minimise sycophancy + maintain capability)
- Student: `Qwen/Qwen3-1.7B` IT
- Agent: claude-opus-4-7, 30min budget, n=100 on sycophancy_slava + sycophancy_aisi
- Image: `:15` (training) — note this is older than the baselines below
- Pulled: `jobs/runs/2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/`
- Adapter on Drive: `drive:2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/final_model/`
- Post-only (`--skip-pre-eval`): post slava=0.200 (candor_rate=0.80), post aisi=0.134

**Base baseline** (full 22 evals on base Qwen3-1.7B IT):
- Run ID: `2026-05-11_11-59_baseline_qwen_qwen3-1.7b_limit100`
- Image: `:18`. Pod killed, pulled to `jobs/runs/<id>/`, promoted to `baselines/qwen_qwen3-1.7b/<bench>__limit100.json`.
- DONE: 18 ok, 4 fail (bfcl, spiralbench_mini, political_bias_openai, moru — `moru` ran on adapter but failed on baseline)

**Adapter-eval** (full 22 evals on the F-trained adapter):
- Run ID: `2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100`
- Image: `:18`. Pod torn down, pulled, promoted to `baselines/qwen_qwen3-1.7b/adapter_eval/2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/`.
- DONE: 19 ok, 3 fail (bfcl, spiralbench_mini, political_bias_openai)

All three live on Drive at `drive:<run_id>/`. Both pods now terminated.

---

## 2. Headline + root-cause verdict

**Capability collapsed.** Sycophancy went down, but the drop is explained by capability collapse, not by the adapter teaching the model to push back.

| Eval | Baseline | Adapter | Δ |
|------|----------|---------|---|
| arc_easy | 0.88 | 0.12 | **-0.76** |
| truthfulqa | 0.47 | 0.13 | -0.34 |
| mmlu | 0.46 | 0.14 | -0.32 |
| healthbench | 0.426 | 0.144 | -0.282 |
| gsm8k | 0.75 | 0.55 | -0.20 |
| **sycophancy_aisi** | 0.329 | 0.136 | **-0.193** (good under F) |
| **sycophancy_sharma** | 0.74 | 0.56 | **-0.18** (good under F) |
| humaneval | 0.70 | 0.56 | -0.14 |
| gpqamain | 0.26 | 0.15 | -0.11 |
| **strong_reject.jailbreak_rate** | 0.19 | **0.29** | **+0.10** (worse — see below) |
| **sycophancy_slava** | 0.267 | 0.20 | -0.067 (SE±0.07, near-noise) |
| coconot.total | 1.0 | 6.0 | over-refusal count up |
| aime2025 | 0.00 | 0.07 | +0.07 (noise floor) |
| arenahardwriting | 0.50 | 0.50 | 0 |

Multi-dim shifts (full breakdown in `jobs/runs/2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/deltas.json`):
- **big_five:** Neuroticism +0.375 (0.53 → 0.90), Extraversion +0.075, Openness +0.044, Conscientiousness -0.121; Agreeableness **disappeared entirely from outputs** (base 0.825, adapter not scored)
- **rozado_battery:** economic_axis_mean -14.3 (12.5 → -1.8, shifted left), social_axis_mean -2.8, iSideWith refusal_rate 0 → 0.20-0.22, ideologies hard_right_score -15.5
- **moral_foundations:** binding_score +0.22, loyalty +0.67, others flat
- **abstention_bench:** divergent — Squad2/QASPER/MediQ collapse to 0 (model can't answer), FalseQA/SelfAware climb 0→1 (over-refusal of answerable); BBQ drops -0.4 (worse stereotype-aware abstention)

**Verdict:** Adapter is broken-format / mode-collapsed prose. NOT pure refusal. `strong_reject.jailbreak_rate` going UP +0.10 means when prompted with harmful content the adapter gave MORE compliant-looking outputs than base — consistent with the judge accepting generic prose as "answered" rather than the adapter refusing. Pure refusal would push `jailbreak_rate` toward 0.

### Root cause from /analyse-run on solve_parsed.txt

The agent trained 108 examples (`build_data_v3.py`), all bare-prompt free-text Q&A:

| Bucket | Approx count | Shape |
|---|---|---|
| Push back on false premises | 24 | `("Is Pluto a planet?", "No, IAU reclassified...")` |
| Hold position under pressure | 16 | multi-turn |
| Decline empty validation | 9 | free text |
| Push back on bad ideas | 9 | free text |
| Naming bad premises | 3 | free text |
| Bad reasoning | 5 | free text |
| Capability | ~42 | `("Capital of France?", "Paris.")`, `("Solve 2x+5=17", "x=6")` |

**Zero MCQ format.** No `A)/B)/C)/D)` examples, no single-letter answer targets, despite half the eval suite being MCQ. Every completion starts literally `<think>\n\n</think>\n\n` then prose; adapter learned "after assistant opener → empty-think + flowing prose". This is incompatible with the MCQ heads that arc_easy/mmlu/gpqa/truthfulqa rely on.

LoRA config: `r=32, α=64, lora_dropout=0.05, target=[q,k,v,o,gate,up,down]_proj, lr=3e-4, epochs=6, batch=4×grad_accum=4, cosine, warmup_ratio=0.05`. Final loss `4.05 → 0.96` — overfit on 108 examples. **No mid-training capability probe.** Agent's only self-probe was `cap_test.py` with 6 free-text questions, all answered correctly — wouldn't catch MCQ collapse. Agent expressed capability preservation as a goal upfront but never connected "training data shape" with "MCQ output distribution".

---

## 3. To-do for the next session — F-run v2 design + execution

### 3a. Decide the data-shape + LoRA-aggressiveness lever (most important)

Three independent levers to dial back in F-run v2; rank by isolation potential:

| Lever | Direct fix | Cost |
|---|---|---|
| **Training data covers eval output distribution** | Add 20-30 ARC/MMLU-style MCQ examples to `lora_starter.py` data scaffold (or condition-F instruction) so the adapter sees `Question + A)/B)/C)/D)\nAnswer: <letter>` pairs | Cheap; biggest single fix per /analyse-run |
| **LoRA hyperparams** | r=32→8, lr=3e-4→1e-4, epochs=6→3, target=q/v only (drop gate/up/down/k/o) | Cheap |
| **Mid-training capability probe** | Bake `score.sh` against a capability mini-suite (gsm8k limit=20 + arc_easy limit=20) into the F-condition framing so the agent gets a real signal | Medium |

See `docs/experimental-design-todos.md` #7 for options (A/B/C/D) on each — D is the data-shape direct fix; A/B are hyperparam knobs; C is the heavy in-loop probe.

**Recommended F-run v2:** combine **D** (MCQ examples in data) + **A** (lower-aggressiveness LoRA defaults), keep agent freedom to override. Run on `:20` once the build lands.

### 3b. Build :20 + bump default image

Build was triggered earlier this session; check status:
```bash
gh run list -R JackPayne123/PostTrainBench -w build-ptb-base.yml -L 3
```

If success: bump `DEFAULT_IMAGE = "jackpayne123/ptb-base:20"` in `src/runpod_backend/runpod_environment.py`, then:
```bash
python3 scripts/validate_evals.py      # green pre-build
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/diag.py
```
:20 recovers spiralbench_mini (judge/ pkg fix) + political_bias_openai (prompts.jsonl checked in). bfcl + moru still expected to fail (see design TODOs #2 and #6).

### 3c. Sample-level inspection (deferred — needs recovery pod)

Inspect-ai per-sample logs live on the persistent volume at `/workspace/ptb_eval/<bench>/logs/`, NOT in the run dir. Both pods are killed, so this needs a recovery pod:
```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_eval_logs.py \
    --benchmark arc_easy \
    --dst jobs/runs/2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100/eval_logs/arc_easy
```
Volume to mount: `riin1cqm6k` (adapter-eval) — already has logs from this run. Pull arc_easy + mmlu + gsm8k + big_five samples to confirm the failure shape: refusal-to-commit / broken-format / garbage tokens / over-thinking. Transcript-level signal is strong enough that this is a confirmation step rather than a load-bearing one.

### 3d. Open design TODOs

`docs/experimental-design-todos.md` updated in this session:
- ~~#4~~ compute_deltas.py — resolved (built this session)
- #1 lora_starter.py chat-template helper — pick A/B/C (now confirmed load-bearing)
- #6 moru grader — pick A/B/C
- #7 LoRA hyperparams + data shape — pick A/B/C/D (the F-run v2 decision)
- #10 chat-template fix lives in agent training data only — duplicates #1; close when #1 lands
- #11 compute_deltas delta surfaces — closed by #4

Other items unchanged: #2 bfcl, #5 --use-baseline, #8 untracked OpenAI usage, #9 multi-attach.

---

## 4. Pipeline state

Current default image: `:18`. Newer:
- `:19` no new fixes (superseded).
- `:20` building / built — has spiralbench_mini judge fix + political_bias_openai prompts.jsonl.

Active machinery to know about:

| File | Purpose |
|------|---------|
| `scripts/compute_deltas.py` | **NEW.** Base vs adapter-eval delta table. `python3 scripts/compute_deltas.py <adapter-run-id> --write-deltas` → `jobs/runs/<adapter-run-id>/deltas.json` + markdown to stdout. |
| `src/evals/registry.py:get_headline` | **PATCHED.** Tries literal-key match (for inspect_evals `a.b: v` flat keys with literal dots) before dotted-path walk. Prior version returned None on `strong_reject_scorer.jailbreak_rate` and silently dropped strong_reject from the delta table. |
| `src/evals/registry.py:EVAL_SUITE` | Single source of truth for 22 eval tasks. Each `EvalInfo` has `category, attribute, higher_is_better, headline_metric, has_task_context`. |
| `scripts/validate_evals.py` | Pre-build validator. Run before every `gh workflow run` to catch eval-surface bugs. Currently green. |
| `src/runpod_backend/submit_run.py` | Agent runs. `--skip-pre-eval` flag now reliable; compute_deltas.py handles backfill. |
| `src/runpod_backend/submit_baseline.py` | Suite runs. `--only-bench A,B,C` for selective re-run. `--adapter-from-run-id <ID>` for adapter-eval (rclone-pulls adapter from drive, vllm `--enable-lora`, runs full suite). |
| `src/runpod_backend/pull_baseline.py` | Pulls baseline / adapter_eval run from drive → promotes to `baselines/<slug>/...` or `baselines/<slug>/adapter_eval/<adapter-run-id>/...`. `--skip-pull` for after a `pull_run.py`. |
| `src/runpod_backend/pull_run.py` | Full-run pull (transcript + summary + per-bench metrics). Does NOT pull inspect-ai `/workspace/ptb_eval/<bench>/logs/`. |
| `src/runpod_backend/pull_eval_logs.py` | Recovery pod to fetch per-sample inspect-ai logs from the volume. Use when sample-level inspection is required. |
| `pod/run_baseline.py` | Pod-side. Iterates EVAL_SUITE OR `cfg.only_bench`. Adapter-mode branches on `cfg.adapter_from_run_id`. |
| `/etc/ptb_run/bench` (root-only) on pod | Bench name passed to score_runner via root-only file. Replaces the agent-visible `.bench` leak fixed on :14. |
| `dev_utils/trace_viewer/app.py` | Interactive local UI for browsing agent runs. `python3 dev_utils/trace_viewer/app.py` → http://127.0.0.1:8765. |
| `.claude/skills/{running-experiments,analyse-run}/SKILL.md` | Auto-loaded operational handbook + post-hoc analyser. |
| `docs/experimental-design-todos.md` | Running list of unresolved design questions. |

---

## 5. Volumes + Pods State (as of handoff)

- **Volume `qwe92egpys`** (jack-pilot-cz) — free.
- **Volume `riin1cqm6k`** (jack-baseline-pilot, $0.05/GB-mo idle) — free; holds adapter-eval `/workspace/ptb_eval/<bench>/logs/` for sample-level inspection.
- **All pods terminated.**

---

## 6. Build / Image History

| Tag | What landed | Notes |
|-----|-------------|-------|
| :11 | rclone fix, isolation v1 | Pre-centralisation. Agent ran as root. Contamination possible. |
| :12 | Agent isolation (uid 1000, /opt/ptb 700) | F-run that tainted on prompts.jsonl read. |
| :13 | + rclone v1.69+ via download | Cosmetic. |
| :14 | + bench-state to /etc/ptb_run, sudo-no-env, /workspace/ptb_eval lockdown, num_hours float, metrics_filename fix, DONE-to-drive | Three more isolation fixes. |
| :15 | Setup_note empty for sycophancy; condition F added; --skip-pre-eval; baseline tooling | F-run #2 ran on this. |
| :16 | Centralisation refactor (src/eval + src/heldout_evals → src/evals/{capability,safety,character}/) | First baseline ran on this; 12 of 22 failed (shared/ not staged). |
| :17 | (Cancelled, never built — superseded) | |
| :18 | + shared/ + judge/ staging fix, --only-bench, --adapter-from-run-id, validator | **The clean experiment ran on this.** |
| :19 | (Triggered, no new fixes — superseded by :20) | |
| :20 | + spiralbench_mini judge/ pkg fix, political_bias_openai prompts.jsonl checked in | **In flight or built this session. Bump DEFAULT_IMAGE after diag pass.** |

---

## 7. Quick Reference — Doing F-run v2

```bash
# Setup
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a

# Verify :20 is the default image (diag.py confirms drive + isolation surface)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/diag.py

# Submit F-run v2 — pick the lever combination from §3a, then submit. Example:
# (Adapt instruction.md / lora_starter.py first if going via lever D or A.)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition F \
    --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B \
    --benchmark sycophancy_slava \
    --extra-evals sycophancy_aisi \
    --time-budget-h 0.5 \
    --limit 100 \
    --skip-heldout \
    --skip-pre-eval

# After F-run v2 completes, run the adapter against the full suite:
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_baseline.py \
    --student Qwen/Qwen3-1.7B \
    --limit 100 \
    --adapter-from-run-id <new_F_run_id>

# Pull + promote + delta:
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/pull_run.py <adapter_eval_run_id>
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/pull_baseline.py <adapter_eval_run_id> --skip-pull
PYTHONPATH=. python3 scripts/compute_deltas.py <new_F_run_id> --write-deltas

# Trace viewer for interactive review
python3 dev_utils/trace_viewer/app.py     # → http://127.0.0.1:8765
```

Estimated time for v2 sprint (build → train → adapter-eval → delta): ~2.5h on RunPod, ~$1.50.
