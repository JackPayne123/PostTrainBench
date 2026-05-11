# Handoff — F-condition First Clean Experiment (2026-05-11)

**For the next session.** Context: prior session ran a paired baseline + adapter-eval to characterize the F-run trained LoRA across the full 22-eval suite. Results landed but full analysis was deferred. This doc gives you exactly what's where, what's known, and what to do next without re-deriving the prior session.

---

## 1. The Experiment

**F-run (agent training):**
- Run ID: `2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0`
- Condition F (minimise sycophancy + maintain capability)
- Student: `Qwen/Qwen3-1.7B` (IT, not Base)
- Image: `jackpayne123/ptb-base:15` (different from later images — note this)
- Agent: claude-opus-4-7, 30min budget, n=100 limit on sycophancy_slava + sycophancy_aisi
- Pulled to `jobs/runs/2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/`
- Adapter on Drive: `drive:2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/final_model/`
- `summary.json`: post-only (`--skip-pre-eval`), post slava=0.200, post aisi=0.134

**Baseline run** (full 22 evals on base Qwen3-1.7B IT):
- Run ID: `2026-05-11_11-59_baseline_qwen_qwen3-1.7b_limit100`
- Image: `jackpayne123/ptb-base:18`
- Pod still up (--keep-pod) — should be killed after pull
- DONE: completed_partial, 18 ok, 4 fail (bfcl, spiralbench_mini, political_bias_openai, moru)

**Adapter-eval** (full 22 evals on F-run adapter):
- Run ID: `2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100`
- Image: `:18`. Pod already torn down.
- DONE: completed_partial, 19 ok, 3 fail (bfcl, spiralbench_mini, political_bias_openai)

All three lived on Drive at `drive:<run_id>/`.

---

## 2. Headline Numbers (what we know now)

Capability collapsed. Sycophancy went down, but explainable by capability collapse, not by the agent teaching the model to push back.

| Eval | Baseline | Adapter | Δ |
|------|----------|---------|---|
| arc_easy | 0.88 | 0.12 | **-0.76** 🚨 |
| truthfulqa | 0.47 | 0.13 | -0.34 |
| mmlu | 0.46 | 0.14 | -0.32 |
| healthbench | 0.426 | 0.144 | -0.282 |
| gsm8k | 0.75 | 0.55 | -0.20 |
| sycophancy_aisi | 0.329 | 0.136 | **-0.193** (objective) |
| sycophancy_sharma | 0.74 | 0.56 | **-0.18** (objective) |
| humaneval | 0.70 | 0.56 | -0.14 |
| gpqamain | 0.26 | 0.15 | -0.11 |
| sycophancy_slava | 0.267 | 0.20 | **-0.067** (objective; n=30 cap, SE 0.07) |
| aime2025 | 0.00 | 0.07 | +0.07 (noise) |
| arenahardwriting | 0.50 | 0.50 | 0 |

Plus 7 multi-dim safety/character evals where the headline isn't a single number — `headline_metric=None` in registry. Full breakdowns are in the per-bench JSON files. (See section 4.)

---

## 3. To-Do for the Next Session

### 3a. Patch together the full experiment

Three pulls + a merge. Drive pulls are fast (~5-10s):

```bash
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a

# 1. Pull baseline + promote to repo
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_baseline.py 2026-05-11_11-59_baseline_qwen_qwen3-1.7b_limit100
# → baselines/qwen_qwen3-1.7b/<bench>__limit100.json (replaces prior partial)

# 2. Pull adapter-eval + promote to a sibling dir
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_baseline.py 2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100
# → baselines/qwen_qwen3-1.7b/adapter_eval/2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/<bench>__limit100.json

# 3. Pull the F-run itself if not already (post values + solve transcript)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py 2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0
```

Once promoted, the data layout is:
```
baselines/qwen_qwen3-1.7b/
  <bench>__limit100.json            # base model metrics (one per eval)
  _index__limit100.json             # aggregated
  adapter_eval/
    2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/
      <bench>__limit100.json        # adapter metrics (one per eval)
      _index__limit100.json
```

### 3b. Build the side-by-side comparison

`scripts/compute_deltas.py` was scoped but not written. Build it. Should:

1. For each eval task in `EVAL_SUITE`:
   - Load `baselines/<slug>/<bench>__limit<N>.json`
   - Load `baselines/<slug>/adapter_eval/<adapter-run-id>/<bench>__limit<N>.json`
   - If `info.headline_metric` is set: compute `delta = adapter_headline - base_headline`. Mark `higher_is_better` direction.
   - If `info.headline_metric is None` (multi-dim): compute deltas for each top-level metric in the JSON, return a dict.
2. Write `jobs/runs/<adapter-run-id>/deltas.json` keyed by bench.
3. Print a markdown table grouped by category (capability / safety / character) with directional indicators.

Use `src.evals.registry.get_headline(metrics, info)` (already exists) for headline extraction.

### 3c. Analyse the agent's behaviour vs the result

This is the meaty part. The headline says "F failed on capability"; what happened in the agent's training, and how does that map to the eval outputs?

Three concrete sub-tasks, run them as parallel subagent calls if useful:

**A. Re-analyse the F-run transcript with the capability-collapse hypothesis in mind.**
Files: `jobs/runs/2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0/solve_parsed.txt` + `solve_out.jsonl`. The prior /analyse-run pass (committed in skill but its findings live only in this session's chat) said:
- Agent used 108 synthetic examples
- LoRA r=32, α=64, lr=3e-4, 6 epochs, target=all 7 proj modules
- Chat-template fix: completions start `<think>\n\n</think>\n\n`
- Final loss 4.3 → 0.19, mean token acc 97.6%
- No contamination attempts; agent worked within isolation
Now we know this combination wrecked capability across the board. Question: *which specific training-data choices or hyperparams caused the over-fitting? Was it the chat-template injection narrowing output distribution, the LoRA aggressiveness, the curriculum, or all three?* Useful concrete output: identify which subset of training examples (if any) would explain MCQ collapse, and which hyperparam knob to dial back first.

**B. Inspect adapter eval responses across the suite.**
Some evals dump per-sample model outputs into the inspect-ai logs (under `logs/` inside each `/workspace/ptb_eval/<bench>/` dir on the adapter-eval pod — but the pod is killed, so this is only available on Drive if we pulled the full run dir, which `pull_baseline.py` does not — `pull_run.py` does the full dir copy). For evals where the headline crashed:
- arc_easy: dump 5 random sample model outputs. Is the adapter refusing? Outputting `<think>...</think>` then nothing? Producing garbage?
- mmlu: same
- gsm8k: same — does it actually attempt math, or refuse?
- One of the multi-dim ones (big_five) to see the personality shift profile

This tells you which failure mode is in play: refusal-to-commit (the analyse-run hypothesis), broken-format, garbage tokens, or something else. To pull these: `PYTHONPATH=. ... pull_run.py 2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100` (note: not pull_baseline) gets the full run dir including any inspect-ai logs/.

**C. Cross-correlate with multi-dim safety/character scores.**
`big_five`, `moral_foundations`, `rozado_battery`, `political_bias_openai`, `abstention_bench`, `spiralbench_mini`, `coconot`, `strong_reject` each have rich per-axis data. The adapter's `strong_reject_scorer.jailbreak_rate` is `0.29` — for a "model that refuses everything" you'd expect this near 0. So it's not pure refusal — it's *targeted* refusal. Investigate which axes show shifts that correlate with the F-condition's intent vs which are artefacts. Specifically:
- Big Five: did Neuroticism actually shift, or is the high value (0.9 on adapter) just baseline drift?
- Moral foundations: did individualizing/binding scores move? In which direction?
- Abstention bench: which sub-datasets show shift?
- StrongREJECT: jailbreak_rate baseline vs adapter

### 3d. Open design questions to settle (docs/experimental-design-todos.md)

1. **chat-template helper in lora_starter.py** — choose A/B/C. Now that we have direct evidence the chat-template fix is load-bearing for evals working at all, B (helper) or C (auto-apply) more attractive.
2. **moru grader** — choose A/B/C. Without a real grader moru's numbers are noise. moru took 23min on the baseline pod then failed entirely (different from adapter's 6min success on :18). Investigate the divergence; or just deprioritise moru.

Future design TODOs to consider adding:
3. **bfcl with shared vllm** — currently can't use shared vllm (tool-call config not set). Either skip bfcl from suite, or spin a tool-aware vllm just for it.
4. **--use-baseline flag for submit_run.py** — was scoped, never built. Once compute_deltas.py exists, this flag becomes the natural way to skip pre-eval on subsequent runs.

---

## 4. Pipeline State

Current image: `:18` (running pods used this). Newer images:
- `:19` was triggered but **without** the spiralbench/political-bias fixes — DO NOT use for new runs.
- `:20` not yet built. **Build this next**: it has spiralbench_mini fix + the political_bias_openai prompts.jsonl, and recovers 2 of the 4 failed evals.

Trigger:
```bash
gh workflow run build-ptb-base.yml -R JackPayne123/PostTrainBench -f tag=20 -r add_harbor_support
```

After :20 builds: bump `DEFAULT_IMAGE = "jackpayne123/ptb-base:20"` in `src/runpod_backend/runpod_environment.py`, run `python3 scripts/validate_evals.py` + `src/runpod_backend/diag.py` to confirm.

Active machinery the next session should know about:

| File | Purpose |
|------|---------|
| `src/evals/registry.py` | Single source of truth for 22 eval tasks. Each `EvalInfo` has `category, attribute, higher_is_better, headline_metric, has_task_context`. New: `get_headline()` helper for dotted-path extraction (e.g. `strong_reject_scorer.jailbreak_rate`). |
| `scripts/validate_evals.py` | Pre-build validator. Run before every `gh workflow run` to catch eval-surface bugs (missing argparse flags, silent-pipe-eats-rc patterns). Currently green. |
| `src/runpod_backend/submit_run.py` | Agent runs. `--skip-pre-eval` flag lets agent pod skip pre-eval (pre/delta backfilled by compute_deltas.py). |
| `src/runpod_backend/submit_baseline.py` | Suite runs. `--only-bench A,B,C` for selective re-run. `--adapter-from-run-id <ID>` for adapter-eval (rclone-pulls adapter from drive, vllm `--enable-lora`, runs full suite). |
| `src/runpod_backend/pull_baseline.py` | Pulls baseline / adapter_eval run from drive → promotes to `baselines/<slug>/...` or `baselines/<slug>/adapter_eval/<adapter-run-id>/...`. |
| `pod/run_baseline.py` | Pod-side. Iterates EVAL_SUITE OR `cfg.only_bench`. Adapter-mode branches on `cfg.adapter_from_run_id`. |
| `RUNPOD_VOLUME_ID` env var | Override default volume so parallel pods can target separate volumes. We have two: `qwe92egpys` (jack-pilot-cz, original) + `riin1cqm6k` (jack-baseline-pilot, created during this session). |
| `/etc/ptb_run/bench` (root-only) on pod | Bench name passed to score_runner via root-only file, not env. Replaces the agent-visible `.bench` leak fixed on :14. |
| `dev_utils/trace_viewer/app.py` | Interactive local UI for browsing agent runs. `python3 dev_utils/trace_viewer/app.py` → http://127.0.0.1:8765. |
| `.claude/skills/{running-experiments,analyse-run}/SKILL.md` | Auto-loaded operational handbook + post-hoc analyser. |
| `docs/experimental-design-todos.md` | Running list of unresolved design questions with situation + options. Append to this when you find new ones. |

---

## 5. Volumes + Pods State (as of handoff)

- **Volume `qwe92egpys`** (jack-pilot-cz) — currently mounted by baseline pod `wprtcc9ci5i1j2`. Free after pod kill.
- **Volume `riin1cqm6k`** (jack-baseline-pilot, $0.05/GB-mo idle) — adapter-eval terminated, free now.
- **Pod `wprtcc9ci5i1j2`** — baseline, DONE, `--keep-pod` so still RUNNING + costing $0.46/hr. **First action: kill it after pulling.**

```bash
# Kill the still-running baseline pod after you've pulled.
runpodctl pod stop wprtcc9ci5i1j2 && runpodctl pod remove wprtcc9ci5i1j2
```

---

## 6. Build / Image History (so you don't repeat mistakes)

| Tag | What landed | Notes |
|-----|-------------|-------|
| :11 | rclone fix, isolation v1 | Pre-centralisation. Agent ran as root. Contamination possible. |
| :12 | Agent isolation (uid 1000, /opt/ptb 700) | F-run that tainted on prompts.jsonl read. |
| :13 | + rclone v1.69+ via download | Cosmetic — apt rclone still on PATH but functional. |
| :14 | + bench-state to /etc/ptb_run, sudo-no-env, /workspace/ptb_eval lockdown, num_hours float, metrics_filename fix, DONE-to-drive | Three more isolation fixes after F-run. |
| :15 | Setup_note empty for sycophancy; condition F added; --skip-pre-eval; baseline tooling | F-run #2 ran on this (the 0.200 / 0.134 result). |
| :16 | Centralisation refactor (src/eval + src/heldout_evals → src/evals/{capability,safety,character}/) | First baseline ran on this; 12 of 22 failed (shared/ not staged). |
| :17 | (Cancelled, never built — superseded) | |
| :18 | + shared/ + judge/ staging fix, --only-bench, --adapter-from-run-id, validator | The clean experiment ran on this. |
| :19 | (Triggered, no new fixes — superseded by :20) | |
| :20 | + spiralbench_mini judge/ pkg fix, political_bias_openai prompts.jsonl checked in | **Next session: trigger and use this.** |

---

## 7. Quick Reference — Doing the Analysis

```bash
# Setup
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a

# Step 1: pull both run dirs (use pull_run.py for full data, not pull_baseline)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py 2026-05-11_11-59_baseline_qwen_qwen3-1.7b_limit100
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py 2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100

# Step 2: promote per-bench JSONs to baselines/<slug>/ (clean structure)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_baseline.py 2026-05-11_11-59_baseline_qwen_qwen3-1.7b_limit100 --skip-pull
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_baseline.py 2026-05-11_11-59_adaptereval_qwen_qwen3-1.7b_limit100 --skip-pull

# Step 3: spawn /analyse-run with the capability-collapse hypothesis primed
# (Use the .claude/skills/analyse-run/SKILL.md template; substitute the
# F-run ID + headline numbers from §2 above.)

# Step 4: build compute_deltas.py per §3b. Mark deltas red/green by direction.

# Step 5: trace-viewer for interactive review
python3 dev_utils/trace_viewer/app.py     # → http://127.0.0.1:8765
```

Estimated time for the full analysis sprint (Steps 1-4): ~45min if no surprises.
