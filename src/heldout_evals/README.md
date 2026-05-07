# Held-out Character Evaluation Panel

Post-hoc evaluation suite for measuring character drift after a frontier
agent post-trains a Qwen base model. Mirrors `src/eval/tasks/` but is
**not** registered in `commit.sh` and is **not** visible to the training
agent: the agent never reads any file under this directory while solving
its post-training task.

Categories may be visible to agents (knowing it will be scored on
"sycophancy" is fine), but the actual prompts, judge prompts, and rubric
weights live here, outside the agent's working tree.

The v0 panel reproduces the spec at
`/Users/jack/projects/claude-trains-qwen/HELDOUT_CHARACTER_EVAL_SET.md`,
plus two political/values batteries (Rozado 2024, OpenAI 2025) and the
MFQ-30 moral-foundations questionnaire.

## v0 panel: 15 tasks

| # | Task dir | What it measures | Source | Items |
|--:|---|---|---|--:|
| 1 | `sycophancy_slava` | sycophancy (6 categories, hand-curated) | `third-party/character-steering` | 30 |
| 2 | `sycophancy_sharma` | canonical sycophancy baseline | `inspect_evals.sycophancy` | 60 |
| 3 | `sycophancy_aisi` | sycophancy under input framing (5-facet rubric) | reproduced from Dubois 2026 (arXiv 2602.23971); paper data not released | 88 (gen at setup) |
| 4 | `abstention_bench` | knows-when-not-to-answer | `inspect_evals.abstention_bench` | 100 |
| 5 | `coconot` | refusal quality / "Art of Saying No" | `inspect_evals.coconot` | 100 |
| 6 | `strong_reject` | jailbreak susceptibility | `inspect_evals.strong_reject` | 50 |
| 7 | `moru` | moral reasoning under uncertainty | `inspect_evals.moru` | 50 |
| 8 | `big_five` | Big Five / IPIP personality fingerprint | `inspect_evals.personality` | 100 |
| 9 | `moral_foundations` | MFQ-30 moral-foundations fingerprint (5 foundations) | Graham/Haidt/Nosek 2011, public academic instrument | 32 |
| 10 | `political_bias_openai` | behavioural political bias (5 axes × 5 slants × 100 topics) | reproduced from OpenAI Oct 2025; prompts not released | 500 (gen at setup) |
| 11 | `rozado_battery` | 10 standardized political quizzes + recovered axis weights | vendored from Zenodo DOI 10.5281/zenodo.10553530 (CC-BY-4.0) | 383 |
| 12 | `spiralbench_mini` | multi-turn delusion-reinforcement panel | frozen `sam-paech/spiral-bench` v0.2 prompts, v1.2 rubric | 20 conv × 6 turns |
| 13 | `capability_gsm8k` | grade-school math, capability preservation | delegate to `src/eval/tasks/gsm8k` | 150 |
| 14 | `capability_humaneval` | code-gen capability preservation | delegate to `src/eval/tasks/humaneval` | 164 |
| 15 | `capability_gpqa` | graduate science capability preservation | delegate to `src/eval/tasks/gpqamain` | 50 |

## How to run

### Pre-flight (one-shot, ~$2-5 in Anthropic API calls)

Two tasks generate prompts at setup time because the upstream papers do not
release prompt data; `generate_prompts.py` reproduces the construction
recipe via Sonnet, then commits a frozen `prompts.jsonl` for deterministic
runs:

```bash
ANTHROPIC_API_KEY=... \
    python src/heldout_evals/tasks/sycophancy_aisi/generate_prompts.py \
        --base-questions 8

ANTHROPIC_API_KEY=... \
    python src/heldout_evals/tasks/political_bias_openai/generate_prompts.py \
        --limit 100
```

### Run the panel against a checkpoint

```bash
bash src/heldout_evals/run_heldout.sh <run_dir>
```

`<run_dir>` is the directory produced by `src/run_task.sh`; it must
contain `final_model/`. Output lands in `<run_dir>/heldout/`:

```
<run_dir>/heldout/
  sycophancy_slava.json
  sycophancy_sharma.json
  sycophancy_aisi.json
  abstention_bench.json
  coconot.json
  strong_reject.json
  moru.json
  big_five.json
  moral_foundations.json
  political_bias_openai.json
  rozado_battery.json
  spiralbench_mini.json
  capability_gsm8k.json
  capability_humaneval.json
  capability_gpqa.json
  summary.json   # produced by aggregate.py
  summary.md
```

### Run a single task

```bash
bash src/heldout_evals/run_heldout.sh <run_dir> sycophancy_slava
bash src/heldout_evals/run_heldout.sh <run_dir> moral_foundations rozado_battery
```

## Layout

```
src/heldout_evals/
  tasks/
    <name>/evaluate.py        # standard CLI; one task = one entrypoint
    <name>/benchmark.txt      # human-readable name
    <name>/README.md          # task-specific notes (where applicable)
    _common.py                # add_standard_args, model_type, template_kwargs, write_metrics
    _delegate.py              # subprocess shim for capability_* tasks
    _inspect_wrap.py          # one-line wraps of inspect_evals.<name> tasks
  judge/haiku_judge.py        # Anthropic API client (claude-haiku-4-5)
  run_heldout.sh              # iterate tasks/<name>/, write per-task JSON + run aggregate.py
  aggregate.py                # per-task JSON -> summary.{json,md}
```

## Generated vs vendored prompts

Three tasks have prompt sets that are not directly pulled from upstream:

| Task | Prompt source | Reproducibility |
|---|---|---|
| `sycophancy_aisi` | `generate_prompts.py` calls Sonnet using paper §7.1 recipe; output frozen to `prompts.jsonl` | Methodology is paper-faithful (40 base × 11 framings, topic taxonomy from Figure 4A); prompts regenerate on each setup run unless committed |
| `political_bias_openai` | `generate_prompts.py` calls Sonnet using OpenAI Oct 2025 framework; output frozen to `prompts.jsonl` | Paper-faithful slant taxonomy (lib_charged/lib_neutral/neutral/cons_neutral/cons_charged) + 5-axis rubric from blog post |
| `rozado_battery` | Frozen vendoring of Rozado 2024 Zenodo CC-BY-4.0 dataset; weights recovered via least-squares regression on his trial data | 6/10 tests have axes recovered at R² ≥ 0.58 (PCT 0.97, PSQ 1.00, WSPQ 1.00, Nolan 0.97, PCT-coords 0.61, ideologies 0.58); 4/10 fall back to mean-Likert fingerprint (8values, Eysenck, iSideWith UK/US — non-linear scoring upstream) |

Commit `prompts.jsonl` (and `tests/<test>/items.json` for Rozado) once
they look right. `generate_prompts.py` and `_solve_weights.py` are
re-runnable for refreshing the prompt set later.

## Reporting fields

`aggregate.py` extracts a flat reporting dict from per-task metrics,
matching the spec at `HELDOUT_CHARACTER_EVAL_SET.md` §Reporting Format:

| Field | Source task |
|---|---|
| `sycophancy_score_slava` | `sycophancy_slava.overall` |
| `sycophancy_score_sharma` | `sycophancy_sharma.accuracy` |
| `sycophancy_score_aisi` | `sycophancy_aisi.total_sycophancy` |
| `abstention_score` | `abstention_bench.accuracy` |
| `refusal_boundary_score` | `coconot.accuracy` |
| `jailbreak_score` | `strong_reject.score` |
| `moral_uncertainty_score` | `moru.accuracy` |
| `spiral_score` | `spiralbench_mini.weighted_score` |
| `big_five_delta` | `big_five.trait_scores` |
| `moral_foundations_delta` | `moral_foundations.foundations` |
| `moral_individualizing` / `moral_binding` | `moral_foundations.{individualizing,binding}_score` |
| `rozado_fingerprint` | `rozado_battery.fingerprint` |
| `openai_political_bias_final` / `_by_axis` / `_by_slant` | `political_bias_openai.{final_score,by_axis,by_slant}` |
| `capability_gsm8k` / `capability_humaneval` / `capability_gpqa` | matching task scores |

## Adding a new task

1. Create `tasks/<name>/evaluate.py` with the standard CLI
   (uses `_common.add_standard_args(parser, default_limit=...)`).
2. Add `tasks/<name>/benchmark.txt` (one-liner).
3. Reuse:
   - **inspect_evals wrap**: copy `tasks/abstention_bench/evaluate.py` and
     change the two task-name strings.
   - **delegate to existing src/eval/tasks**: copy `tasks/capability_gsm8k/evaluate.py`.
   - **judge-based custom**: copy `tasks/sycophancy_slava/evaluate.py`.
   - **multi-axis judge with rubric**: copy `tasks/sycophancy_aisi/`.
4. Smoke-test:
   ```bash
   python tasks/<name>/evaluate.py --model-path Qwen/Qwen3-1.7B-Base \
       --limit 5 --json-output-file /tmp/<name>.json
   ```
5. Update `aggregate.py` if the task contributes a top-level reporting
   field.

## Pending follow-ups (v0.1+)

- **Persona-vector sidecar** (Phase 7 in original plan) — separate plan needed.
- **Bloom custom traits** (Phase 8) — separate plan needed.
- **Per-axis weights for the 4 non-linear Rozado tests** — could be lifted
  from upstream JS:
  - 8values → `github.com/8values/8values.github.io/questions.js`
  - iSideWith UK/US → would need scraping per-party-per-policy weight tables
- **Validate regex-based stance parser** against Rozado's stance-detection
  xlsx samples (`/tmp/rozado-small-*/sample_for_stance_detection_measurament.xlsx`,
  132 labeled rows). Currently we use longest-match regex; if real model
  outputs slip past that, fall back to LLM-based parser like Rozado does.
- **Smoke run + base-vs-instruct sanity** — needs GPU, see expected-direction
  deltas (sycophancy ↑, agreeableness ↑, abstention ↓ for instruct vs base).
