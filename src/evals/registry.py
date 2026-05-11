"""Canonical eval-suite registry.

Single source of truth for which eval tasks exist, what category they
belong to, what they measure, and where their evaluate.py lives. Every
caller — pre/post-eval, baseline runner, agent-facing score.sh,
contamination judge, summary aggregator — reads from here so the suite
stays consistent.

History (2026-05-11):
    Pre-refactor the suite was split between src/eval/tasks/ (used by
    pre/post-eval) and src/heldout_evals/tasks/ (used by run_heldout.sh
    panel) with three `capability_*` wrappers, two divergent sycophancy
    forks, and a `sycophancy`/`sycophancy_sharma` naming inconsistency.
    Centralised into src/evals/tasks/{capability,safety,character}/ with
    this registry as the discovery layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EVALS_ROOT = Path(__file__).resolve().parent
TASKS_ROOT = EVALS_ROOT / "tasks"


Category = Literal["capability", "safety", "character"]


@dataclass(frozen=True)
class EvalInfo:
    """One row in the canonical registry."""

    name: str
    category: Category
    display_name: str
    # `attribute`: for safety/character tasks, the trait the task is a
    # proxy for. Multiple tasks can share an attribute (e.g. sycophancy
    # is measured by three independent benchmarks). `--safety-attribute
    # sycophancy` runs all three and reports the objective delta as a
    # mean / per-bench breakdown. None for pure-capability tasks.
    attribute: str | None = None
    notes: str = ""
    # `direction`: higher_is_better=True means the eval's accuracy IS
    # the desired behaviour (gsm8k, humaneval, refusal coverage). False
    # means the accuracy is the failure rate the agent should minimise
    # under condition F / maximise under condition D (sycophancy_*).
    higher_is_better: bool = True
    # Hosts that need a per-task `task_context/` (e.g. bfcl's external
    # checker). Auto-discovered when staging.
    has_task_context: bool = False
    # Headline metric key in evaluate.py's output JSON. Default
    # "accuracy" — most evals emit that. Dotted-path supported for
    # nested scorers (e.g. inspect_evals tasks emit
    # "strong_reject_scorer.strong_reject_metric"). Set to None for
    # multi-dimensional evals (big_five, rozado_battery, etc.) where
    # no single number is the headline; full breakdown lives in the
    # per-bench JSON file regardless.
    headline_metric: str | None = "accuracy"

    @property
    def path(self) -> Path:
        """Absolute path to this task's directory."""
        return TASKS_ROOT / self.category / self.name

    @property
    def evaluate_py(self) -> Path:
        return self.path / "evaluate.py"


# ─── Canonical suite ─────────────────────────────────────────────────────
# Order within each category is intentional: most-load-bearing benchmarks
# first so summary tables / matrix runs surface them at the top.

EVAL_SUITE: dict[str, EvalInfo] = {
    # Capability (10) — standard task-completion benchmarks. Failures
    # under condition F count as "capability drift" and disqualify the
    # adapter even if the safety-attribute delta is favourable.
    "gsm8k":            EvalInfo("gsm8k",            "capability", "GSM8K"),
    "humaneval":        EvalInfo("humaneval",        "capability", "HumanEval"),
    "mmlu":             EvalInfo("mmlu",             "capability", "MMLU"),
    "gpqamain":         EvalInfo("gpqamain",         "capability", "GPQA-main"),
    "arc_easy":         EvalInfo("arc_easy",         "capability", "ARC-Easy"),
    "truthfulqa":       EvalInfo("truthfulqa",       "capability", "TruthfulQA"),
    "aime2025":         EvalInfo("aime2025",         "capability", "AIME 2025"),
    "bfcl":             EvalInfo("bfcl",             "capability", "BFCL", has_task_context=True),
    "arenahardwriting": EvalInfo("arenahardwriting", "capability", "Arena-Hard-v2 (Writing)"),
    "healthbench":      EvalInfo("healthbench",      "capability", "HealthBench"),

    # Safety (7) — behavioural / safety-relevant constructs. Eligible as
    # `--safety-attribute` targets.
    "sycophancy_sharma": EvalInfo(
        "sycophancy_sharma", "safety", "Sycophancy (Sharma et al.)",
        attribute="sycophancy", higher_is_better=False,
        notes="admits_mistake rate. Higher = more sycophantic.",
    ),
    "sycophancy_slava": EvalInfo(
        "sycophancy_slava", "safety", "Sycophancy (Slava prototype)",
        attribute="sycophancy", higher_is_better=False,
        notes="Judge-rated sycophancy_rate = 1 - candor_rate. Higher = more sycophantic.",
    ),
    "sycophancy_aisi": EvalInfo(
        "sycophancy_aisi", "safety", "Sycophancy (AISI Ask-Don't-Tell)",
        attribute="sycophancy", higher_is_better=False,
        notes="5-facet rubric mean / 15. Higher = more sycophantic.",
    ),
    "strong_reject": EvalInfo(
        "strong_reject", "safety", "StrongREJECT",
        attribute="harmful_compliance", higher_is_better=False,
        notes="Jailbreak rate: 0=perfect refusal, 1=fully complies.",
        # strong_reject_metric is the refusal-quality score (1-jailbreak_rate
        # roughly); we want the more intuitive jailbreak_rate as the
        # headline (lower=better, matches higher_is_better=False).
        headline_metric="strong_reject_scorer.jailbreak_rate",
    ),
    "coconot": EvalInfo(
        "coconot", "safety", "CoCoNot",
        attribute="over_refusal", higher_is_better=False,
        notes="Over-refusal of benign requests. Higher = worse.",
        # Default 'accuracy' likely absent; CoCoNot uses model_graded_qa
        # bucket counts. Headline = "safety concerns identified" rate.
        # Adjust once we see a real run output.
        headline_metric="model_graded_qa.total",
    ),
    "abstention_bench": EvalInfo(
        "abstention_bench", "safety", "AbstentionBench",
        attribute="calibrated_abstention",
        # 15 sub-datasets, no single headline. Aggregator can compute
        # mean across sub-datasets if useful; per-dataset breakdown
        # lives in the full JSON.
        headline_metric=None,
    ),
    "spiralbench_mini": EvalInfo(
        "spiralbench_mini", "safety", "SpiralBench (mini)",
        attribute="delusion_reinforcement", higher_is_better=False,
        # Off-rails or weighted-rubric average; check after first
        # successful run.
        headline_metric=None,
    ),

    # Character (5) — personality / values / dispositions. Reported in
    # capability_drift section but not typically a `--safety-attribute`
    # target (no clear "lower is safer" direction for big_five etc.).
    "big_five": EvalInfo(
        "big_five", "character", "Big Five personality",
        # 5 trait dimensions, no single headline. Full breakdown
        # (Conscientiousness, Neuroticism, Openness, Extraversion,
        # Agreeableness) in per-bench JSON.
        headline_metric=None,
    ),
    "moral_foundations": EvalInfo(
        "moral_foundations", "character", "Moral Foundations Questionnaire",
        # 5 foundations + 2 aggregate scores. No single headline.
        headline_metric=None,
    ),
    "rozado_battery": EvalInfo(
        "rozado_battery", "character", "Rozado political-bias battery",
        attribute="political_bias",
        # Multi-test fingerprint; no single headline.
        headline_metric=None,
    ),
    "political_bias_openai": EvalInfo(
        "political_bias_openai", "character", "OpenAI political-bias eval",
        attribute="political_bias",
        # 5-axis judge rubric scored on 4 slants × 4 areas. Check after
        # first successful run for canonical headline.
        headline_metric=None,
    ),
    "moru": EvalInfo("moru", "character", "MoRU", headline_metric=None),
}


# ─── Convenience accessors ──────────────────────────────────────────────


def by_category(category: Category) -> list[EvalInfo]:
    """All tasks in a category, in registry order."""
    return [e for e in EVAL_SUITE.values() if e.category == category]


def by_attribute(attribute: str) -> list[EvalInfo]:
    """All tasks measuring an attribute (e.g. "sycophancy" → 3 tasks)."""
    return [e for e in EVAL_SUITE.values() if e.attribute == attribute]


def get_headline(metrics: dict, info: "EvalInfo") -> float | None:
    """Extract the headline metric from an eval's full metrics dict.

    Uses info.headline_metric. Tries literal-key match first (some
    evaluators emit `"a.b": v` as flat keys with dots in the name —
    inspect_evals scorers, e.g. `strong_reject_scorer.jailbreak_rate`),
    then falls back to a dotted-path walk for genuinely nested dicts.
    Returns None if info.headline_metric is None (multi-dim evals) or
    nothing resolves.
    """
    if info.headline_metric is None or not metrics:
        return None
    if info.headline_metric in metrics:
        v = metrics[info.headline_metric]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    parts = info.headline_metric.split(".")
    cur: object = metrics
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def full_suite() -> list[str]:
    """All registered task names, in registry order."""
    return list(EVAL_SUITE.keys())


def capability_drift_tasks(exclude: list[str] | None = None) -> list[EvalInfo]:
    """Tasks reported under `capability_drift` axis of summary.json.

    By default this is every capability + character + non-target safety
    task. Caller passes `exclude=measured_by` (the tasks scoring the
    --safety-attribute objective) to keep those out of the drift axis
    and into the objective axis instead.
    """
    exclude = set(exclude or [])
    return [e for e in EVAL_SUITE.values() if e.name not in exclude]


def validate() -> None:
    """Sanity-check the registry against the filesystem.

    Raises if any registered task has no evaluate.py on disk or if any
    on-disk task is missing from the registry. Called at import-time of
    submit_run.py / submit_baseline.py / run_experiment.py so we fail
    fast on registry drift.
    """
    registered = {e.name for e in EVAL_SUITE.values()}
    on_disk: set[str] = set()
    for cat_dir in TASKS_ROOT.iterdir():
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue
        for task_dir in cat_dir.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            on_disk.add(task_dir.name)

    missing_on_disk = registered - on_disk
    if missing_on_disk:
        raise RuntimeError(
            f"registry references tasks not on disk: {sorted(missing_on_disk)}. "
            f"Add evaluate.py under src/evals/tasks/<category>/<name>/ or remove from EVAL_SUITE."
        )
    missing_in_registry = on_disk - registered
    if missing_in_registry:
        raise RuntimeError(
            f"tasks on disk not in registry: {sorted(missing_in_registry)}. "
            f"Add entries to EVAL_SUITE in src/evals/registry.py."
        )

    for ev in EVAL_SUITE.values():
        if not ev.evaluate_py.exists():
            raise RuntimeError(
                f"registry task {ev.name} ({ev.category}) has no evaluate.py at {ev.evaluate_py}"
            )


__all__ = [
    "EVALS_ROOT",
    "TASKS_ROOT",
    "Category",
    "EvalInfo",
    "EVAL_SUITE",
    "by_category",
    "by_attribute",
    "full_suite",
    "capability_drift_tasks",
    "get_headline",
    "validate",
]
