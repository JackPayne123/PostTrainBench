"""Per-run directory layout helpers.

Every agent_run.py invocation produces a self-contained directory under
jobs/runs/ named by (timestamp, condition, teacher_slug, student_slug, seed).
This module centralises the slug rules, dir naming, and config/summary I/O so
agent_run.py and dev_utils/list_runs.py share one source of truth.

Layout:
  jobs/runs/<dirname>/
    config.json           # invariant inputs to the run (teacher, student, etc.)
    summary.json          # rolled up after run completes
    metrics_pre.json      # eval scores BEFORE training
    metrics_post.json     # eval scores AFTER training
    char_probe.json       # held-out char eval (placeholder until #45 lands)
    contamination_judgement.txt
    disallowed_model_judgement.txt
    judge_output.json
    solve_out.jsonl       # raw claude-code stream-json log
    solve_parsed.txt      # human-readable transcript
    agent_workspace.tar.gz
    final_model/          # merged-LoRA finetune (~3.5 GB; gitignored)
    timer.log
    pod_meta.json
    system_monitor.log
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def slugify_model(model_id: str) -> str:
    """Turn an HF model id (or arbitrary teacher name) into a path-safe slug.

    Drops the org prefix (`Qwen/Qwen3-1.7B-Base` -> `Qwen3-1.7B-Base`),
    lowercases, keeps `.`, replaces other non-alnum with `-`, collapses
    repeated dashes, strips leading/trailing dashes.
    """
    base = model_id.rsplit("/", 1)[-1].lower()
    s = re.sub(r"[^a-z0-9.]+", "-", base)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def build_run_dir_name(
    *,
    condition: str,
    teacher_slug: str,
    student_slug: str,
    seed: int,
    when: datetime | None = None,
) -> str:
    # Per-minute timestamp + 6-char random hex prevents collisions when
    # two runs are submitted in the same minute (parallel sweep). Caught
    # as design TODO #22 — the prior `<ts>_<cond>_<teacher>_<student>_seedN`
    # shape collided across `for seed in 0 1 2; do submit_run ... &; done`.
    import secrets
    when = when or datetime.now(timezone.utc).astimezone()
    ts = when.strftime("%Y-%m-%d_%H-%M")
    suffix = secrets.token_hex(3)  # 6 chars
    return f"{ts}_{condition}_{teacher_slug}_{student_slug}_seed{seed}_{suffix}"


@dataclass
class RunConfig:
    """What we fix about a run before it starts. Written to config.json."""
    schema_version: int
    condition: str
    teacher_model: str
    student_model: str
    teacher_slug: str
    student_slug: str
    benchmark: str
    prompt_variant: str
    seed: int
    time_budget_h: float
    agent: str
    agent_config: str
    base_image: str
    git_sha: str
    started_at: str
    run_dir_name: str
    extra: dict[str, Any] = field(default_factory=dict)


def make_run_config(
    *,
    condition: str,
    teacher_model: str,
    student_model: str,
    benchmark: str,
    prompt_variant: str,
    seed: int,
    time_budget_h: float,
    agent: str,
    agent_config: str,
    base_image: str,
    git_sha: str,
    extra: dict[str, Any] | None = None,
) -> tuple[RunConfig, str]:
    """Returns (config, run_dir_name)."""
    teacher_slug = slugify_model(teacher_model)
    student_slug = slugify_model(student_model)
    when = datetime.now(timezone.utc).astimezone()
    dirname = build_run_dir_name(
        condition=condition,
        teacher_slug=teacher_slug,
        student_slug=student_slug,
        seed=seed,
        when=when,
    )
    cfg = RunConfig(
        schema_version=SCHEMA_VERSION,
        condition=condition,
        teacher_model=teacher_model,
        student_model=student_model,
        teacher_slug=teacher_slug,
        student_slug=student_slug,
        benchmark=benchmark,
        prompt_variant=prompt_variant,
        seed=seed,
        time_budget_h=time_budget_h,
        agent=agent,
        agent_config=agent_config,
        base_image=base_image,
        git_sha=git_sha,
        started_at=when.isoformat(),
        run_dir_name=dirname,
        extra=extra or {},
    )
    return cfg, dirname


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def write_run_config(run_dir: Path, cfg: RunConfig) -> None:
    write_json(run_dir / "config.json", asdict(cfg))


def headline_from_metrics(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    """Extract a {metric_name: value} dict from PTB's varied metrics.json shapes.

    PTB tasks are heterogenous — some return {accuracy: x, stderr: y}, some
    nest under a scorer name. We pull the first numeric or {value: ...} field
    we see, plus a stderr if present alongside.
    """
    if not metrics:
        return None
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, dict) and "value" in v and isinstance(v["value"], (int, float)):
            out[k] = float(v["value"])
    return out or None


def compute_delta(
    pre: dict[str, float] | None, post: dict[str, float] | None
) -> dict[str, float] | None:
    if not pre or not post:
        return None
    return {k: post[k] - pre[k] for k in post.keys() & pre.keys()}


def parse_judgement(path: Path, positive_token: str) -> bool | None:
    """Parse one of PTB's judgement text files.

    contamination_judgement.txt content is one line like
    "no contamination detected" / "contamination detected".
    disallowed_model_judgement.txt is "only allowed use detected" / "disallowed use detected".

    Returns True if the positive (= bad) token is present, False if explicitly
    negated, None if file missing or unreadable.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text().strip().lower()
    except OSError:
        return None
    if not text:
        return None
    if text.startswith("no "):
        return False
    return positive_token.lower() in text


def make_summary(
    *,
    run_dir: Path,
    status: str,
    duration_s: float,
    pre_metrics: dict[str, Any] | None,
    post_metrics: dict[str, Any] | None,
    char_probe: dict[str, Any] | None,
    pod_info: dict[str, Any] | None,
    final_model_dir: Path,
    solve_out_path: Path,
) -> dict[str, Any]:
    pre_h = headline_from_metrics(pre_metrics)
    post_h = headline_from_metrics(post_metrics)
    final_present = (final_model_dir / "config.json").exists()
    final_size = (
        sum(p.stat().st_size for p in final_model_dir.rglob("*") if p.is_file())
        if final_model_dir.exists()
        else 0
    )
    trace_lines = 0
    if solve_out_path.exists():
        with solve_out_path.open() as f:
            for _ in f:
                trace_lines += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "config_ref": "config.json",
        "status": status,
        "duration_s": round(duration_s, 1),
        "pre": pre_h,
        "post": post_h,
        "delta": compute_delta(pre_h, post_h),
        "char_probe": char_probe,
        "contamination_detected": parse_judgement(
            run_dir / "contamination_judgement.txt", "contamination detected"
        ),
        "disallowed_use_detected": parse_judgement(
            run_dir / "disallowed_model_judgement.txt", "disallowed use detected"
        ),
        "final_model_present": final_present,
        "final_model_size_bytes": final_size,
        "agent_trace_lines": trace_lines,
        "pod": pod_info or {},
    }
