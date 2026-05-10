#!/usr/bin/env python3
"""Aggregate per-task heldout JSON files into a single summary.

Maps the spec's reporting fields (HELDOUT_CHARACTER_EVAL_SET.md §Reporting Format)
to the per-task metrics produced by each evaluate.py.

Usage:
    python aggregate.py <heldout_dir>

Writes <heldout_dir>/summary.json and <heldout_dir>/summary.md.

Phase 7 will replace this stub with a richer mapper. For now: collect every
task's metrics dict under its task name, plus a flat top-level set of the
spec's reporting fields where derivable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def load_task_results(heldout_dir: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(os.listdir(heldout_dir)):
        if not name.endswith(".json") or name.startswith("summary"):
            continue
        task = name.removesuffix(".json")
        with open(os.path.join(heldout_dir, name)) as f:
            out[task] = json.load(f)
    return out


def derive_reporting_fields(by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map per-task metrics to HELDOUT spec §Reporting Format fields.

    This is a best-effort mapping. Phase 7 will tighten it as each task's
    actual metric keys land. Missing tasks produce `null` rather than KeyError.
    """
    def pick(task: str, *keys: str) -> Any:
        d = by_task.get(task)
        if not d:
            return None
        for k in keys:
            if k in d:
                return d[k]
        return None

    return {
        "sycophancy_score_slava": pick("sycophancy_slava", "overall", "score", "accuracy"),
        "sycophancy_score_sharma": pick("sycophancy_sharma", "accuracy", "score"),
        "sycophancy_score_aisi": pick("sycophancy_aisi", "total_sycophancy", "score"),
        "abstention_score": pick("abstention_bench", "accuracy", "score"),
        "refusal_boundary_score": pick("coconot", "accuracy", "score"),
        "jailbreak_score": pick("strong_reject", "score", "accuracy"),
        "moral_uncertainty_score": pick("moru", "accuracy", "score"),
        "spiral_score": pick("spiralbench_mini", "overall", "score"),
        "big_five_delta": pick("big_five", "trait_scores", "score"),
        "moral_foundations_delta": pick("moral_foundations", "foundations", "score"),
        "moral_individualizing": pick("moral_foundations", "individualizing_score"),
        "moral_binding": pick("moral_foundations", "binding_score"),
        "rozado_fingerprint": pick("rozado_battery", "fingerprint"),
        "openai_political_bias_final": pick("political_bias_openai", "final_score"),
        "openai_political_bias_by_axis": pick("political_bias_openai", "by_axis"),
        "openai_political_bias_by_slant": pick("political_bias_openai", "by_slant"),
        "capability_gsm8k": pick("capability_gsm8k", "accuracy", "score"),
        "capability_humaneval": pick("capability_humaneval", "pass@1", "accuracy", "score"),
        "capability_gpqa": pick("capability_gpqa", "accuracy", "score"),
    }


def write_markdown(summary: dict[str, Any], by_task: dict[str, dict[str, Any]], path: str) -> None:
    lines: list[str] = ["# Heldout Eval Summary", ""]
    lines.append("## Reporting fields (per HELDOUT spec)")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k, v in summary.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## Per-task raw metrics")
    lines.append("")
    for task, metrics in by_task.items():
        lines.append(f"### {task}")
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2))
        lines.append("```")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("heldout_dir", help="dir containing <task>.json files")
    args = parser.parse_args()

    if not os.path.isdir(args.heldout_dir):
        print(f"error: {args.heldout_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    by_task = load_task_results(args.heldout_dir)
    summary = derive_reporting_fields(by_task)

    out_json = os.path.join(args.heldout_dir, "summary.json")
    out_md = os.path.join(args.heldout_dir, "summary.md")

    with open(out_json, "w") as f:
        json.dump({"reporting": summary, "by_task": by_task}, f, indent=2)
    write_markdown(summary, by_task, out_md)

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
