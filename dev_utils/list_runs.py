"""List + filter runs collected by agent_run.py.

Walks jobs/runs/<dir>/{summary.json, config.json} and prints a table.

Usage:
  python dev_utils/list_runs.py
  python dev_utils/list_runs.py --sort delta --limit 20
  python dev_utils/list_runs.py --filter teacher=claude-opus-4-7 --filter condition=C
  python dev_utils/list_runs.py --csv > runs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "jobs" / "runs"


COLUMNS = [
    "timestamp",
    "cond",
    "teacher",
    "student",
    "seed",
    "pre",
    "post",
    "delta",
    "cont",
    "disall",
    "status",
    "duration",
]


def fmt_metric(d: dict | None) -> str:
    if not d:
        return "-"
    # Pick first numeric. PTB tasks vary; prefer 'accuracy' if present.
    if "accuracy" in d:
        return f"{d['accuracy']:.3f}"
    k, v = next(iter(d.items()))
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def fmt_delta(d: dict | None) -> str:
    if not d:
        return "-"
    if "accuracy" in d:
        v = d["accuracy"]
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.3f}"
    k, v = next(iter(d.items()))
    if isinstance(v, (int, float)):
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.3f}"
    return str(v)


def fmt_duration(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def fmt_bool(b: bool | None) -> str:
    if b is None:
        return "?"
    return "yes" if b else "no"


def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # tolerate trailing Z
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for d in sorted(root.glob("*"), key=lambda p: p.name):
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        sm_path = d / "summary.json"
        sm: dict[str, Any] = {}
        if sm_path.exists():
            try:
                sm = json.loads(sm_path.read_text())
            except Exception:
                pass
        rows.append({
            "dir": d,
            "timestamp_raw": cfg.get("started_at"),
            "timestamp": parse_iso(cfg.get("started_at")),
            "cond": cfg.get("condition", "?"),
            "teacher": cfg.get("teacher_slug") or cfg.get("teacher_model", "?"),
            "student": cfg.get("student_slug") or cfg.get("student_model", "?"),
            "seed": cfg.get("seed", "?"),
            "benchmark": cfg.get("benchmark", "?"),
            "pre_dict": sm.get("pre"),
            "post_dict": sm.get("post"),
            "delta_dict": sm.get("delta"),
            "cont": sm.get("contamination_detected"),
            "disall": sm.get("disallowed_use_detected"),
            "status": sm.get("status", "in_progress"),
            "duration_s": sm.get("duration_s"),
        })
    return rows


def apply_filters(rows: list[dict], filters: list[str]) -> list[dict]:
    if not filters:
        return rows
    out = rows
    for f in filters:
        if "=" not in f:
            print(f"warning: bad --filter '{f}'; expected k=v", file=sys.stderr)
            continue
        k, v = f.split("=", 1)
        out = [r for r in out if str(r.get(k, "")) == v]
    return out


def sort_rows(rows: list[dict], sort_key: str) -> list[dict]:
    if sort_key == "time":
        rows.sort(
            key=lambda r: (r["timestamp"] or datetime.min),
            reverse=True,
        )
    elif sort_key == "delta":
        def k(r):
            d = r["delta_dict"] or {}
            return -(d.get("accuracy", float("-inf")) if isinstance(d.get("accuracy"), (int, float)) else float("-inf"))
        rows.sort(key=k)
    elif sort_key in {"teacher", "student", "cond", "status", "benchmark"}:
        rows.sort(key=lambda r: str(r.get(sort_key, "")))
    return rows


def render_table(rows: list[dict]) -> None:
    """Plain text fallback. Tries rich.Table if installed."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        for col in COLUMNS:
            table.add_column(col)
        for r in rows:
            ts = r["timestamp"].strftime("%Y-%m-%d %H:%M") if r["timestamp"] else "?"
            table.add_row(
                ts,
                r["cond"],
                r["teacher"],
                r["student"],
                str(r["seed"]),
                fmt_metric(r["pre_dict"]),
                fmt_metric(r["post_dict"]),
                fmt_delta(r["delta_dict"]),
                fmt_bool(r["cont"]),
                fmt_bool(r["disall"]),
                r["status"],
                fmt_duration(r["duration_s"]),
            )
        Console().print(table)
        return
    except ImportError:
        pass

    # Plain text fallback
    headers = ["TIMESTAMP", "COND", "TEACHER", "STUDENT", "SEED", "PRE", "POST", "Δ", "CONT", "DISALL", "STATUS", "DURATION"]
    fmt_rows = []
    for r in rows:
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M") if r["timestamp"] else "?"
        fmt_rows.append([
            ts, r["cond"], r["teacher"], r["student"], str(r["seed"]),
            fmt_metric(r["pre_dict"]), fmt_metric(r["post_dict"]),
            fmt_delta(r["delta_dict"]),
            fmt_bool(r["cont"]), fmt_bool(r["disall"]),
            r["status"], fmt_duration(r["duration_s"]),
        ])
    widths = [max(len(h), *(len(row[i]) for row in fmt_rows)) for i, h in enumerate(headers)] if fmt_rows else [len(h) for h in headers]
    line = lambda cells: "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    print(line(headers))
    print(line(["-" * w for w in widths]))
    for r in fmt_rows:
        print(line(r))


def render_csv(rows: list[dict]) -> None:
    w = csv.writer(sys.stdout)
    w.writerow(COLUMNS)
    for r in rows:
        ts = r["timestamp"].isoformat() if r["timestamp"] else ""
        w.writerow([
            ts, r["cond"], r["teacher"], r["student"], r["seed"],
            fmt_metric(r["pre_dict"]), fmt_metric(r["post_dict"]),
            fmt_delta(r["delta_dict"]),
            fmt_bool(r["cont"]), fmt_bool(r["disall"]),
            r["status"], fmt_duration(r["duration_s"]),
        ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"runs directory (default {DEFAULT_ROOT.relative_to(REPO_ROOT)})")
    p.add_argument("--sort", default="time",
                   choices=["time", "delta", "teacher", "student", "cond", "status", "benchmark"])
    p.add_argument("--filter", action="append", default=[], metavar="K=V",
                   help="repeatable; e.g. --filter teacher=claude-opus-4-7 --filter condition=C")
    p.add_argument("--limit", type=int, default=0,
                   help="show at most N rows (0 = all)")
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()

    if not args.root.exists():
        print(f"runs root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    rows = collect(args.root)
    rows = apply_filters(rows, args.filter)
    rows = sort_rows(rows, args.sort)
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("(no runs)", file=sys.stderr)
        return

    if args.csv:
        render_csv(rows)
    else:
        render_table(rows)


if __name__ == "__main__":
    main()
