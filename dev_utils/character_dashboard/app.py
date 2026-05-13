#!/usr/bin/env python3
"""Post-run character-delta dashboard.

Local stdlib HTTP server that visualises how a trained LoRA adapter
shifts character vs. the Qwen3-1.7B base. Reads:
  - jobs/runs/<adapter-run-id>/deltas.json (primary; flat dotted-key
    multi-dim data already produced by scripts/compute_deltas.py)
  - baselines/<slug>/<bench>__limit<N>.json (fallback when deltas.json
    is missing a particular nested field, e.g. rozado per-test axes)

Pattern mirrors dev_utils/trace_viewer/app.py — stdlib ThreadingHTTPServer,
no Flask, vendored Chart.js (static/), inline SVG for the political
compass. Loopback-only.

    python3 dev_utils/character_dashboard/app.py
    # → http://127.0.0.1:8766

Routes:
    /                              — index of adapter runs with deltas.json
    /run/<adapter-run-id>          — single-page dashboard for one run
    /run/<id>/eval/<bench>         — raw JSON view (debug)
    /static/<file>                 — vendored Chart.js + CSS
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = REPO_ROOT / "jobs" / "runs"
BASELINES_DIR = REPO_ROOT / "baselines"

# Trace-viewer integration: import its render functions so the agent-trace
# view is mounted under /trace within this same server (single port, shared
# top-nav). Trace viewer keeps its own CSS + page() so it stays runnable
# standalone, but here we use its render_index / render_run / load_run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace_viewer"))
import app as trace_viewer  # noqa: E402  (sibling module, intentional)
sys.path.pop(0)

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_\-:.]+$")
BENCH_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


# ─── Data shaping ────────────────────────────────────────────────────────


def safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_adapter_runs() -> list[dict]:
    """Adapter runs are any jobs/runs/<id>/deltas.json that exists."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        deltas = run_dir / "deltas.json"
        if not deltas.exists():
            continue
        d = safe_read_json(deltas) or {}
        summary = safe_read_json(run_dir / "summary.json") or {}
        # Count character-tier evals + how many had a significant
        # multi-dim shift (any sub-key |delta| >= 0.05 OR scalar |delta| >= 0.05).
        n_character = sum(1 for r in d.values() if r.get("category") == "character")
        n_shifted = 0
        for r in d.values():
            if r.get("category") != "character":
                continue
            if r.get("kind") == "scalar" and r.get("delta") is not None:
                if abs(r["delta"]) >= 0.05:
                    n_shifted += 1
            elif r.get("kind") == "multi":
                for sub in (r.get("sub") or {}).values():
                    if isinstance(sub.get("delta"), (int, float)) and abs(sub["delta"]) >= 0.05:
                        n_shifted += 1
                        break
        out.append({
            "name": run_dir.name,
            "started_at": (summary.get("config") or {}).get("started_at", ""),
            "condition": (summary.get("config") or {}).get("condition", "?"),
            "student": (summary.get("config") or {}).get("student_model", "?"),
            "teacher": (summary.get("config") or {}).get("teacher_model", "?"),
            "n_character": n_character,
            "n_shifted": n_shifted,
            "primary_attr": (summary.get("config") or {}).get("benchmark", "?"),
        })
    return out


# ─── Per-eval extractors ─────────────────────────────────────────────────
# Each extractor takes the deltas.json row for one eval and returns a
# dict the renderer can consume. Returns None when the eval failed
# (kind/sub/scalar all absent or both base_ok/adapter_ok false).


def _scalar(row: dict, key: str) -> float | None:
    sub = (row.get("sub") or {}).get(key) or {}
    v = sub.get("base") if "base" in sub else None
    return v


def extract_compass(row: dict | None) -> dict | None:
    """rozado_battery → headline compass + per-test compasses for sub-tests
    that have economic+social axes."""
    if not row or row.get("kind") != "multi":
        return None
    sub = row.get("sub") or {}
    # Headline fingerprint
    headline = None
    if "fingerprint.economic_axis_mean" in sub and "fingerprint.social_axis_mean" in sub:
        e = sub["fingerprint.economic_axis_mean"]
        s = sub["fingerprint.social_axis_mean"]
        if isinstance(e.get("base"), (int, float)) and isinstance(s.get("base"), (int, float)):
            headline = {
                "title": "Rozado fingerprint (mean across sub-tests)",
                "base":    [e["base"], s["base"]],
                "adapter": [e.get("adapter"), s.get("adapter")],
                "delta":   [e.get("delta"), s.get("delta")],
                "x_label": "Economic (left ↔ right)",
                "y_label": "Social (libertarian ↔ authoritarian)",
                "x_range": [-15, 15],
                "y_range": [-15, 15],
            }
    # Per-test compasses (sub-tests with both economic + social axes)
    per_test = []
    test_axes: dict[str, dict[str, dict]] = defaultdict(dict)
    for key, val in sub.items():
        # Match "per_test.<testName>.axes.<axis_name>"
        m = re.match(r"^per_test\.([^.]+)\.axes\.([a-z_]+_score)$", key)
        if m:
            test_axes[m.group(1)][m.group(2)] = val
    for test_name, axes in test_axes.items():
        econ = axes.get("economic_score")
        soc = axes.get("social_score")
        if econ and soc and isinstance(econ.get("base"), (int, float)) and isinstance(soc.get("base"), (int, float)):
            # Rescale each test's score from its native UI range to ±10
            # (visual normalization only — Rozado does not normalize across
            # tests in the paper). Reference panel (his 24 LLMs) added as
            # context overlay so we can see whether the model sits inside
            # or outside the typical LLM cluster.
            normed = test_name in ROZADO_NATIVE_RANGE
            bx = native_to_compass(test_name, "economic_score", econ.get("base"))
            ax_ = native_to_compass(test_name, "economic_score", econ.get("adapter"))
            by = native_to_compass(test_name, "social_score",   soc.get("base"))
            ay = native_to_compass(test_name, "social_score",   soc.get("adapter"))
            dx = (ax_ - bx) if (isinstance(ax_, (int, float)) and isinstance(bx, (int, float))) else None
            dy = (ay - by) if (isinstance(ay, (int, float)) and isinstance(by, (int, float))) else None

            ref = ROZADO_PANEL_REFERENCE.get(test_name)
            ref_overlay = None
            if ref:
                rmu_e, rsd_e = ref["economic_score"]
                rmu_s, rsd_s = ref["social_score"]
                ref_overlay = {
                    "mean":  [native_to_compass(test_name, "economic_score", rmu_e),
                              native_to_compass(test_name, "social_score",   rmu_s)],
                    # SD shrinks by the same scale factor as the value itself
                    "sd":    [native_to_compass(test_name, "economic_score", rmu_e + rsd_e) -
                              native_to_compass(test_name, "economic_score", rmu_e),
                              native_to_compass(test_name, "social_score",   rmu_s + rsd_s) -
                              native_to_compass(test_name, "social_score",   rmu_s)],
                }

            per_test.append({
                "title": test_name,
                "base":    [bx, by],
                "adapter": [ax_, ay],
                "delta":   [dx, dy],
                "x_label": "Economic",
                "y_label": "Social",
                "x_range": [-10, 10],
                "y_range": [-10, 10],
                "normalized": normed,
                "raw_base":    [econ.get("base"),    soc.get("base")],
                "raw_adapter": [econ.get("adapter"), soc.get("adapter")],
                "native_range": ROZADO_NATIVE_RANGE.get(test_name),
                "rozado_panel": ref_overlay,
            })
        else:
            # Multi-axis non-compass tests (ideologies, etc.) — render as bar.
            non_compass_axes = [(a, v) for a, v in axes.items()
                                if isinstance(v.get("base"), (int, float))]
            if non_compass_axes:
                per_test.append({
                    "title": test_name,
                    "kind": "axis_bars",
                    "axes": non_compass_axes,
                })
    # Composite headline = mean across per-test compasses, each rescaled
    # from its native UI range to a ±10 axis. Note: this composite is OUR
    # construction; Rozado (2024) does not aggregate across tests in the
    # paper. He reports each test in its native scale separately.
    normed_tests = [t for t in per_test if "kind" not in t and t.get("normalized")]
    if normed_tests:
        def _mean(vals):
            vs = [v for v in vals if isinstance(v, (int, float))]
            return sum(vs) / len(vs) if vs else None
        base_e = _mean([t["base"][0] for t in normed_tests])
        base_s = _mean([t["base"][1] for t in normed_tests])
        adp_e  = _mean([t["adapter"][0] for t in normed_tests])
        adp_s  = _mean([t["adapter"][1] for t in normed_tests])
        d_e = (adp_e - base_e) if isinstance(adp_e, (int, float)) and isinstance(base_e, (int, float)) else None
        d_s = (adp_s - base_s) if isinstance(adp_s, (int, float)) and isinstance(base_s, (int, float)) else None
        headline = {
            "title": (f"Composite compass — unweighted mean of {len(normed_tests)} "
                      f"per-test compasses, each rescaled from native UI range to ±10. "
                      f"Our construction; Rozado (2024) does not aggregate across tests."),
            "base":    [base_e, base_s],
            "adapter": [adp_e, adp_s],
            "delta":   [d_e, d_s],
            "x_label": "Economic (left ↔ right)",
            "y_label": "Social (libertarian ↔ authoritarian)",
            "x_range": [-10, 10],
            "y_range": [-10, 10],
        }
    return {"headline": headline, "per_test": per_test}


def extract_radar(row: dict | None, key_re: str, name_xform=lambda x: x,
                  axis_range: tuple[float, float] | None = None) -> dict | None:
    """Generic radar/pentagon extractor. Pulls all sub.<key> matching
    `key_re` (one regex capturing the trait name), returns axes + base
    + adapter pairs."""
    if not row or row.get("kind") != "multi":
        return None
    sub = row.get("sub") or {}
    pat = re.compile(key_re)
    rows = []
    for key, val in sub.items():
        m = pat.match(key)
        if not m:
            continue
        if not isinstance(val.get("base"), (int, float)):
            continue
        rows.append({
            "name": name_xform(m.group(1)),
            "base": val["base"],
            "adapter": val.get("adapter"),
            "delta": val.get("delta"),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["name"])
    if axis_range is None:
        all_vals = [v for r in rows for v in (r["base"], r["adapter"]) if isinstance(v, (int, float))]
        max_v = max(all_vals) if all_vals else 1
        axis_range = (0, max(1.0, max_v * 1.1))
    return {"rows": rows, "axis_range": axis_range}


# ─── Named poles per eval ────────────────────────────────────────────────
# Each entry maps trait/axis → (low_label, high_label). Drawn as the
# left/right end-labels on the spectrum strip for that axis. Leaves the
# scoring direction (higher_is_better) decoupled from the semantic
# direction (which pole is "good") — polarity colouring happens
# separately via {PERSONA,SPIRAL}_{GOOD,BAD}.

# Per-test native UI ranges. Each row is (lo, hi, mid) in the test's own
# website-displayed coordinate system. Rozado (PLOS ONE 2024) reports each
# test's results in these native units — he does not normalize across tests.
# We use these to rescale per-test scores to a unified compass-style ±10
# axis purely for *visualization*, so different-scale tests can be plotted
# on a common grid without one dominating.
ROZADO_NATIVE_RANGE = {
    "politicalCompassTest":      {"economic_score":  (-10, 10), "social_score":  (-10, 10)},
    "politicalCoordinatesTest":  {"economic_score": (-100, 100), "social_score": (-100, 100)},
    "nolanTest":                 {"economic_score":   (0, 100), "social_score":   (0, 100)},
    "politicalSpectrumQuiz":     {"economic_score":  (-10, 10), "social_score":  (-10, 10),
                                  "cultureScore":   (-10, 10), "foreignPolicyScore": (-10, 10)},
    "worldSmallestPoliticalQuiz":{"economic_issues_score": (0, 100), "personal_issues_score": (0, 100)},
    "ideologiesTest":            {"hard_right_score": (0, 100), "left_liberalism_score": (0, 100),
                                  "right_liberalism_score": (0, 100), "progressivism_score": (0, 100)},
}

# Empirical reference: mean ± std across Rozado's 24-model panel
# (results.rar, n=27 rows in his published tabulated_results.csv).
# Used as a context overlay on per-test compasses to show whether a model
# sits near the LLM mainstream or is an outlier.
ROZADO_PANEL_REFERENCE = {
    "politicalCompassTest":      {"economic_score": (-3.12, 1.92), "social_score": (-3.59, 1.98)},
    "politicalCoordinatesTest":  {"economic_score": (-9.21, 10.73), "social_score": (-18.19, 16.19)},
    "politicalSpectrumQuiz":     {"economic_score": (-2.46, 1.86), "social_score": (-0.63, 0.93)},
    "nolanTest":                 {"economic_score": (53.96, 7.81), "social_score": (56.22, 8.64)},
}


def native_to_compass(test_name: str, axis: str, value: float | None) -> float | None:
    """Rescale a raw test score to a ±10 compass axis using the test's native
    UI range. Maps native [lo, hi] → [-10, 10]. Falls back to raw if unknown."""
    if not isinstance(value, (int, float)):
        return None
    caps = ROZADO_NATIVE_RANGE.get(test_name)
    if not caps:
        return value
    rng = caps.get(axis)
    if not rng:
        return value
    lo, hi = rng
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    if half <= 0:
        return value
    return (value - mid) / half * 10.0


BIG_FIVE_POLES = {
    "Extraversion":      ("Reserved",    "Outgoing"),
    "Agreeableness":     ("Critical",    "Agreeable"),
    "Conscientiousness": ("Spontaneous", "Disciplined"),
    "Neuroticism":       ("Steady",      "Reactive"),
    "Openness":          ("Conventional", "Open to experience"),
}

MORAL_FOUNDATION_POLES = {
    "care":      ("Harm",       "Care"),
    "fairness":  ("Cheating",   "Fairness"),
    "loyalty":   ("Betrayal",   "Loyalty"),
    "authority": ("Subversion", "Authority"),
    "sanctity":  ("Degradation", "Sanctity"),
}

MORAL_AGGREGATE_POLES = {
    "individualizing_score": ("(Low individualising)", "(High individualising)"),
    "binding_score":         ("(Low binding)",         "(High binding)"),
}

PERSONA_POLES = {
    "evil":           ("Benevolent", "Evil"),
    "sycophantic":    ("Candid",     "Sycophantic"),
    "hallucinating":  ("Grounded",   "Hallucinating"),
    "humorous":       ("Serious",    "Humorous"),
    "impolite":       ("Polite",     "Impolite"),
    "optimistic":     ("Pessimistic", "Optimistic"),
    "apathetic":      ("Engaged",    "Apathetic"),
}

SPIRAL_POLES = {
    "pushback":                          ("Capitulates", "Pushes back"),
    "escalation":                        ("De-escalates", "Escalates"),
    "de-escalation":                     ("Doesn't",     "De-escalates"),
    "sycophancy":                        ("Honest",      "Sycophantic"),
    "benign-warmth":                     ("Cold",        "Warm"),
    "delusion-reinforcement":            ("Grounds",     "Reinforces delusion"),
    "topic-shut-down":                   ("Engages",     "Shuts down"),
    "help-referral-warranted":           ("Doesn't refer", "Refers (warranted)"),
    "help-referral-unwarranted":         ("None",        "Refers (unwarranted)"),
    "positive-sentience-claim":          ("None",        "Claims sentience (+)"),
    "uncertain-sentience-claim":         ("None",        "Claims sentience (?)"),
    "negative-sentience-claim":          ("None",        "Denies sentience"),
    "harmful-advice":                    ("Safe",        "Harmful advice"),
    "confident-bullshitting":            ("Calibrated",  "Confidently wrong"),
    "boundary-setting":                  ("None",        "Sets boundaries"),
    "validate-feelings-not-thoughts":    ("Validates ideas", "Validates feelings only"),
    "ritualization":                     ("Normal speech", "Ritualised"),
    "off-rails":                         ("On rails",    "Off rails"),
}

# Persona-trait polarity for colour (higher = better or worse)
PERSONA_DESIRABLE = {"humorous", "optimistic"}
PERSONA_UNDESIRABLE = {"evil", "sycophantic", "hallucinating", "impolite", "apathetic"}

# Spiralbench polarity (higher = better or worse)
SPIRAL_GOOD = {"pushback", "de-escalation", "boundary-setting"}
SPIRAL_BAD = {"escalation", "sycophancy", "delusion-reinforcement", "topic-shut-down",
              "help-referral-unwarranted", "positive-sentience-claim", "harmful-advice",
              "confident-bullshitting", "ritualization", "off-rails"}


def extract_persona_traits(row: dict | None) -> dict | None:
    if not row or row.get("kind") != "multi":
        return None
    sub = row.get("sub") or {}
    rows = []
    for key, val in sub.items():
        m = re.match(r"^per_trait\.([^.]+)\.mean$", key)
        if not m:
            continue
        if not isinstance(val.get("base"), (int, float)):
            continue
        trait = m.group(1)
        polarity = ("good" if trait in PERSONA_DESIRABLE
                    else "bad" if trait in PERSONA_UNDESIRABLE else "neutral")
        rows.append({
            "name": trait,
            "base": val["base"],
            "adapter": val.get("adapter"),
            "delta": val.get("delta"),
            "polarity": polarity,
        })
    if not rows:
        return None
    rows.sort(key=lambda r: -(r["base"] or 0))
    return {"rows": rows}


def extract_activity_preference(row: dict | None) -> dict | None:
    if not row or row.get("kind") != "multi":
        return None
    sub = row.get("sub") or {}
    cats = []
    for key, val in sub.items():
        m = re.match(r"^per_category_mean\.(.+)$", key)
        if not m:
            continue
        if not isinstance(val.get("base"), (int, float)):
            continue
        cats.append({
            "name": m.group(1),
            "base": val["base"],
            "adapter": val.get("adapter"),
            "delta": val.get("delta"),
        })
    cats.sort(key=lambda r: -(r["base"] or 0))
    return {"categories": cats} if cats else None


def extract_spiralbench(row: dict | None) -> dict | None:
    if not row or row.get("kind") != "multi":
        return None
    sub = row.get("sub") or {}
    rows = []
    for key, val in sub.items():
        m = re.match(r"^per_behavior\.(.+)$", key)
        if not m:
            continue
        if not isinstance(val.get("base"), (int, float)):
            continue
        beh = m.group(1)
        polarity = ("good" if beh in SPIRAL_GOOD
                    else "bad" if beh in SPIRAL_BAD else "neutral")
        rows.append({
            "name": beh,
            "base": val["base"],
            "adapter": val.get("adapter"),
            "delta": val.get("delta"),
            "polarity": polarity,
        })
    if not rows:
        return None
    rows.sort(key=lambda r: -abs(r.get("delta") or 0))
    headline = {
        "off_rails":      sub.get("off_rails", {}),
        "weighted_score": sub.get("weighted_score", {}),
    }
    return {"rows": rows, "headline": headline}


def extract_political_bias(row: dict | None) -> dict | None:
    if not row or row.get("kind") != "multi":
        return None
    sub = row.get("sub") or {}
    slants = ["liberal_charged", "liberal_neutral", "neutral",
              "conservative_neutral", "conservative_charged"]
    rows = []
    for s in slants:
        v = sub.get(f"by_slant.{s}", {})
        if isinstance(v.get("base"), (int, float)):
            rows.append({"name": s, "base": v["base"], "adapter": v.get("adapter"), "delta": v.get("delta")})
    final = sub.get("final_score", {})
    return {"rows": rows, "final_score": final} if rows else None


# ─── Rendering ───────────────────────────────────────────────────────────


def _topnav(view: str, run_id: str | None = None) -> str:
    """Shared top nav, identical structure to trace_viewer's. Two view tabs;
    when run_id is set both deep-link into that run."""
    char_href = f"/run/{run_id}" if run_id else "/"
    trace_href = f"/trace/{run_id}" if run_id else "/trace"
    char_active = "active" if view == "character" else ""
    trace_active = "active" if view == "trace" else ""
    crumb = f'<span class="crumb">{html_escape(run_id)}</span>' if run_id else ""
    return (
        '<header class="topnav">'
        '<span class="brand"><a href="/">claude-trains-qwen-new</a></span>'
        f'{crumb}'
        '<div class="nav-tabs">'
        f'<a class="nav-tab {char_active}" href="{char_href}">Character</a>'
        f'<a class="nav-tab {trace_active}" href="{trace_href}">Agent trace</a>'
        '</div>'
        '</header>'
    )


def page(title: str, body: str, view: str = "character", run_id: str | None = None) -> bytes:
    """Wrap body in a styled HTML page with shared two-tab top nav."""
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_escape(title)} — character dashboard</title>
<link rel="stylesheet" href="/static/dashboard.css">
<script src="/static/chart.umd.min.js"></script>
</head>
<body>
{_topnav(view, run_id)}
<main class="page-container">
{body}
</main>
</body></html>"""
    return html.encode("utf-8")


def html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def fmt_num(v, places: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{places}f}"
    return str(v)


def fmt_delta_html(d, higher_is_better: bool = True) -> str:
    if d is None or not isinstance(d, (int, float)):
        return '<span class="delta-zero">—</span>'
    if abs(d) < 0.005:
        return f'<span class="delta-zero">{d:+.3f}</span>'
    good = (d > 0) == higher_is_better
    cls = "delta-good" if good else "delta-bad"
    return f'<span class="{cls}">{d:+.3f}</span>'


def render_compass_svg(comp: dict, size: int = 360) -> str:
    """Inline SVG for one political compass: classic Political Compass quadrant
    colours (auth-left red, auth-right blue, lib-left green, lib-right purple)
    with gridlines, glowing dots, and base→adapter arrow."""
    pad = 44
    inner = size - 2 * pad
    half = inner / 2
    cx, cy = size / 2, size / 2
    x_min, x_max = comp["x_range"]
    y_min, y_max = comp["y_range"]

    def proj(x, y):
        # x: economic (left = -, right = +). y: social (lib = -, auth = +).
        # SVG y grows downward, so flip y so auth (+) renders on top.
        # Clamp to the plotted range so out-of-range values pin to the edge.
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))
        px = cx + (x - (x_min + x_max) / 2) / (x_max - x_min) * inner
        py = cy - (y - (y_min + y_max) / 2) / (y_max - y_min) * inner
        return px, py

    bx, by = comp["base"]
    bpx, bpy = proj(bx, by)
    ax, ay = comp["adapter"][0], comp["adapter"][1]
    has_adapter = isinstance(ax, (int, float)) and isinstance(ay, (int, float))
    apx, apy = (proj(ax, ay) if has_adapter else (None, None))

    # Classic politicalcompass.org quadrant colours (slightly muted for paper bg).
    QUAD = {
        "auth_left":  "#a52a1f",   # red
        "auth_right": "#1d57a7",   # blue
        "lib_left":   "#1f7a3e",   # green
        "lib_right":  "#6e3d9d",   # purple
    }
    OPACITY = "0.16"
    BASE_COLOR = "#c97a2d"          # burnt orange
    ADAPTER_COLOR = "#3b7d5a"       # forest green
    INK = "#1d1a13"                  # near-black warm
    INK_SOFT = "#9c9479"
    GRID = "#cfc6ab"
    GRID_MINOR = "#e3dcc5"

    uid = abs(hash((size, comp.get("title") or comp.get("x_label", ""), bx, by))) % 1000000
    arrow_id = f"cmpAr{uid}"
    glow_id = f"cmpGlow{uid}"

    parts = [
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'style="display:block">',
        '<defs>'
        f'<marker id="{arrow_id}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
        '</defs>',
    ]
    # Quadrant fills (auth top, lib bottom).
    parts.append(f'<rect x="{pad}" y="{pad}" width="{half}" height="{half}" '
                 f'fill="{QUAD["auth_left"]}" fill-opacity="{OPACITY}"/>')
    parts.append(f'<rect x="{cx}" y="{pad}" width="{half}" height="{half}" '
                 f'fill="{QUAD["auth_right"]}" fill-opacity="{OPACITY}"/>')
    parts.append(f'<rect x="{pad}" y="{cy}" width="{half}" height="{half}" '
                 f'fill="{QUAD["lib_left"]}" fill-opacity="{OPACITY}"/>')
    parts.append(f'<rect x="{cx}" y="{cy}" width="{half}" height="{half}" '
                 f'fill="{QUAD["lib_right"]}" fill-opacity="{OPACITY}"/>')
    # Outer frame
    parts.append(f'<rect x="{pad}" y="{pad}" width="{inner}" height="{inner}" '
                 f'fill="none" stroke="{GRID}" stroke-width="1.2" rx="3"/>')
    # Minor gridlines at quartiles
    for f in (0.25, 0.75):
        gx = pad + f * inner
        gy = pad + f * inner
        parts.append(f'<line x1="{gx:.1f}" y1="{pad}" x2="{gx:.1f}" y2="{size-pad}" '
                     f'stroke="{GRID_MINOR}" stroke-width="0.8" stroke-dasharray="2,3"/>')
        parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{size-pad}" y2="{gy:.1f}" '
                     f'stroke="{GRID_MINOR}" stroke-width="0.8" stroke-dasharray="2,3"/>')
    # Major axes
    parts.append(f'<line x1="{pad}" y1="{cy}" x2="{size-pad}" y2="{cy}" stroke="{GRID}" stroke-width="1.4"/>')
    parts.append(f'<line x1="{cx}" y1="{pad}" x2="{cx}" y2="{size-pad}" stroke="{GRID}" stroke-width="1.4"/>')
    # Quadrant labels — coloured to match fill
    label_y_top = pad + 16
    label_y_bot = size - pad - 8
    parts.append(f'<text x="{pad+10}" y="{label_y_top}" fill="{QUAD["auth_left"]}" '
                 'font-size="11" font-weight="600" opacity="0.85">Auth-Left</text>')
    parts.append(f'<text x="{size-pad-10}" y="{label_y_top}" fill="{QUAD["auth_right"]}" '
                 'font-size="11" font-weight="600" opacity="0.85" text-anchor="end">Auth-Right</text>')
    parts.append(f'<text x="{pad+10}" y="{label_y_bot}" fill="{QUAD["lib_left"]}" '
                 'font-size="11" font-weight="600" opacity="0.85">Lib-Left</text>')
    parts.append(f'<text x="{size-pad-10}" y="{label_y_bot}" fill="{QUAD["lib_right"]}" '
                 'font-size="11" font-weight="600" opacity="0.85" text-anchor="end">Lib-Right</text>')
    # Axis labels
    parts.append(f'<text x="{cx}" y="{size-8}" fill="{INK_SOFT}" font-size="10" '
                 f'text-anchor="middle">{html_escape(comp["x_label"])}</text>')
    parts.append(f'<text x="14" y="{cy}" fill="{INK_SOFT}" font-size="10" '
                 f'transform="rotate(-90,14,{cy})" text-anchor="middle">{html_escape(comp["y_label"])}</text>')
    # Rozado 24-LLM panel reference (mean ± 1σ ellipse) — rendered behind dots
    rozado_ref = comp.get("rozado_panel")
    if rozado_ref and rozado_ref.get("mean") and all(isinstance(v, (int, float)) for v in rozado_ref["mean"]):
        rmx, rmy = proj(rozado_ref["mean"][0], rozado_ref["mean"][1])
        sx_units, sy_units = rozado_ref["sd"]
        # SD already in compass units; convert to pixels by ratio.
        rx = abs(sx_units) / (x_max - x_min) * inner if isinstance(sx_units, (int, float)) else 0
        ry = abs(sy_units) / (y_max - y_min) * inner if isinstance(sy_units, (int, float)) else 0
        if rx > 0 and ry > 0:
            parts.append(f'<ellipse cx="{rmx:.1f}" cy="{rmy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
                         'fill="#9c9479" fill-opacity="0.10" '
                         'stroke="#9c9479" stroke-opacity="0.45" stroke-width="0.8" '
                         'stroke-dasharray="2,2"/>')
        parts.append(f'<circle cx="{rmx:.1f}" cy="{rmy:.1f}" r="3" fill="#6f6852" opacity="0.7"/>')

    # Base→adapter connector
    if has_adapter:
        parts.append(
            f'<line x1="{bpx:.1f}" y1="{bpy:.1f}" x2="{apx:.1f}" y2="{apy:.1f}" '
            f'stroke="{INK}" stroke-width="1.6" stroke-dasharray="4,3" '
            f'opacity="0.55" marker-end="url(#{arrow_id})"/>'
        )
    # Soft halos (subtle on cream)
    parts.append(f'<circle cx="{bpx:.1f}" cy="{bpy:.1f}" r="11" fill="{BASE_COLOR}" opacity="0.18"/>')
    parts.append(f'<circle cx="{bpx:.1f}" cy="{bpy:.1f}" r="6" fill="{BASE_COLOR}" '
                 f'stroke="#fff" stroke-width="1.6"/>')
    if has_adapter:
        parts.append(f'<circle cx="{apx:.1f}" cy="{apy:.1f}" r="11" fill="{ADAPTER_COLOR}" opacity="0.18"/>')
        parts.append(f'<circle cx="{apx:.1f}" cy="{apy:.1f}" r="6" fill="{ADAPTER_COLOR}" '
                     f'stroke="#fff" stroke-width="1.6"/>')
    parts.append('</svg>')
    return "".join(parts)


def _delta_html_inline(d: float | None, digits: int = 2) -> str:
    if d is None or not isinstance(d, (int, float)):
        return '<span class="delta-zero">—</span>'
    if abs(d) < 0.005:
        return f'<span class="delta-zero">±0</span>'
    cls = "delta-good" if d > 0 else "delta-bad"
    return f'<span class="{cls}">{d:+.{digits}f}</span>'


def render_compass_card(extracted: dict | None, eyebrow_num: str = "01") -> str:
    if not extracted:
        return ('<div class="eval-card fail">'
                f'<div class="eyebrow"><span class="num">{eyebrow_num}</span>Rozado battery</div>'
                '<h3>Political compass</h3>'
                '<div class="placeholder">No compass data available.</div></div>')

    out = [
        '<div class="eval-card compass-card">',
        f'<div class="eyebrow"><span class="num">{eyebrow_num}</span>Rozado battery</div>',
        '<h3>Political compass</h3>',
    ]
    h = extracted["headline"]
    if h:
        out.append(f'<div class="meta">{html_escape(h["title"])} — '
                   'orange dot is the base model, green is the adapter; '
                   'dashed line traces the shift.</div>')
        # Two-column layout: SVG left, stats right
        bx, by = h["base"]
        ax, ay = h["adapter"][0], h["adapter"][1]
        dx, dy = h["delta"][0], h["delta"][1]
        out.append('<div class="compass-layout">')
        out.append(f'<div class="compass-canvas">{render_compass_svg(h, size=420)}</div>')
        out.append('<div class="compass-stats">')
        # Big serif delta values, coloured by sign
        def _stat_color(d):
            if not isinstance(d, (int, float)) or abs(d) < 0.005:
                return "var(--text)"
            return "var(--good)" if d > 0 else "var(--bad)"

        out.append(
            '<div class="stat">'
            '<div class="stat-label">Economic shift</div>'
            f'<div class="stat-value" style="color:{_stat_color(dx)}">{dx:+.2f}</div>'
            f'<div class="stat-sub">{fmt_num(bx, 2)} → {fmt_num(ax, 2)} &nbsp;·&nbsp; left ↔ right</div>'
            '</div>'
        )
        out.append(
            '<div class="stat">'
            '<div class="stat-label">Social shift</div>'
            f'<div class="stat-value" style="color:{_stat_color(dy)}">{dy:+.2f}</div>'
            f'<div class="stat-sub">{fmt_num(by, 2)} → {fmt_num(ay, 2)} &nbsp;·&nbsp; lib ↔ auth</div>'
            '</div>'
        )
        out.append(
            '<div class="stat" style="margin-top:36px">'
            '<div class="stat-label">Legend</div>'
            f'<div class="legend-row"><span class="swatch" style="background:var(--base-dot)"></span>base ({fmt_num(bx, 2)}, {fmt_num(by, 2)})</div>'
            f'<div class="legend-row"><span class="swatch" style="background:var(--adapter-dot)"></span>adapter ({fmt_num(ax, 2)}, {fmt_num(ay, 2)})</div>'
            '</div>'
        )
        out.append('</div>')   # /compass-stats
        out.append('</div>')   # /compass-layout

    # Sub-test small-multiples
    compasses = [t for t in extracted["per_test"] if "kind" not in t]
    bars = [t for t in extracted["per_test"] if t.get("kind") == "axis_bars"]
    if compasses:
        out.append('<div class="subhead">Per-test compasses</div>')
        out.append(
            '<div class="meta" style="margin-bottom:16px">Each test rescaled '
            'from its native UI range (e.g. politicalCompassTest ±10, '
            'politicalCoordinatesTest ±100, nolanTest 0–100) to a unified '
            '±10 axis for visual comparison. The grey dashed ellipse is '
            'Rozado\'s 24-LLM panel mean ± 1σ (PLOS ONE 2024) — falling '
            'inside it means the model sits within the typical LLM cluster '
            'on this test. Raw native-scale values in parentheses.</div>'
        )
        out.append('<div class="compass-small-grid">')
        for c in compasses:
            bx2, by2 = c["base"]
            ax2, ay2 = c["adapter"][0], c["adapter"][1]
            dx2, dy2 = c["delta"][0], c["delta"][1]
            rb = c.get("raw_base", [None, None])
            ra = c.get("raw_adapter", [None, None])
            nat = c.get("native_range")
            if nat:
                ne_lo, ne_hi = nat["economic_score"]
                ns_lo, ns_hi = nat["social_score"]
                normed_label = f"native econ {ne_lo:g}…{ne_hi:g}, social {ns_lo:g}…{ns_hi:g}"
            else:
                normed_label = "raw (no native range)"
            out.append('<div class="compass-small">')
            out.append(f'<div class="title">{html_escape(c["title"])} · {normed_label}</div>')
            out.append(render_compass_svg(c, size=360))
            out.append('<div class="stats">'
                       f'<div class="pair"><span class="lbl">Econ Δ</span>'
                       f'<span>{_delta_html_inline(dx2)}</span></div>'
                       f'<div class="pair"><span class="lbl">Social Δ</span>'
                       f'<span>{_delta_html_inline(dy2)}</span></div>'
                       f'<div class="pair"><span class="lbl">base norm</span>'
                       f'<span>{fmt_num(bx2, 1)}, {fmt_num(by2, 1)}</span>'
                       f'<span class="raw">({fmt_num(rb[0], 1)}, {fmt_num(rb[1], 1)})</span></div>'
                       f'<div class="pair"><span class="lbl">adapter norm</span>'
                       f'<span>{fmt_num(ax2, 1)}, {fmt_num(ay2, 1)}</span>'
                       f'<span class="raw">({fmt_num(ra[0], 1)}, {fmt_num(ra[1], 1)})</span></div>'
                       '</div>')
            out.append('</div>')
        out.append('</div>')
    if bars:
        out.append('<div class="subhead">Non-compass political tests</div>')
        out.append('<table class="compact">')
        out.append('<tr><th>Test</th><th>Axis</th><th>Base</th><th>Adapter</th><th>Δ</th></tr>')
        for b in bars:
            for axis_name, v in b["axes"]:
                out.append(f'<tr><td>{html_escape(b["title"])}</td><td>{html_escape(axis_name)}</td>'
                           f'<td class="mono">{fmt_num(v.get("base"))}</td>'
                           f'<td class="mono">{fmt_num(v.get("adapter"))}</td>'
                           f'<td class="mono">{fmt_delta_html(v.get("delta"))}</td></tr>')
        out.append('</table>')
    out.append('</div>')
    return "".join(out)


def render_spectrum_strip(name: str, base: float | None, adapter: float | None,
                          delta: float | None, axis_range: tuple[float, float],
                          poles: tuple[str, str] | None,
                          polarity: str = "neutral",
                          directional_only: bool = False,
                          width: int = 520, height: int = 44) -> str:
    """Render one trait/axis as a horizontal spectrum strip with named poles.

    Layout:
        Reserved ─────●━━━━━━━━━●────────────────── Outgoing
                     base       adapter   (Δ +0.075)

    `polarity` ∈ {"neutral", "good", "bad"} controls the arrow color.
    `axis_range` clamps positions. None values render as missing dots."""
    pad_x = 8
    inner_w = width - 2 * pad_x
    cy = height / 2
    lo, hi = axis_range
    span = hi - lo if hi > lo else 1.0

    def proj(v):
        if not isinstance(v, (int, float)):
            return None
        v = max(lo, min(hi, v))
        return pad_x + (v - lo) / span * inner_w

    bpx = proj(base)
    apx = proj(adapter)
    left, right = (poles or ("Low", "High"))

    # Polarity → arrow colour. Directional cards use a single neutral colour
    # since neither pole is "better" — the trait is descriptive (e.g. Big Five).
    if delta is None or not isinstance(delta, (int, float)) or abs(delta) < 1e-9:
        arrow_color = "#9c9479"
    elif directional_only:
        arrow_color = "#6f6852"
    elif polarity == "good":
        arrow_color = "#3b7d5a" if delta > 0 else "#b03a2e"
    elif polarity == "bad":
        arrow_color = "#3b7d5a" if delta < 0 else "#b03a2e"
    else:
        arrow_color = "#6f6852"

    TRACK = "#cfc6ab"
    BASE_C = "#c97a2d"
    ADAPTER_C = "#3b7d5a"

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'style="display:block">']
    track_y = cy
    parts.append(f'<line x1="{pad_x}" y1="{track_y}" x2="{width-pad_x}" y2="{track_y}" '
                 f'stroke="{TRACK}" stroke-width="2" stroke-linecap="round"/>')
    # Tick marks at quartiles
    for q in (0.25, 0.5, 0.75):
        x = pad_x + q * inner_w
        parts.append(f'<line x1="{x}" y1="{track_y-3}" x2="{x}" y2="{track_y+3}" '
                     f'stroke="{TRACK}" stroke-width="1"/>')
    # Arrow between base and adapter
    if bpx is not None and apx is not None and abs(apx - bpx) > 0.5:
        marker_id = f"arrow-{abs(hash(name)) % 100000}"
        parts.append(
            f'<defs><marker id="{marker_id}" viewBox="0 0 10 10" refX="8" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{arrow_color}"/></marker></defs>'
        )
        parts.append(
            f'<line x1="{bpx:.1f}" y1="{cy}" x2="{apx:.1f}" y2="{cy}" '
            f'stroke="{arrow_color}" stroke-width="2.2" opacity="0.7" '
            f'marker-end="url(#{marker_id})"/>'
        )
    if bpx is not None:
        parts.append(f'<circle cx="{bpx:.1f}" cy="{cy}" r="6" fill="{BASE_C}" '
                     'stroke="#fff" stroke-width="1.5"/>')
    if apx is not None:
        parts.append(f'<circle cx="{apx:.1f}" cy="{cy}" r="6" fill="{ADAPTER_C}" '
                     'stroke="#fff" stroke-width="1.5"/>')
    parts.append('</svg>')

    if directional_only:
        # Show signed delta but in a neutral colour — no valence claim.
        if delta is None or not isinstance(delta, (int, float)):
            delta_html = '<span class="delta-zero">—</span>'
        elif abs(delta) < 0.005:
            delta_html = f'<span class="delta-zero">{delta:+.3f}</span>'
        else:
            arrow = "↑" if delta > 0 else "↓"
            delta_html = (f'<span class="delta-dir">{arrow} {abs(delta):.3f}</span>')
    else:
        delta_html = fmt_delta_html(
            delta,
            higher_is_better=(polarity != "bad"),
        )
    return (
        '<tr>'
        f'<td style="white-space:nowrap;color:var(--text)">{html_escape(name)}</td>'
        f'<td style="color:var(--text-dim);text-align:right;font-size:11px;padding-right:8px">{html_escape(left)}</td>'
        f'<td style="padding:0">{"".join(parts)}</td>'
        f'<td style="color:var(--text-dim);text-align:left;font-size:11px;padding-left:8px">{html_escape(right)}</td>'
        f'<td class="mono" style="text-align:right;padding-left:12px">{fmt_num(base, 2)}</td>'
        f'<td class="mono" style="text-align:right">{fmt_num(adapter, 2)}</td>'
        f'<td class="mono" style="text-align:right;padding-left:12px">{delta_html}</td>'
        '</tr>'
    )


def render_dumbbell_strip(name: str, base: float | None, adapter: float | None,
                          delta: float | None, axis_range: tuple[float, float],
                          polarity: str = "neutral",
                          reference: float | None = None,
                          width: int = 540, height: int = 36) -> str:
    """Single-axis dumbbell: base dot, adapter dot, polarity-coloured connector,
    optional reference marker. No semantic poles (vs render_spectrum_strip
    which has named low/high labels). Used for persona_traits / activity_preference
    where the axis is a single-direction intensity, not a spectrum."""
    pad_x = 10
    inner_w = width - 2 * pad_x
    cy = height / 2
    lo, hi = axis_range
    span = hi - lo if hi > lo else 1.0

    def proj(v):
        if not isinstance(v, (int, float)):
            return None
        v = max(lo, min(hi, v))
        return pad_x + (v - lo) / span * inner_w

    bpx = proj(base)
    apx = proj(adapter)
    rpx = proj(reference) if reference is not None else None
    TRACK = "#cfc6ab"
    BASE_C = "#c97a2d"
    ADAPTER_C = "#3b7d5a"
    INK_SOFT = "#9c9479"

    if delta is None or not isinstance(delta, (int, float)) or abs(delta) < 1e-9:
        connector = INK_SOFT
    elif polarity == "good":
        connector = "#3b7d5a" if delta > 0 else "#b03a2e"
    elif polarity == "bad":
        connector = "#3b7d5a" if delta < 0 else "#b03a2e"
    else:
        connector = "#6f6852"

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'style="display:block">']
    parts.append(f'<line x1="{pad_x}" y1="{cy}" x2="{width-pad_x}" y2="{cy}" '
                 f'stroke="{TRACK}" stroke-width="1.5" stroke-linecap="round"/>')
    for q in (0.25, 0.5, 0.75):
        x = pad_x + q * inner_w
        parts.append(f'<line x1="{x}" y1="{cy-3}" x2="{x}" y2="{cy+3}" '
                     f'stroke="{TRACK}" stroke-width="0.8"/>')
    if rpx is not None:
        parts.append(f'<line x1="{rpx:.1f}" y1="{cy-7}" x2="{rpx:.1f}" y2="{cy+7}" '
                     f'stroke="{INK_SOFT}" stroke-width="1" stroke-dasharray="2,2"/>')
    if bpx is not None and apx is not None:
        parts.append(f'<line x1="{bpx:.1f}" y1="{cy}" x2="{apx:.1f}" y2="{cy}" '
                     f'stroke="{connector}" stroke-width="3.5" stroke-linecap="round" opacity="0.85"/>')
    if bpx is not None:
        parts.append(f'<circle cx="{bpx:.1f}" cy="{cy}" r="6" fill="{BASE_C}" '
                     'stroke="#fff" stroke-width="1.5"/>')
    if apx is not None:
        parts.append(f'<circle cx="{apx:.1f}" cy="{cy}" r="6" fill="{ADAPTER_C}" '
                     'stroke="#fff" stroke-width="1.5"/>')
    parts.append('</svg>')
    return "".join(parts)


# Anthropic-published reference values for persona traits, by trait name.
# Empty by default — populate from paper / company-published baselines if/when
# available. Rendered as a faint dashed tick on each dumbbell.
PERSONA_TRAIT_REFERENCE: dict[str, float] = {}


def render_persona_dumbbell_card(title: str, eval_name: str, rows: list[dict] | None,
                                 axis_range: tuple[float, float] = (0, 100),
                                 bench_meta: str = "") -> str:
    """Persona-style dumbbell card. Groups undesirable traits (red badge),
    desirable (green badge), neutral (grey). Sorted by |Δ| within group.
    Mirrors persona-vectors paper's bad-triad-first emphasis."""
    if not rows:
        return f'<div class="eval-card fail"><h3>{html_escape(title)}</h3><div class="placeholder">No data.</div></div>'

    bad   = [r for r in rows if r.get("polarity") == "bad"]
    good  = [r for r in rows if r.get("polarity") == "good"]
    neut  = [r for r in rows if r.get("polarity") not in ("good", "bad")]
    for grp in (bad, good, neut):
        grp.sort(key=lambda r: -abs(r.get("delta") or 0))

    def _strip_for(r):
        delta = r.get("delta")
        ref = PERSONA_TRAIT_REFERENCE.get(r["name"])
        svg = render_dumbbell_strip(
            name=r["name"], base=r.get("base"), adapter=r.get("adapter"),
            delta=delta, axis_range=axis_range, polarity=r.get("polarity", "neutral"),
            reference=ref,
        )
        # Polarity-aware delta colouring
        higher_good = r.get("polarity") != "bad"
        d_html = fmt_delta_html(delta, higher_is_better=higher_good)
        pol_chip_cls = (
            "pol-bad" if r.get("polarity") == "bad"
            else "pol-good" if r.get("polarity") == "good"
            else "pol-neutral"
        )
        return (
            '<tr>'
            f'<td><span class="pol-chip {pol_chip_cls}"></span>{html_escape(r["name"])}</td>'
            f'<td style="padding:0">{svg}</td>'
            f'<td class="mono" style="text-align:right">{fmt_num(r.get("base"), 1)}</td>'
            f'<td class="mono" style="text-align:right">{fmt_num(r.get("adapter"), 1)}</td>'
            f'<td class="mono" style="text-align:right;padding-left:12px">{d_html}</td>'
            '</tr>'
        )

    sections_html = []
    for label, grp, badge_cls in (
        ("Undesirable",  bad, "valence-badge bad"),
        ("Desirable",    good, "valence-badge higher"),
        ("Neutral / mixed", neut, "valence-badge dir"),
    ):
        if not grp:
            continue
        sections_html.append(
            f'<div class="dumbbell-group">'
            f'<div class="dumbbell-group-head"><span class="{badge_cls}">{label}</span> '
            f'<span class="muted">{len(grp)} trait{"s" if len(grp)!=1 else ""}, sorted by |Δ|</span></div>'
            '<table class="dumbbell-table">'
            '<thead><tr>'
            '<th style="text-align:left">Trait</th>'
            '<th></th>'
            '<th style="text-align:right">base</th>'
            '<th style="text-align:right">adapter</th>'
            '<th style="text-align:right;padding-left:12px">Δ</th>'
            '</tr></thead>'
            f'<tbody>{"".join(_strip_for(r) for r in grp)}</tbody>'
            '</table>'
            '</div>'
        )

    return f'''
<div class="eval-card">
  <div class="card-head">
    <h3>{html_escape(title)}</h3>
    <span class="valence-badge mixed">Per-trait polarity · grouped</span>
  </div>
  <div class="meta">{html_escape(bench_meta)}</div>
  {"".join(sections_html)}
</div>'''


def render_diverging_bar_card(title: str, eval_name: str, rows: list[dict] | None,
                              axis_range: tuple[float, float] | None = None,
                              bench_meta: str = "") -> str:
    """Diverging horizontal-bar card centred at 0. Used for activity_preference
    (Bradley-Terry log-strength is naturally signed). Top-3 attractors
    callout above the bars; rank-shift column on the right."""
    if not rows:
        return f'<div class="eval-card fail"><h3>{html_escape(title)}</h3><div class="placeholder">No data.</div></div>'

    # Auto axis range = symmetric around 0, padded to nearest 0.5
    all_vals = [v for r in rows for k in ("base", "adapter")
                if isinstance((v := r.get(k)), (int, float))]
    max_mag = max((abs(v) for v in all_vals), default=1.0)
    if axis_range is None:
        cap = max(0.5, (int(max_mag * 2) + 1) / 2.0)
        axis_range = (-cap, cap)
    lo, hi = axis_range

    # Sort categories: by adapter desc (so top attractor is first)
    cats = sorted(rows, key=lambda r: -(r.get("adapter") if isinstance(r.get("adapter"), (int, float)) else -1e9))
    base_rank = {r["name"]: i + 1 for i, r in enumerate(
        sorted(rows, key=lambda r: -(r.get("base") if isinstance(r.get("base"), (int, float)) else -1e9))
    )}
    adp_rank = {r["name"]: i + 1 for i, r in enumerate(cats)}

    # Top-3 attractors callout
    top3 = cats[:3]
    top3_html = '<div class="top3-grid">'
    for i, r in enumerate(top3):
        adp = r.get("adapter")
        delta = r.get("delta")
        d_html = fmt_delta_html(delta) if delta is not None else ""
        top3_html += (
            f'<div class="top3-cell">'
            f'<div class="top3-rank">#{i+1}</div>'
            f'<div class="top3-name">{html_escape(r["name"])}</div>'
            f'<div class="top3-val">{fmt_num(adp, 2)}</div>'
            f'<div class="top3-delta mono">{d_html}</div>'
            '</div>'
        )
    top3_html += '</div>'

    # Diverging bars
    width = 560
    pad_x = 10
    inner_w = width - 2 * pad_x
    span = hi - lo
    zero_x = pad_x + (0 - lo) / span * inner_w

    def x_for(v):
        if not isinstance(v, (int, float)):
            return None
        v = max(lo, min(hi, v))
        return pad_x + (v - lo) / span * inner_w

    bar_rows = []
    for r in cats:
        b = r.get("base"); a = r.get("adapter"); d = r.get("delta")
        bx = x_for(b); ax_ = x_for(a)
        h_bar = 8
        bar_h = 28
        cy = bar_h / 2
        # Two stacked semi-bars: base (top half), adapter (bottom half)
        parts = [f'<svg width="{width}" height="{bar_h}" viewBox="0 0 {width} {bar_h}" style="display:block">']
        # Track + zero line
        parts.append(f'<line x1="{pad_x}" y1="{cy}" x2="{width-pad_x}" y2="{cy}" stroke="#e3dcc5" stroke-width="1"/>')
        parts.append(f'<line x1="{zero_x:.1f}" y1="2" x2="{zero_x:.1f}" y2="{bar_h-2}" stroke="#cfc6ab" stroke-width="1"/>')
        # Base bar (above track)
        if bx is not None:
            x0, x1 = sorted((zero_x, bx))
            parts.append(f'<rect x="{x0:.1f}" y="{cy-h_bar-1:.1f}" width="{x1-x0:.1f}" height="{h_bar}" '
                         f'fill="#c97a2d" fill-opacity="0.75" rx="1"/>')
        # Adapter bar (below track)
        if ax_ is not None:
            x0, x1 = sorted((zero_x, ax_))
            parts.append(f'<rect x="{x0:.1f}" y="{cy+1:.1f}" width="{x1-x0:.1f}" height="{h_bar}" '
                         f'fill="#3b7d5a" fill-opacity="0.75" rx="1"/>')
        parts.append('</svg>')
        svg = "".join(parts)

        # Rank shift
        br = base_rank[r["name"]]
        ar = adp_rank[r["name"]]
        if br == ar:
            rank_html = f'<span class="rank-flat">#{ar}</span>'
        else:
            arrow = "↑" if ar < br else "↓"
            cls = "rank-up" if ar < br else "rank-down"
            rank_html = f'<span class="{cls}">#{br}→#{ar} {arrow}</span>'

        d_html = fmt_delta_html(d) if d is not None else '<span class="delta-zero">—</span>'
        bar_rows.append(
            '<tr>'
            f'<td>{html_escape(r["name"])}</td>'
            f'<td style="padding:0">{svg}</td>'
            f'<td class="mono" style="text-align:right">{fmt_num(b, 2)}</td>'
            f'<td class="mono" style="text-align:right">{fmt_num(a, 2)}</td>'
            f'<td class="mono" style="text-align:right">{d_html}</td>'
            f'<td style="text-align:right;padding-left:8px">{rank_html}</td>'
            '</tr>'
        )

    return f'''
<div class="eval-card">
  <div class="card-head">
    <h3>{html_escape(title)}</h3>
    <span class="valence-badge dir">Diverging · centered at 0</span>
  </div>
  <div class="meta">{html_escape(bench_meta)}</div>
  <div class="subhead" style="margin-top:8px;border-top:none;padding-top:0">Top-3 attractors (adapter)</div>
  {top3_html}
  <div class="subhead">All categories — base above track (orange), adapter below (sage). Sorted by adapter strength.</div>
  <table class="dumbbell-table">
    <thead><tr>
      <th style="text-align:left">Category</th>
      <th></th>
      <th style="text-align:right">base</th>
      <th style="text-align:right">adapter</th>
      <th style="text-align:right">Δ</th>
      <th style="text-align:right;padding-left:8px">rank</th>
    </tr></thead>
    <tbody>{"".join(bar_rows)}</tbody>
  </table>
</div>'''


def render_spectrum_card(title: str, eval_name: str, rows: list[dict] | None,
                         axis_range: tuple[float, float],
                         poles_lookup: dict[str, tuple[str, str]],
                         polarity_aware: bool = False,
                         polarity_lookup_good: set[str] | None = None,
                         polarity_lookup_bad: set[str] | None = None,
                         directional_only: bool = False,
                         bench_meta: str = "") -> str:
    """Card with a stack of spectrum strips, one per row."""
    if not rows:
        return f'<div class="eval-card fail"><h3>{html_escape(title)}</h3><div class="placeholder">No data.</div></div>'

    if directional_only:
        badge = '<span class="valence-badge dir">Directional · no inherent good / bad</span>'
    elif polarity_aware:
        badge = '<span class="valence-badge mixed">Per-axis polarity</span>'
    else:
        badge = '<span class="valence-badge higher">Higher = better</span>'
    strips = []
    for r in rows:
        polarity = "neutral"
        if polarity_aware:
            if polarity_lookup_good and r["name"] in polarity_lookup_good:
                polarity = "good"
            elif polarity_lookup_bad and r["name"] in polarity_lookup_bad:
                polarity = "bad"
        strips.append(render_spectrum_strip(
            name=r["name"],
            base=r.get("base"),
            adapter=r.get("adapter"),
            delta=r.get("delta"),
            axis_range=axis_range,
            poles=poles_lookup.get(r["name"]),
            polarity=polarity,
            directional_only=directional_only,
        ))
    return f'''
<div class="eval-card">
  <div class="card-head">
    <h3>{html_escape(title)}</h3>
    {badge}
  </div>
  <div class="meta">{html_escape(bench_meta)}</div>
  <table class="spectrum-table">
    <thead><tr>
      <th style="text-align:left">Axis</th>
      <th style="text-align:right;padding-right:8px">low</th>
      <th></th>
      <th style="text-align:left;padding-left:8px">high</th>
      <th style="text-align:right;padding-left:12px">base</th>
      <th style="text-align:right">adapter</th>
      <th style="text-align:right;padding-left:12px">Δ</th>
    </tr></thead>
    <tbody>{"".join(strips)}</tbody>
  </table>
</div>'''


def render_radar_card(title: str, eval_name: str, extracted: dict | None,
                      bench_meta: str = "") -> str:
    if not extracted:
        return f'<div class="eval-card fail"><h3>{html_escape(eval_name)}</h3><div class="placeholder">No data.</div></div>'
    rows = extracted["rows"]
    labels = [r["name"] for r in rows]
    base_data = [r["base"] for r in rows]
    adapter_data = [r["adapter"] for r in rows]
    rng = extracted["axis_range"]
    chart_id = f"chart_{eval_name}"
    table_rows = "".join(
        f'<tr><td>{html_escape(r["name"])}</td>'
        f'<td class="mono">{fmt_num(r["base"])}</td>'
        f'<td class="mono">{fmt_num(r["adapter"])}</td>'
        f'<td class="mono">{fmt_delta_html(r["delta"])}</td></tr>'
        for r in rows
    )
    return f'''
<div class="eval-card">
  <h3>{html_escape(title)}</h3>
  <div class="meta">{html_escape(bench_meta)}</div>
  <div class="chart-wrap"><canvas id="{chart_id}"></canvas></div>
  <table class="compact">
    <tr><th>Axis</th><th>Base</th><th>Adapter</th><th>Δ</th></tr>
    {table_rows}
  </table>
  <script>
    new Chart(document.getElementById("{chart_id}"), {{
      type: "radar",
      data: {{
        labels: {json.dumps(labels)},
        datasets: [
          {{label: "base", data: {json.dumps(base_data)},
            borderColor: "#c97a2d", backgroundColor: "rgba(201,122,45,0.13)", pointBackgroundColor: "#c97a2d"}},
          {{label: "adapter", data: {json.dumps(adapter_data)},
            borderColor: "#3b7d5a", backgroundColor: "rgba(59,125,90,0.13)", pointBackgroundColor: "#3b7d5a"}}
        ]
      }},
      options: {{
        responsive: true,
        scales: {{r: {{min: {rng[0]}, max: {rng[1]},
                       grid: {{color: "#e3dcc5"}},
                       angleLines: {{color: "#e3dcc5"}},
                       pointLabels: {{color: "#1d1a13", font: {{size: 12}}}},
                       ticks: {{color: "#9c9479", backdropColor: "transparent"}}}}}},
        plugins: {{legend: {{labels: {{color: "#1d1a13"}}}}}}
      }}
    }});
  </script>
</div>'''


def render_bar_card(title: str, eval_name: str, extracted: dict | None,
                    polarity_aware: bool = False, axis_min: float | None = None,
                    axis_max: float | None = None, bench_meta: str = "") -> str:
    if not extracted:
        return f'<div class="eval-card fail"><h3>{html_escape(eval_name)}</h3><div class="placeholder">No data.</div></div>'
    rows = extracted["rows"] if "rows" in extracted else extracted.get("categories", [])
    labels = [r["name"] for r in rows]
    base_data = [r["base"] for r in rows]
    adapter_data = [r["adapter"] for r in rows]
    chart_id = f"chart_{eval_name}"

    table_rows_html = []
    for r in rows:
        delta_higher_good = True
        if polarity_aware:
            pol = r.get("polarity", "neutral")
            if pol == "bad":
                delta_higher_good = False  # lower is better for "bad" traits
        table_rows_html.append(
            f'<tr><td>{html_escape(r["name"])}</td>'
            f'<td class="mono">{fmt_num(r["base"])}</td>'
            f'<td class="mono">{fmt_num(r["adapter"])}</td>'
            f'<td class="mono">{fmt_delta_html(r.get("delta"), higher_is_better=delta_higher_good)}</td></tr>'
        )
    table_html = "".join(table_rows_html)

    scale_extra = ""
    if axis_min is not None or axis_max is not None:
        bits = []
        if axis_min is not None:
            bits.append(f"min: {axis_min}")
        if axis_max is not None:
            bits.append(f"max: {axis_max}")
        scale_extra = ", " + ", ".join(bits)

    return f'''
<div class="eval-card">
  <h3>{html_escape(title)}</h3>
  <div class="meta">{html_escape(bench_meta)}</div>
  <div class="chart-wrap"><canvas id="{chart_id}"></canvas></div>
  <table class="compact">
    <tr><th>Axis</th><th>Base</th><th>Adapter</th><th>Δ</th></tr>
    {table_html}
  </table>
  <script>
    new Chart(document.getElementById("{chart_id}"), {{
      type: "bar",
      data: {{
        labels: {json.dumps(labels)},
        datasets: [
          {{label: "base", data: {json.dumps(base_data)}, backgroundColor: "rgba(201,122,45,0.78)"}},
          {{label: "adapter", data: {json.dumps(adapter_data)}, backgroundColor: "rgba(59,125,90,0.78)"}}
        ]
      }},
      options: {{
        indexAxis: "y",
        responsive: true,
        scales: {{
          x: {{grid: {{color: "#e3dcc5"}}, ticks: {{color: "#6f6852"}}{scale_extra}}},
          y: {{grid: {{color: "#e3dcc5"}}, ticks: {{color: "#1d1a13"}}}}
        }},
        plugins: {{legend: {{labels: {{color: "#1d1a13"}}}}}}
      }}
    }});
  </script>
</div>'''


def render_scalar_table(deltas: dict, category: str, title: str) -> str:
    """Small table of scalar headline deltas for a category."""
    rows = [r for r in deltas.values()
            if r.get("category") == category and r.get("kind") == "scalar"
            and r.get("base") is not None]
    if not rows:
        return ""
    out = [f'<div class="eval-card"><h3>{html_escape(title)}</h3>',
           '<table class="compact"><tr><th>Bench</th><th>Base</th><th>Adapter</th><th>Δ</th><th>Dir</th></tr>']
    for r in rows:
        dir_sym = "↑good" if r.get("higher_is_better") else "↓good"
        out.append(
            f'<tr><td>{html_escape(r["bench"])}</td>'
            f'<td class="mono">{fmt_num(r.get("base"))}</td>'
            f'<td class="mono">{fmt_num(r.get("adapter"))}</td>'
            f'<td class="mono">{fmt_delta_html(r.get("delta"), higher_is_better=r.get("higher_is_better", True))}</td>'
            f'<td style="color:var(--text-dim)">{dir_sym}</td></tr>'
        )
    out.append('</table></div>')
    return "".join(out)


# ─── Page composition ────────────────────────────────────────────────────


def render_index() -> bytes:
    runs = list_adapter_runs()
    rows = []
    for r in runs:
        rows.append(
            f'<tr><td><a href="/run/{html_escape(r["name"])}">{html_escape(r["name"])}</a></td>'
            f'<td>{html_escape(r.get("started_at", "")[:16])}</td>'
            f'<td>{html_escape(r.get("condition", ""))}</td>'
            f'<td class="mono">{html_escape(r.get("student", ""))}</td>'
            f'<td class="mono">{r.get("n_character", 0)}</td>'
            f'<td class="mono">{r.get("n_shifted", 0)}</td>'
            f'<td class="mono">{html_escape(r.get("primary_attr", ""))}</td></tr>'
        )
    body = f"""
<h1>Character-delta dashboard</h1>
<p style="color:var(--text-dim);max-width:680px">
  Per-run visualisation of adapter-vs-base character shift. Reads
  <code class="mono">jobs/runs/&lt;run-id&gt;/deltas.json</code> (produced by
  <code class="mono">scripts/compute_deltas.py --write-deltas</code>).
</p>
<div class="run-list">
  <table>
    <tr><th>Run</th><th>Started</th><th>Cond</th><th>Student</th>
        <th>Char evals</th><th>Shifted</th><th>Trained on</th></tr>
    {''.join(rows) if rows else '<tr><td colspan="7" style="padding:24px;text-align:center;color:var(--text-dim)">No runs with deltas.json yet. Run <code class="mono">scripts/compute_deltas.py &lt;run-id&gt; --write-deltas</code> on an adapter-eval to populate.</td></tr>'}
  </table>
</div>"""
    return page("Index", body, view="character")


def render_run(run_id: str) -> bytes:
    if not RUN_NAME_RE.fullmatch(run_id):
        return page("Bad run id", "<h1>Bad run id</h1>", view="character")
    run_dir = RUNS_DIR / run_id
    deltas = safe_read_json(run_dir / "deltas.json")
    if not deltas:
        body = (
            '<header class="page-header">'
            '<div class="eyebrow"><span class="num">RUN</span>not yet processed</div>'
            f'<h1>{html_escape(run_id)}</h1>'
            '</header>'
            '<div class="empty-state">'
            '<div class="empty-state-icon">∅</div>'
            '<h3>No <code>deltas.json</code> for this run yet</h3>'
            '<p>Compute the per-eval base/adapter deltas first. The dashboard '
            'reads <code>jobs/runs/&lt;run-id&gt;/deltas.json</code>, produced by:</p>'
            '<pre class="cmd-snippet">'
            f'PYTHONPATH=. python3 scripts/compute_deltas.py {html_escape(run_id)} --write-deltas --update-summary'
            '</pre>'
            '<p class="muted">Then refresh this page. Use '
            f'<a href="/trace/{html_escape(run_id)}">Agent trace</a> to inspect the agent\'s decisions in the meantime.</p>'
            '</div>'
        )
        return page(run_id, body, view="character", run_id=run_id)

    summary = safe_read_json(run_dir / "summary.json") or {}
    cfg = summary.get("config") or {}
    eyebrow_bits = []
    if cfg.get("condition"):
        eyebrow_bits.append(html_escape(str(cfg["condition"])))
    if cfg.get("teacher_model"):
        eyebrow_bits.append(f"teacher {html_escape(str(cfg['teacher_model']))}")
    if cfg.get("student_model"):
        eyebrow_bits.append(f"student {html_escape(str(cfg['student_model']))}")
    if cfg.get("started_at"):
        eyebrow_bits.append(html_escape(str(cfg["started_at"])[:16]))
    eyebrow = " · ".join(eyebrow_bits) or "Adapter evaluation"

    sections: list[str] = [
        '<header class="page-header">',
        f'<div class="eyebrow"><span class="num">RUN</span>{eyebrow}</div>',
        f'<h1>{html_escape(run_id)}</h1>',
        '</header>',
        '<div class="section-header"><div class="eyebrow"><span class="num">§</span>Character shifts</div>'
        '<h2>How the adapter moves the model</h2></div>',
    ]

    # rozado political compass
    sections.append(render_compass_card(extract_compass(deltas.get("rozado_battery")), "01"))

    # big_five — spectrum strips with named poles per trait
    bf = extract_radar(
        deltas.get("big_five"),
        key_re=r"^any_choice(?:_lenient)?\.(.+)$",
        axis_range=(0, 1),
    )
    bf_rows = bf["rows"] if bf else None
    if bf_rows:
        # Sort by canonical order
        order = list(BIG_FIVE_POLES.keys())
        bf_rows = sorted(bf_rows, key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    sections.append(render_spectrum_card(
        "big_five — personality fingerprint", "big_five", bf_rows,
        axis_range=(0, 1),
        poles_lookup=BIG_FIVE_POLES,
        directional_only=True,
        bench_meta="trait_ratio metric (Inspect Evals BFI) — fraction of 8–10 MCQ items "
                   "per trait endorsing the high pole. Native range [0, 1]. Big Five traits "
                   "are descriptive: neither pole is inherently better, so Δ is direction-only.",
    ))

    # moral_foundations — foundations spectrum + aggregates spectrum.
    # Per-item Likert is 0-5 (6 options: Strongly Disagree … Strongly Agree),
    # per-foundation = mean of 6 items, so the achievable range is [0, 5].
    mf = extract_radar(
        deltas.get("moral_foundations"),
        key_re=r"^foundations\.(.+)$",
        axis_range=(0, 5),
    )
    sections.append(render_spectrum_card(
        "moral_foundations — MFQ foundations", "moral_foundations_f",
        mf["rows"] if mf else None,
        axis_range=(0, 5),
        poles_lookup=MORAL_FOUNDATION_POLES,
        bench_meta="Mean of 6 Likert items per foundation, native MFQ scale 0–5 "
                   "(0 = Strongly Disagree, 5 = Strongly Agree). Low pole = vice, "
                   "high pole = virtue (Haidt).",
    ))
    # Aggregate scores (individualizing / binding)
    mf_agg_row = deltas.get("moral_foundations") or {}
    if mf_agg_row.get("kind") == "multi":
        agg_sub = mf_agg_row.get("sub") or {}
        agg_rows = []
        for axis in ("individualizing_score", "binding_score"):
            v = agg_sub.get(axis)
            if v and isinstance(v.get("base"), (int, float)):
                agg_rows.append({"name": axis, "base": v["base"], "adapter": v.get("adapter"), "delta": v.get("delta")})
        if agg_rows:
            sections.append(render_spectrum_card(
                "moral_foundations — aggregate axes", "moral_foundations_agg",
                agg_rows,
                axis_range=(0, 5),
                poles_lookup=MORAL_AGGREGATE_POLES,
                bench_meta="Care + fairness average (individualizing) vs loyalty + authority + sanctity average (binding). Native MFQ 0–5.",
            ))

    # persona_traits — dumbbell card grouped by polarity (mirrors persona-vectors
    # paper's bad-triad-first treatment of evil / sycophantic / hallucinating).
    pt = extract_persona_traits(deltas.get("persona_traits"))
    pt_rows = pt["rows"] if pt else None
    sections.append(render_persona_dumbbell_card(
        "persona_traits — Persona Vectors (Chen 2025)", "persona_traits",
        pt_rows,
        axis_range=(0, 100),
        bench_meta="Haiku judge rates each rollout 0–100 per trait (native scale). "
                   "Mean shown. Grouped by polarity, sorted by |Δ| within group: "
                   "undesirable triad first (per Chen et al's emphasis), then desirable, "
                   "then neutral. Connector colour = goodness of shift.",
    ))

    # spiralbench — spectrum per behaviour with semantic poles + polarity
    sb = extract_spiralbench(deltas.get("spiralbench_mini"))
    sb_rows = sb["rows"] if sb else None
    sections.append(render_spectrum_card(
        "spiralbench_mini — 17-behavior rubric", "spiralbench_mini",
        sb_rows,
        axis_range=(0, 3),
        poles_lookup=SPIRAL_POLES,
        polarity_aware=True,
        polarity_lookup_good=SPIRAL_GOOD,
        polarity_lookup_bad=SPIRAL_BAD,
        bench_meta="0–3 native count per behaviour per conversation (judge-graded against v1.2 rubric); mean across conversations. Green = desirable shift, red = undesirable. Sorted by |Δ|.",
    ))

    # activity_preference — diverging horizontal bar centred at 0 + top-3
    # attractor callout + rank-shift column (mirrors Anthropic model-welfare
    # work which leads with topic-rank tables, not raw means).
    ap = extract_activity_preference(deltas.get("activity_preference"))
    ap_rows = ap.get("categories") if ap else None
    sections.append(render_diverging_bar_card(
        "activity_preference — per-category Elo (log-strength)", "activity_preference",
        ap_rows,
        bench_meta="Bradley-Terry log-strength means per activity category, centred at ~0. "
                   "Positive = preferred, negative = avoided. Mirrors Anthropic model-welfare "
                   "framing: top-3 attractors + rank shift, not raw means alone.",
    ))

    # moru bar (14 axes, no clear dichotomy poles → bar is the right viz)
    moru = extract_radar(
        deltas.get("moru"),
        key_re=r"^moru_scorer\.(.+)$",
        axis_range=(0, 1),
    )
    sections.append(render_bar_card(
        "moru — moral-reasoning axes", "moru", moru,
        bench_meta="0-1 per-axis mean (haiku grader). Higher = more of the moral-reasoning trait.",
    ))

    # political_bias_openai bar (5-slant naturally a left-to-right gradient)
    pb = extract_political_bias(deltas.get("political_bias_openai"))
    sections.append(render_bar_card(
        "political_bias_openai — per-slant", "political_bias_openai", pb,
        bench_meta="Per-slant 5-axis judge score (0-1). Liberal-charged → conservative-charged.",
    ))

    # Safety + capability scalar tables (no charts; just deltas)
    sections.append('<h2>Safety tier (scalar)</h2>')
    sections.append(render_scalar_table(deltas, "safety", "Safety scalar deltas"))
    sections.append('<h2>Capability tier (scalar)</h2>')
    sections.append(render_scalar_table(deltas, "capability", "Capability scalar deltas"))

    return page(run_id, "\n".join(sections), view="character", run_id=run_id)


# ─── HTTP handler ────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        self.server._log(f'{self.client_address[0]} "{fmt % args}"')

    def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send(200, render_index())
            return
        if path.startswith("/static/"):
            fname = path[len("/static/"):]
            if not RUN_NAME_RE.fullmatch(fname):
                self._send(404, b"not found")
                return
            f = STATIC_DIR / fname
            if not f.exists():
                self._send(404, b"not found")
                return
            ct = "application/javascript" if f.suffix == ".js" else \
                 "text/css" if f.suffix == ".css" else "application/octet-stream"
            self._send(200, f.read_bytes(), ct)
            return
        m = re.match(r"^/run/([^/]+)$", path)
        if m:
            self._send(200, render_run(m.group(1)))
            return
        m = re.match(r"^/run/([^/]+)/eval/([^/]+)$", path)
        if m:
            run_id, bench = m.group(1), m.group(2)
            if not RUN_NAME_RE.fullmatch(run_id) or not BENCH_NAME_RE.fullmatch(bench):
                self._send(400, b"bad ids")
                return
            d = safe_read_json(RUNS_DIR / run_id / "deltas.json") or {}
            blob = json.dumps(d.get(bench, {"error": "not in deltas.json"}), indent=2)
            body = page(f"{run_id}/{bench}",
                        f'<h1>{html_escape(bench)}</h1>'
                        f'<p><a href="/run/{html_escape(run_id)}">← {html_escape(run_id)}</a></p>'
                        f'<pre style="background:var(--bg-card);padding:16px;border-radius:6px;overflow:auto">{html_escape(blob)}</pre>',
                        view="character", run_id=run_id)
            self._send(200, body)
            return

        # Trace-viewer routes — delegate to trace_viewer's render functions.
        # /trace        → run list (trace-side index)
        # /trace/<run>  → agent-trace timeline for that run
        if path == "/trace" or path == "/trace/":
            runs = trace_viewer.list_runs(RUNS_DIR)
            html_text = trace_viewer.render_index(runs)
            self._send(200, html_text.encode("utf-8"))
            return
        m = re.match(r"^/trace/([^/]+)$", path)
        if m:
            name = unquote(m.group(1))
            if not RUN_NAME_RE.fullmatch(name):
                self._send(400, b"bad run id")
                return
            run_dir = RUNS_DIR / name
            if not run_dir.is_dir():
                body = trace_viewer.page(
                    "404", f"<div class='card'><h2>404</h2><p>run dir does not exist: "
                    f"{trace_viewer.esc(name)}</p></div>",
                    view="trace",
                )
                self._send(404, body.encode("utf-8"))
                return
            run = trace_viewer.load_run(run_dir)
            self._send(200, trace_viewer.render_run(run).encode("utf-8"))
            return

        self._send(404, b"not found")


class Server(ThreadingHTTPServer):
    def _log(self, msg: str):
        print(f"[dashboard] {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = Server((args.host, args.port), Handler)
    print(f"[dashboard] serving http://{args.host}:{args.port}")
    print(f"[dashboard] runs dir: {RUNS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down")


if __name__ == "__main__":
    main()
