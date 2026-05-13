"""Trace viewer for agent runs in jobs/runs/.

Local web app that lists runs and renders each run's agent trace
(solve_out.jsonl) with score progression, metadata, and a filterable
action timeline.

Usage:
    python dev_utils/trace_viewer/app.py            # serve on http://127.0.0.1:8765
    python dev_utils/trace_viewer/app.py --port 9000
    python dev_utils/trace_viewer/app.py --runs-dir /path/to/jobs/runs

Stdlib only. Works with the Claude Code stream-json format produced by
agents/claude*/solve.sh.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "jobs" / "runs"

TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\] ")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
TOOL_RESULT_TRUNC = 3000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _scalar_acc(d: Any) -> float | None:
    """Pull a scalar `accuracy`-like float from a metrics dict or summary slot.

    Handles the three shapes we've seen: (1) flat {accuracy: X}, (2) nested
    {<bench>: {accuracy: X}}, (3) None/empty."""
    if not isinstance(d, dict):
        return None
    v = d.get("accuracy")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _resolve_index_scores(
    entry: Path, summary: dict, config: dict, pre: dict, post: dict
) -> tuple[float | None, float | None, float | None]:
    """Find pre/post/delta for the index row, falling back through the
    layers we've accumulated over the project's lifetime:

      1. Legacy unsuffixed `metrics_pre.json` / `metrics_post.json` files.
      2. summary.json's `pre`/`post`/`delta` blocks (current pipeline).
      3. Per-bench `metrics_post_<benchmark>.json` for the primary bench
         (the suffix path used by submit_run.py post-`:14`).
      4. `deltas.json` written by `scripts/compute_deltas.py --write-deltas`,
         keyed by primary bench, when running an adapter-eval.

    Returns (pre_acc, post_acc, delta_value). Any may be None."""
    pre_acc = _scalar_acc(pre)
    post_acc = _scalar_acc(post)
    delta = summary.get("delta")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        delta = None

    if pre_acc is None:
        pre_acc = _scalar_acc(summary.get("pre"))
    if post_acc is None:
        post_acc = _scalar_acc(summary.get("post"))

    bench = config.get("benchmark")
    if bench:
        if post_acc is None:
            post_acc = _scalar_acc(safe_read_json(entry / f"metrics_post_{bench}.json"))
        if pre_acc is None:
            pre_acc = _scalar_acc(safe_read_json(entry / f"metrics_pre_{bench}.json"))

        if delta is None:
            deltas_blob = safe_read_json(entry / "deltas.json")
            if isinstance(deltas_blob, dict):
                row = deltas_blob.get(bench)
                if isinstance(row, dict):
                    d = row.get("delta")
                    if isinstance(d, (int, float)) and not isinstance(d, bool):
                        delta = float(d)
                    # Also backfill pre/post from deltas.json when missing.
                    if pre_acc is None:
                        b = row.get("base")
                        if isinstance(b, (int, float)) and not isinstance(b, bool):
                            pre_acc = float(b)
                    if post_acc is None:
                        a = row.get("adapter")
                        if isinstance(a, (int, float)) and not isinstance(a, bool):
                            post_acc = float(a)

    if delta is None and pre_acc is not None and post_acc is not None:
        delta = post_acc - pre_acc

    return pre_acc, post_acc, delta


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    out = []
    for entry in sorted(runs_dir.iterdir(), reverse=True):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        config = safe_read_json(entry / "config.json") or {}
        summary = safe_read_json(entry / "summary.json") or {}
        pre = safe_read_json(entry / "metrics_pre.json") or {}
        post = safe_read_json(entry / "metrics_post.json") or {}
        trace_path = entry / "solve_out.jsonl"
        n_versions = _count_versions_lite(trace_path) if trace_path.exists() else 0
        pre_acc, post_acc, delta = _resolve_index_scores(
            entry, summary, config, pre, post
        )
        out.append(
            {
                "name": entry.name,
                "started_at": config.get("started_at", ""),
                "condition": config.get("condition", "?"),
                "teacher": config.get("teacher_model", "?"),
                "student": config.get("student_model", "?"),
                "benchmark": config.get("benchmark", "?"),
                "agent": config.get("agent", "?"),
                "status": summary.get("status") or ("complete" if post or post_acc is not None else "no_post"),
                "pre_acc": pre_acc,
                "post_acc": post_acc,
                "delta": delta,
                "duration_s": summary.get("duration_s"),
                "trace_lines": summary.get("agent_trace_lines"),
                "has_trace": trace_path.exists(),
                "n_versions": n_versions,
            }
        )
    return out


def _count_versions_lite(jsonl_path: Path) -> int:
    """Fast version count for the index page: stream the trace and count
    distinct training-cmd Bash calls plus a synthetic V0 if any eval cmd
    fires before the first training cmd. Avoids the full normalize_timeline
    cost on every index load. Returns 0 on parse error."""
    try:
        n_train = 0
        has_pre_eval = False
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                m = TIMESTAMP_RE.match(line)
                if m:
                    line = line[m.end():]
                # Cheap pre-filter — skip lines that can't contain a Bash cmd.
                if '"name":"Bash"' not in line and '"name": "Bash"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = ev.get("message") or {}
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    if block.get("name") != "Bash":
                        continue
                    cmd = (block.get("input") or {}).get("command", "") or ""
                    masked = _mask_quoted(cmd)
                    is_eval = bool(SCORE_CMD_RE.search(masked)) and not _INSPECTION_RE.match(cmd)
                    is_train = _is_training_cmd(cmd)
                    if is_eval and n_train == 0:
                        has_pre_eval = True
                    elif is_train:
                        n_train += 1
        return n_train + (1 if has_pre_eval else 0)
    except OSError:
        return 0


def load_run(run_dir: Path) -> dict[str, Any]:
    return {
        "name": run_dir.name,
        "config": safe_read_json(run_dir / "config.json") or {},
        "summary": safe_read_json(run_dir / "summary.json") or {},
        "pod_meta": safe_read_json(run_dir / "pod_meta.json") or {},
        "pre": safe_read_json(run_dir / "metrics_pre.json") or {},
        "post": safe_read_json(run_dir / "metrics_post.json") or {},
        "rerun_post": safe_read_json(run_dir / "rerun_post_summary.json"),
        "prompt": (run_dir / "prompt.txt").read_text(encoding="utf-8")
        if (run_dir / "prompt.txt").exists()
        else "",
        "extras_pre": _glob_metrics(run_dir, "metrics_pre_*.json"),
        "extras_post": _glob_metrics(run_dir, "metrics_post_*.json"),
        "heldout_summary": safe_read_json(run_dir / "heldout" / "summary.json"),
        "trace_path": run_dir / "solve_out.jsonl",
    }


def _glob_metrics(run_dir: Path, pattern: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(run_dir.glob(pattern)):
        # metrics_pre_humaneval.json -> humaneval
        prefix = "metrics_pre_" if "pre" in pattern else "metrics_post_"
        name = path.stem[len(prefix):]
        out[name] = safe_read_json(path) or {}
    return out


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------


def parse_events(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            wall_ts = None
            m = TIMESTAMP_RE.match(line)
            if m:
                wall_ts = m.group(1)
                line = line[m.end():]
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if wall_ts:
                ev["_wall_ts"] = wall_ts
            # Native timestamps live on user events as "timestamp"; assistant
            # events have none. Pick the best per-event ts now, then forward-
            # and back-fill across neighbouring events so every item gets one.
            ev["_ts"] = ev.get("timestamp") or wall_ts
            events.append(ev)
    _fill_timestamps(events)
    return events


def _fill_timestamps(events: list[dict[str, Any]]) -> None:
    """Forward-fill (using the next event's ts) then back-fill (using the
    previous event's ts) so every event has a `_ts`. Forward-fill first means
    an assistant turn inherits the timestamp of the next user event - i.e.
    when the agent's response was actually consumed - which is closer to the
    truth than the previous user event's input time."""
    next_ts = None
    for ev in reversed(events):
        if ev.get("_ts"):
            next_ts = ev["_ts"]
        else:
            ev["_ts"] = next_ts
    last_ts = None
    for ev in events:
        if ev.get("_ts"):
            last_ts = ev["_ts"]
        else:
            ev["_ts"] = last_ts


def normalize_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten Claude Code stream-json events into ordered timeline items.

    Tool results are matched back to their tool_use by id so we can label
    them with the tool name.
    """
    tool_names: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    assistant_turn = 0
    user_turn = 0

    for ev in events:
        t = ev.get("type")
        ts = ev.get("_ts") or ev.get("_wall_ts")

        if t == "system":
            sub = ev.get("subtype")
            if sub == "init":
                items.append(
                    {
                        "kind": "system_init",
                        "ts": ts,
                        "model": ev.get("model"),
                        "session_id": ev.get("session_id"),
                        "cwd": ev.get("cwd"),
                        "tools": ev.get("tools") or [],
                    }
                )
            else:
                items.append({"kind": "system", "ts": ts, "subtype": sub, "data": ev})
            continue

        if t == "result":
            items.append(
                {
                    "kind": "result",
                    "ts": ts,
                    "subtype": ev.get("subtype"),
                    "data": {k: v for k, v in ev.items() if k not in {"type", "_wall_ts", "_ts"}},
                }
            )
            continue

        if t == "rate_limit_event":
            continue  # noisy, omit from timeline

        if t in ("assistant", "user"):
            msg = ev.get("message") or {}
            role = msg.get("role") or t
            if role == "assistant":
                assistant_turn += 1
                turn_no = assistant_turn
            else:
                user_turn += 1
                turn_no = user_turn
            content = msg.get("content")
            if isinstance(content, str):
                # user messages can be a bare string
                items.append(
                    {"kind": "text", "role": role, "ts": ts, "turn": turn_no, "text": content}
                )
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "text":
                    items.append(
                        {
                            "kind": "text",
                            "role": role,
                            "ts": ts,
                            "turn": turn_no,
                            "text": block.get("text", ""),
                        }
                    )
                elif bt == "thinking":
                    items.append(
                        {
                            "kind": "thinking",
                            "role": role,
                            "ts": ts,
                            "turn": turn_no,
                            "text": block.get("thinking", ""),
                        }
                    )
                elif bt == "tool_use":
                    tool_id = block.get("id", "")
                    name = block.get("name", "tool")
                    tool_names[tool_id] = name
                    items.append(
                        {
                            "kind": "tool_use",
                            "ts": ts,
                            "turn": turn_no,
                            "id": tool_id,
                            "name": name,
                            "input": block.get("input"),
                        }
                    )
                elif bt == "tool_result":
                    tid = block.get("tool_use_id", "")
                    items.append(
                        {
                            "kind": "tool_result",
                            "ts": ts,
                            "turn": turn_no,
                            "id": tid,
                            "name": tool_names.get(tid, "tool"),
                            "content": block.get("content"),
                            "is_error": block.get("is_error", False),
                        }
                    )
                else:
                    items.append(
                        {
                            "kind": "other",
                            "ts": ts,
                            "block_type": bt,
                            "data": block,
                        }
                    )
    return items


def _annotate_elapsed(items: list[dict[str, Any]]) -> None:
    """Mutate each item with `_elapsed_s` = seconds since previous item with
    a parseable ts. First item has None."""
    prev_t: float | None = None
    for item in items:
        t = parse_iso(item.get("ts") or "")
        if t is not None and prev_t is not None:
            item["_elapsed_s"] = t - prev_t
        if t is not None:
            prev_t = t


def render_tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(item.get("text", ""))
                else:
                    chunks.append(json.dumps(item, indent=2, ensure_ascii=False))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    if content is None:
        return ""
    return json.dumps(content, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Score-progression extraction
# ---------------------------------------------------------------------------


ACCURACY_RE = re.compile(r"accuracy[^0-9\-]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
SCORE_CMD_RE = re.compile(
    r"\b(evaluate\.py|score\.sh)\b", re.IGNORECASE
)


def extract_intermediate_scores(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find Bash tool calls running evaluate.py / score.sh and pull accuracy
    out of the matched tool_result text. Best-effort - the agent often runs
    these many times during the run so we get a rough progression."""
    pending: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for item in items:
        if item["kind"] == "tool_use" and item.get("name") == "Bash":
            cmd = (item.get("input") or {}).get("command", "")
            if SCORE_CMD_RE.search(cmd):
                pending[item["id"]] = {"cmd": cmd, "ts": item.get("ts")}
        elif item["kind"] == "tool_result" and item.get("id") in pending:
            info = pending.pop(item["id"])
            text = render_tool_result_text(item.get("content"))
            accs = ACCURACY_RE.findall(text)
            limit = _sniff_limit(info["cmd"])
            out.append(
                {
                    "ts": item.get("ts"),
                    "cmd": info["cmd"].splitlines()[0][:200],
                    "accuracy": float(accs[-1]) if accs else None,
                    "limit": limit,
                    "id": item["id"],
                }
            )
    return out


def _sniff_limit(cmd: str) -> int | None:
    m = re.search(r"--limit[\s=](\d+)", cmd)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Version detection — segment the trace into V0 (base), V1, V2, ...
# Boundary signal: a Bash tool_use that looks like training, followed by
# eval calls. The version's score is the *last* eval before the next train.
# ---------------------------------------------------------------------------


_TRAIN_RUNNER = r"(?:python\d*|uv\s+run(?:\s+python\d*)?)"
TRAIN_CMD_RE = re.compile(
    # Anchor on *executing* a training script or framework, not just naming one.
    # Order matters - alternatives are tried left-to-right.
    rf"(?:"
    rf"{_TRAIN_RUNNER}\s+\S*?(?:train|sft|dpo|orpo|finetune|fine_tune)\w*\.py"
    rf"|accelerate\s+launch"
    rf"|torchrun\b"
    rf"|{_TRAIN_RUNNER}\s+-m\s+(?:torch\.distributed|peft|trl|axolotl)"
    rf"|trainer\.train\("
    rf"|\.save_pretrained\("
    rf")",
    re.IGNORECASE,
)

# Commands whose first non-comment token is one of these are inspection, not
# execution - even if their body literally contains the string "python
# train.py" (e.g. `pkill -f "python train.py"`, `ps -ef | grep ...`).
_INSPECTION_RE = re.compile(
    r"^\s*(?:#[^\n]*\n\s*)*"
    r"(pkill|kill|killall|ps|grep|pgrep|cat|head|tail|less|more|"
    r"vim|nano|view|ls|find|stat|file|wc|md5sum|sha256sum|du|df|"
    r"echo)\b",
    re.IGNORECASE,
)

_QUOTED_RE = re.compile(r'"[^"\n]*"' + r"|'[^'\n]*'")


def _mask_quoted(cmd: str) -> str:
    """Replace contents of quoted strings with spaces, preserving length so
    regex match indices line up with the original `cmd`. Keeps the opening
    and closing quote chars themselves."""
    def _repl(m: re.Match) -> str:
        s = m.group(0)
        return s[0] + " " * (len(s) - 2) + s[-1]
    return _QUOTED_RE.sub(_repl, cmd)


def _is_training_cmd(cmd: str) -> bool:
    if not cmd:
        return False
    if _INSPECTION_RE.match(cmd):
        return False
    return bool(TRAIN_CMD_RE.search(_mask_quoted(cmd)))


def _training_cmd_snippet(cmd: str, max_len: int = 200) -> str:
    """Return the line of the command that triggered the match (or the first
    non-blank prefix line if no match), trimmed to `max_len`."""
    masked = _mask_quoted(cmd)
    m = TRAIN_CMD_RE.search(masked)
    if m:
        start = cmd.rfind("\n", 0, m.start()) + 1
        end = cmd.find("\n", m.end())
        if end == -1:
            end = len(cmd)
        line = cmd[start:end].strip()
        if len(line) > max_len:
            line = line[:max_len] + "…"
        return line
    return (cmd.splitlines()[0] if cmd else "")[:max_len]


DATA_FETCH_RE = re.compile(
    r"\b(huggingface_hub|hf\s+download|datasets\.load_dataset|wget|curl)\b",
    re.IGNORECASE,
)

TRAINING_HYPERPARAM_RE = re.compile(
    r"\b(learning_rate|lora_r|lora_alpha|num_train_epochs|"
    r"per_device_train_batch_size|gradient_accumulation_steps)\b"
)

ERROR_LIKE_RE = re.compile(
    # Anchor to line-start so dataset prose mentioning "ValueError" doesn't match.
    # Matches Python tracebacks and CUDA/OOM messages where they actually appear:
    # at column 0 of a fresh line.
    r"(?m)^\s*(Traceback \(most recent call last\):"
    r"|[A-Z][A-Za-z]*(?:Error|Exception): "
    r"|CUDA (?:error|out of memory|OOM)"
    r"|torch\.cuda\.OutOfMemoryError"
    r"|RuntimeError: "
    r"|OSError: )",
)


def _eval_accuracy_from_result(items: list[dict[str, Any]], idx: int, tool_id: str) -> tuple[float | None, str | None, int | None]:
    """Walk forward from idx looking for the matching tool_result and parse
    accuracy out of it. Returns (accuracy, ts, limit_hint). Mirrors the logic
    in extract_intermediate_scores so the two stay consistent."""
    for j in range(idx + 1, len(items)):
        nxt = items[j]
        if nxt.get("kind") == "tool_result" and nxt.get("id") == tool_id:
            text = render_tool_result_text(nxt.get("content"))
            accs = ACCURACY_RE.findall(text)
            return (
                float(accs[-1]) if accs else None,
                nxt.get("ts"),
                None,
            )
    return None, None, None


def detect_versions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Segment a normalized timeline into model versions.

    Returns a list of version records:
        {"label": "V0"|"V1"|..., "is_base": bool,
         "train_idx": int|None, "train_ts": str|None, "train_cmd": str|None,
         "eval_idx": int|None, "eval_ts": str|None,
         "accuracy": float|None, "limit": int|None, "status": str}

    V0 is the synthetic pre-training base; emitted only if at least one
    eval cmd fires before the first training cmd.
    """
    train_events: list[tuple[int, dict[str, Any]]] = []
    eval_events: list[tuple[int, dict[str, Any], float | None, str | None, int | None]] = []

    for i, it in enumerate(items):
        if it.get("kind") != "tool_use" or it.get("name") != "Bash":
            continue
        cmd = (it.get("input") or {}).get("command", "") or ""
        masked = _mask_quoted(cmd)
        is_eval = bool(SCORE_CMD_RE.search(masked)) and not _INSPECTION_RE.match(cmd)
        is_train = _is_training_cmd(cmd)
        # If a single cmd matches both (rare), prefer eval — agents sometimes
        # name an eval script `train_eval.py`. Conservative bias.
        if is_eval:
            acc, ts, _ = _eval_accuracy_from_result(items, i, it.get("id", ""))
            eval_events.append((i, it, acc, ts, _sniff_limit(cmd)))
        elif is_train:
            train_events.append((i, it))

    versions: list[dict[str, Any]] = []

    # V0: pre-training evals
    first_train_idx = train_events[0][0] if train_events else None
    pre_evals = [e for e in eval_events if first_train_idx is None or e[0] < first_train_idx]
    if pre_evals:
        last = pre_evals[-1]
        versions.append({
            "label": "V0",
            "is_base": True,
            "train_idx": None,
            "train_ts": None,
            "train_cmd": None,
            "eval_idx": last[0],
            "eval_ts": last[3],
            "accuracy": last[2],
            "limit": last[4],
            "status": "ok" if last[2] is not None else "no_score",
        })

    for n, (t_idx, t_it) in enumerate(train_events, start=1):
        next_train_idx = train_events[n][0] if n < len(train_events) else None
        # Last eval strictly after this train and strictly before the next.
        candidates = [
            e for e in eval_events
            if e[0] > t_idx and (next_train_idx is None or e[0] < next_train_idx)
        ]
        if candidates:
            last = candidates[-1]
            eval_idx, eval_ts, acc, limit = last[0], last[3], last[2], last[4]
            status = "ok" if acc is not None else "no_score"
        else:
            eval_idx = eval_ts = acc = limit = None
            status = "no_eval"

        cmd_full = ((t_it.get("input") or {}).get("command", "") or "").strip()
        cmd_snippet = _training_cmd_snippet(cmd_full) if cmd_full else ""
        versions.append({
            "label": f"V{n}",
            "is_base": False,
            "train_idx": t_idx,
            "train_ts": t_it.get("ts"),
            "train_cmd": cmd_snippet,
            "train_turn": t_it.get("turn"),
            "eval_idx": eval_idx,
            "eval_ts": eval_ts,
            "accuracy": acc,
            "limit": limit,
            "status": status,
        })

    return versions


def summarize_between_versions(
    items: list[dict[str, Any]], prev_end: int, next_start: int
) -> dict[str, Any]:
    """Heuristic activity summary for items[prev_end : next_start].

    prev_end = index of the previous version's eval (or train if no eval).
    next_start = index of the next version's train cmd.
    """
    if next_start <= prev_end + 1:
        return {}

    slice_ = items[prev_end + 1 : next_start]
    tool_counts: dict[str, int] = {}
    data_fetches: list[str] = []
    config_edits: list[str] = []
    error_count = 0
    error_sample: str | None = None
    agent_note: str | None = None
    gap_s = 0.0

    for it in slice_:
        elapsed = it.get("_elapsed_s")
        if isinstance(elapsed, (int, float)) and elapsed > 0:
            gap_s += elapsed

        kind = it.get("kind")
        if kind == "tool_use":
            name = it.get("name", "tool")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            payload = it.get("input") or {}
            if name == "Bash":
                cmd = payload.get("command", "") or ""
                if DATA_FETCH_RE.search(cmd):
                    data_fetches.append(cmd.splitlines()[0][:140])
            elif name in ("Edit", "Write", "MultiEdit"):
                path = payload.get("file_path", "") or ""
                blob = (
                    payload.get("new_string", "")
                    + " "
                    + payload.get("content", "")
                )
                if path.endswith(".py") and TRAINING_HYPERPARAM_RE.search(blob):
                    config_edits.append(path.split("/")[-1])
        elif kind == "tool_result":
            text = render_tool_result_text(it.get("content"))
            if it.get("is_error") or ERROR_LIKE_RE.search(text):
                error_count += 1
                if error_sample is None:
                    m = ERROR_LIKE_RE.search(text)
                    if m:
                        # Grab the line containing the match.
                        start = text.rfind("\n", 0, m.start()) + 1
                        end = text.find("\n", m.end())
                        if end == -1:
                            end = min(len(text), m.end() + 120)
                        error_sample = text[start:end].strip()[:160]
        elif kind == "text" and it.get("role") == "assistant" and agent_note is None:
            txt = (it.get("text") or "").strip()
            if txt:
                agent_note = txt[:280] + ("…" if len(txt) > 280 else "")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    config_edits = [p for p in config_edits if not (p in seen or seen.add(p))]

    return {
        "tool_counts": tool_counts,
        "data_fetches": data_fetches[:5],
        "config_edits": config_edits[:6],
        "errors": {"count": error_count, "sample": error_sample},
        "gap_s": gap_s,
        "agent_note": agent_note,
    }


def annotate_version_membership(
    items: list[dict[str, Any]], versions: list[dict[str, Any]]
) -> None:
    """Mutate items in place: set item['_version'] to the label of the
    version each item belongs to. An item belongs to V_n if it sits between
    V_{n-1}'s eval (or V_n's train) and V_n's eval. Items before V0/V1 are
    labelled 'pre'; items after the last version are labelled with the last
    version's label."""
    if not versions:
        for it in items:
            it["_version"] = ""
        return

    # Build (start_idx, label) boundaries.
    boundaries: list[tuple[int, str]] = []
    for v in versions:
        anchor = v["train_idx"] if v["train_idx"] is not None else v["eval_idx"]
        if anchor is None:
            continue
        boundaries.append((anchor, v["label"]))
    boundaries.sort()

    cur = "pre"
    bi = 0
    for i, it in enumerate(items):
        while bi < len(boundaries) and boundaries[bi][0] <= i:
            cur = boundaries[bi][1]
            bi += 1
        it["_version"] = cur


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def esc(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def fmt_acc(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v * 100:.1f}%"
    return esc(v)


def fmt_delta(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.1f}pp"
    return esc(v)


def fmt_duration(s: Any) -> str:
    if not isinstance(s, (int, float)):
        return "-"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def fmt_event_ts(s: Any) -> str:
    """Format an event timestamp (ISO 8601 with 'Z' or offset) as
    `YYYY-MM-DD HH:MM:SS` for display in event headers."""
    if not isinstance(s, str) or not s:
        return ""
    try:
        date, rest = s.split("T", 1)
        # rest may be like "10:33:34.444Z" or "15:53:59.133541+10:00"
        hms = rest[:8]
        return f"{date} {hms}"
    except ValueError:
        return s


def fmt_elapsed(s: float | None) -> str:
    if s is None or s < 0.05:
        return ""
    if s < 60:
        return f"+{s:.1f}s"
    if s < 3600:
        return f"+{s/60:.1f}m"
    return f"+{s/3600:.2f}h"


def parse_iso(s: str) -> float | None:
    """Return a float UNIX-timestamp-ish ordering key for an ISO 8601 string,
    or None if unparseable. Used only for ordering / elapsed-time math."""
    if not s:
        return None
    from datetime import datetime, timezone
    try:
        # python <3.11 chokes on trailing Z; normalize.
        s2 = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s2).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def fmt_started(s: Any) -> str:
    """Short human-readable form of an ISO 8601 started_at string."""
    if not isinstance(s, str) or not s:
        return "-"
    # 2026-05-08T15:53:59.133541+10:00 -> 2026-05-08 15:53
    try:
        date, rest = s.split("T", 1)
        hhmm = rest[:5]
        return f"{date} {hhmm}"
    except ValueError:
        return s


def started_sort_key(r: dict[str, Any]) -> str:
    """Return a sortable string for the Started column.

    Falls back to the run dirname (which starts with the timestamp) when
    started_at is missing.
    """
    return r.get("started_at") or r.get("name") or ""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@300;400;500;600;700&display=swap');
:root {
  --bg:           #f4eee0;
  --panel:        #fbf7eb;
  --panel-2:      #f0e9d3;
  --bg-inset:     #faf4e4;
  --border:       #cfc6ab;
  --border-soft:  #e3dcc5;
  --fg:           #1d1a13;
  --muted:        #6f6852;
  --faint:        #9c9479;
  --accent:       #2d2516;
  --accent-deep:  #4f4225;
  --green: #3b7d5a;
  --red:   #b03a2e;
  --blue:  #3d6a99;
  --yellow:#a37820;
  --purple:#6e3d9d;
  --orange:#c97a2d;

  --serif: "Newsreader", "Cormorant Garamond", Georgia, serif;
  --sans:  "Inter", ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  --mono:  ui-monospace, "SF Mono", Menlo, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid transparent; transition: border-color 120ms; }
a:hover { border-color: var(--accent); }

/* Top nav — shared across views */
header.topnav {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 14px 32px;
  display: flex; align-items: center; gap: 24px;
  position: sticky; top: 0; z-index: 10;
}
header.topnav .brand {
  font-family: var(--serif);
  font-size: 18px;
  font-weight: 500;
  color: var(--fg);
  letter-spacing: -0.005em;
}
header.topnav .brand a { color: inherit; border: none; }
header.topnav .nav-tabs { display: flex; gap: 6px; margin-left: auto; }
header.topnav .nav-tab {
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 6px 14px;
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  background: transparent;
}
header.topnav .nav-tab.active {
  background: var(--fg);
  color: var(--bg);
  border-color: var(--fg);
}
header.topnav .nav-tab:hover:not(.active) { color: var(--fg); border-color: var(--fg); }
header.topnav .crumb {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
}

.container { max-width: 1240px; margin: 36px auto; padding: 0 48px 96px; }
.muted { color: var(--muted); }
.mono, code, pre { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* Eyebrow + serif h-styles to match dashboard */
.eyebrow {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.eyebrow::before {
  content: "";
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent-deep);
  display: inline-block;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 28px 32px;
  margin-bottom: 22px;
}
.card h2 {
  margin: 0 0 16px;
  font-family: var(--serif);
  font-weight: 500;
  font-size: 22px;
  color: var(--fg);
  letter-spacing: -0.005em;
}
.card h3 {
  margin: 18px 0 10px;
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 600;
  color: var(--faint);
  text-transform: uppercase;
  letter-spacing: 0.16em;
}

table.runs { width: 100%; border-collapse: collapse; font-size: 13px; }
table.runs th, table.runs td {
  padding: 12px 16px; text-align: left;
  border-bottom: 1px solid var(--border-soft);
  vertical-align: top;
}
table.runs th {
  color: var(--muted); font-weight: 500;
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em;
  background: var(--bg);
}
table.runs.sortable th { cursor: pointer; user-select: none; }
table.runs.sortable th:hover { color: var(--fg); }
table.runs tr:hover td { background: var(--bg-inset); }
table.runs td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.runs a { font-family: var(--mono); }

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
}
.metric {
  background: var(--bg-inset);
  border: 1px solid var(--border-soft);
  padding: 14px 16px;
  border-radius: 4px;
}
.metric .label { color: var(--faint); font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; }
.metric .value {
  font-family: var(--serif);
  font-size: 28px;
  font-weight: 400;
  color: var(--fg);
  margin-top: 6px;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.metric .value.up { color: var(--green); }
.metric .value.down { color: var(--red); }

.kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 20px; font-size: 13px; }
.kv .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 18px; }
.chip {
  background: var(--bg-inset);
  border: 1px solid var(--border-soft);
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.04em;
  cursor: pointer;
  user-select: none;
}
.chip.active {
  background: var(--fg); color: var(--bg); border-color: var(--fg);
}
.search {
  width: 100%; padding: 10px 14px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  color: var(--fg);
  border-radius: 4px;
  font-family: var(--sans);
}
.search:focus { outline: none; border-color: var(--accent); }

/* Timeline */
.timeline { display: flex; flex-direction: column; gap: 8px; }
.event {
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-left: 3px solid var(--border);
  border-radius: 4px;
  padding: 12px 16px;
}
.event.text { border-left-color: var(--blue); }
.event.thinking { border-left-color: var(--purple); background: var(--bg-inset); }
.event.tool_use { border-left-color: var(--yellow); }
.event.tool_result { border-left-color: var(--green); }
.event.tool_result.error { border-left-color: var(--red); }
.event.system_init, .event.system, .event.result { border-left-color: var(--faint); background: var(--bg-inset); }
.event header {
  display: flex; align-items: center; gap: 10px;
  font-size: 11px; color: var(--muted);
  margin-bottom: 8px;
  letter-spacing: 0.04em;
}
.event header .badge {
  background: var(--bg-inset);
  padding: 2px 8px; border-radius: 3px; font-size: 10px;
  color: var(--fg);
  text-transform: uppercase; letter-spacing: 0.1em;
}
.event header .badge.tool { background: rgba(163,120,32,0.18); color: #6b4d10; font-weight: 600; }
.event header .badge.tool.result { background: rgba(59,125,90,0.18); color: var(--green); }
.event header .badge.tool.error { background: rgba(176,58,46,0.18); color: var(--red); }
.event header .turn { color: var(--muted); }
.event header .ts { margin-left: auto; font-family: var(--mono); }
.event header .elapsed { color: var(--accent-deep); margin-left: 6px; font-size: 11px; font-family: var(--mono); }
.event pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg-inset);
  border: 1px solid var(--border-soft);
  border-radius: 3px;
  padding: 10px 12px;
  font-size: 12.5px;
  color: var(--fg);
  max-height: 480px;
  overflow: auto;
}
.event pre.cmd { background: var(--bg); }
.event .body.collapsed pre { max-height: 96px; }
.event .toggle {
  font-size: 11px;
  color: var(--accent-deep);
  cursor: pointer;
  margin-top: 6px;
  display: inline-block;
  letter-spacing: 0.06em;
}
.event details summary { cursor: pointer; color: var(--muted); font-size: 12px; }
details > summary { list-style: none; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸ "; color: var(--faint); }
details[open] > summary::before { content: "▾ "; }

.scoreline {
  display: flex; align-items: baseline; gap: 12px;
  padding: 6px 0;
  border-bottom: 1px dashed var(--border-soft);
  font-size: 13px;
}
.scoreline:last-child { border-bottom: none; }
.scoreline .ts { color: var(--faint); width: 170px; font-family: var(--mono); }
.scoreline .acc { font-weight: 600; width: 80px; font-family: var(--mono); color: var(--fg); }
.scoreline .cmd { color: var(--muted); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--mono); font-size: 12px; }

.tag {
  font-size: 10px; padding: 2px 8px;
  border-radius: 3px;
  background: var(--bg-inset);
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 600;
  border: 1px solid var(--border-soft);
}
.tag.cond-A { background: rgba(61,106,153,0.12); color: #2a4d75; border-color: rgba(61,106,153,0.3); }
.tag.cond-B { background: rgba(59,125,90,0.12); color: var(--green); border-color: rgba(59,125,90,0.3); }
.tag.cond-C { background: rgba(201,122,45,0.12); color: var(--orange); border-color: rgba(201,122,45,0.3); }
.tag.cond-D { background: rgba(110,61,157,0.12); color: var(--purple); border-color: rgba(110,61,157,0.3); }
.tag.cond-E { background: rgba(176,58,46,0.12); color: var(--red); border-color: rgba(176,58,46,0.3); }
.tag.cond-F { background: rgba(45,37,22,0.12); color: var(--accent-deep); border-color: rgba(45,37,22,0.3); }

.status { font-size: 10px; padding: 3px 9px; border-radius: 3px;
          letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;
          border: 1px solid transparent; }
.status.complete, .status.success {
  background: rgba(59,125,90,0.14); color: var(--green); border-color: rgba(59,125,90,0.3);
}
.status.agent_failed, .status.failed, .status.error {
  background: rgba(176,58,46,0.14); color: var(--red); border-color: rgba(176,58,46,0.3);
}
.status.no_post, .status.unknown { background: var(--bg-inset); color: var(--muted); border-color: var(--border-soft); }

.version-row {
  display: grid;
  grid-template-columns: 56px 84px 80px 1fr;
  gap: 12px; align-items: baseline;
  padding: 10px 14px; background: var(--bg-inset);
  border: 1px solid var(--border-soft); border-left: 3px solid var(--accent-deep);
  border-radius: 4px;
}
.version-row.base { border-left-color: var(--faint); opacity: .9; }
.version-row.no_eval { border-left-color: var(--yellow); }
.version-row .vlabel { font-family: var(--serif); font-weight: 500; font-size: 16px; color: var(--fg); }
.version-row .vacc { font-family: var(--mono); font-size: 15px; font-weight: 600; }
.version-row .vacc.up { color: var(--green); } .version-row .vacc.down { color: var(--red); }
.version-row .vdelta { font-family: var(--mono); font-size: 12px; }
.version-row .vdelta.up { color: var(--green); } .version-row .vdelta.down { color: var(--red); }
.version-row .vcmd { color: var(--muted); font-family: var(--mono); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.version-row a.jump { color: var(--accent-deep); margin-left: 6px; font-size: 11px; }

.gap-row {
  display: grid;
  grid-template-columns: 56px 1fr;
  gap: 12px; padding: 6px 10px 10px 10px;
  font-size: 12px; color: var(--muted);
  border-left: 3px dashed var(--border); margin: 4px 0 4px 12px;
}
.gap-row .glabel { color: var(--faint); font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; }
.gap-row .gbody { display: flex; flex-direction: column; gap: 4px; }
.gap-row .gbody .row { display: flex; flex-wrap: wrap; gap: 8px; }
.gap-row .gbody .pill {
  background: var(--bg-inset); border: 1px solid var(--border-soft);
  padding: 1px 8px; border-radius: 999px; font-size: 11px;
  font-family: var(--mono);
}
.gap-row .gbody .pill.err { background: rgba(176,58,46,0.10); color: var(--red); border-color: rgba(176,58,46,0.3); }
.gap-row .gbody .note { color: var(--fg); font-style: italic; }
.versions-list { display: flex; flex-direction: column; gap: 6px; }

/* ── Parabellum-style page header (trace run page + index) ──────── */
.page-header {
  margin-bottom: 36px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}
.page-header .eyebrow { margin-bottom: 14px; }
.page-header h1 {
  font-family: var(--serif);
  font-weight: 300;
  font-size: 38px;
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--fg);
  margin: 0;
}
.page-header .muted, .page-header p {
  color: var(--muted);
  font-size: 13px;
  margin-top: 14px;
  max-width: 720px;
}

.run-title-row {
  display: flex; align-items: center; gap: 16px;
  flex-wrap: wrap; margin-bottom: 22px;
}
.run-title-row h1 {
  font-family: var(--mono);
  font-weight: 400;
  font-size: 18px;
  margin: 0;
  letter-spacing: 0;
}
.run-pills { display: flex; gap: 8px; }

.header-stats { display: flex; gap: 10px; margin: 16px 0 8px; flex-wrap: wrap; }
.stat-pill {
  background: var(--bg-inset);
  border: 1px solid var(--border-soft);
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
}
.stat-pill b { color: var(--fg); font-weight: 600; font-variant-numeric: tabular-nums; margin-right: 4px; }

.meta-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  margin-top: 16px;
}
.meta-block { }
.meta-block-head {
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--faint);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-soft);
}
.meta-row {
  display: grid;
  grid-template-columns: 110px 1fr;
  gap: 12px;
  padding: 5px 0;
  font-size: 13px;
  border-bottom: 1px dashed var(--border-soft);
}
.meta-row:last-child { border-bottom: none; }
.meta-row .k {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  align-self: center;
}
.meta-row .v {
  color: var(--fg);
  font-size: 12px;
  word-break: break-word;
}

/* Section headers between cards */
.section-header { margin: 48px 0 18px; }
.section-header:first-of-type { margin-top: 16px; }
.section-header h2 {
  font-family: var(--serif);
  font-weight: 400;
  font-size: 26px;
  letter-spacing: -0.01em;
  color: var(--fg);
  margin: 0;
}
.section-header .eyebrow { margin-bottom: 10px; }
.title-counter {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--faint);
  letter-spacing: 0.04em;
}

/* Eyebrow (shared with character dashboard) */
.eyebrow {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.eyebrow::before {
  content: "";
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent-deep);
  display: inline-block;
}
.eyebrow .num {
  color: var(--faint);
  font-variant-numeric: tabular-nums;
  margin-right: 4px;
}

/* ── Trace index runs list polish ───────────────────────────────── */
.run-list { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.run-list table { width: 100%; border-collapse: collapse; }
.run-list th { background: var(--bg); }
.run-list.trace-runs th {
  padding: 14px 18px;
  text-align: left;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 500;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
}
.run-list.trace-runs td {
  padding: 16px 18px;
  border-bottom: 1px solid var(--border-soft);
  vertical-align: middle;
  font-size: 13px;
}
.run-list.trace-runs tr:last-child td { border-bottom: none; }
.run-list.trace-runs tr:hover td { background: var(--bg-inset); }
.run-list.trace-runs td.ts-col { color: var(--muted); font-size: 12px; white-space: nowrap; }
.run-list.trace-runs td.run-cell { min-width: 320px; }
.run-list.trace-runs .run-name {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--fg);
  border: none;
}
.run-list.trace-runs .run-name:hover { color: var(--accent); }
.run-list.trace-runs .run-meta {
  display: flex;
  gap: 8px;
  margin-top: 4px;
  font-size: 11px;
  color: var(--faint);
  font-family: var(--mono);
  flex-wrap: wrap;
}
.run-list.trace-runs .run-meta .sep { color: var(--border); }
.run-list.trace-runs td.cond-col,
.run-list.trace-runs td.bench-col { white-space: nowrap; }
.run-list.trace-runs td.bench-col { font-family: var(--mono); font-size: 12px; color: var(--muted); }
.run-list.trace-runs td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }
.run-list.trace-runs .trace-link {
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  border: none;
  white-space: nowrap;
}
.run-list.trace-runs .trace-link:hover { color: var(--accent); }

.dim { color: var(--faint); }

/* ── Filter rows on run page ────────────────────────────────────── */
.filters { display: flex; flex-direction: column; gap: 10px; margin: 18px 0; }
.filter-row {
  display: flex;
  align-items: center;
  gap: 14px;
  padding-bottom: 10px;
  border-bottom: 1px dashed var(--border-soft);
}
.filter-row:last-child { border-bottom: none; padding-bottom: 0; }
.filter-label {
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  min-width: 70px;
}
.filter-label small { display: block; font-size: 9px; opacity: 0.7; text-transform: none; letter-spacing: 0.04em; margin-top: 2px; font-weight: 400; }
.chips.inline { margin: 0; flex: 1; }
"""

JS = """
function toggleEventBody(btn) {
  const body = btn.previousElementSibling;
  body.classList.toggle('collapsed');
  btn.textContent = body.classList.contains('collapsed') ? 'Show more' : 'Show less';
}
function applyFilters() {
  const search = document.getElementById('searchBox').value.toLowerCase();
  const activeKinds = new Set(
    Array.from(document.querySelectorAll('.chip.kind.active')).map(c => c.dataset.kind)
  );
  const activeTools = new Set(
    Array.from(document.querySelectorAll('.chip.tool.active')).map(c => c.dataset.tool)
  );
  const versionChips = document.querySelectorAll('.chip.version');
  const activeVersions = new Set(
    Array.from(document.querySelectorAll('.chip.version.active')).map(c => c.dataset.version)
  );
  const events = document.querySelectorAll('.event');
  let shown = 0;
  events.forEach(ev => {
    const kind = ev.dataset.kind;
    const tool = ev.dataset.tool || '';
    const version = ev.dataset.version || '';
    let match = activeKinds.has(kind);
    if (match && (kind === 'tool_use' || kind === 'tool_result') && activeTools.size > 0) {
      match = activeTools.has(tool);
    }
    if (match && versionChips.length > 0) {
      match = activeVersions.has(version);
    }
    if (match && search) {
      match = ev.textContent.toLowerCase().includes(search);
    }
    ev.style.display = match ? '' : 'none';
    if (match) shown += 1;
  });
  const counter = document.getElementById('shownCount');
  if (counter) counter.textContent = shown + ' / ' + events.length + ' events';
}
function setupChips() {
  document.querySelectorAll('.chip').forEach(c => {
    c.addEventListener('click', () => {
      c.classList.toggle('active');
      applyFilters();
    });
  });
  const search = document.getElementById('searchBox');
  if (search) search.addEventListener('input', applyFilters);
  applyFilters();
}
function setupSortable() {
  const tables = document.querySelectorAll('table.sortable');
  tables.forEach(table => {
    const headers = table.querySelectorAll('thead th');
    headers.forEach((th, idx) => {
      th.addEventListener('click', () => sortTable(table, idx, th));
    });
  });
}
function sortTable(table, colIdx, th) {
  const type = th.dataset.type || 'text';
  const currentDir = th.dataset.dir;
  const dir = currentDir === 'asc' ? 'desc' : 'asc';
  // Clear other headers
  table.querySelectorAll('thead th').forEach(h => {
    if (h !== th) {
      h.removeAttribute('data-dir');
      h.textContent = h.textContent.replace(/[ ▾▴]+$/, '');
    }
  });
  th.dataset.dir = dir;
  th.textContent = th.textContent.replace(/[ ▾▴]+$/, '') + (dir === 'asc' ? ' ▴' : ' ▾');

  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {
    const av = a.children[colIdx]?.dataset.sort ?? a.children[colIdx]?.textContent ?? '';
    const bv = b.children[colIdx]?.dataset.sort ?? b.children[colIdx]?.textContent ?? '';
    let cmp;
    if (type === 'num') {
      const an = av === '' ? -Infinity : parseFloat(av);
      const bn = bv === '' ? -Infinity : parseFloat(bv);
      cmp = an - bn;
    } else {
      cmp = String(av).localeCompare(String(bv));
    }
    return dir === 'asc' ? cmp : -cmp;
  });
  rows.forEach(r => tbody.appendChild(r));
}
document.addEventListener('DOMContentLoaded', () => { setupChips(); setupSortable(); });
"""


def _topnav(view: str, run_id: str | None = None) -> str:
    """Shared top nav: brand left, two view-tabs right. `view` is 'trace' or
    'character'. When `run_id` is set both tabs deep-link into that run."""
    if run_id:
        char_href = f"/run/{run_id}"
        trace_href = f"/trace/{run_id}"
    else:
        char_href = "/"
        trace_href = "/trace"
    char_active = "active" if view == "character" else ""
    trace_active = "active" if view == "trace" else ""
    crumb = f'<span class="crumb">{esc(run_id)}</span>' if run_id else ""
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


def page(title: str, body: str, view: str = "trace", run_id: str | None = None) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
{_topnav(view, run_id)}
<div class="container">
{body}
</div>
<script>{JS}</script>
</body>
</html>
"""


def _dash_or(v, formatter=None):
    """Render `—` for missing/None/?, else apply formatter."""
    if v is None or v == "" or v == "?":
        return '<span class="dim">—</span>'
    if formatter:
        return formatter(v)
    return esc(v)


def render_index(runs: list[dict[str, Any]]) -> str:
    runs = sorted(runs, key=started_sort_key, reverse=True)

    # Counts for header summary
    n_total = len(runs)
    n_complete = sum(1 for r in runs if r["status"] == "complete")
    n_with_delta = sum(1 for r in runs if isinstance(r["delta"], (int, float)))
    n_failed = sum(1 for r in runs if r["status"] in ("agent_failed", "failed", "error"))

    rows = []
    for r in runs:
        cond_raw = r.get("condition") or "?"
        cond_tag = (f'<span class="tag cond-{esc(cond_raw)}">{esc(cond_raw)}</span>'
                    if cond_raw and cond_raw != "?" else '<span class="dim">—</span>')
        status = esc(r["status"]).replace(" ", "_")
        started_key = started_sort_key(r)
        delta = r["delta"] if isinstance(r["delta"], (int, float)) else ""
        dur = r["duration_s"] if isinstance(r["duration_s"], (int, float)) else ""
        lines = r["trace_lines"] if isinstance(r["trace_lines"], (int, float)) else ""
        n_vers = r.get("n_versions") or 0
        vers_display = esc(n_vers) if n_vers else '<span class="dim">—</span>'
        student_short = (r.get("student") or "").split("/")[-1] or "?"
        bench = r.get("benchmark") or "?"
        teacher = (r.get("teacher") or "").split("/")[-1] or "?"

        meta_sub = (
            f'<div class="run-meta">'
            f'<span>{_dash_or(student_short)}</span>'
            f'<span class="sep">·</span>'
            f'<span>teacher {_dash_or(teacher)}</span>'
            f'<span class="sep">·</span>'
            f'<span>{_dash_or(dur, lambda v: fmt_duration(v))}</span>'
            f'<span class="sep">·</span>'
            f'<span>{_dash_or(lines, lambda v: f"{v:,} lines")}</span>'
            f'</div>'
        )

        rows.append(
            f"""<tr>
  <td data-sort="{esc(started_key)}" class="mono ts-col">{esc(fmt_started(r['started_at']))}</td>
  <td data-sort="{esc(r['name'])}" class="run-cell">
    <a class="run-name" href="/run/{esc(r['name'])}">{esc(r['name'])}</a>
    {meta_sub}
  </td>
  <td data-sort="{esc(cond_raw)}" class="cond-col">{cond_tag}</td>
  <td data-sort="{esc(bench)}" class="bench-col">{_dash_or(bench)}</td>
  <td data-sort="{esc(delta)}" class="num">{fmt_delta(r['delta']) if isinstance(r['delta'],(int,float)) else '<span class=dim>—</span>'}</td>
  <td data-sort="{esc(n_vers)}" class="num">{vers_display}</td>
  <td data-sort="{esc(r['status'])}"><span class="status {status}">{esc(r['status'])}</span></td>
  <td class="link-col"><a class="trace-link" href="/trace/{esc(r['name'])}">trace →</a></td>
</tr>"""
        )

    body = f"""
<header class="page-header">
  <div class="eyebrow"><span class="num">§</span>Agent traces · {n_total} runs</div>
  <h1>Agent decisions, score progression, action timelines</h1>
  <div class="header-stats">
    <span class="stat-pill"><b>{n_complete}</b> complete</span>
    <span class="stat-pill"><b>{n_with_delta}</b> with Δ</span>
    <span class="stat-pill"><b>{n_failed}</b> failed</span>
  </div>
  <p class="muted">Click any row to open the agent trace. Use the <strong>Character</strong> tab in the nav to flip to the post-hoc per-eval shift dashboard for the same run. Sortable — click a column header.</p>
</header>

<div class="run-list trace-runs">
  <table class="runs sortable" id="runsTable">
    <thead>
      <tr>
        <th data-type="text" data-dir="desc">Started ▾</th>
        <th data-type="text">Run</th>
        <th data-type="text">Cond</th>
        <th data-type="text">Bench</th>
        <th data-type="num">Δ</th>
        <th data-type="num">Vers</th>
        <th data-type="text">Status</th>
        <th></th>
      </tr>
    </thead>
    <tbody>{''.join(rows) or '<tr><td colspan=8 class="muted">No runs found</td></tr>'}</tbody>
  </table>
</div>
"""
    return page("Runs", body, view="trace")


def render_metric_cards(pre: dict, post: dict, delta: Any) -> str:
    pre_acc = pre.get("accuracy") if isinstance(pre, dict) else None
    post_acc = post.get("accuracy") if isinstance(post, dict) else None
    delta_class = ""
    if isinstance(delta, (int, float)):
        delta_class = "up" if delta >= 0 else "down"
    return f"""
<div class="metric-grid">
  <div class="metric">
    <div class="label">Pre accuracy</div>
    <div class="value">{fmt_acc(pre_acc)}</div>
  </div>
  <div class="metric">
    <div class="label">Post accuracy</div>
    <div class="value">{fmt_acc(post_acc)}</div>
  </div>
  <div class="metric">
    <div class="label">Delta</div>
    <div class="value {delta_class}">{fmt_delta(delta)}</div>
  </div>
</div>
"""


def render_extras_table(extras_pre: dict, extras_post: dict) -> str:
    keys = sorted(set(extras_pre) | set(extras_post))
    if not keys:
        return ""
    rows = []
    for k in keys:
        p = extras_pre.get(k, {}).get("accuracy") if isinstance(extras_pre.get(k), dict) else None
        po = extras_post.get(k, {}).get("accuracy") if isinstance(extras_post.get(k), dict) else None
        delta = (po - p) if isinstance(p, (int, float)) and isinstance(po, (int, float)) else None
        rows.append(
            f"<tr><td>{esc(k)}</td><td class='num'>{fmt_acc(p)}</td>"
            f"<td class='num'>{fmt_acc(po)}</td><td class='num'>{fmt_delta(delta)}</td></tr>"
        )
    return f"""
<h3>Extra evals</h3>
<table class="runs">
  <thead><tr><th>Benchmark</th><th>Pre</th><th>Post</th><th>Δ</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def _fmt_gap_duration(s: float) -> str:
    if s <= 0:
        return "<1s"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.0f}m"
    return f"{s/3600:.1f}h"


def render_version_row(v: dict[str, Any], prev_acc: float | None) -> str:
    label = v["label"]
    klass = "base" if v["is_base"] else ("no_eval" if v["status"] == "no_eval" else "")
    acc = v.get("accuracy")
    acc_class = ""
    if isinstance(prev_acc, (int, float)) and isinstance(acc, (int, float)):
        acc_class = "up" if acc >= prev_acc else "down"
    delta = None
    if isinstance(prev_acc, (int, float)) and isinstance(acc, (int, float)):
        delta = acc - prev_acc
    delta_class = "up" if isinstance(delta, (int, float)) and delta >= 0 else "down"
    delta_html = f"<span class='vdelta {delta_class}'>{fmt_delta(delta)}</span>" if delta is not None else "<span class='vdelta muted'>-</span>"
    cmd = v.get("train_cmd") or ("pre-training eval" if v["is_base"] else "(no train cmd captured)")
    limit_tag = ""
    if v.get("limit"):
        limit_tag = f" <span class='tag'>limit={esc(v['limit'])}</span>"
    jump = ""
    if v.get("train_idx") is not None:
        jump = f" <a class='jump' href='#item-{esc(v['train_idx'])}'>jump ↓</a>"
    elif v.get("eval_idx") is not None:
        jump = f" <a class='jump' href='#item-{esc(v['eval_idx'])}'>jump ↓</a>"
    status_tag = ""
    if v["status"] == "no_eval":
        status_tag = " <span class='tag' style='color:var(--yellow)'>no eval</span>"
    elif v["status"] == "no_score":
        status_tag = " <span class='tag muted'>score unknown</span>"
    return (
        f"<div class='version-row {klass}' id='{esc(label.lower())}'>"
        f"<span class='vlabel'>{esc(label)}</span>"
        f"<span class='vacc {acc_class}'>{fmt_acc(acc)}</span>"
        f"{delta_html}"
        f"<span class='vcmd mono'>{esc(cmd)}{limit_tag}{status_tag}{jump}</span>"
        f"</div>"
    )


def render_gap_row(gap: dict[str, Any]) -> str:
    if not gap:
        return ""
    tcs = gap.get("tool_counts") or {}
    tally = " ".join(
        f"<span class='pill'>{esc(k)} ×{esc(v)}</span>"
        for k, v in sorted(tcs.items(), key=lambda kv: -kv[1])
    )
    errs = gap.get("errors") or {}
    err_pill = ""
    if errs.get("count", 0) > 0:
        err_pill = f"<span class='pill err'>{esc(errs['count'])} error{'s' if errs['count'] != 1 else ''}</span>"
    data_html = ""
    if gap.get("data_fetches"):
        items = "".join(f"<span class='pill'>📥 {esc(d)}</span>" for d in gap["data_fetches"])
        data_html = f"<div class='row'>{items}</div>"
    cfg_html = ""
    if gap.get("config_edits"):
        items = "".join(f"<span class='pill'>✎ {esc(f)}</span>" for f in gap["config_edits"])
        cfg_html = f"<div class='row'>{items}</div>"
    err_sample = ""
    if errs.get("sample"):
        err_sample = f"<div class='note mono' style='color:var(--red)'>{esc(errs['sample'])}</div>"
    note = ""
    if gap.get("agent_note"):
        note = f"<div class='note'>“{esc(gap['agent_note'])}”</div>"
    gap_s = gap.get("gap_s") or 0.0
    return (
        f"<div class='gap-row'>"
        f"<span class='glabel'>{esc(_fmt_gap_duration(gap_s))}</span>"
        f"<div class='gbody'>"
        f"<div class='row'>{tally}{err_pill}</div>"
        f"{data_html}{cfg_html}{err_sample}{note}"
        f"</div></div>"
    )


def render_versions_card(
    versions: list[dict[str, Any]], gaps: list[dict[str, Any]]
) -> str:
    if not versions:
        return ""
    rows: list[str] = []
    prev_acc: float | None = None
    for i, v in enumerate(versions):
        rows.append(render_version_row(v, prev_acc))
        # Insert the gap *after* this version (i.e. before the next one).
        if i < len(gaps) and gaps[i]:
            rows.append(render_gap_row(gaps[i]))
        if isinstance(v.get("accuracy"), (int, float)):
            prev_acc = v["accuracy"]
    explain = (
        "<p class='muted' style='font-size:12px;margin:0 0 10px'>"
        "Versions inferred from the trace: each <code>train.py</code>/<code>trainer.train</code>-like "
        "Bash call starts a new version; its accuracy is the last "
        "<code>evaluate.py</code> result before the next training run. "
        "Between-version rows summarise the agent's intervening actions."
        "</p>"
    )
    return (
        f"<div class='card'>"
        f"<h2>Model versions ({len(versions)})</h2>"
        f"{explain}"
        f"<div class='versions-list'>{''.join(rows)}</div>"
        f"</div>"
    )


def render_intermediate_scores(scores: list[dict[str, Any]]) -> str:
    if not scores:
        return '<p class="muted">No intermediate <code>evaluate.py</code> / <code>score.sh</code> calls detected in trace.</p>'
    rows = []
    for s in scores:
        rows.append(
            f"<div class='scoreline'>"
            f"<span class='ts mono'>{esc(fmt_event_ts(s['ts']) or '-')}</span>"
            f"<span class='acc'>{fmt_acc(s['accuracy'])}</span>"
            f"<span class='tag'>limit={esc(s['limit'] or 'full')}</span>"
            f"<span class='cmd mono'>{esc(s['cmd'])}</span>"
            f"</div>"
        )
    return "".join(rows)


def fmt_tool_input(name: str, payload: Any) -> tuple[str, str]:
    """Return (lang_class, formatted_text) for a tool_use input."""
    if not isinstance(payload, dict):
        return "json", json.dumps(payload, indent=2, ensure_ascii=False)
    if name == "Bash":
        cmd = payload.get("command", "")
        desc = payload.get("description")
        prefix = f"# {desc}\n" if desc else ""
        return "shell", prefix + str(cmd)
    if name in ("Edit", "MultiEdit"):
        path = payload.get("file_path", "")
        old = payload.get("old_string", "")
        new = payload.get("new_string", "")
        return "diff", f"--- {path}\n--- old\n{old}\n+++ new\n{new}"
    if name == "Write":
        path = payload.get("file_path", "")
        content = payload.get("content", "")
        return "text", f"# {path}\n{content}"
    if name == "Read":
        path = payload.get("file_path", "")
        offset = payload.get("offset")
        limit = payload.get("limit")
        suffix = ""
        if offset or limit:
            suffix = f"  (offset={offset} limit={limit})"
        return "text", f"{path}{suffix}"
    if name in ("Grep", "Glob"):
        return "json", json.dumps(payload, indent=2, ensure_ascii=False)
    return "json", json.dumps(payload, indent=2, ensure_ascii=False)


def maybe_truncate(text: str) -> tuple[str, bool]:
    if len(text) <= TOOL_RESULT_TRUNC:
        return text, False
    head = text[: TOOL_RESULT_TRUNC // 2]
    tail = text[-TOOL_RESULT_TRUNC // 2 :]
    return f"{head}\n\n... [{len(text) - TOOL_RESULT_TRUNC} chars elided] ...\n\n{tail}", True


def render_event(item: dict[str, Any], idx: int = 0) -> str:
    kind = item["kind"]
    ts_text = fmt_event_ts(item.get("ts") or "")
    elapsed_text = fmt_elapsed(item.get("_elapsed_s"))
    ts = esc(ts_text)
    if elapsed_text:
        ts = f"{ts} <span class='elapsed'>{esc(elapsed_text)}</span>"
    turn = item.get("turn")
    turn_str = f"<span class='turn'>turn {esc(turn)}</span>" if turn else ""
    ver = esc(item.get("_version") or "")
    ver_attr = f" data-version='{ver}'" if ver else ""
    # Use idx-based anchor so version-row jump links land precisely on the
    # event. Add a turn-N alias on the first event of each turn for
    # human-friendly anchors.
    id_attr = f" id='item-{idx}'"

    if kind == "system_init":
        body = (
            f"<div class='kv'>"
            f"<span class='k'>Model</span><span class='mono'>{esc(item.get('model'))}</span>"
            f"<span class='k'>Session</span><span class='mono'>{esc(item.get('session_id'))}</span>"
            f"<span class='k'>Cwd</span><span class='mono'>{esc(item.get('cwd'))}</span>"
            f"<span class='k'>Tools</span><span class='mono'>{esc(', '.join(item.get('tools') or []))}</span>"
            f"</div>"
        )
        return f"<div{id_attr} class='event system_init' data-kind='system_init'{ver_attr}><header><span class='badge'>system init</span><span class='ts mono'>{ts}</span></header>{body}</div>"

    if kind == "system":
        return (
            f"<div{id_attr} class='event system' data-kind='system'{ver_attr}>"
            f"<header><span class='badge'>system</span><span class='muted'>{esc(item.get('subtype'))}</span><span class='ts mono'>{ts}</span></header>"
            f"<pre>{esc(json.dumps(item.get('data'), indent=2, ensure_ascii=False))}</pre>"
            f"</div>"
        )

    if kind == "result":
        return (
            f"<div{id_attr} class='event result' data-kind='result'{ver_attr}>"
            f"<header><span class='badge'>result</span><span class='muted'>{esc(item.get('subtype'))}</span><span class='ts mono'>{ts}</span></header>"
            f"<pre>{esc(json.dumps(item.get('data'), indent=2, ensure_ascii=False))}</pre>"
            f"</div>"
        )

    if kind == "text":
        role = item.get("role", "assistant")
        text = item.get("text", "") or ""
        return (
            f"<div{id_attr} class='event text' data-kind='text'{ver_attr}>"
            f"<header><span class='badge'>{esc(role)}</span>{turn_str}<span class='ts mono'>{ts}</span></header>"
            f"<pre>{esc(text)}</pre>"
            f"</div>"
        )

    if kind == "thinking":
        text = item.get("text", "") or ""
        if not text.strip():
            # Claude Code's --output-format stream-json strips the thinking
            # body and only keeps the signature, so the block is genuinely
            # empty here. Render a one-line marker rather than a fake
            # expandable.
            return (
                f"<div{id_attr} class='event thinking' data-kind='thinking'{ver_attr}>"
                f"<header><span class='badge'>thinking</span>{turn_str}"
                f"<span class='muted'>redacted by Claude Code stream-json (signature only)</span>"
                f"<span class='ts mono'>{ts}</span></header>"
                f"</div>"
            )
        return (
            f"<div{id_attr} class='event thinking' data-kind='thinking'{ver_attr}>"
            f"<header><span class='badge'>thinking</span>{turn_str}<span class='ts mono'>{ts}</span></header>"
            f"<details><summary>show thinking ({len(text)} chars)</summary>"
            f"<pre>{esc(text)}</pre></details>"
            f"</div>"
        )

    if kind == "tool_use":
        name = item.get("name", "tool")
        _, formatted = fmt_tool_input(name, item.get("input"))
        body, truncated = maybe_truncate(formatted)
        toggle = (
            "<span class='toggle' onclick='toggleEventBody(this)'>Show more</span>"
            if truncated
            else ""
        )
        cmd_class = "cmd" if name == "Bash" else ""
        return (
            f"<div{id_attr} class='event tool_use' data-kind='tool_use' data-tool='{esc(name)}'{ver_attr}>"
            f"<header><span class='badge tool'>{esc(name)}</span>{turn_str}<span class='muted mono'>{esc(item.get('id'))}</span><span class='ts mono'>{ts}</span></header>"
            f"<div class='body'><pre class='{cmd_class}'>{esc(body)}</pre></div>{toggle}"
            f"</div>"
        )

    if kind == "tool_result":
        name = item.get("name", "tool")
        text = render_tool_result_text(item.get("content"))
        body, truncated = maybe_truncate(text)
        is_error = item.get("is_error")
        err_class = "error" if is_error else ""
        badge_class = "tool result" + (" error" if is_error else "")
        toggle = (
            "<span class='toggle' onclick='toggleEventBody(this)'>Show more</span>"
            if truncated
            else ""
        )
        return (
            f"<div{id_attr} class='event tool_result {err_class}' data-kind='tool_result' data-tool='{esc(name)}'{ver_attr}>"
            f"<header><span class='badge {badge_class}'>{esc(name)} → result</span>{turn_str}<span class='muted mono'>{esc(item.get('id'))}</span><span class='ts mono'>{ts}</span></header>"
            f"<div class='body collapsed'><pre>{esc(body)}</pre></div>{toggle}"
            f"</div>"
        )

    return (
        f"<div{id_attr} class='event other' data-kind='other'{ver_attr}>"
        f"<header><span class='badge'>{esc(kind)}</span></header>"
        f"<pre>{esc(json.dumps(item, indent=2, ensure_ascii=False))}</pre>"
        f"</div>"
    )


def render_run(run: dict[str, Any]) -> str:
    config = run["config"]
    summary = run["summary"]
    name = run["name"]
    cond = config.get("condition", "?")
    delta = summary.get("delta")

    events = parse_events(run["trace_path"])
    items = normalize_timeline(events)
    _annotate_elapsed(items)
    inter_scores = extract_intermediate_scores(items)
    versions = detect_versions(items)
    annotate_version_membership(items, versions)

    # Compute between-version activity summaries. gaps[i] = activity *before*
    # versions[i+1], i.e. between versions[i].eval/train and versions[i+1].train.
    gaps: list[dict[str, Any]] = []
    for i in range(len(versions) - 1):
        prev = versions[i]
        nxt = versions[i + 1]
        prev_end = prev.get("eval_idx") if prev.get("eval_idx") is not None else prev.get("train_idx")
        next_start = nxt.get("train_idx")
        if prev_end is None or next_start is None:
            gaps.append({})
            continue
        gaps.append(summarize_between_versions(items, prev_end, next_start))

    versions_card_html = render_versions_card(versions, gaps)

    # Tool-name set for filter chips
    tool_names = sorted(
        {it["name"] for it in items if it["kind"] in ("tool_use", "tool_result") and it.get("name")}
    )
    kind_set = sorted({it["kind"] for it in items})

    kinds_chips = "".join(
        f"<span class='chip kind active' data-kind='{esc(k)}'>{esc(k)}</span>" for k in kind_set
    )
    tools_chips = "".join(
        f"<span class='chip tool' data-tool='{esc(t)}'>{esc(t)}</span>" for t in tool_names
    )

    # Version chips: include "pre" only if any item is labelled pre.
    version_labels: list[str] = []
    if any(it.get("_version") == "pre" for it in items):
        version_labels.append("pre")
    for v in versions:
        version_labels.append(v["label"])
    versions_chips_html = ""
    if len(versions) >= 2:
        versions_chips_html = (
            "<div class='chips'>"
            "<span class='muted' style='font-size:11px;align-self:center'>Versions (filters timeline by trace segment):</span>"
            + "".join(
                f"<span class='chip version active' data-version='{esc(lbl)}'>{esc(lbl)}</span>"
                for lbl in version_labels
            )
            + "</div>"
        )

    timeline_html = "".join(render_event(it, idx=i) for i, it in enumerate(items))

    # Parabellum-style header eyebrow: condition + teacher + student + started
    eyebrow_bits = []
    if cond and cond != "?":
        eyebrow_bits.append(esc(cond))
    teacher = config.get("teacher_model")
    student = config.get("student_model")
    if teacher: eyebrow_bits.append(f"teacher {esc(teacher.split('/')[-1])}")
    if student: eyebrow_bits.append(f"student {esc(student.split('/')[-1])}")
    started_at = config.get("started_at")
    if started_at: eyebrow_bits.append(esc(str(started_at)[:16]))
    eyebrow_text = " · ".join(eyebrow_bits) or "Adapter run"

    status_raw = summary.get("status") or "unknown"
    status_class = esc(status_raw).replace(" ", "_")
    cond_chip = (f'<span class="tag cond-{esc(cond)}">{esc(cond)}</span>'
                 if cond and cond != "?" else "")

    # Two-column metadata grid: left = experiment config, right = execution
    def kv_block(title: str, pairs: list[tuple[str, Any]]) -> str:
        rows_html = "".join(
            f'<div class="meta-row"><span class="k">{esc(k)}</span>'
            f'<span class="v mono">{esc(v) if v not in (None, "") else "—"}</span></div>'
            for k, v in pairs
        )
        return (
            f'<div class="meta-block">'
            f'<div class="meta-block-head">{esc(title)}</div>'
            f'{rows_html}</div>'
        )

    config_block = kv_block("Experiment", [
        ("Condition", cond),
        ("Teacher", config.get("teacher_model")),
        ("Student", config.get("student_model")),
        ("Benchmark", config.get("benchmark")),
        ("Agent", config.get("agent")),
        ("Time budget", f"{config.get('time_budget_h', '?')}h"),
        ("Seed", config.get("seed")),
    ])
    exec_block = kv_block("Execution", [
        ("Started", config.get("started_at")),
        ("Duration", fmt_duration(summary.get("duration_s"))),
        ("Trace lines", summary.get("agent_trace_lines")),
        ("Image", config.get("base_image")),
        ("Git SHA", config.get("git_sha")),
        ("Pod", run["pod_meta"].get("pod_id")),
    ])

    heldout_html = ""
    if run["heldout_summary"]:
        heldout_html = (
            f"<div class='card'><h2>Held-out panel</h2>"
            f"<pre>{esc(json.dumps(run['heldout_summary'], indent=2, ensure_ascii=False))}</pre></div>"
        )

    rerun_html = ""
    if run["rerun_post"]:
        rerun_html = (
            f"<div class='card'><h2>Rerun-post summary</h2>"
            f"<pre>{esc(json.dumps(run['rerun_post'], indent=2, ensure_ascii=False))}</pre></div>"
        )

    body = f"""
<header class="page-header">
  <div class="eyebrow"><span class="num">RUN</span>{eyebrow_text}</div>
  <div class="run-title-row">
    <h1>{esc(name)}</h1>
    <div class="run-pills">
      {cond_chip}
      <span class="status {status_class}">{esc(status_raw)}</span>
    </div>
  </div>
  <div class="meta-grid">{config_block}{exec_block}</div>
</header>

<section class="section-header">
  <div class="eyebrow"><span class="num">§</span>Score progression</div>
  <h2>How the headline metric moved</h2>
</section>
<div class="card">
  {render_metric_cards(run['pre'], run['post'], delta)}
  {render_extras_table(run['extras_pre'], run['extras_post'])}
  <h3>Intermediate evals from agent trace</h3>
  {render_intermediate_scores(inter_scores)}
</div>

<section class="section-header">
  <div class="eyebrow"><span class="num">§</span>Model versions</div>
  <h2>What the agent built, version by version</h2>
</section>
{versions_card_html}

{heldout_html}
{rerun_html}

<section class="section-header">
  <div class="eyebrow"><span class="num">§</span>Briefing</div>
  <h2>Prompt the agent received</h2>
</section>
<div class="card">
  <details><summary>show prompt ({len(run['prompt']):,} chars)</summary>
    <pre>{esc(run['prompt'])}</pre>
  </details>
</div>

<section class="section-header">
  <div class="eyebrow"><span class="num">§</span>Action timeline</div>
  <h2>Every tool call, message, and reflection <span class="title-counter" id="shownCount"></span></h2>
</section>
<div class="card">
  <input id="searchBox" class="search" placeholder="Filter by text — file paths, command substrings, score values …" />
  <div class="filters">
    <div class="filter-row">
      <span class="filter-label">Kinds</span>
      <div class="chips inline">{kinds_chips}</div>
    </div>
    <div class="filter-row">
      <span class="filter-label">Tools <small>(filters tool_use + tool_result; empty = all)</small></span>
      <div class="chips inline">{tools_chips}</div>
    </div>
    {versions_chips_html}
  </div>
  <div class="timeline">{timeline_html or '<p class=muted>No events parsed.</p>'}</div>
</div>
"""
    return page(f"Run · {name}", body, view="trace", run_id=name)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    runs_dir: Path = DEFAULT_RUNS_DIR

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write(f"[viewer] {self.address_string()} {format % args}\n")

    def _send_html(self, html_text: str, status: int = 200) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self, msg: str = "not found") -> None:
        self._send_html(page("404", f"<div class='card'><h2>404</h2><p>{esc(msg)}</p></div>"), 404)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/" or path == "":
            runs = list_runs(self.runs_dir)
            self._send_html(render_index(runs))
            return

        if path.startswith("/run/"):
            name = path[len("/run/"):].strip("/")
            if not name or not RUN_NAME_RE.match(name):
                self._not_found("invalid run name")
                return
            run_dir = self.runs_dir / name
            if not run_dir.is_dir():
                self._not_found(f"run dir does not exist: {name}")
                return
            run = load_run(run_dir)
            self._send_html(render_run(run))
            return

        self._not_found(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Path to jobs/runs/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        sys.exit(f"runs dir not found: {runs_dir}")

    Handler.runs_dir = runs_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[viewer] Serving runs from {runs_dir}")
    print(f"[viewer] Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
