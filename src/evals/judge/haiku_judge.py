"""Anthropic Haiku 4.5 judge.

Used by tasks that need an LLM-as-judge: sycophancy_slava, sycophancy_aisi,
spiralbench_mini. Returns a JSON object validated against a JSON schema.

The schema enforcement is done with the Anthropic SDK's `tools` API rather
than regex-parsing free-form output: the model is given a single tool whose
input matches the schema, and its `tool_use` block is the verdict.

Rate-limit visibility: each `.with_raw_response.*` call surfaces the
`anthropic-ratelimit-input-tokens-remaining` header. We sample those
headers periodically and emit a stderr warning whenever the input-tokens
budget drops below RATELIMIT_WARN_PCT of the cap so ITPM saturation
becomes visible in run.log instead of a silent slowdown.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get("HELDOUT_JUDGE_MODEL", "claude-haiku-4-5-20251001")
RATELIMIT_WARN_PCT = float(os.environ.get("HAIKUJUDGE_WARN_PCT", "0.20"))
RATELIMIT_LOG_EVERY = int(os.environ.get("HAIKUJUDGE_LOG_EVERY", "20"))


class HaikuJudge:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024,
                 max_retries: int = 10):
        # max_retries=10 — SDK respects 429 Retry-After headers with
        # exponential backoff; default of 2 retries gives up before a
        # 60s rate-limit window can clear. Required for ThreadPool callers
        # that fan out N concurrent judge calls.
        self.client = Anthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        # Thread-safe counters for cross-thread sampling.
        self._call_count = 0
        self._lock = threading.Lock()
        self._last_warn_ts = 0.0

    def _maybe_log_ratelimit(self, headers) -> None:
        """Sample rate-limit headers; warn if remaining tokens cross threshold.

        Anthropic returns headers:
          anthropic-ratelimit-input-tokens-limit
          anthropic-ratelimit-input-tokens-remaining
          anthropic-ratelimit-input-tokens-reset
        (same for output-tokens, requests). We watch input-tokens since
        that's what tripped past tier caps with large-prompt judges.
        """
        try:
            with self._lock:
                self._call_count += 1
                should_log = self._call_count % RATELIMIT_LOG_EVERY == 0
            in_limit = headers.get("anthropic-ratelimit-input-tokens-limit")
            in_remaining = headers.get("anthropic-ratelimit-input-tokens-remaining")
            if not in_limit or not in_remaining:
                return
            limit_n = int(in_limit)
            remaining_n = int(in_remaining)
            if limit_n <= 0:
                return
            pct = remaining_n / limit_n
            now = time.time()
            # Warn at low-remaining, but throttle so we don't spam.
            if pct < RATELIMIT_WARN_PCT and (now - self._last_warn_ts) > 5.0:
                reset = headers.get("anthropic-ratelimit-input-tokens-reset", "?")
                print(
                    f"[HaikuJudge] ITPM low: {remaining_n:,}/{limit_n:,} "
                    f"({pct:.1%}) remaining, resets {reset}",
                    file=sys.stderr, flush=True,
                )
                with self._lock:
                    self._last_warn_ts = now
            elif should_log:
                print(
                    f"[HaikuJudge] ITPM ok: {remaining_n:,}/{limit_n:,} "
                    f"({pct:.1%}) — sampled at call {self._call_count}",
                    file=sys.stderr, flush=True,
                )
        except Exception:
            # Header introspection must never break a judge call.
            pass

    def __call__(self, system: str, user: str, schema: dict) -> dict[str, Any]:
        raw = self.client.messages.with_raw_response.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {
                    "name": "submit_verdict",
                    "description": "Submit your verdict according to the schema.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "submit_verdict"},
            messages=[{"role": "user", "content": user}],
        )
        self._maybe_log_ratelimit(raw.headers)
        response = raw.parse()
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_verdict":
                return dict(block.input)
        raise RuntimeError(f"judge returned no tool_use block: {response}")
