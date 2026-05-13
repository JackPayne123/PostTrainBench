"""Shared progress + phase-timing helpers for the slow benches.

Goal: structured stdout logging so eval_<phase>_<bench>.log files surface
WHERE wall time goes. Today most custom-code benches (persona_traits,
spiralbench_mini, healthbench, political_bias_openai, sycophancy_aisi)
are silent through their judge phase; the pod-side run.log only sees a
single OK line at completion. With this helper we get:

    [progress] persona_traits gen           starting (n=1400, workers=8)
    [progress] persona_traits gen           done=140/1400  (10.0%)  rate=2.3/s  elapsed=60.8s  eta=549.2s
    [progress] persona_traits gen           done=280/1400  (20.0%)  rate=2.3/s  elapsed=121.5s  eta=488.4s
    ...
    [progress] persona_traits gen           done=1400/1400 (100%)   rate=2.3/s  elapsed=611.2s
    [progress] persona_traits judge         starting (n=1400, workers=4)
    [progress] persona_traits judge         done=350/1400  (25.0%)  rate=5.8/s  elapsed=60.1s  eta=180.3s
    ...

Lines are stdout-flushed so the inspect-ai/.log surface picks them up
even when log_realtime=False (which we keep off because it floods the
log).

Two patterns:

1. ProgressTimer — manual instrumentation around an explicit loop, e.g.
   `for ... in ...: tick()`. Use for ThreadPoolExecutor as_completed
   loops and synchronous per-sample loops.

2. async_progress_gather — for asyncio.gather() patterns, wraps awaitables
   with a counter and emits periodic progress.
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Awaitable, Callable, Iterable, TypeVar

T = TypeVar("T")


def _log(bench: str, phase: str, msg: str) -> None:
    """Stdout-flushed progress line with consistent prefix."""
    print(f"[progress] {bench} {phase:14s} {msg}", flush=True)


class ProgressTimer:
    """Tracks N/total progress and prints periodic + final lines.

    Usage:
        with ProgressTimer("persona_traits", "judge", total=1400,
                           workers=4, every_pct=5) as p:
            for ... in ...:
                ...
                p.tick()
    """

    def __init__(
        self,
        bench: str,
        phase: str,
        total: int,
        workers: int | None = None,
        every_pct: float = 10.0,
        every_seconds: float = 30.0,
    ):
        self.bench = bench
        self.phase = phase
        self.total = total
        self.workers = workers
        self.every_pct = every_pct
        self.every_seconds = every_seconds
        self.done = 0
        self.t0: float = 0.0
        self._last_pct_logged = 0.0
        self._last_time_logged = 0.0

    def __enter__(self) -> "ProgressTimer":
        self.t0 = time.monotonic()
        self._last_time_logged = self.t0
        workers_str = f", workers={self.workers}" if self.workers else ""
        _log(self.bench, self.phase, f"starting (n={self.total}{workers_str})")
        return self

    def tick(self, n: int = 1) -> None:
        """Advance the counter by `n` and emit a progress line if either
        the every_pct or every_seconds threshold has been crossed."""
        self.done += n
        now = time.monotonic()
        pct = (self.done / self.total) * 100 if self.total else 100.0
        pct_crossed = pct - self._last_pct_logged >= self.every_pct
        time_crossed = (now - self._last_time_logged) >= self.every_seconds
        if pct_crossed or time_crossed or self.done >= self.total:
            self._emit(now, pct)
            self._last_pct_logged = pct
            self._last_time_logged = now

    def _emit(self, now: float, pct: float) -> None:
        elapsed = now - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else 0
        eta_str = f"  eta={eta:.1f}s" if self.done < self.total else ""
        _log(
            self.bench, self.phase,
            f"done={self.done}/{self.total}  ({pct:.1f}%)  "
            f"rate={rate:.2f}/s  elapsed={elapsed:.1f}s{eta_str}"
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.monotonic() - self.t0
        if exc_type is not None:
            _log(self.bench, self.phase, f"FAILED after {elapsed:.1f}s ({exc_type.__name__}: {exc_val})")
            return
        if self.done == 0:
            _log(self.bench, self.phase, f"done — but tick was never called (elapsed {elapsed:.1f}s)")
            return
        # Always emit a final line, even if last tick already did.
        rate = self.done / elapsed if elapsed > 0 else 0
        _log(
            self.bench, self.phase,
            f"done={self.done}/{self.total} (100.0%)  rate={rate:.2f}/s  elapsed={elapsed:.1f}s  ✓"
        )


async def async_progress_gather(
    bench: str,
    phase: str,
    awaitables: list[Awaitable[T]],
    workers: int | None = None,
    every_pct: float = 10.0,
    every_seconds: float = 30.0,
) -> list[T]:
    """asyncio.gather() with periodic progress logging.

    Each awaitable is wrapped so that completion ticks the counter.
    Returns results in the same order as the input list.
    """
    total = len(awaitables)
    with ProgressTimer(bench, phase, total=total, workers=workers,
                       every_pct=every_pct, every_seconds=every_seconds) as p:
        async def _wrap(aw: Awaitable[T]) -> T:
            res = await aw
            p.tick()
            return res
        return await asyncio.gather(*(_wrap(a) for a in awaitables))


__all__ = ["ProgressTimer", "async_progress_gather"]
