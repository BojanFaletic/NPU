"""Minimal per-op wall-clock profiler. Zero cost when disabled.

Usage:
    from npu.profiler import profile, PROF

    PROF.enable()                  # turn on (once, e.g. at CLI parse)
    with profile("wqkv"):
        ...                        # code to time
    PROF.report()                  # aggregated table to stdout
    PROF.reset()                   # clear counters

When disabled, `profile(name)` returns a no-op context manager with no
allocation — the `with` block becomes a pair of attribute lookups.
"""
from __future__ import annotations
import time
from contextlib import contextmanager
from collections import defaultdict


class _Profiler:
    __slots__ = ("enabled", "_ns", "_cnt")

    def __init__(self):
        self.enabled = False
        self._ns: dict[str, int] = defaultdict(int)
        self._cnt: dict[str, int] = defaultdict(int)

    def enable(self):  self.enabled = True
    def disable(self): self.enabled = False

    def reset(self):
        self._ns.clear()
        self._cnt.clear()

    def add(self, name: str, ns: int):
        self._ns[name] += ns
        self._cnt[name] += 1

    def report(self, total_label: str = "total") -> None:
        if not self._ns:
            print("[profiler] no samples recorded")
            return
        rows = sorted(self._ns.items(), key=lambda kv: -kv[1])
        total = self._ns.get(total_label, sum(self._ns.values()))
        print(f"{'op':<24} {'total_ms':>10} {'calls':>7} {'ms/call':>10} {'%':>6}")
        for name, ns in rows:
            ms = ns / 1e6
            c = self._cnt[name]
            pct = 100.0 * ns / total if total > 0 else 0.0
            print(f"{name:<24} {ms:>10.2f} {c:>7d} {ms/c:>10.3f} {pct:>5.1f}%")


PROF = _Profiler()


class _TimerCM:
    __slots__ = ("name", "t0")
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.t0 = time.perf_counter_ns()
        return self
    def __exit__(self, *exc):
        PROF.add(self.name, time.perf_counter_ns() - self.t0)


class _NoopCM:
    __slots__ = ()
    def __enter__(self): return self
    def __exit__(self, *exc): return False


_NOOP = _NoopCM()


def profile(name: str):
    return _TimerCM(name) if PROF.enabled else _NOOP
