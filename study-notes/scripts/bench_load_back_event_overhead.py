"""Microbench: host-side overhead added by the load-back CUDA-event timing.

Compares the OLD instrumentation (one ``time.perf_counter()`` pair) against
the NEW one (two ``Event(enable_timing=True)`` allocations + two ``record()``
calls on a stream). Reports p50/p99/mean in microseconds.

Run on the target GPU box, e.g.::

    python scripts/bench_load_back_event_overhead.py --iters 100000

Throwaway script — not part of the package.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def _percentile(sorted_xs, p):
    idx = min(len(sorted_xs) - 1, int(len(sorted_xs) * p))
    return sorted_xs[idx]


def _summarize(label, samples_us):
    samples_us.sort()
    p50 = _percentile(samples_us, 0.50)
    p99 = _percentile(samples_us, 0.99)
    mean = statistics.fmean(samples_us)
    print(
        f"{label:<40s} mean={mean:8.3f} us  p50={p50:8.3f} us  p99={p99:8.3f} us"
    )
    return p50, p99, mean


def bench_old(iters: int):
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1e6)
    return out


def bench_new(iters: int, stream: torch.cuda.Stream):
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        e_start = torch.cuda.Event(enable_timing=True)
        e_finish = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e_start.record()
            e_finish.record()
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1e6)
    return out


def bench_event_alloc_only(iters: int):
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        torch.cuda.Event(enable_timing=True)
        torch.cuda.Event(enable_timing=True)
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1e6)
    return out


def bench_record_only(iters: int, stream: torch.cuda.Stream):
    e_start = torch.cuda.Event(enable_timing=True)
    e_finish = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            e_start.record()
            e_finish.record()
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1e6)
    return out


def bench_new_drained(iters: int, stream: torch.cuda.Stream, sync_every: int = 1):
    """Mimic realistic scenario: each merged op only enqueues 2 events; the
    event queue is drained by DMA completion before the next op arrives.

    We approximate that by ``torch.cuda.synchronize()`` every ``sync_every``
    iters so the event pool never deepens.
    """
    out = []
    for i in range(iters):
        if i % sync_every == 0:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        e_start = torch.cuda.Event(enable_timing=True)
        e_finish = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e_start.record()
            e_finish.record()
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1e6)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__} (cuda {torch.version.cuda})")
    print(f"iters={args.iters} warmup={args.warmup}")
    print()

    stream = torch.cuda.Stream()

    # warmup
    bench_new(args.warmup, stream)

    old = bench_old(args.iters)
    new = bench_new(args.iters, stream)
    alloc = bench_event_alloc_only(args.iters)
    rec = bench_record_only(args.iters, stream)
    # Realistic: queue drained by DMA completion between merged ops.
    drained = bench_new_drained(args.iters, stream, sync_every=1)

    p50_old, _, _ = _summarize("OLD (perf_counter pair)", old)
    p50_new, _, _ = _summarize("NEW back-to-back (queue saturates)", new)
    _summarize("  └ component: 2x Event(enable_timing=True)", alloc)
    _summarize("  └ component: 2x event.record()", rec)
    p50_drained, _, _ = _summarize("NEW drained (sync per op, realistic)", drained)
    print()
    print(
        f"Net hot-path delta per merged load-back op (saturated): "
        f"~{p50_new - p50_old:+.2f} us (p50)"
    )
    print(
        f"Net hot-path delta per merged load-back op (drained):   "
        f"~{p50_drained - p50_old:+.2f} us (p50)"
    )
    print(
        "Reference DMA cost from PR table (H20): 50MB->1.79ms, "
        "500MB->9.60ms (i.e. 1.79e3..9.60e3 us)."
    )


if __name__ == "__main__":
    main()
