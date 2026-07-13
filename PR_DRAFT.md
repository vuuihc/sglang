<!--
PR draft for branch fix/load-back-metric-accurate-dma-time.
Suggested title:
  fix(hicache): measure load-back duration with CUDA events
Suggested CC line at the bottom of the PR description (or in a follow-up comment):
  cc @ShawnKung (original metric author, #10225)
     @fzyzcjy (observability CODEOWNER)
     @xiezhq-hermann @hzh0425 (mem_cache CODEOWNERS)
-->

## Motivation

`sglang:load_back_duration_seconds` is currently measured by wrapping `time.perf_counter()` around the synchronous Python dispatch in `HiRadixCache.load_back()`. Because the actual H2D transfer runs asynchronously on the load stream, the metric only captures CPU-side scheduling overhead (hundreds of μs) and completely misses the real DMA duration (ms scale), which is what users actually experience.

On NVIDIA H20 with a controlled pinned-memory H2D simulation (single-op, no merge):

| Payload | OLD (`perf_counter`, dispatch) | NEW (CUDA event, real DMA) |
|---------|-------------------------------:|---------------------------:|
| 50 MB   | ~0.50 ms                       | 1.79 ms                    |
| 100 MB  | ~0.58 ms                       | 3.05 ms                    |
| 200 MB  | ~0.69 ms                       | 4.34 ms                    |
| 500 MB  | ~0.80 ms                       | 9.60 ms                    |

OLD is roughly constant in dispatch overhead; NEW scales linearly with payload as expected for H2D bandwidth.

## Modifications

- `cache_controller.py`: add `timing_event_supported()` (one-shot capability probe, cached) and `make_timing_event()`. Allocate a dedicated `(ack_start_event, ack_finish_event)` pair per `start_loading()` call, both with `enable_timing=True`. The pair is intentionally not drawn from the recycled `LayerDoneCounter` pool so a stale ack cannot be overwritten when its slot is reused. Add `num_tokens` field to `HiCacheAck` (defaults to 0 for back-compat).
- `hiradix_cache.py` / `hi_mamba_radix_cache.py` / `unified_radix_cache.py`: in `loading_check()`, report token count and (when the probe passed) the CUDA `elapsed_time` once the DMA has completed. Remove the inaccurate `perf_counter` wrapper from `HiRadixCache.load_back()`. Refactor the existing tuple-unpack `for _, finish_event, ack_list in ...` to attribute access on `HiCacheAck` (forced by the new field).
- `hybrid_cache_controller.py`: same timing-event-pair pattern in `start_loading()`. Write path untouched.
- `metrics_collector.py`: documentation tweak only.

Backends where `Event(enable_timing=True)` is unsupported log a single startup warning, still report `num_tokens`, and silently skip the duration observation.

### Metric semantics

`sglang:load_back_duration_seconds` now covers, per merged load-back op:
- The actual H2D DMA on the load stream.
- Any wait on the load stream behind earlier load-back ops (since `ack_finish_event` is queued after the new layer copies and naturally absorbs that wait).
- The small cross-stream sync overhead.

It does **not** cover the time an op spends in `cache_controller.load_queue` between `load()` enqueue and `start_loading()` drain. In normal operation this is sub-millisecond — `load()` is O(1) Python (`load_queue.append`), and the queue is drained inside the same prefill scheduler step that filled it (`scheduler.py` calls `ready_to_load_host_cache()` right after batch formation). The wait is bounded by the number of hicache-hit requests in the batch, not by the size of any individual request, and is negligible against the DMA cost itself.

## Accuracy Tests

Not applicable — metric-only change. No model forward / kernel paths touched.

## Speed Tests and Profiling

Hot-path overhead: one `perf_counter()` pair removed, two `Event(enable_timing=True)` allocations + two `record()` calls added per merged load-back op. Negligible against the DMA cost itself (which is what the metric measures).

## Test Plan

New unit test `test/registered/unit/mem_cache/test_hicache_load_back_timing.py`:
- `make_timing_event` returns an event that supports `elapsed_time` after a recorded H2D op.
- `HiRadixCache.loading_check` observes both real DMA duration (`> 0`) and the merged-op token count for a completed ack.

Locally verified on RTX 4090D (sglang 0.5.12.post1, torch 2.9.1+cu128):

```
test_elapsed_time_works ... ok
test_loading_check_observes_duration_and_tokens ... ok
Ran 2 tests in 2.734s — OK
```

On a 4 MB H2D probe through the same code path, `elapsed_time` returned 10.92 ms and `observe_load_back_duration` was called with `0.010916` s — three orders of magnitude above the ~μs `perf_counter` dispatch time the old metric was emitting, which is exactly the gap this PR is fixing.

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [ ] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

---

cc @ShawnKung (original `load_back_duration_seconds` author, #10225) @fzyzcjy (observability CODEOWNER) @xiezhq-hermann @hzh0425 (mem_cache CODEOWNERS)
