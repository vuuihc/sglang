[Bug Fix][Compilation] Use a deterministic computation_graph dump filename to stop unbounded per-startup accumulation

## Motivation

When piecewise `torch.compile` is enabled, `SGLangBackend.__call__` dumps a human-readable copy of the split graph module for debugging (a depyf-style dump). The dump path embeds a wall-clock timestamp in the filename:

```python
if rank == 0:
    graph_path = os.path.join(
        local_cache_dir, f"computation_graph_{time.time()}.py"
    )
    if not os.path.exists(graph_path):
        ...
        with open(graph_path, "w") as f:
            f.write(src)
```

Because `time.time()` changes on every process, the filename is unique on every startup, so `os.path.exists(graph_path)` is always `False` and the guard never skips anything. Each startup therefore writes a brand-new `computation_graph_<timestamp>.py` into `local_cache_dir`, and nothing ever removes them. The dump is guarded only by `rank == 0` — not by any debug flag — so it runs on every compile-enabled boot.

`SGLangBackend.__call__` asserts it runs at most once per instance:

```python
assert not self._called, "SGLangBackend can only be called once"
```

So within a process the write happens once per backend instance (e.g. `backbone`, `eagle_head`), and the intent of the `if not os.path.exists(...)` guard is clearly "write once, don't rewrite". The timestamp defeats that intent:

```text
1. First boot writes computation_graph_1700000000.1234.py.
2. The exists-check passed only because that exact path was new.
3. Next boot computes a different timestamp, so the check passes again.
4. Over N restarts the cache dir accumulates N copies of essentially the same dump, never garbage-collected.
```

The file is a pure debug artifact: `git grep computation_graph` shows the write site is the only reference in the tree — nothing reads, globs, or cleans these files — so the filename can be made deterministic without breaking any consumer.

This change removes the timestamp so the dump lands at a fixed, per-`model_tag` path. As a side effect the existing `os.path.exists` guard becomes meaningful again (write-once instead of always-miss), and the cache directory stays bounded across restarts.

No tracking issue; happy to open one if preferred.

## Modifications

- In `python/sglang/srt/compilation/backend.py`, change the graph dump filename from `f"computation_graph_{time.time()}.py"` to a fixed `"computation_graph.py"`. The dump already lives under a `local_cache_dir` that is scoped by cache hash + rank + `model_tag`, so a fixed name is unique per compiled region and simply overwrites/skips on subsequent boots instead of accumulating.
- No import changes: `time` is still used elsewhere in the file (`compilation_start_time = time.time()`, elapsed-time logging).

### Optional follow-up (raising for reviewer input, not included here)

This dump is a debugging aid. It currently runs on every compile-enabled startup, gated only by `rank == 0`. It may be cleaner to also gate it behind the existing `compile_config.get_enable_debug_mode()` flag (already used in `cuda_piecewise_backend.py`) so it is only written when debugging. I left that out of this PR to keep the change surgical and avoid altering default behavior — let me know if you'd like it folded in.

## Tests

- Ran `pre-commit run --files python/sglang/srt/compilation/backend.py` locally — hooks pass.
- Ran `python3 -m compileall -q python/sglang/srt/compilation/backend.py`.
- No focused `pytest` was added/run: the dump is emitted from `SGLangBackend.__call__`, which requires a real Dynamo-produced `fx.GraphModule`, an initialized `torch.distributed` group, and the CUDA `torch.compile` path — i.e. GPU + a full compile pass — so it is not practically reachable from a CPU-only unit test. The fix is a static, single-token filename change whose behavior (deterministic path → `os.path.exists` guard becomes a true write-once) is verifiable by inspection.
- Verified the leak reasoning statically: `git grep -n computation_graph` returns only the write site, confirming no reader/cleaner depends on the timestamped name.

## Speed Tests and Profiling

N/A. This is a correctness/hygiene fix that removes unbounded disk growth in the compile cache directory. It does not change compiled kernels, cache hit behavior, or the serving hot path.

## Checklist

- [x] Format your code according to the Code Formatting with Pre-Commit.
- [ ] Add unit tests as outlined in the Running Unit Tests. (Not practical — dump path requires GPU + full compile pass; see Tests.)
- [ ] Update documentation as needed, including docstrings or example tutorials. (N/A)
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed. (N/A — static hygiene fix)
- [ ] For reviewer assignment, see the Reviewer Assignment Guide.
