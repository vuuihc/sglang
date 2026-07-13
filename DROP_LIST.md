# Drop list — sglang contribution-target scan (2026-06-01)

Candidates that came out of the 3 parallel scans but were filtered out of
the final ranked list. Kept here so future re-runs of the skill don't
re-surface them.

## Dropped after severity reality check (Phase 6)

### `managers/load_snapshot.py:373-385` — mmap leak on partial-init exception
- Why dropped: triggering the leak requires an exception in
  `HEADER_STRUCT.pack_into` or `_write_payload` during `__init__`. Production
  hit rate is near zero. Real-world impact is bounded (Python's
  refcount-based cleanup of mmap still runs at function exit). Not worth a
  first PR — the "reproduce" path is awkward and the reviewer will push back
  on impact.

### `connector/base_connector.py:23-29, 62-72` — `signal.signal()` closure pins `self`
- Why dropped: the fix (drop the per-instance signal installation; rely on
  `__enter__`/`__exit__` + `atexit`) is a *behavior* change, not a code
  cleanup. The existing handler chain is intentional (last handler wins,
  chains to second-to-last). Removing it could regress an established
  cleanup path. Risk too high for a first PR; better raised as a
  "Question:" issue first.

### `layers/rotary_embedding/mrope_rope_index.py:737-739, 773-775` — 3× `.item()` on mrope
- Why dropped: only fires on the multimodal prefill path, which is
  exercised by a small fraction of contributors. The fix is correct but
  the impact is small in absolute terms (3 syncs per image row in a
  prefill call). Out-competed by #2 / #3 which touch the same `.item()`
  pattern in a hotter scheduler path.

### `layers/attention/dual_chunk_flashattention_backend.py:184-185, 629-630, 1038-1039` — 3 sites
- Why dropped: three separate sites in a single 1730-line file. Each fix
  individually is fine, but the file is dense and changes here need
  closer review. Higher reviewer cost than #2/#3. Can be revisited after
  the simpler scheduler-path syncs land.

### `layers/attention/vision.py:376, 442, 496, 668, 778` — `seq_lens.max().item()` × 5
- Why dropped: same reason as above — 5 sites in a multimodal path with
  limited reviewer overlap. The fix is mechanical but the *blast radius*
  is the entire vision attention forward.

### `layers/attention/flashinfer_backend.py:1379` — `bool(block.all())` in SWA loop
- Why dropped: agent self-flagged as `NEEDS-PROFILE`. The block size is
  `extend_len × paged_len` and the loop is in the SWA prefill path. Needs
  a profile to confirm the sync cost before committing to the fix. Not
  a good first PR until profiled.

### `layers/attention/flashattention_backend.py:2491` — `seqlens.max().item()` in CUDA-graph replay
- Why dropped: only fires in CUDA-graph replay path; the agent self-flagged
  `MAYBE` because reordering the existing D2H transfer on the next line
  (`cu_seqlens_q.cpu().numpy()`) could regress other paths. Low priority.

### `layers/attention/hybrid_linear_attn_backend.py:242-245` — `bool(... .any())` per-step
- Why dropped: only fires when `mamba_track_mask` is set on the hybrid
  attention path. The agent self-flagged `MAYBE` pending confirmation
  that a CPU `mamba_track_mask_cpu` view exists. Need a verification pass
  before promoting.

### `mem_cache/swa_memory_pool.py:229` — `bool(torch.all(row_mask).item())`
- Why dropped: low impact. The `row_mask` is a per-row tensor; one
  boolean sync per `_filter_swa_cpu_copy` call. The path is called
  during SWA offload, not per-decode-step. Save for later.

### `mem_cache/radix_cache.py:746-754` / `mamba_radix_cache.py:1211-1221` — `_delete_leaf` doesn't clear `parent`
- Why dropped: agent self-flagged as `LOW` and explicitly said "not a hard
  leak; just a code-hygiene inconsistency". The PR motivation is weak —
  reviewers will ask "does this fix anything observable?" Hard to defend.

### `managers/schedule_batch.py:736-741` — `Req <-> SamplingParams.custom_params` cycle
- Why dropped: the cycle is intentional (it's how the custom logit
  processor looks up the Req by id). Python's cyclic GC handles it. Agent
  self-flagged `LOW` and noted "Not a real leak". Confirmed by reading
  the call sites — `__req__` is read inside the custom logit processor's
  call, not a passive reference.

## Dropped after typo scan (Phase 4) — 0 findings

The full typo / 1-line correctness scan of
`python/sglang/srt/{managers,mem_cache,layers,entrypoints,disaggregation,eplb,sampling,connector,distributed}` +
`python/sglang/{server_args.py,utils.py,check_env.py,launch_server.py,__init__.py,cli/}` +
`pyproject.toml` returned **0 HIGH or MED confidence findings**.

What was checked:

- `weihgt/wieght/parameter/initalize/...` typo scan in identifiers — 0 hits in scope.
- `raise NotImplemented` (not `NotImplementedError`) — 0 hits.
- `hasattr(..., "x")` followed by `.x` typo — all attribute names match.
- `assert (a, b)` always-truthy tuple pitfall — 0 hits; all `assert` with
  parenthesized args are either `or` expressions or shape comparisons.
- `is` / `is not` vs `==` on strings — only hits compare to actual `bool`
  singletons (technically correct).
- Off-by-one `range(len(x))` with `x[i+1]` — 0 hits; all use `len(x) - 1`.
- Missing `await` on coroutines in `async def` — 0 hits; all properly
  awaited or scheduled via `asyncio.create_task`.
- CLI default mismatch in `server_args.py` — none.
- Bare `except:` — all 23 hits are intentional import-fallback / path-choice
  blocks.

**Out-of-scope typo candidates** (would need a follow-up scan with
`models/` included or `sgl-kernel/` C++ included):

- `models/deepseek_vl2.py` and `models/deepseek_ocr.py` contain 9 hits of
  `view_seperator` (a misspelled `nn.Parameter` name). This matches the
  #25786 precedent very well, but `models/` is the wrong directory for a
  first PR (per-file review cost is high).
- `sgl-kernel/` C++/CUDA was not scanned; the #25695 precedent (missing
  `template` keyword) was a C++ fix, so this directory is a known-fertile
  territory for typo-style fixes that a follow-up scan should cover.

## Strategy switch signal (from the skill's Phase 8)

The typo scan returning zero findings in scope is a strong signal that
sglang's `srt/` hygiene is good. The next time the skill is run:

1. **Include `models/` in scope** — that directory contains the
   `view_seperator` cluster.
2. **Include `sgl-kernel/` C++ in scope** — that directory contains
   kernel/header typos that match the #25695 precedent.
3. **Run a dynamic test pass** — for the GPU sync candidates that
   self-flagged `NEEDS-PROFILE` (rank `flashinfer_backend.py:1379`,
   `hybrid_linear_attn_backend.py:242-245`), the actual sync cost
   is unknown without a profile. A small profiling PR (with a chart)
   is a higher-quality contribution than guessing.

## Dedup evidence (Phase 5)

For each HIGH/MED candidate, the following queries returned 0
unrelated-results hits in sgl-project/sglang's open PRs / issues
(verified via GitHub public search API on 2026-06-01):

| Candidate | Search terms tested | Hits |
|-----------|----------------------|------|
| prefill_delayer `.item()` | `prefill_delayer sync`, `tp0_info cpu` | 0 |
| dp_attn sync | `dp_attn tp0_info`, `tp0_info cpu` | 0 |
| unified_radix_cache lru_cache | `lru_cache maxsize`, `unified_radix_cache` | 0 |
| dsa_indexer seq_lens | `dsa_indexer seq_lens`, `seq_lens_cpu` | 0 |
| mamba_radix_cache slot leak | `mamba_pool free`, `mamba_radix_cache` | 0 |
| decode bootstrap leak | `bootstrap_room leak`, `kv_receiver abort` | 0 |

The prefill_delayer file has 2 open related PRs (#26321, #23398, #23709)
but none of them touch the `.item()` sync pattern. #24768 (the NCCL
all-gather change) is merged but does not touch the candidate sites.
