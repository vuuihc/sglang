# sglang — Contribution Targets (scan run 2026-06-01)

> Output of the `find-contribution-targets` skill against `sgl-project/sglang` @ local
> HEAD `f60710a1d`. Workspace: `/Users/vuuihc/Workspace/study/sglang`.
>
> **Pre-existing work in this workspace**: `PR_DRAFT.md` is a complete draft PR for
> `fix(hicache): measure load-back duration with CUDA events` in the `mem_cache/`
> package. The rank-1 candidate below sits in the same package and would be a
> natural follow-up PR after that lands.

## TL;DR

| # | Rank | Module | Lines | Risk | Why |
|---|------|--------|------:|------|-----|
| 1 | @lru_cache on recursive instance method | `mem_cache/` | 1–3 | low | classic Python anti-pattern, drop the decorator |
| 2 | 8× `.item()` D2H syncs on `tp0_info` | `managers/prefill_delayer.py` | ~6 | low–med | 8 syncs/step → 1; clean win |
| 3 | 3× D2H syncs on the same tensor | `managers/scheduler_components/dp_attn.py` | ~3 | low | same root cause as #2; pair or merge |
| 4 | `seq_lens[i].item()` inside per-batch loop | `layers/attention/dsa/dsa_indexer.py` | 2 | low | `seq_lens_cpu[i]` is right there |
| 5 | mamba slot leak on `insert()` exception | `mem_cache/mamba_radix_cache.py` | ~5 | med | missing try/except; real GPU-memory leak |
| 6 | 3 dict entries leaked per failed bootstrap | `disaggregation/decode.py` | 1 | med | `abort()` doesn't free; `clear()` does |

A 4-PR pipeline: ship #1 → #2 → #4 → #5 (in `mem_cache/` first, then scheduler, then attention, then mamba). #3 pairs with #2. #6 is a quick win for disagg users.

## Repo quick facts

- `python/sglang/srt/` is the hot path (srt = "SGLang Runtime"). Sub-packages of interest: `managers/` (scheduler, tokenizers, schedules), `mem_cache/` (radix/mamba/unified tree cache + hicache), `layers/attention/` (multiple backends), `disaggregation/` (PD), `eplb/` (expert load balancing), `connector/` (KV transfer).
- `sgl-kernel/` is C++/CUDA, `sgl-model-gateway/` and `rust/` are Rust. **Out of scope for first-time-contributor PRs.**
- Commit style: `[Module] description` — examples in recent log: `[AMD]`, `[NPU]`, `[bugfix]`, `[CI]`, `[attn backend]`, `[diffusion]`, `[PP]`, `[mem_cache]`, `[tokenizer]`, `[Bugfix]`, `[BugFix]`, `[Fix]`, `(none)`. Both Chinese and English are common; recent log is heavy on English.
- PR description template: Motivation / Modifications / Accuracy Tests / Speed Tests / Test Plan / Checklist (see `PR_DRAFT.md` for the exact template).
- Pre-commit: `.pre-commit-config.yaml` exists; lint with `pre-commit run --files <paths>` before pushing.

## First-PR archaeology (Phase 2)

- 358 merged PRs in the last 500 closed PRs; 94 distinct authors contributed exactly 1 merged PR in that window.
- The "single-PR-in-window" sample is dominated by **CONTRIBUTOR**-association PRs — meaning first-time or rarely-active external contributors. Examples that match the profile of the candidates below:
  - #26534 `fix: use req.req_pool_idx instead of loop variable for req_to_token i…` (CONTRIBUTOR) — same flavor as #4 (replace a per-iter value with a cheaper equivalent).
  - #26521 `fix: copy seq_lens in TRTLLM MHA draft decode cuda graph capture` (CONTRIBUTOR) — same flavor as #2/#3 (sync-related fix in attention path).
  - #26590 `[BugFix] preserve cached token details in multi-tokenizer output` (CONTRIBUTOR) — same flavor as #1 (small targeted cache-state fix).
- Typical merged PR size for a new contributor: **1–30 lines**. Almost all of the 6 candidates above are in that range.
- Precedent for the rank-1 candidate: classic `@lru_cache` on bound method is a well-documented Python anti-pattern (CPython docs explicitly warn about pinning). It is exactly the kind of thing reviewers accept quickly.

## Issue/PR hunt (Phase 3)

- `good first issue` (50 open): mostly feature requests requiring broader design understanding (TBO + shared experts, JIT cache unification, diffusion refactors). The 4 oldest are from 2024 and almost certainly stale. Not a fertile source for *small* first PRs.
- `help wanted` (17 open): same — old, big-picture.
- `bug` (6 open): all model/hardware-specific (Gemma-4 NVFP4, Qwen3.5 PD, DeepSeek tracking). Not relevant.
- **Conclusion**: labels in sglang are dominated by *feature* and *tracking* issues, not small bug candidates. The static scan is the right path here, not the label-filtered tracker.

## Severity reality check (Phase 6) — top 6

### #1 — `@lru_cache(maxsize=1)` on `UnifiedTreeNode.get_prefix_hash_values` — MED → strong first PR

**Verified**: `python/sglang/srt/mem_cache/unified_radix_cache.py:116-121`.

```python
@lru_cache(maxsize=1)
def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
    if node is None or node.hash_value is None:
        return []

    return node.get_prefix_hash_values(node.parent) + node.hash_value
```

- **Why it's an anti-pattern**: (1) `self` is part of the cache key. The `lru_cache` lives at the class object, so it pins the live `UnifiedTreeNode` instance for the duration of the program. (2) It's recursive and is called with *different* `node` arguments (line 121). With `maxsize=1`, every new `node` evicts the previous entry, so the cache provides no real memoization — only an extra dict lookup, and a hidden long-lived reference. (3) Returns a `list[str]` (a mutable), which is then mutated by callers — `lru_cache` on a function that returns mutable state is itself a hazard.
- **Why it's a great first PR**: the fix is mechanical (drop the decorator and either inline the recursion, or use an explicit dict invalidated in `_reset_full` / `_split_node`); the patch is 1–3 lines; the test that "every call after the first is a miss" is one line; and it ships a small but real code-hygiene improvement that any reviewer will recognize.
- **Risk**: low. The function is currently called from `unified_radix_cache.py` and a few adjacent matchers; removing the cache just means callers get the actual O(depth) walk they were already paying for on the first call. No performance regression in practice (cache is a miss after the first call anyway).

### #2 — 8× `.item()` syncs in `PrefillDelayer._negotiate_should_allow_prefill_pure` — MED

**Verified**: `python/sglang/srt/managers/prefill_delayer.py:163, 165, 170, 174, 175, 195, 196, 197`. The function reads columns of `tp0_info` (a `(dp_size × attn_tp_size × 5)` GPU tensor) and reduces each to a Python scalar with a separate `.item()` call. **8 separate D2H syncs per prefill decision step**.

- **Why it's a candidate**: the surrounding code does an all-gather into `tp0_info` and then reads many columns. Once. Per step. The pattern of "8× `.item()`" is the kind of thing a reviewer will accept as a 1-PR cleanup, similar to the precedent #26521.
- **Not a dup of #24768**: #24768 changed the all-gather group (gloo CPU → NCCL device) but did **not** touch the `.item()` calls. The file at HEAD `f60710a1d` still has all 8 syncs.
- **Dedup**: searched GitHub for `prefill_delayer sync`, `tp0_info cpu`, etc. — 0 hits. The only open PRs touching this file (#26321, #23398, #23709) modify the trigger condition, not the sync pattern. Clean.
- **Risk**: low–med. The fix collapses 8 syncs to 1 `.cpu().tolist()`. Performance impact is small in absolute terms (~tens of μs/step on a small tensor) but the cleanliness of the patch is the point.

### #3 — 3× D2H syncs on the same tensor in `MLPSyncInfo.all_gather` — MED

**Verified**: `python/sglang/srt/managers/scheduler_components/dp_attn.py:104-108`.

```python
cpu_data = tp0_info[:, :2].cpu()         # sync #1 (needed for tolist)
self.global_num_tokens = cpu_data[:, 0].tolist()
self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
self.can_cuda_graph = bool(tp0_info[:, 2].min().item())   # sync #2
self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())  # sync #3
```

- **Why it's a candidate**: same root cause as #2 (multiple syncs on a tensor that's already been gathered). Fix is 1 line: extend the slice to `[:4]`, do `.cpu().tolist()` once, read `min` / `max` from the CPU list. 3 syncs → 1.
- **Pairs naturally with #2**. Could be one PR ("collapse redundant D2H syncs in scheduler hot path") or two smaller PRs (one per file). Two smaller PRs is the project's preferred style — see #24768 + #26521 history.

### #4 — `seq_lens[i].item()` in per-batch loop — MED

**Verified**: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1149` (and a parallel pattern at line 999).

```python
for i in range(forward_batch.batch_size):
    seq_len = forward_batch.seq_lens[i].item()   # sync per iter
    q_len = (
        forward_batch.extend_seq_lens_cpu[i]
        if forward_batch.forward_mode.is_extend()
        else 1
    )
```

- **Why it's a candidate**: `forward_batch.seq_lens_cpu` exists (verified in `managers/schedule_batch.py:1747, 1842, 2361, 2441, etc.`). Replace `forward_batch.seq_lens[i].item()` with `forward_batch.seq_lens_cpu[i]`. 1-line change. Per-iter sync removed; the only cost is reading the CPU tensor. The function is in the DSA / Hisparse prefill path.
- **Precedent**: #26521 (`fix: copy seq_lens in TRTLLM MHA draft decode cuda graph capture`) is from the same family — same author association (CONTRIBUTOR), same flavor of fix.
- **Risk**: low. There are two sites (line 999 and line 1149) — both should be in the same PR.

### #5 — mamba slot leak on `MambaRadixCache.cache_unfinished_req` exception — MED

**Verified**: `python/sglang/srt/mem_cache/mamba_radix_cache.py:665-681`.

```python
mamba_value_donated = self._alloc_mamba_slot()
self.req_to_token_pool.mamba_pool.copy_from(
    req.mamba_pool_idx.unsqueeze(0), mamba_value_donated
)

result = self.insert(InsertParams(... mamba_value=mamba_value_donated ...))
new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
if mamba_exist:
    self.req_to_token_pool.mamba_pool.free(mamba_value_donated)
```

- **Why it's a candidate**: if `self.insert(...)` raises (driver error, OOM, radix invariant violation), the slot is allocated and never returned to `mamba_pool`. The success path frees it; the failure path doesn't.
- **Why it's a real leak**: a sustained rate of `cache_unfinished_req` failures (e.g. on a misbehaving workload) monotonically decreases the available mamba slots. Eventually `mamba_pool` is full and the server falls over.
- **Fix**: try/except around `self.insert(...)` that calls `mamba_pool.free(mamba_value_donated)` on the way out. ~5 lines. Precedent pattern: `try/finally` is the Pythonic equivalent of RAII and is used elsewhere in sglang's allocator path.
- **Risk**: med. The path is only taken when `enable_mamba_extra_buffer=True`, so the regression blast radius is bounded, but the test surface for "insert raises" needs at least a monkey-patch test.
- **Reproduction**: `with self.enable_mamba_extra_buffer, monkeypatch self.insert to raise, fire requests in a loop, query self.req_to_token_pool.mamba_pool available slots` — they will monotonically decrease.

### #6 — 3 dict entries leaked per failed bootstrap (disagg decode) — HIGH impact / MED risk

**Verified**: `python/sglang/srt/disaggregation/decode.py:670-671`. The retry-exhaustion path calls `decode_req.kv_receiver.abort()`, which only marks the room failed (per `common/conn.py:1075` — `abort()` is a failure-recorder stub). It does **not** call `kv_receiver.clear()`, which is the only path that actually pops `bootstrap_room` from `kv_mgr.request_status`, `kv_mgr.required_prefill_response_num_table`, and `kv_mgr.prefill_response_tracker` (lines 1070-1073). Each retry-exhausted request leaves 3 dead entries in long-lived manager dicts. Under flaky bootstrap (network blips, slow prefill instances), the leaked entries grow unboundedly for the server's lifetime.

- **Why it's a candidate**: this is the highest-impact leak in the list. The fix is **1 line**: replace `decode_req.kv_receiver.abort()` with `decode_req.kv_receiver.clear()`. The author must verify `clear()` is idempotent (a quick read of `conn.py:1070-1073` should confirm it).
- **Risk**: med. Disagg paths are sensitive; an incorrect fix could either (a) cause double-free of bootstrap_room or (b) break the failure-recording that downstream observability depends on. The PR should ideally include a unit test that exercises the retry-exhaustion path with a fake black-hole bootstrap address and asserts that `len(kv_mgr.request_status)` returns to baseline.
- **Reproduction**: in a 2-node disagg setup, point the decode-side bootstrap address at a black-hole port, send N requests, wait for `_max_ensure_retries` to elapse (default 15 scheduling cycles), inspect `len(kv_manager.request_status)` — it's `3*N` larger than it should be.

## Drop list

See `DROP_LIST.md` for the candidates that didn't make the cut, with one-line reasoning each.

## Outputs

- `CONTRIBUTION_TARGETS.md` — this file
- `FIRST_PR_DRAFT.md` — patch + description for rank-1
- `DROP_LIST.md` — what was filtered
- `RUN_LOG.jsonl` — one-line summary for scheduler/loop diffing
