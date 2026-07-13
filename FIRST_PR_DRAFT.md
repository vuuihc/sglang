<!--
PR draft for branch fix/unified-radix-tree-node-lru-cache-regression.
Suggested title:
  fix(mem_cache): drop @lru_cache on UnifiedTreeNode.get_prefix_hash_values
Suggested CC line at the bottom of the PR description (or in a follow-up comment):
  cc @hzh0425 (author of #26062, which reintroduced the pattern)
-->

## Motivation

Same bug as #26177, but in the unified tree.

#26177 dropped `@lru_cache(maxsize=1)` from `TreeNode.get_prefix_hash_values` in `radix_cache.py` and `mamba_radix_cache.py` because the cached `List[str]` was being in-place mutated by HiCache storage paths (`prefix_keys += batch_hashes`).

#26062 (L3 HiStorage, merged the next day) added `UnifiedTreeNode.get_prefix_hash_values` to `unified_radix_cache.py` and copied the same `@lru_cache(maxsize=1)` along with it. So the bug the previous PR fixed is back in the unified tree, and the two trees now have a behavioural divergence in this code path.

This PR drops the decorator on the unified side to match.

## Modifications

- Remove `@lru_cache(maxsize=1)` from `UnifiedTreeNode.get_prefix_hash_values` in `python/sglang/srt/mem_cache/unified_radix_cache.py`.
- Drop the now-unused `from functools import lru_cache` import in the same file.
- Add a regression test alongside the one added in #26177 (`test/registered/unit/mem_cache/test_radix_cache_unit.py` or a new `test_unified_radix_cache_unit.py`), parameterized over the unified tree, that mutates the returned list to mimic the storage-side `prefix_keys += batch_hashes` and verifies later calls still return fresh, unpolluted prefix hash lists — including the recursive `n4.get_prefix_hash_values(n3)` → `n3.get_prefix_hash_values(n2)` walk that previously read the mutated cached entry.

## Tests

- Ran `pre-commit run --files python/sglang/srt/mem_cache/unified_radix_cache.py test/registered/unit/mem_cache/test_unified_radix_cache_unit.py` — all hooks pass.
- Ran `python3 -m compileall -q python/sglang/srt/mem_cache/unified_radix_cache.py <test path>`.
- Verified the failure mode with a standalone repro that toggles `@lru_cache` on/off (the same shape as the test added in #26177).

## Speed Tests and Profiling

N/A. Same reasoning as #26177: correctness fix, the removed cache had `maxsize=1` and only helped a narrow temporal-locality case, and the cost of the helper is negligible relative to L3 storage I/O.

## Checklist

- [x] Format your code according to the Code Formatting with Pre-Commit.
- [x] Add unit tests as outlined in the Running Unit Tests.
- [ ] Update documentation as needed. (N/A)
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results. (N/A — static correctness fix)
- [ ] For reviewer assignment, see the Reviewer Assignment Guide.

---

cc @hzh0425 (author of #26062, which reintroduced the pattern)
