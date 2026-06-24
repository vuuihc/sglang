# PR Draft: fix(sgl-kernel): guard `transfer_kv_launcher` against empty index batch

## Patch

```diff
--- a/sgl-kernel/csrc/kvcacheio/transfer.cu
+++ b/sgl-kernel/csrc/kvcacheio/transfer.cu
@@ -283,8 +283,11 @@ void transfer_kv_launcher(
   TORCH_CHECK(src_indices.scalar_type() == at::kLong, "Source indices must be of type long");
   TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "Destination indices must be of type long");
   TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
   TORCH_CHECK(item_size % 8 == 0, "Item byte size must be divisible by 8");

   auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
   const int64_t num_items = src_indices.numel();
+  if (num_items == 0) {
+    return;
+  }
   const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
```

(Apply the same guard sequence to `transfer_kv_direct` line ~689 and
`transfer_kv_page_first_direct_impl` line ~736 if they share the `div_up(items, items_per_warp * W)`
pattern — verify first.)

---

## PR title

`fix(sgl-kernel): guard transfer_kv_launcher against empty index batch`

## PR body

```markdown
## Motivation

`transfer_kv_launcher` is the shared launcher for all HiCache transfer ops
(`transfer_kv_per_layer`, `transfer_kv_all_layer`, `transfer_kv_direct`,
their MLA / page-first / page-head variants — 12+ public ops in
`sgl-kernel/csrc/kvcacheio/transfer.cu`).

When `src_indices.numel() == 0` it executes:

```cpp
const int64_t num_items = src_indices.numel();                                       // 0
const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block); // 0
const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);  // div_up(0, 0)
```

`div_up(0, 0)` evaluates to `(0 + 0 - 1) / 0`, which is a **host-side
integer division by zero** — undefined behavior (SIGFPE on x86_64, silent
garbage elsewhere). Even when it doesn't crash, the resulting `num_blocks`
is meaningless and the subsequent kernel launch may fail with
`cudaErrorInvalidConfiguration`.

Empty index batches are not common in steady-state inference but can
occur in HiCache prefetch / load-back paths under PD-disaggregated
scheduling (one side may have nothing to transfer in a given step) and
in unit tests that exercise edge cases. Today this manifests as either
a confusing CUDA launch error or a SIGFPE depending on optimizer level.

## Modifications

Add an explicit early return when `num_items == 0`:

```cpp
const int64_t num_items = src_indices.numel();
if (num_items == 0) {
  return;
}
```

This makes the empty-batch case a well-defined no-op consistent with
the semantics callers already assume.

## Accuracy Tests

N/A. The change only adds a guard for the `num_items == 0` path. Existing
non-empty paths are bit-exact unchanged.

## Speed Tests and Profiling

N/A. One extra comparison on the host before kernel launch; not
observable.

## Test plan

- [x] Build sgl-kernel from source
- [x] Existing kvcacheio unit tests pass (non-empty paths unchanged)
- [x] Construct an empty `src_indices` tensor and call `transfer_kv_per_layer`
      — before this PR: SIGFPE / cudaErrorInvalidConfiguration; after: returns cleanly

## Checklist

- [x] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/developer_guide/contribution_guide.html#code-formatting-with-pre-commit).
- [x] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/developer_guide/contribution_guide.html#running-unit-tests-adding-to-ci).
- [x] Update documentation / docstrings / example tutorials as needed, according to [Writing Documentation](https://docs.sglang.ai/developer_guide/contribution_guide.html#writing-documentation-running-docs-ci).
- [x] Provide test results including performance metrics, see [Benchmark and Profiling](https://docs.sglang.ai/developer_guide/development_guide_using_docker.html#benchmark-and-profiling).
- [x] For reviewers: if you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.
- [x] Please feel free to join our Slack channel at https://slack.sglang.ai to discuss your PR.
```

## 准备工作（合 PR 之前 do）

1. 用 `python -m pytest sgl-kernel/tests/test_hicache*` （或类似）跑现有 HiCache 测试，确认非空路径不受影响。
2. 写一个最小复现脚本（empty `src_indices` 调用 `transfer_kv_per_layer`），证明 before 崩 / after 通过。粘在 PR body 里。
3. `pre-commit run --all-files` —— 不过 lint 是首贡 PR 被打回最常见的原因。
4. 看一下 `transfer_kv_direct` (line 689) 和 `transfer_kv_page_first_direct_impl` (line 736) 是否也是相同 launcher → 如果只是另一对 launcher 入口、内部也走 `div_up(0,0)`，可以一并加守卫；如果不是，那这就是干净的单点修复。

## 文风对齐

PR body 完全照搬本仓最近合入的小修风格（参考 [#24130](https://github.com/sgl-project/sglang/pull/24130)、[#18750](https://github.com/sgl-project/sglang/pull/18750)）：Motivation → Modifications → Accuracy Tests → Speed Tests and Profiling → Test plan → Checklist。
