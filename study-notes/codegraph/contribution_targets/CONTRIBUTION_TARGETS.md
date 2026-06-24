# sgl-kernel/csrc 贡献点扫描报告

**扫描日期**: 2026-05-28
**扫描范围**: `sgl-kernel/csrc/` （排除 `cpu/`, `musa/`, `metal/`, `cutlass_extensions/`, `3rdparty/`, 测试）
**目标**: 1 个小型 first-PR（1-30 行）

## 仓库 baseline（你需要知道的）

- **首次贡献者典型 PR 尺寸**: 1-30 行，1-3 个文件
- **强精度模板**: [#18750](https://github.com/sgl-project/sglang/pull/18750)（SM `==` → `>=`）、[#24130](https://github.com/sgl-project/sglang/pull/24130)（FlashMLA CMake guard）
- **在飞且不要碰**:
  - AOT → JIT kernel 迁移（@Johnsonms / @xingsy97 / @BBuf / @celve / @mmangkad）
  - DeepSeek V4 / SM120 / AMD V4（@hnyls2002 / @AdamPlatin123 / @AgainstEntropy）
  - CPU AVX/AMX/RVV（@jaylisde / @blzheng / @chunyuan-w / @ChenTim1011）

## ✅ 推荐候选（已通过 dedup + git blame 验证）

### 🏆 Rank #1 — `transfer.cu` 空索引时 host 端整数除 0（HiCache）

| 字段 | 内容 |
|---|---|
| **文件** | [`sgl-kernel/csrc/kvcacheio/transfer.cu:288-290`](sgl-kernel/csrc/kvcacheio/transfer.cu#L288-L290) |
| **分类** | edge_case / resource_lifecycle |
| **预估 diff** | +1/-0，1 文件 |
| **风险** | 极低 |
| **PR-readiness** | clear |
| **去重状态** | ✅ 无冲突 PR / issue（搜过 `transfer_kv` / `num_items` / `kvcacheio`） |

**Bug**: `transfer_kv_launcher` 在 `src_indices.numel() == 0` 时：
```cpp
const int64_t num_items = src_indices.numel();                                       // 0
const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block); // 0
const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);  // div_up(0, 0)
```
`div_up(0, 0)` = `(0 + 0 - 1) / 0` → **host 端整数除以 0**（SIGFPE / UB）。

**爆炸半径**: 12+ 个公共 op 都走这个 launcher（`transfer_kv_per_layer`, `transfer_kv_all_layer`, `transfer_kv_direct`, `*_mla`, `*_pf_lf`, `*_lf_ph` 等）。所有 HiCache load/back 路径都共享。

**Why intentional? 不**: PR [#8236](https://github.com/sgl-project/sglang/pull/8236)（AMD HiCache 支持，2025-08-28）引入 launcher 时就没加零守卫，是遗漏不是设计。

**你的优势**: 你当前分支就是 `fix/load-back-metric-accurate-dma-time`，正好在 HiCache load-back 路径，描述里可以直接说"在测我们的 load-back 计时改动时发现这个 edge case"。

**修复**（4 行）:
```cpp
TORCH_CHECK(item_size % 8 == 0, "Item byte size must be divisible by 8");

auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
const int64_t num_items = src_indices.numel();
if (num_items == 0) return;   // <-- add: empty index batch is a no-op
const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
```

**Risk of false-positive**: caller 大多在外面非空判断了，但有些 HiCache prefetch 路径在并发场景下会落到 0 长度 batch（特别是 PD 解耦时一侧空了）。即使从未触发，也是廉价的防御性修。完整 PR 描述见 `FIRST_PR_DRAFT.md`。

---

### Rank #2 — `moe_sum` / `moe_sum_reduce` 空 batch 启动错误（PR #23636 的姊妹修复）

| 字段 | 内容 |
|---|---|
| **文件** | `sgl-kernel/csrc/moe/moe_sum.cu:27`、`sgl-kernel/csrc/moe/moe_sum_reduce.cu:227` |
| **分类** | edge_case |
| **预估 diff** | +2/-0，2 文件 |
| **去重状态** | 🟡 同模式 PR [#23636](https://github.com/sgl-project/sglang/pull/23636)（4 行守卫 `activation.cu`）目前 OPEN 未合 |

**Bug**: `num_tokens == 0` 时 `dim3 grid(0)` → `cudaErrorInvalidConfiguration`。

**为什么是 #2 而不是 #1**: 同模式 PR #23636 已开 1 个月未合，说明这条 lane 的合入节奏不快。如果先合 #23636 再单独发姊妹 PR 风险最小；如果想一次过，最好先 ping #23636 作者协调。

---

### Rank #3 — `per_token_quant_fp8` 空 batch 启动错误

| 字段 | 内容 |
|---|---|
| **文件** | `sgl-kernel/csrc/gemm/per_token_quant_fp8.cu:286` |
| **分类** | edge_case |
| **预估 diff** | +1/-0，1 文件 |
| **去重状态** | ✅ 无冲突 |

**Bug**: `dim3 grid(num_tokens)` 在 `num_tokens=0` 时 `cudaErrorInvalidConfiguration`。FP8 推理热路径，DP>1 时空闲 rank 容易触发。

---

## ❌ Drop list（已排除的候选 + 原因）

| 候选 | 文件 | 排除原因 |
|---|---|---|
| `activation.cu` silu/gelu 零守卫 | `elementwise/activation.cu` | **DUPLICATED** — PR [#23636](https://github.com/sgl-project/sglang/pull/23636) 已覆盖全部 4 个函数 |
| `segment_packbits` 死代码删除 | `speculative/packbit.cu` | **PARTIAL** — 看起来真死（只剩 `noqa: F401` 导入），但 PR [#19088](https://github.com/sgl-project/sglang/pull/19088) 在尝试把它移到 JIT，删之前要问作者意图；可单独开 issue 询问，别直接删 |
| `concat_mla.cu:197` int 溢出 | `elementwise/concat_mla.cu:197` | **PARTIAL** — 已有 PR [#12453](https://github.com/sgl-project/sglang/pull/12453) 修过 "long inputs" 但 line 197 的 `int` 截断仍在；只在 `a.size(0)*a.size(1)` > 2^31 时触发，正常推理 batch×seq 不到，证据较弱 |
| `cutlass_mla_kernel.cu:223` SM `== 100` | `attention/cutlass_mla_kernel.cu:223` | **needs-design-discussion** — 注释明说 SM103a 精度有问题，可能是有意的 |
| `concat_mla_k` 零守卫 | `elementwise/concat_mla.cu:88-100` | **PARTIAL** — concat_mla 已 JIT 化（[`python/sglang/jit_kernel/concat_mla.py`](python/sglang/jit_kernel/concat_mla.py)），AOT 路径还活着但热度下降 |
| `pos_enc.cu` rotary 零守卫 | `elementwise/pos_enc.cu:175` | **低优** — rotary 在 QKV proj 之后调用，上游已非空，触发概率低 |
| `gptq_kernel.cu` `cudaMemcpyAsync` 流问题 | `gemm/gptq/gptq_kernel.cu:1886-1889` | **低优** — 在 weight-loading 启动期，不在热路径 |
| `cutlass_moe_helper.cu` dead bounds check | `moe/cutlass_moe_helper.cu:34-38` | **needs-design-discussion** — 实际 num_experts 永远 ≤ 1024，纯 cosmetic |
| `moe_sum_reduce.cu:265` 冗余 stream 查询 | `moe/moe_sum_reduce.cu:265` | **太微** — 50-100 ns 节省，无观测影响 |
| `moe_align_kernel.cu:328-329` no-op ceil_to_warp | `moe/moe_align_kernel.cu:328-329` | **lane conflict** — moe_align 已被 @xingsy97 移到 JIT（PR #19704），AOT 在 deletion 路径上 |
| `kvcacheio/transfer.cu:701-748` `.cpu()` 同步 | `kvcacheio/transfer.cu:701` | **lane conflict** — 跟你自己的 `fix/load-back-metric-accurate-dma-time` 分支冲突，需自己协调 |
| `merge_state_v2` `.to(float32)` 守卫 | Python only | **过于微小** — PyTorch 内部已 short-circuit |

## Phase 8 follow-on（如果 Rank #1 合入后想接着做）

- **Rank #2**: 等 PR #23636 合入后，立刻 file 姊妹 PR 覆盖 `moe_sum.cu`、`moe_sum_reduce.cu`、`per_token_quant_fp8.cu`，可以 cite #23636 作为模板。
- **开 issue 询问 segment_packbits 现状** — 无风险高信任的"第一次接触"。可同时 cc PR #19088 作者，问"AOT 还需要吗？我看不到 call site"。
- **跑本地 stress test 验证 transfer.cu 修复** — 用你自己的 load-back benchmark 脚本 [`scripts/bench_load_back_event_overhead.py`](scripts/bench_load_back_event_overhead.py) 构造空 batch 复现。
