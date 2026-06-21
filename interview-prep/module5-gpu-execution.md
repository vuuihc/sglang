# 模块五：GPU 执行优化

> 基于 SGLang 代码库的面试备考精华笔记

---

## Q17：CUDA Graph 的工作原理

CUDA Graph 把一次 forward pass 中所有 GPU kernel 的调用顺序记录成一张 DAG，replay 时用**一次 CPU 调用**替代原来的多次 kernel launch，消除 CPU-GPU 之间的调度等待气泡。

### 为什么不同 batch size 需要不同的图

关键原因是 **tensor 内存地址在 capture 时被烤进图里**：

```python
# cuda_graph_runner.py
capture 时：graph 记录了具体的 GPU 内存地址（如 input_ids 在 0x7f3a...）
replay 时：把新数据写入同一块预分配 buffer（地址相同），然后 graph.replay()
```

不同 batch size 需要不同大小的 buffer → 地址不同 → 需要不同的图。同时 kernel launch 的 grid dimensions 也随 batch size 变化。

SGLang 做法：预先为一批离散的 batch size（1, 2, 4, 8, 16...）分别 capture 图，运行时 padding 到最近的档位。

---

## Q18：FlashAttention 核心思想

**IO-Aware Attention**：标准 attention 要把完整的 `(N, N)` attention score 矩阵写到 HBM，FlashAttention 通过 online softmax 避免了这一点。

```
标准 attention：
  写 (N, N) score matrix 到 HBM：O(N²) 内存
  N=32K → 32K×32K×2 bytes ≈ 2 GB，HBM 读写各一次

FlashAttention（Tiling + Online Softmax）：
  外层 for q_block in (0, N_q, block_m):
    内层 for kv_block in (0, N_kv, block_n):
      scores = Q_block @ K_block^T          ← 在 SRAM 内完成
      online softmax update(m, l, acc)      ← 维护 running max/sum
  output[q_block] = acc / l                ← 只写一次 output 到 HBM

HBM 访问从 O(N²) 降到 O(N)，N=32K 时节省约 1000× 搬运量
```

是一种 fused kernel，把 QK matmul、softmax、V matmul 融合为一个 kernel，消除中间结果的 HBM 读写。

---

## Q19：Prefill 是 Compute-bound，Decode 是 Memory-bound

用 Roofline 模型判断：

```
H100 旁路点（ridge point）：
  peak FLOPS / peak bandwidth ≈ 990 TFLOPs / 3.35 TB/s ≈ 295 FLOPs/byte
  arithmetic intensity > 295 → compute-bound
  arithmetic intensity < 295 → bandwidth-bound

FFN matmul：(N, 4096) × (4096, 16384)
  权重大小（每次都要读）：4096 × 16384 × 2 bytes = 128 MB
  计算量：N × 4096 × 16384 × 2 FLOPs

Prefill（N=1024）：
  intensity = 1024 × 134M FLOPs / 128 MB ≈ 1070 FLOPs/byte >> 295
  → compute-bound ✓

Decode（N=1）：
  intensity = 134M FLOPs / 128 MB ≈ 1 FLOPs/byte << 295
  → bandwidth-bound ✓
```

Decode 的瓶颈是把权重从 HBM 搬进 SRAM，GPU 的 tensor core 大量空闲。这是量化（减少权重字节）和 MLA（减少 KV cache 带宽）在 decode 场景下收益显著的根本原因。

---

## Q20：量化（Quantization）

**动机**：Decode 是 bandwidth-bound，减少权重位宽 = 直接减少搬运量 = 接近线性加速。

```
FP16 → FP8：每权重 2 bytes → 1 byte，搬运带宽减半，decode 速度接近 2×
FP16 → INT4：每权重 2 bytes → 0.5 byte，理论 4× 带宽节省
```

### FP8 W8A8 实现（`layers/quantization/w8a8_fp8.py`）

```
Weight：离线量化（checkpoint 里已是 FP8），per-channel 存 scale
Activation：动态量化（每次 forward 实时），per-token

Forward 流程：
  input (BF16) → per_token_quant → input_fp8 + scale_in
  weight_fp8 × weight_scale（已存）
  FP8 GEMM（CUTLASS kernel，H100+ 原生支持）
  → output 自动 dequant 回 BF16

硬件要求：SM 89+（H100 等）
```

### 支持格式

- W8A8 FP8 / W8A8 INT8（`layers/quantization/` 目录）
- W4A8、W4A16（AWQ、GPTQ、Marlin）
- MXFP4、W4FP4 等

**FP8 vs INT8**：FP8 保留了浮点的动态范围，对 activation outlier 更鲁棒，是当前主流量化格式（DeepSeek-V3 训练即使用 FP8）。

---

## Q21：CudaGraphRunner vs PiecewiseCudaGraphRunner

### CudaGraphRunner（标准版，用于 decode）

把整个 forward pass capture 成**一张**完整 CUDA graph。

**限制**：只支持 decode（batch × 1 token，形状固定）。Prefill token 数量可变 + attention KV cache slot 动态，无法整体 capture。

### PiecewiseCudaGraphRunner（分段版，支持 prefill）

在特定**分割点**处把 forward 切断，每段分别 capture 成独立小图，分割点处的 op 在图外正常执行。

**分割点由 `@register_split_op()` 标注：**

```python
# parallel_state.py:157 ← TP AllReduce
@register_split_op()
def inplace_all_reduce(tensor, group_name): ...

# radix_attention.py:151 ← attention（KV cache 动态）
@register_split_op()
def forward_extend(...): ...

# MoE 显式添加：
compile_config.add_split_op("sglang.moe_forward_piecewise_cuda_graph_impl")
```

**一个 transformer layer 的切分结构：**

```
[CUDA Graph 片段]：QKV 投影 (ColumnParallelLinear)
─── 分割点：attention kernel（KV slot 动态）───
[CUDA Graph 片段]：O 投影 (RowParallelLinear)
─── 分割点：AllReduce（TP 通信）───
[CUDA Graph 片段]：FFN gate/up + activation
─── 分割点：AllReduce（TP 通信）───
[CUDA Graph 片段]：FFN down projection
```

**解决的问题：**

| 问题 | CudaGraphRunner | PiecewiseCudaGraphRunner |
|---|---|---|
| Prefill 支持 | ❌（仅 decode） | ✅（按 token 数量 bucket） |
| 动态 attention | ❌ 整图 capture 失败 | ✅ 分割点外执行 |
| TP AllReduce | 图内 capture 有限制 | ✅ 分割点外执行 |
| torch.compile 融合 | 有限 | ✅ 每段配合 compile 做 kernel fusion |

核心设计：把"可静态 capture 的计算密集段"与"必须动态执行的通信/attention"分离，前者用图加速，后者正常执行，交替运行。
