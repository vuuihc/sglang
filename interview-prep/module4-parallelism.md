# 模块四：并行与分布式

> 基于 SGLang 代码库的面试备考精华笔记

---

## Q13：Tensor Parallelism 的通信位置与 collective 类型

### ColumnParallelLinear + RowParallelLinear 必须配对

```
ColumnParallelLinear（QKV 投影 / FFN gate/up）：
  输入全量复制给每个 GPU
  → matmul → 每 GPU 得到输出的一列分片
  → 默认不通信（gather_output=False）
  → 若 gather_output=True 才做 AllGather（少数情况）

RowParallelLinear（out_proj / FFN down）：
  输入按行分片（对接 Column 的列输出）
  → matmul → 每 GPU 得到部分 sum
  → AllReduce（把所有 GPU 的部分 sum 加起来）
```

代码：`layers/linear.py:467`（Column → optional AllGather），`layers/linear.py:1543`（Row → AllReduce）。

### 关键认知

- 通信确实在 matmul 之后，但 Column 用 AllGather，Row 用 AllReduce
- Column + Row 配对才构成完整 TP 块
- 一个 transformer layer 通常只有一次 AllReduce（在 Row 之后），不是每个 matmul 后都通信
- AllGather 在输出需要完整结果时使用（如最后一层输出），中间层通常不做

```python
# ColumnParallelLinear.forward (linear.py:467)
output_parallel = self.quant_method.apply(self, input_, bias)
if self.gather_output:
    output = tensor_model_parallel_all_gather(output_parallel)  # AllGather，非默认
else:
    output = output_parallel  # 默认：保持分片

# RowParallelLinear.forward (linear.py:1543)
if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
    output = tensor_model_parallel_all_reduce(output_parallel)  # AllReduce
```

---

## Q14：MoE Expert Parallelism 的通信模式

**不是单向 scatter，是双向 All-to-All。**

```
step 1 dispatch（All-to-All）：
  每个 GPU 把本地 token 路由到各 expert 所在 GPU
  dispatch_indx 记录：每个 token 去哪个 GPU/expert

step 2 expert compute：
  各 GPU 对本地持有的 expert 做 FFN 计算（无通信）

step 3 combine（All-to-All）：
  把计算结果发回 token 来源 GPU
  combine_indx 记录：结果回到哪里、权重如何加权合并
```

代码：`layers/moe/topk.py:62-76`，`dispatch_indx` 和 `combine_indx` 对称存在。

moe_a2a_backend（`server_args.py:608`）：支持 deepep、ascend_fuseep 等不同后端。

### 与 TP 的区别

| | TP | EP |
|--|----|----|
| 分片对象 | 权重矩阵（按列/行切） | Expert（不同 GPU 持有不同 expert） |
| token 流向 | 每个 GPU 处理全部 token，权重部分 | token 被路由到特定 GPU，权重完整 |
| 通信模式 | AllReduce / AllGather | All-to-All × 2 |
| 通信时机 | 每个 FFN sub-layer 后 | dispatch 前 + combine 后 |

---

## Q15：MLA（Multi-head Latent Attention）压缩机制

### 核心思想

MLA 把 KV cache 从 `num_heads × head_dim × 2` 压缩到 `kv_lora_rank + qk_rope_head_dim`。

DeepSeek-V2 具体数字：`kv_lora_rank = 512`，`hidden_dim = 7168`，压缩比约 14x。

```python
# deepseek_v2.py:1370+
kv_a_proj_with_mqa:  # hidden_dim → kv_lora_rank + qk_rope_head_dim
kv_a_layernorm:      # RMSNorm on kv_lora_rank（压缩后做归一化）
```

### RoPE 解耦（关键设计）

标准 MHA 中 RoPE 施加于 K，但压缩后的 latent 不能直接加 RoPE（解压后才能施加）。

MLA 的解法：把 K 拆成两部分：
- `kv_lora_rank`：无 RoPE，**可以压缩存入 KV cache**
- `qk_rope_head_dim`：带 RoPE，不压缩，也存入 KV cache（但维度小）

KV cache 存储的是 `[kv_lora_rank + qk_rope_head_dim]`，而不是解压后的全量 KV。

### Absorb 优化

推理时可以把解压矩阵 absorb 进 Q/O 投影（矩阵合并），不存解压后的 K/V，直接用压缩 latent 做 attention，进一步减少显存和计算。

---

## Q16：Pipeline Parallelism 的 Bubble 比例

### 公式

```
bubble_ratio = (p - 1) / (m + p - 1)

p = pipeline stages（流水线级数）
m = microbatches（一次 step 切成几个 microbatch）
```

### 直觉

- m >> p 时：`bubble_ratio → 1/m → 0`，bubble 可忽略
- m = 1 时：`bubble_ratio = (p-1)/p`，接近全量空转（只有 1/p 有效计算）
- p = 1 时：无 bubble（单卡，但也没有 PP 加速）

**结论：** microbatch 数量 m 越大（"连续跑的时间越长"），首尾 bubble 的比例越小。这也是为什么 PP 适合大 batch 训练，推理场景（batch 小）PP 效率较低。

### SGLang 现状

SGLang 推理以 **TP + EP** 为主要并行策略，PP 支持有限（主要依赖 HF 的 `device_map` 做 naive PP）。EP 在 MoE 模型（DeepSeek 系列）中是核心并行维度。

---

## 并行策略总结对比

| 并行类型 | 分片对象 | 通信类型 | 适用场景 |
|---------|---------|---------|---------|
| TP | 权重矩阵（列/行） | AllReduce + AllGather | 单节点多卡，减少每卡显存 |
| EP | Expert（整块） | All-to-All × 2 | MoE 模型，不同 GPU 持有不同 expert |
| PP | 层（按深度切） | P2P（send/recv） | 多节点，模型太大装不进单节点 |
| DP | 数据（不同 batch） | AllReduce（梯度） | 训练为主；推理 DP = 多 replica |
