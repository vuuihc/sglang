# 06 · DSA 的 indexer 与 IndexCache 工作原理

> 对应 **Q9**:indexer / IndexCache 具体怎么工作。

## 0. 背景:DSA 要解决什么

标准注意力是 **稠密** 的:第 N 个 token 要和前面所有 N−1 个 token 算注意力,
长上下文下复杂度 O(N²),KV 越长越慢。

**DSA(DeepSeek Sparse Attention)** 的想法:
> 对每个 query,**并不是所有历史 token 都重要**。先用一个**极轻量的打分器**快速挑出
> 最相关的 **top-k** 个历史 token(比如 k=2048),**只对这 k 个**做真正的(昂贵的)MLA 注意力。

这个"轻量打分器"就是 **lightning indexer**;"挑 top-k"的结果可以跨层复用,就是 **IndexCache**。

## 1. Lightning Indexer:一个迷你 MQA 打分器

看代码 [dsa_indexer.py:302-381](../../python/sglang/srt/layers/attention/dsa/dsa_indexer.py) 的 `Indexer.__init__`,它有自己一套**很小**的投影:

```python
self.wq_b         = ReplicatedLinear(q_lora_rank, n_heads * head_dim)  # indexer 自己的 query
self.wk           = ReplicatedLinear(hidden_size, head_dim)            # 只 1 份 key → MQA！
self.weights_proj = ReplicatedLinear(hidden_size, n_heads)            # 每个 indexer head 的权重
self.k_norm       = LayerNorm(head_dim)
self.rotary_emb   = get_rope_wrapper(rope_head_dim, ...)
self.softmax_scale = head_dim ** -0.5
```

关键点:
- **`head_dim` 很小**(indexer 专用的小维度),所以打分极便宜——这是"lightning"的由来。
- **`wk` 输出只有 1 份 key(MQA 风格)**:所有 indexer head 共享同一份 key,
  进一步省算力和显存。代码里那套 `_MQA_LOGITS_*` 常量就是在管这个打分 logits 的显存预算。
- indexer 用的是 **FP8** 量化的 q/k(`q_fp8`、`kv_cache_fp8`,见 589 行附近),
  打分这一步本身也用低精度加速。

### 打分 → 选 top-k 的流程

对位置 i 的 query,indexer 大致做:

```
1. q_idx = wq_b(q_lora)             # indexer query, 形状 [n_heads, head_dim]
2. k_idx = k_norm(wk(hidden)) + RoPE  # indexer key(每个历史 token 1 份, MQA)
3. logits[i, j] = Σ_head  w[i,head] · (q_idx[head] · k_idx[j]) · softmax_scale
                  # weights_proj 给每个 head 一个权重 w，再对 head 求和
4. topk_indices = top_k(logits[i, :], k = index_topk)   # 选出最相关的 k 个历史 token
```

得到的 `topk_indices` 交给主注意力后端:**MLA 只在这 k 个被选中的 KV 上做注意力**,
而不是全部历史。复杂度从 O(N²) 降到 O(N·k),k 固定(如 2048),长上下文收益巨大。

> 对比:主 MLA 注意力用的是大维度、多 head、潜在压缩的 KV;
> indexer 是另一套**独立的小投影**,唯一职责是"快速给历史 token 排个序、挑出 top-k"。
> 它**不参与最终输出的数值计算**,只决定"看哪些 token"。

## 2. 短序列自动退回稠密 MHA

挑 top-k 是有固定开销的。序列很短时(默认 < 2048 token),
"挑 2048 个中的 top-2048"没意义,稀疏化反而亏。所以 DSA 后端会**自动**对短 prefill 退回标准 MHA
(`MHA_ONE_SHOT`,一次算完所有 token),无需手动开关。
阈值可用 `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` 调大(见 [deepseek_v32.md](../basic_usage/deepseek_v32.md)),
阈值越大越偏向稠密(更快但可能掉精度)。

## 3. IndexCache:跨层复用 top-k(`index_topk_pattern`)

### 动机
每一层都跑一次 indexer 打分 + top-k,**本身也有成本**。
观察:**相邻层选出来的 top-k 历史 token 高度相似**(语义上"重要的 token"层间变化不大)。
于是:**让一部分层直接复用前面某层算好的 top-k,跳过自己的 indexer 计算。** 这就是 IndexCache。

### 怎么配
GLM-5 推荐(见 [deepseek_v32.md](../basic_usage/deepseek_v32.md)):

```bash
--json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
```

这个字符串**长度 = 层数**,每个字符对应一层:

| 字符 | 含义 | 成本 |
|---|---|---|
| **`F`** = Fresh | 这一层**自己跑 indexer**,重新计算 top-k | 贵(完整打分) |
| **`S`** = Shared | 这一层**复用上游 F 层算好的 top-k**,跳过 indexer | 便宜(几乎免费) |

代码佐证 [server_args.py:2378-2388](../../python/sglang/srt/server_args.py):`"S" in index_topk_pattern`
被称作 "**shared layers**",且注释说"shared layers 依赖跨层传播的 topk indices"——
即 S 层不自己算,靠 F 层把 indices 传下来。`index_topk_freq` 是另一种等价表达(每隔几层 fresh 一次)。

### 数字直觉
设 76 层,pattern 里约一半是 `S`:
- 全 `F`:76 次 indexer 计算。
- 一半 `S`:约 38 次 → **indexer 开销直接砍半**,而最终注意力质量几乎不变
  (因为 top-k 集合层间本就高度重合)。
- 文档定性:"negligible accuracy loss"(可忽略的精度损失)换"better tradeoff between speedup and performance"。

### 一个坑(面试加分)
IndexCache **不兼容 `--enable-two-batch-overlap`(TBO)**:
TBO 的算子路径不会跨层传播 topk indices,S 层会"没有 indices 还硬跑稀疏注意力"→ 报错。
代码在 [server_args.py:2382](../../python/sglang/srt/server_args.py) 显式 `raise ValueError` 拦截。
所以开 IndexCache 时别同时开 TBO。

## 4. 全链路串一遍

```
hidden_states
   │
   ▼ (每层) 这一层是 F 还是 S？
   │     F：lightning indexer 打分 → 选 top-k → 缓存这组 indices
   │     S：直接取上游 F 层缓存的 top-k indices
   ▼
   主 MLA 注意力：只在这 top-k 个历史 KV 上计算   ← O(N·k)，长上下文加速核心
   ▼
   FFN / MoE …
```

## 5. 一句面试话

> "DSA 的 indexer 是一个独立的迷你 MQA 打分器(自己的小投影、单份 key、FP8 打分),职责只有一个:为每个 query 从全部历史 token 里快速挑出 top-k 最相关的,主 MLA 注意力只对这 k 个算,把长上下文复杂度从 O(N²) 降到 O(N·k)。IndexCache 进一步发现相邻层挑出的 top-k 高度重合,于是用 `index_topk_pattern` 让标 `S` 的层复用标 `F` 的层算好的 indices、跳过自己的 indexer,几乎不掉精度地把 indexer 开销砍半。注意它和 TBO 互斥。"
</content>
