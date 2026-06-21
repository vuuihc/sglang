# 模块四：并行与分布式 + MLA 深度解析

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

### 为什么是 All-to-All 而不是 one-to-all

EP 必须与 DP 配合：batch 中的 token 分布在各 GPU（DP 分片），每个 token 路由到 top-k 个 expert，这些 expert 可能在任意 GPU 上。因此**每个 GPU 都要向其他每个 GPU 发送 token，也要接收**，是真正的多对多。

如果 token 全在一张卡上（无 DP），dispatch 方向是 one-to-many，combine 是 many-to-one。但这种情况在实际 EP 部署中不存在。

### 与 TP 的区别

| | TP | EP |
|--|----|----|
| 分片对象 | 权重矩阵（按列/行切） | Expert（不同 GPU 持有不同 expert） |
| token 流向 | 每个 GPU 处理全部 token，权重部分 | token 被路由到特定 GPU，权重完整 |
| 通信模式 | AllReduce / AllGather | All-to-All × 2 |
| 通信时机 | 每个 FFN sub-layer 后 | dispatch 前 + combine 后 |

---

## Q15：MLA（Multi-head Latent Attention）深度解析

### 1. 核心参数（DeepSeek-V2）

```
hidden_size      = 5120
num_heads        = 128
qk_nope_head_dim = 128   ← Q/K 中不带 RoPE 的部分
qk_rope_head_dim = 64    ← Q/K 中带 RoPE 的部分
v_head_dim       = 128
kv_lora_rank     = 512   ← KV 压缩维度（关键）
q_lora_rank      = 1536  ← Q 也做了低秩压缩
```

### 2. Q 也有低秩压缩（q_b_proj）

MLA 对 Q 同样做了 LoRA 式压缩（代码：`deepseek_v2.py:1445`）：

```
标准 MHA Q：h(5120) → W_Q(5120,24576) → [128 heads, 192]  参数量：1.26 亿

MLA Q（两步）：
  q_a_proj：h(5120) → c_Q(1536)                参数量：787万
  q_a_layernorm：RMSNorm(c_Q)
  q_b_proj：c_Q(1536) → [128 heads, 192]        参数量：3774万
  合计：4561万，节省约 63%
```

### 3. KV cache 存什么

```
输入 h(5120) → kv_a_proj(5120→576) → [c_KV(512) | k_pe(64)]
                                         ↓           ↓
                                      无 RoPE      带 RoPE
                                      进 cache     进 cache

KV cache 每 token 存：576 个值
标准 MHA 每 token 存：128 × 128 × 2 = 32768 个值
压缩比：57×
```

### 4. E2E Forward（以 decode 单 token 为例）

```
Step 1  输入投影（1 次 forward）
  h(1,5120) → fused_a_proj → [c_Q(1,1536) | c_KV(1,512) | k_pe(1,64)]
  写入 KV cache：[c_KV(512) | k_pe(64)] = 576 值

Step 2  Q 展开
  c_Q(1,1536) → q_a_layernorm → q_b_proj → [q_nope(1,128h,128) | q_pe(1,128h,64)]
  对 q_pe 施加 RoPE（当前位置）

Step 3  从 cache 读历史 K/V
  c_KV(seq,512),  k_pe(seq,64) ← 读出所有历史 token

Step 4  Attention（标准多头）
  K = [解压k_nope(seq,128h,128) | k_pe_rotated(seq,128h,64)]
  Q @ K^T → scores(1,128h,seq) → softmax → attn_weights
  attn_weights @ V(seq,128h,128) → output(1,128h,128)

Step 5  输出
  concat all heads(1,16384) → o_proj(16384→5120)
```

### 5. Absorb 优化：把"解压 K/V"这步彻底消掉

**问题：** Step 3 读出 c_KV(seq,512) 后，要解压成 K_nope(seq,128h,128)，这是个大张量。

**数学变换（以头 h 为例，标注每步 shape）：**

```
原始 attention score：
  scores_h = Q_nope_h (1,128)  @  K_nope_h^T (128,seq)

代入 K_nope_h 的定义  K_nope_h = c_KV (seq,512) @ W_kv_b_K_h (512,128)：

  = Q_nope_h (1,128)
    @
    [c_KV (seq,512) @ W_kv_b_K_h (512,128)]^T

展开转置  [AB]^T = B^T A^T：

  = Q_nope_h (1,128)  @  W_kv_b_K_h^T (128,512)  @  c_KV^T (512,seq)

重新加括号（结合律）：

  = [Q_nope_h (1,128)  @  W_kv_b_K_h^T (128,512)]  @  c_KV^T (512,seq)
     └──────────────── Q_absorbed_h (1,512) ────────┘

  = Q_absorbed_h (1,512)  @  c_KV^T (512,seq)  →  scores_h (1,seq)
```

W_kv_b_K_h 从 **K 侧消失**，被挪进了 Q 侧。K 侧只剩 c_KV(seq,512)，无头编号。

**Offline 权重合并（模型加载时一次性）：**
```
W_absorb_h = W_q_b_nope_h (1536,128) @ W_kv_b_K_h^T (128,512) → (1536,512)
替换 q_b_proj 的 nope 权重，forward 时直接：
  c_Q (1,1536) @ W_absorb_h (1536,512) → Q_absorbed_h (1,512)
```

V 侧同理（加权 c_KV 后再过 W_kv_b_V，并与 o_proj 合并）。

### 6. Absorb 后 K/V 的真实形状：MQA

```
Q_absorbed_all: (1, 128 heads, 512)  ← 128 个不同的 Q（W_absorb_h per head 不同）
c_KV (cache):   (seq, 1 head,  512)  ← 只有 1 头，被 128 个 Q 头共享！

Attention：(1,128h,512) @ broadcast(seq,1h,512)^T → (1,128h,seq)
           128 个头 scores 不同，因为 Q_absorbed_h 不同
           K 侧始终只有 1 头，无需解压
```

128 头的多样性全部在 Q 侧体现（每头有不同的 W_absorb_h），K/V 侧是 MQA（1 头）。

### 7. Absorb 的计算量 tradeoff

```
标准 MHA attention FLOPs：
  Q(1,128h,128) @ K^T(128h,128,seq) + attn @ V(128h,seq,128)
  = 2 × 128 × 128 × seq = 32768 × seq

MLA Absorb attention FLOPs：
  Q_absorbed(1,128h,512) @ c_KV^T(512,seq) + attn @ c_KV(seq,512)
  = 2 × 128 × 512 × seq = 131072 × seq   ← FLOPs 多 4×

但 KV cache 读取带宽：
  标准 MHA：seq × 32768 values
  MLA：      seq × 576 values             ← 带宽少 57×
```

Decode 是 memory-bandwidth bound，GPU compute 单元大量空闲。多出的 4× FLOPs 打到空闲 compute 上，不增加 wall clock time；少搬运 57× 数据直接减少耗时。Absorb 本质是**用计算换带宽**，在 decode 场景下是合算的。

---

## Q16：Pipeline Parallelism 的 Bubble 比例

### Microbatch 是什么

PP 把模型按层切成 P 段。若不拆 batch，各 GPU 只能顺序等待（GPU 0 算完才给 GPU 1），大量空转。

引入 m 个 microbatch（把 batch 切成 m 小块依次送入）：

```
t1: GPU0:mb1  GPU1:---  GPU2:---  GPU3:---
t2: GPU0:mb2  GPU1:mb1  GPU2:---  GPU3:---
t3: GPU0:mb3  GPU1:mb2  GPU2:mb1  GPU3:---
t4: GPU0:mb4  GPU1:mb3  GPU2:mb2  GPU3:mb1  ← 流水线满
t5: GPU0:---  GPU1:mb4  GPU2:mb3  GPU3:mb2
t6: GPU0:---  GPU1:---  GPU2:mb4  GPU3:mb3
t7: GPU0:---  GPU1:---  GPU2:---  GPU3:mb4

总时长 = m + p - 1 = 4 + 4 - 1 = 7
有效计算 = m = 4，空转 = p - 1 = 3
```

### 公式

```
bubble_ratio = (p - 1) / (m + p - 1)

m=1, p=4：3/4 = 75% 空转
m=4, p=4：3/7 = 43%
m=8, p=4：3/11 = 27%
m→∞：    → 0%
```

m 越大，流水线利用率越高。PP 适合大 batch 训练，推理 batch 小时效率较低。SGLang 推理以 TP + EP 为主，PP 支持有限。

---

---

## EP 补充：常见追问

### EP + DP 的具体例子（为什么需要 DP）

```
4 GPU，64 expert，每 GPU 持有 16 个 expert：
  GPU 0：expert 0-15
  GPU 1：expert 16-31
  GPU 2：expert 32-47
  GPU 3：expert 48-63

没有 DP（所有 token 在 GPU 0）：
  Dispatch：GPU 0 → GPU 1, 2, 3    one-to-many
  Combine： GPU 1, 2, 3 → GPU 0    many-to-one
  GPU 3 可能全程空闲，GPU 0 是瓶颈

加上 DP（每个 GPU 处理不同请求）：
  GPU 0 有 token A, B；GPU 1 有 token C, D...
  token A（在 GPU 0）→ expert 17(GPU 1) + expert 35(GPU 2)
  token C（在 GPU 1）→ expert 3(GPU 0) + expert 49(GPU 3)
  → 每个 GPU 都要发送也要接收 → 真正的 All-to-All
```

### EP 和 TP 能同时用吗

可以，DeepSeek 标准部署就是 TP + EP：
- Attention 层：TP（节点内多卡，AllReduce）
- MoE FFN 层：EP（跨节点，All-to-All × 2）
- 单个大 expert 本身也可以再做 TP

### EP 路由是 per-token 独立的

```
router_logits = token_hidden_states @ W_router  # (total_tokens, num_experts)
top_k = argmax(router_logits, k=2)              # (total_tokens, 2) ← 每 token 独立选

token A：expert 3, expert 17
token B：expert 5, expert 22    ← 和 A 不同
token C：expert 8, expert 17    ← 和 A 部分重叠
```

All-to-All 把每个 token 精准发到它选中的 top-k expert GPU，不是广播后 discard。

### dispatch + combine 是两次独立的 All-to-All 调用

```
all_to_all_dispatch()   ← 第一次：token → expert GPU
expert_compute()        ← FFN 计算，无通信
all_to_all_combine()    ← 第二次：结果 → 回到来源 GPU，加权合并
```

deepep 库里就是 `dispatch()` 和 `combine()` 两个独立 API。

### All-to-All 的同步是天然的

All-to-All 不需要额外协调：每次 run_batch 里，attention 层的 TP AllReduce 已经是一个同步点，AllReduce 结束时所有 GPU 同时完成 attention，自然一起进入 MoE FFN 层发起 All-to-All。

---

## Mixed Batch 与 Attention Kernel 差异

### Mixed batch 的实际执行

```
线性层（QKV 投影、FFN 等）：
  所有 token（prefill + decode）拼成一个大 tensor：
  input: (total_tokens, hidden_dim)
  → 一个大矩阵乘，GPU 一视同仁
  → TP AllReduce 也是对整个 tensor 做一次

Attention 层（唯一分叉的地方）：
  prefill token：FlashAttention extend kernel（token 间互相 attend，causal mask）
  decode token：Paged attention decode kernel（单 Q 读 KV cache）
  → SGLang 用 FlashInfer 分别处理，结果拼回 (total_tokens, hidden_dim) 继续走

EP All-to-All：
  router 对所有 token 各自独立算 top-k → 一次 dispatch → 一次 combine
  prefill 和 decode token 混在同一次 All-to-All 里
```

### Prefill vs Decode Attention 的 kernel 差异

| | Prefill | Decode |
|--|---------|--------|
| Q 规模 | N_q 个 token（可能几千） | 每请求 1 个 token |
| 瓶颈 | Compute-bound（Q 间互相 attend，O(N²)） | Bandwidth-bound（读 KV cache）|
| Tiling | block_m（Q 维度）+ block_n（KV 维度） | 无 block_m（Q=1），split-k 切 KV 维度 |
| 并行来源 | Q blocks 间并行，SM 自然打满 | split-k 把 KV 序列切块交给多 SM 并行 |

**FlashAttention（Prefill）：**
```
for q_block in (0, N_q, block_m):        # 外层：Q 维度 tile
    for kv_block in (0, N_kv, block_n):  # 内层：KV 维度 tile
        partial = Q_block @ K_block^T
        online softmax update(m, l, acc)  # 不写 HBM，SRAM 内累加
    output[q_block] = acc / l
```

**FlashDecoding（Decode，split-k）：**
```
# 多个 thread block 并行，各自负责 KV 序列的一段：
block_i 处理 KV[i*C : (i+1)*C]：
    partial_out_i, m_i, l_i = softmax(Q @ K_chunk^T) @ V_chunk

# 归约（log-sum-exp 合并，满足结合律，可树形归约 O(log C)）：
merge(A, B):
    m_new = max(m_A, m_B)
    alpha_A = exp(m_A - m_new);  alpha_B = exp(m_B - m_new)
    l_new   = alpha_A * l_A + alpha_B * l_B
    out_new = (alpha_A * l_A * out_A + alpha_B * l_B * out_B) / l_new
```

merge 操作满足结合律，可以两两迭代 m_i / l_i / acc，与 FlashAttention 内层的 online softmax update 是同一个操作。

---

## 并行策略总结对比

| 并行类型 | 分片对象 | 通信类型 | 适用场景 |
|---------|---------|---------|---------|
| TP | 权重矩阵（列/行） | AllReduce + AllGather | 单节点多卡，减少每卡显存 |
| EP | Expert（整块） | All-to-All × 2 | MoE 模型，不同 GPU 持有不同 expert |
| PP | 层（按深度切） | P2P（send/recv） | 多节点，模型太大装不进单节点 |
| DP | 数据（不同 batch） | AllReduce（梯度） | 训练为主；推理 DP = 多 replica |
