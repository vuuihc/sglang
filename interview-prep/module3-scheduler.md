# 模块三：调度器设计

> 基于 SGLang 代码库的面试备考精华笔记

---

## 核心概念：两个数据结构

```
ScheduleBatch（CPU 侧，调度器管理）
  reqs: List[Req]         Python 对象，包含完整请求状态
  prefix_lens: List[int]  每个请求的缓存前缀长度
  extend_lens: List[int]  每个请求本次需计算的 token 数
  seq_lens: torch.Tensor  每个请求的当前序列总长度
  out_cache_loc: Tensor   已分配的 KV slot 编号
  forward_mode: ForwardMode

         │  ForwardBatch.init_new(batch, model_runner)
         ▼

ForwardBatch（GPU 侧，ModelRunner 执行）
  input_ids: Tensor         GPU tensor
  req_pool_indices: Tensor  GPU tensor
  seq_lens: Tensor          GPU tensor
  out_cache_loc: Tensor     GPU tensor
  extend_prefix_lens: Tensor  仅 prefill 模式填充
  extend_seq_lens: Tensor     仅 prefill 模式填充
```

---

## Q8：Continuous Batching 解决了什么

Static batching 有两个问题：

1. **Padding 浪费**：batch 内所有序列填充到最长长度，短序列的 padding token 消耗算力
2. **等待气泡**：整批请求全部完成才接受新请求，GPU 在等人

CB 的核心是解决第二点：**请求随时加入和离开 running batch**，不用等一批全部完成。

注意："100% 计算密度"是过度表述：
- CUDA Graph 要求固定 batch size，维护多个图仍有离散误差
- Decode 阶段天然 memory-bandwidth bound，这是 auto-regressive 的固有特性，不是调度问题

准确说法：CB 让 GPU **几乎不空转**，消除了 GPU 等人的 bubble。

---

## Q9：Prefill vs Decode 计算特性与 PD 拆分

| | Prefill | Decode |
|-|---------|--------|
| seqlen | 长（几百到几千） | 1（每次生成一个 token） |
| 瓶颈类型 | Compute-bound（矩阵乘法主导） | Memory-bandwidth bound（读 KV cache + 权重） |
| 适合硬件 | 高 FLOPS 设备 | 高带宽设备 |

**混部的问题（head-of-line blocking）：**
长 prefill 占满一个 forward slot，期间 decode 请求无法出 token，TPOT 飙升，影响用户体验。

**PD Disaggregation 的代价：**
KV cache 需要通过网络从 prefill 机器传到 decode 机器（SGLang 支持 mooncake/nixl/mori 等传输后端，`disaggregation/` 目录）。传输延迟计入 TTFT，所以 PD 只在 prefill/decode 负载足够大时才合算。

---

## Q10：Chunked Prefill

**动机：** 长 prefill（比如 32K token）一次 forward 要占用很长时间，期间 decode 请求全部 block。

**解法：** 把 prefill 切成固定大小的 chunk（`rem_chunk_tokens`，一般 4096 token），每个 iteration 最多处理一个 chunk。超出部分下一轮继续。

**SGLang 里的实现：** `ForwardMode.MIXED`（`forward_batch_info.py:82`）——同一次 GPU forward 里既有 prefill chunk 又有 decode 请求，两者混合执行。

**新的调度复杂性：**
- 同一请求跨多个 iteration 的状态需要保留（`ScheduleBatch.chunked_req` 字段）
- `extend_input_len` 在 chunk 截断时会被修改为 chunk 大小，不是完整 extend 长度
- 要防止 chunked 请求在多轮 radix tree insert 时自引用导致 hit_count 虚高（代码里 `chunked=True` 跳过 hit_count 更新）

---

## Q11：ScheduleBatch → ForwardBatch 转换做了什么

**主要做两件事，不是过滤策略。**

### ① 真正的重活在之前：KV cache 分配

`ForwardBatch.init_new()` 调用之前，`scheduler.run_batch()` 里已经：
- `alloc_for_extend()` → 分配 prefill slot，填写 `req_to_token`
- `alloc_for_decode()` → 分配 decode slot，追加 `req_to_token`
- 结果存在 `batch.out_cache_loc`（GPU tensor）

### ② init_new 本身做的事

**CPU Python 对象 → GPU tensor 的结构化打包（`forward_batch_info.py:496`）：**

```python
ret = cls(
    input_ids        = batch.input_ids,           # GPU tensor
    req_pool_indices = batch.req_pool_indices,    # GPU tensor
    seq_lens         = batch.seq_lens,            # GPU tensor
    out_cache_loc    = batch.out_cache_loc,       # GPU tensor（已分配好的 slot 编号）
    ...
)
```

**按 ForwardMode 选择性填充字段（`forward_batch_info.py:462`）：**

```python
if batch.forward_mode.is_decode_or_idle():
    extend_seq_lens = extend_prefix_lens = None   # decode 不需要这些
else:
    extend_seq_lens    = batch.extend_lens        # prefill 才需要
    extend_prefix_lens = batch.prefix_lens
```

ScheduleBatch 是调度器侧的 Python 对象，ForwardBatch 是 GPU kernel 需要的 tensor 集合，两者的分离让调度逻辑（CPU）和执行逻辑（GPU）各自保持干净。

---

## Q12：LPM Starvation 与调度策略

### 默认策略

`server_args.py:408`：`schedule_policy: str = "fcfs"`  
**默认是 FCFS（先来先到），不是 LPM。**

### 策略分类（`schedule_policy.py:125`）

```python
class CacheAwarePolicy(Enum):
    LPM = "lpm"          # Longest Prefix Match，优先调度缓存命中多的请求
    DFS_WEIGHT = "dfs-weight"

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"        # 先来先到（默认）
    LOF = "lof"          # Longest Output First
    RANDOM = "random"
    ROUTING_KEY = "routing-key"
```

### LPM 的 Starvation 问题

`_sort_by_longest_prefix`（`schedule_policy.py:276`）按 `prefix_indices` 长度降序排队。

如果系统里有大量请求共享某个热门 system prompt（命中长前缀），它们会反复排在队列前面，而没有共享前缀的新请求一直被排在后面——**starvation（饥饿）**。

### 自动防饥饿机制（`schedule_policy.py:203`）

```python
def _determine_active_policy(self, waiting_queue):
    if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
        return CacheAgnosticPolicy.FCFS   # 队列超 128 自动回退 FCFS
    return self.policy
```

触发条件是**队列长度 > 128**（不是时间阈值）。队列积压多时：
1. 有 starvation 风险需要公平保障
2. LPM 的 `O(n log n)` 排序 + 每请求 radix tree 查询开销也太大

### In-Batch Prefix Caching 去优先机制

`_compute_prefix_matches` 里的 `temporary_deprioritized`（`schedule_policy.py:234`）：

当 waiting_queue 里多个请求共享相同前缀但没有缓存命中时，只让**第一个**进 batch 做 prefill（它会把结果写入 RadixCache），其余请求**暂时降权**等下一轮——此时后续请求可以直接命中缓存，避免多个请求重复计算同一段前缀。

---

## 延伸："extend" 命名的含义

`extend` 不是 prompt 的总长度，而是**这次 forward 需要实际计算 KV 的 token 数**：

```
extend_input_len = len(fill_ids) - len(prefix_indices)
```

举例：
```
请求输入：[system prompt 80 tokens] + [user question 20 tokens]
RadixCache 命中：前 80 tokens 的 KV 已缓存

fill_ids          = 100 tokens（完整输入序列）
prefix_indices    = 80 slots（已缓存，KV 不需要重算）
extend_input_len  = 20       ← 只计算这 20 个 token 的 KV

ForwardBatch 字段：
  seq_len            = 100   attention 时 attend 的总范围（含 prefix）
  extend_prefix_lens = 80    attention kernel：pos 0-79 读缓存 KV
  extend_seq_lens    = 20    attention kernel：pos 80-99 是本次新算的
```

**命名语义：** 在已有缓存（prefix）的基础上往后"延伸"（extend）计算新 token 的 KV。
对缓存命中 0 的请求，extend = full prefill；对缓存命中 80% 的请求，extend 只有剩下的 20%。

**Decode 模式下 extend 字段为 None**（`forward_batch_info.py:463`）：decode 每次只生成 1 个 token，没有 prefix vs extend 的区分概念。
