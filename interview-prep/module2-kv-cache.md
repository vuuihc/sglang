# 模块二：KV Cache 管理

> 基于 SGLang 代码库的面试备考精华笔记

---

## 核心数据结构全景

```
┌─────────────────────────────────────────────────────────────┐
│ RadixCache（前缀树，CPU侧）                                  │
│   管理 token 序列 → KV slot 索引的映射关系                   │
│   TreeNode.value = slot indices tensor                       │
│   TreeNode.lock_ref = 当前引用该节点的请求数                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ match_prefix → prefix_indices
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ReqToTokenPool（GPU，[max_requests, max_context_len]）       │
│   req_to_token[req_pool_idx, token_pos] = kv_slot_index     │
│   是"请求视角的地址翻译表"                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ slot 编号作为索引
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TokenToKVPool（GPU，物理 KV 存储）                           │
│   [num_layers, 2, num_slots, head_dim]                       │
│   slot 可被多个请求共享（前缀共享的物理载体）                │
└─────────────────────────────────────────────────────────────┘
```

---

## Q4：RadixCache 的 match_prefix 过程

### 数据结构

```python
class TreeNode:
    children: defaultdict(TreeNode)   # key = 第一个 page 的 token tuple
    key: RadixKey                     # 这段路径上的完整 token 序列
    value: torch.Tensor               # 对应的 KV slot indices
    lock_ref: int                     # 当前引用该节点的请求数
    last_access_time: float           # LRU 用
```

children 的 key（`child_key`）是取 token 序列前 `page_size` 个 token 组成的 tuple，所以查找 child 是 **O(1) hash lookup**，不是线性扫描。

### match_prefix 流程（`radix_cache.py:646`）

```python
child_key = key.child_key(self.page_size)           # 前 page_size 个 token → dict key
while len(key) > 0 and child_key in node.children:  # O(1) 字典查找
    child = node.children[child_key]
    prefix_len = child.key.match(key, page_size)     # 完整 key 比较

    if prefix_len < len(child.key):                  # 部分匹配 → split
        new_node = self._split_node(child.key, child, prefix_len)
        value.append(new_node.value)
        break
    else:                                             # 完全匹配 → 继续往下
        value.append(child.value)
        key = key[prefix_len:]
```

**split_node** 把现有 node 从中间断开：创建一个新 node 代表共同前缀，原 node 变成剩余部分的子节点，共享的 value（slot indices）被 clone 给新 node。

**match_prefix 返回的是 slot indices tensor**（`MatchResult.device_indices`），直接用于 write_cache_indices 写入新请求的 req_to_token 行，无需拷贝或重算 KV。

---

## Q5：KV Cache 显存满时的驱逐策略

**默认策略：LRU**（`cache_init_params.py:25`：`eviction_policy: str = "lru"`，`server_args.py:419` 同）

可选策略（`server_args.py:250`）：lru、lfu、slru、priority，代码里还实现了 fifo、filo、mru（`evict_policy.py`）。

**驱逐的物理约束（由树结构保证）：**

- 只能驱逐叶节点（或没有子节点的节点）
- `lock_ref > 0` 的节点（有 running request 在用）不可驱逐
- 驱逐从最少使用的叶节点向根部收缩，自然保护共享度高的前缀

驱逐触发点（`common.py:330`）：

```python
def evict_from_tree_cache(tree_cache, num_tokens):
    if allocator.available_size() < num_tokens:
        tree_cache.evict(EvictParams(num_tokens=num_tokens))
```

先驱逐，再 alloc，保证不直接 OOM。

---

## Q6：两层结构（ReqToTokenPool + TokenToKVPool）的设计原因

### 三个核心原因

**① 支持不连续的 paged 分配**

不同请求的 token 不需要占用物理连续 slot，`req_to_token[req_idx, pos]` 提供 position → slot 的间接寻址，attention kernel 通过这张表找到任意位置的 KV。

**② 前缀共享的实现机制（最关键）**

`write_cache_indices`（`common.py:104`）在 prefill 时做两件事：

```python
# 前缀部分：直接写入 RadixCache 匹配到的 slot 编号（共享！）
req_to_token[req_pool_idx, 0:prefix_len] = prefix_indices

# 后缀部分：写入新分配的 slot 编号
req_to_token[req_pool_idx, prefix_len:seq_len] = out_cache_loc
```

两个请求的 req_to_token 前几列指向**同一批物理 slot**，shared KV 无需拷贝或重算。

**③ slot 生命周期独立于请求生命周期**

多个请求可能共享同一个 slot（前缀共享），slot 不能随任一请求结束就被 free。TreeNode 的 `lock_ref` 追踪引用数，只有所有引用者都释放后，节点的 slot 才进入可驱逐状态。

### 完整请求生命周期数据流

```
① prefill（alloc_for_extend, common.py:429）
   alloc_req_slots()    → ReqToTokenPool 分配一行（req_pool_idx）
   alloc_token_slots()  → TokenToKVPoolAllocator 分配 suffix slot 编号
   write_cache_indices()→ 写 req_to_token：prefix 列 = 共享 slot，suffix 列 = 新 slot

② decode（alloc_for_decode, common.py:524）
   alloc 1个新 slot/request
   req_to_token[req_pool_idx, seq_len] = new_slot   （追加一列）

③ 请求完成（release_kv_cache, common.py:567）
   cache_finished_req() → RadixCache.insert() 把 token→slot 映射写入树（供后续请求复用）
   req_to_token_pool.free(req) → 释放该行
   slot 本身由 RadixCache 管理（lock_ref 归零后才可驱逐）
```

### 前缀共享的具体过程图

```
场景：Req A 先完成，Req B 来时有相同的前 100 token（系统提示）

RadixTree 在 A 完成后：
  root → [token 0..99] node_A.value = [slot7, slot3, ..., slot12]
                      ↘ [token 100..119] → [slot19, slot21, ...]（A 独有）

Req B 到来 → match_prefix 命中前 100 token
  prefix_indices = [slot7, slot3, ..., slot12]    ← 与 A 完全相同

write_cache_indices 写入 Req B 的 req_to_token 行：
  pos 0..99  → [slot7, slot3, ..., slot12]       ← 直接引用，0 拷贝 0 计算
  pos 100..114 → [slot88, slot91, ...]           ← B 独有，新分配

attention kernel for Req B：
  pos 0-99:   slot 7,3,...→ 读 A 算过的 KV 值   ← 无需重算 prefill！
  pos 100-114: slot 88,91,..→ 读 B 自己的 KV
```

---

## Q7：RadixAttention 与 PagedAttention 的关系，及与 vLLM 的对比

### 层次关系（不是"实现"关系）

```
┌────────────────────────────────┐
│  RadixAttention                │  ← KV cache 管理策略（调度器侧，CPU）
│  用前缀树管理哪些 slot 给谁    │    决定 req_to_token 怎么填写
└────────────────┬───────────────┘
                 │ req_to_token 表
                 ▼
┌────────────────────────────────┐
│  Paged Attention               │  ← 存储和计算机制（GPU kernel 侧）
│  FlashInfer / Triton kernel    │    按 req_to_token 查表，算 attention
└────────────────────────────────┘
```

**RadixAttention 不改变 attention 的计算方式**，只影响 `req_to_token` 如何构造。attention kernel 看到的只是一张 slot 编号表，不知道哪些 slot 是共享的。

### SGLang RadixTree vs vLLM APC（Automatic Prefix Caching）

**vLLM APC 的机制：**

```
以 block（16 token）为单位，链式 hash：
  hash(block_i) = sha256(token_ids[i*16:(i+1)*16] + hash(block_{i-1}))

查找：hash → block 字典，命中则复用
驱逐：block 级别 LRU，独立于上下游 block
```

**关键对比：**

```
┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
│                 │ vLLM APC                     │ SGLang RadixAttention        │
├─────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 数据结构        │ hash → block 平坦字典         │ Radix Tree（层级树）          │
│ 匹配粒度        │ 固定 block 边界（16 token）   │ 任意 token（page_size=1 时）  │
│ 层级感知        │ 隐式（链式 hash）             │ 显式（树路径）                │
│ 驱逐一致性      │ 需额外逻辑（孤儿 block 问题） │ 树结构天然保证（只驱逐叶节点）│
│ LoRA 隔离       │ 不支持                        │ extra_key 命名空间隔离        │
└─────────────────┴──────────────────────────────┴──────────────────────────────┘
```

**vLLM 孤儿 block 问题：**

```
多轮对话：Block0(system) → Block1(turn1) → Block2(turn2) → Block3(turn3)

vLLM LRU 可能先驱逐 Block1（LRU 顺序），但保留 Block2、Block3
→ Block2、Block3 的 KV 变成孤儿：前驱缺失，无法被复用
→ 这些 block 占着显存却没有使用价值，直到自己被 LRU 驱逐

SGLang 树结构的保证：
  root → [Block0] → [Block1] → [Block2] → [Block3]
  Block3 是叶节点，优先被驱逐
  Block1 是 Block2 的父节点，Block2 存在时 Block1 不会被驱逐
  一致性由数据结构自动维护
```

---

## page_size 的影响

**默认值：CUDA GPU 上为 1**（`server_args.py:3017-3020`），特殊情况：
- DeepSeek DSA：64
- Qwen3VL + aiter on ROCm：16

### page_size = 1 vs page_size = N 的取舍

| 维度 | page_size=1 | page_size=N（如16） |
|------|-------------|---------------------|
| 匹配精度 | 可精确到任意 token 边界 | 截断到 N 的整数倍 |
| 内碎片 | 零碎片 | 每个序列末尾最多浪费 N-1 个 slot（临时，decode 会续填） |
| 分配开销 | 每 token 一次 alloc | 每 page 一次 alloc，调用次数少 N 倍 |
| 内存访问 | 各 slot 可能散列 | page 内 N slot 连续，对 attention kernel 更友好 |
| 兼容性 | 通用 | 部分 backend（trtllm_mha）强制要求 |

**末尾碎片实际不大**（`alloc_extend_naive` 的 part1 逻辑）：extend 时会先填满上一个 page 的空余位置，新 decode token 也会续写到同一个 page 里，直到 page 填满才分配新 page。不满的 page 只在请求完成时才真正留下碎片。
