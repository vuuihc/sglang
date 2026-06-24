# SGLang 深入浅出（一）：从一棵树到一个缓存——Radix Cache 的设计与实现

> 本系列通过阅读 SGLang 源码，深入理解 LLM 推理引擎的核心设计。第一篇从 Radix Cache 切入，理解 SGLang 如何用一棵基数树实现 KV Cache 的高效复用。

## 1. 从一个问题开始

假设你正在运行一个 LLM 服务，同时处理三个请求：

```
请求 A: [系统提示] + "请翻译以下句子为英文：今天天气很好"
请求 B: [系统提示] + "请翻译以下句子为英文：明天会下雨吗"
请求 C: [系统提示] + "请总结以下文章：..."
```

三个请求共享同一段系统提示（可能长达数千 token）。在传统的推理引擎中，每个请求都会独立地对系统提示做 Prefill 计算，生成各自的 KV Cache。这意味着同一段文本的 KV Cache 被重复计算了三次——GPU 算力和显存都被浪费了。

SGLang 的 RadixAttention 机制正是为了解决这个问题而设计的：**用一棵基数树（Radix Tree）管理所有请求的 KV Cache，自动识别和复用共享前缀**。这个设计发表在 NeurIPS 2024 论文《SGLang: Efficient Execution of Structured Language Model Programs》中 [1]，实现了最高 6.4 倍的吞吐量提升。

## 2. 什么是基数树？

在理解 Radix Cache 之前，先回顾一下基数树（也叫压缩前缀树 / Patricia Tree）。

普通的前缀树（Trie）中，每条边代表一个字符。如果一条路径上没有分叉，就会产生大量只有一个子节点的中间节点，造成空间浪费：

```
普通 Trie:
  root → '你' → '好' → '世' → '界'
                    → '吗'
```

基数树通过**压缩无分叉路径**来解决这个问题——将连续无分叉的边合并为一条：

```
基数树:
  root → '你好' → '世界'
                → '吗'
```

这个"压缩"特性对 KV Cache 管理至关重要：一段连续的 token 序列可以作为一个整体存储在单个节点中，既节省了节点数量，又保持了前缀匹配的效率。

基数树在计算机科学中有广泛应用。最经典的例子是路由表查找——Linux 内核的路由缓存（LC-trie）和 IPv6 路由表都用基数树实现 [2]。SGLang 将这个数据结构引入 LLM 推理，用 token 序列替代 IP 地址前缀，用 KV Cache 张量替代路由条目，实现了异曲同工的高效前缀匹配。

## 3. SGLang 的 Radix Cache：一棵会呼吸的树

### 3.1 核心数据结构

SGLang 的 Radix Cache 定义在 `python/sglang/srt/mem_cache/radix_cache.py` 中。核心数据结构是 `TreeNode`：

```python
class TreeNode:
    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)  # 子节点
        self.parent: TreeNode = None           # 父节点
        self.key: RadixKey = None              # 该节点对应的 token 序列
        self.value: Optional[torch.Tensor] = None  # KV Cache 的 GPU 显存索引
        self.lock_ref = 0                      # 引用计数（被多少请求持有）
        self.last_access_time = time.monotonic()
        self.hit_count = 0
        self.host_value: Optional[torch.Tensor] = None  # CPU 端的 KV Cache 索引
        self.hash_value: Optional[List[str]] = None     # 每个 page 的哈希值
        self.priority = priority
```

每个 `TreeNode` 存储了：
- **key**：该节点代表的 token 序列（`RadixKey` 类型，支持普通模式和 bigram 模式）
- **value**：对应的 KV Cache 在 GPU 显存中的位置索引（一个 `torch.Tensor`）
- **lock_ref**：引用计数，用于保护正在使用的 KV Cache 不被驱逐

而 `RadixCache` 类管理着整棵树：

```python
class RadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.root_node = TreeNode()       # 根节点
        self.evictable_size_ = 0          # 可驱逐的 token 数
        self.protected_size_ = 0          # 受保护的 token 数
        self.eviction_strategy = ...       # 驱逐策略（LRU/LFU/FIFO 等）
```

### 3.2 一棵"活"的树：Match → Insert → Evict

Radix Cache 不是一棵静态的树，它在请求的生命周期中不断变化。三个核心操作构成了它的"呼吸"：

**Match（匹配）**：当一个新请求到来时，沿着树找到最长的共享前缀：

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    key = params.key
    key = key.page_aligned(self.page_size)
    value, last_node = self._match_prefix_helper(self.root_node, key)
    # 返回匹配到的 KV Cache 索引和终止节点
    return MatchResult(device_indices=value, last_device_node=last_node, ...)
```

**Insert（插入）**：请求完成后，将新的 KV Cache 插入树中：

```python
def insert(self, params: InsertParams) -> InsertResult:
    key, value = key.maybe_to_bigram_view(self.is_eagle, value)
    key = key.page_aligned(self.page_size)
    prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
    return InsertResult(prefix_len=prefix_len)
```

**Evict（驱逐）**：当显存不足时，按驱逐策略释放最久未使用的 KV Cache：

```python
def evict(self, params: EvictParams) -> EvictResult:
    # 按策略选择可驱逐的叶子节点，释放其 value
    ...
```

### 3.3 一个完整的例子

让我们用一个具体例子来走一遍完整流程。

**初始状态**：空树，只有根节点。

```
root
```

**请求 A 到来**：`[系统提示] + "翻译：今天天气很好"`

1. `match_prefix`：没有匹配，返回空
2. Prefill 计算整段 token 的 KV Cache
3. `cache_unfinished_req`：将 KV Cache 插入树

```
root → [系统提示 + "翻译：今天天气很好"] (node_A, value=GPU索引)
```

**请求 B 到来**：`[系统提示] + "翻译：明天会下雨吗"`

1. `match_prefix`：匹配到 `[系统提示]` 部分，但树中只有一个大节点，需要 split
2. `_split_node`：将 `node_A` 拆分为共享前缀和独有后缀

```
root → [系统提示] (node_shared, value=GPU索引_1)
         → ["翻译：今天天气很好"] (node_A, value=GPU索引_2)
```

3. 请求 B 只需 Prefill `"翻译：明天会下雨吗"` 部分，复用 `[系统提示]` 的 KV Cache
4. `cache_unfinished_req`：插入 B 的独有部分

```
root → [系统提示] (node_shared)
         → ["翻译：今天天气很好"] (node_A)
         → ["翻译：明天会下雨吗"] (node_B)
```

这就是 Radix Cache 的核心价值：**共享前缀只存储一份 KV Cache，新请求只需计算差异部分**。

## 4. Split：树的动态重组

Split 是 Radix Cache 中最精妙的操作。当新请求的前缀只匹配到某个节点的一部分时，需要将这个节点"劈开"：

```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    # new_node -> child
    new_node = TreeNode(priority=child.priority)
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.key = child.key[:split_len]          # 共享前缀
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]              # 独有后缀
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node

    # hash_value 也需要相应拆分
    new_node.hash_value, child.hash_value = split_node_hash_value(
        child.hash_value, split_len, self.page_size
    )
    return new_node
```

Split 操作的关键性质：
1. **不复制 KV Cache 数据**：`value` 的 clone 只是索引的复制，不是 KV Cache 张量本身的复制
2. **保持前缀完整性**：拆分后，从 root 到任何叶子节点的路径拼接起来，仍然等于原始的完整 token 序列
3. **hash_value 同步拆分**：如果节点已经有了 page 级别的哈希值，也需要对应拆分

这个设计让人联想到 B+ 树的节点分裂——都是在查找路径上动态调整树的结构，以适应新的数据分布。但 Radix Tree 的 split 是由**前缀共享**驱动的，而 B+ 树的 split 是由**容量溢出**驱动的。

## 5. 驱逐：当显存不够时

KV Cache 可以很大——一个 70B 模型处理 8K 上下文，单个请求的 KV Cache 就可能超过 1GB。当 GPU 显存不足时，Radix Cache 需要驱逐不再使用的节点。

SGLang 支持多种驱逐策略：

```python
if self.eviction_policy == "lru":
    self.eviction_strategy = LRUStrategy()
elif self.eviction_policy == "lfu":
    self.eviction_strategy = LFUStrategy()
elif self.eviction_policy == "fifo":
    self.eviction_strategy = FIFOStrategy()
# ... 还有 mru, filo, priority, slru
```

驱逐的核心约束是 **引用计数（lock_ref）**：

```python
def evict(self, params: EvictParams) -> EvictResult:
    # 只驱逐 lock_ref == 0 且是叶子节点的节点
    # 从可驱逐节点中选择，释放其 value（KV Cache 索引归还内存池）
```

一个节点只有在 `lock_ref == 0`（没有被任何活跃请求引用）且是叶子节点（没有子节点依赖它）时才能被驱逐。这类似于垃圾回收中的引用计数机制——但这里是手动的，由调度器在请求完成时调用 `dec_lock_ref`。

驱逐后，节点仍然留在树中（只是 `value` 变成了 `None`），这样如果后续有请求匹配到这个前缀，虽然 KV Cache 已经不在了，但树的结构还在，可以快速定位到匹配位置。这个设计在 `evicted` 属性中体现：

```python
@property
def evicted(self):
    return self.value is None
```

这是一种"软驱逐"策略——树的结构（骨架）保留，只释放实际占用的显存。如果将来需要恢复，可以通过 HiCache 的存储后端从 CPU 或磁盘重新加载（这将是第二篇的主题）。

## 6. 引用计数：保护正在使用的缓存

引用计数是 Radix Cache 正确性的关键保障。当一个请求匹配到某个前缀时，匹配路径上的所有节点引用计数 +1；请求完成时，引用计数 -1。

```python
def inc_lock_ref(self, node, params=None):
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            # 从可驱逐变为受保护
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        node = node.parent
    return IncLockRefResult(delta=delta)
```

注意 `inc_lock_ref` 会沿着 parent 链一直向上到 root——因为如果一个子节点正在被使用，它的所有祖先节点都不能被驱逐。这就像文件系统中的硬链接：只要有一个引用存在，数据就不会被回收。

## 7. Page 对齐：与注意力机制的协作

现代 LLM 推理引擎通常使用 PagedAttention [3] 来管理 KV Cache，将显存划分为固定大小的"页"（page）。SGLang 的 Radix Cache 也遵循这个设计：

```python
key = key.page_aligned(self.page_size)
```

所有插入和匹配操作都会将 token 序列对齐到 page 边界。这意味着：
- 一个 `TreeNode.key` 的长度总是 `page_size` 的整数倍
- 未对齐的尾部 token 不会被缓存（但会保留在请求的 `prefix_indices` 中）
- `hash_value` 也是按 page 计算的，每个 page 一个哈希

`page_size` 通常是 1（token 级别）或更大的值（如 16、64）。更大的 page size 减少了管理开销，但增加了内部碎片。

## 8. Hash 值：为分层缓存铺路

每个 `TreeNode` 可以有一个 `hash_value` 列表，存储每个 page 的 SHA256 哈希：

```python
class RadixKey:
    def hash_page(self, start: end, prior_hash=None) -> str:
        hasher = hashlib.sha256()
        if prior_hash:
            hasher.update(bytes.fromhex(prior_hash))
        for j in range(start, end):
            hasher.update(t[j].to_bytes(4, byteorder="little"))
        return hasher.hexdigest()
```

`hash_value` 的用途是为 HiCache 的分层存储提供寻址依据——通过哈希值可以在 CPU 内存或外存中快速定位一个 page 的 KV Cache 数据，而不需要依赖树的结构。这将在第二篇中详细展开。

这里有一个有趣的递归方法 `get_prefix_hash_values`，它收集从 root 到当前节点的所有祖先的 hash 值：

```python
@lru_cache(maxsize=1)
def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
    if node is None or node.hash_value is None:
        return []
    return node.get_prefix_hash_values(node.parent) + node.hash_value
```

这个方法用于构建传递给存储后端的 `prefix_keys`，让存储后端知道一个 page 在前缀树中的位置。不过，这个 `@lru_cache(maxsize=1)` 装饰器本身存在一些设计问题——我们将在系列的最后一篇中专门讨论。

## 9. 与其他系统的对比

| 特性 | SGLang Radix Cache | vLLM Block Manager [3] | PagedAttention (原始) |
|------|-------------------|----------------------|---------------------|
| 数据结构 | 基数树 | Block 表 + 引用计数 | Block 表 |
| 前缀复用 | 自动（树结构隐含） | 需要显式 copy-on-write | 不支持 |
| 驱逐粒度 | 节点级（可变长度） | Block 级（固定大小） | Block 级 |
| 共享发现 | O(L) 树遍历 | O(1) 哈希表查找 | 不适用 |
| 分支支持 | 原生（树结构） | 有限（COW） | 不支持 |

SGLang 的 Radix Cache 相比 vLLM 的 Block Manager 最大的优势在于**自动前缀复用**。vLLM 使用 copy-on-write 机制共享 block，但需要请求显式声明共享关系（如 parallel sampling）。而 SGLang 的基数树天然支持任意前缀共享——只要两个请求有共同前缀，树结构就会自动将它们合并。

这种设计哲学的差异源于不同的优化目标：vLLM 的 Block Manager 侧重于**内存管理效率**（减少碎片），而 SGLang 的 Radix Cache 侧重于**计算复用效率**（减少重复 Prefill）。

## 10. 小结

Radix Cache 是 SGLang 推理引擎的核心数据结构，它用一棵动态变化的基数树管理所有请求的 KV Cache：

1. **Match**：O(L) 时间找到最长共享前缀，复用已有 KV Cache
2. **Insert**：将新计算的 KV Cache 插入树，自动 split 共享节点
3. **Evict**：按策略驱逐无人引用的节点，释放显存
4. **引用计数**：保护正在使用的缓存不被驱逐
5. **Page 对齐**：与 PagedAttention 协作，按页管理显存
6. **Hash 值**：为分层缓存（HiCache）的存储后端提供寻址

这棵"会呼吸的树"在请求的生命周期中不断 split、merge、evict，始终保持着最优的前缀共享结构。它是 SGLang 实现高吞吐量的基石——论文报告在 few-shot learning 场景下，RadixAttention 带来了最高 5 倍的吞吐量提升 [1]。

在下一篇中，我们将看到这棵树如何延伸到 GPU 显存之外——HiCache 如何将 KV Cache 分层存储在 GPU、CPU 和外存中，实现跨设备的缓存协同。

---

## 参考文献

[1] Zheng, L., Yin, L., Xie, Z., et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024. arXiv:2312.07104

[2] Nilsson, S., Tikkanen, A. "Implementing a Dynamic Compressed Trie." In Proceedings of the 7th International Workshop on Algorithm Engineering, 2004.

[3] Kwon, W., Li, Z., Zhuang, S., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. arXiv:2309.06180

[4] SGLang 源码: https://github.com/sgl-project/sglang

---

*下一篇预告：SGLang 深入浅出（二）：HiCache 分层存储——GPU、CPU 与外存的 KV Cache 协奏*
