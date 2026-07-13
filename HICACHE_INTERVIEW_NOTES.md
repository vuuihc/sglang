# SGLang HiCache 深度梳理（面试讲解版）

> 目标：用这份文档，能在面试里从「为什么需要」→「怎么设计」→「关键工程优化」→「和 vLLM/其他框架对比」→「收益量化」一条线讲清楚 SGLang 的分层 KV Cache（HiCache）。
>
> 所有结论都对照了当前仓库代码，关键实现位置以 `file:line` 给出，方便随时翻源码确认。

---

## 0. 一句话电梯演讲（先背下来）

> HiCache 是 SGLang 的**分层 KV Cache 系统**。它把 RadixAttention 的「前缀复用」思想从 GPU 显存扩展到三级存储：**L1=GPU 显存、L2=主机内存（CPU DRAM）、L3=分布式存储（Mooncake / 3FS / NIXL / AIBrix）**。通过一棵 `HiRadixTree` 统一管理三层 KV 的元数据，配合**预取（prefetch）、写回（write-back）、零拷贝传输、计算-传输 overlap、GPU 辅助 IO kernel** 等优化，在多轮对话和长上下文场景下把 KV cache 容量扩大几个数量级，同时把 prefill 时间和 TTFT 大幅降低，还能让一个集群里所有实例**共享同一份 L3 缓存**。

记忆锚点：**「三级缓存 + 一棵树 + 两个核心动作（预取/写回）+ 一堆传输优化」**。

---

## 1. 背景：为什么需要 HiCache

### 1.1 问题根源：prefill 贵、且大量重复

- LLM 推理分两阶段：**prefill**（把 prompt 编码成 KV cache，计算密集、与序列长度平方相关）和 **decode**（逐 token 生成，访存密集）。
- 当多个请求**共享相同前缀**（system prompt、few-shot 示例、多轮对话历史、RAG 文档），这段前缀的 KV cache 是**完全相同**的。重复计算就是浪费。

### 1.2 SGLang 的起点：RadixAttention

- SGLang 的招牌特性 **RadixAttention**：用一棵 **基数树（Radix Tree）** 组织 GPU 显存里的 KV cache，从根到叶的一条路径 = 一个请求的前缀；共享前缀复用同一批节点 → 避免重复计算 + 节省显存。
- 局限：只用 GPU 空闲显存，**容量极其有限**。热点前缀一多就被挤掉（evict），命中率上不去。

### 1.3 HiCache 的核心 idea

> 借鉴现代 CPU 的三级缓存设计，把「空闲存储」一层层榨干：
> - **L1 = GPU 显存**（最快最小，单实例私有）
> - **L2 = 主机 CPU 内存**（大一两个量级，单实例私有）
> - **L3 = 分布式存储 / KV 池**（巨大、跨实例共享）

类比 CPU：L1/L2 私有给每个核（每个推理实例），L3 共享给所有核（集群所有实例）—— 这正是 HiCache 三层「私有 vs 共享」的划分。

参考博客：<https://lmsys.org/blog/2025-09-10-sglang-hicache/>
官方设计文档：[docs/advanced_features/hicache_design.md](docs/advanced_features/hicache_design.md)

---

## 2. 整体架构与核心数据结构

### 2.1 三层存储

| 层级 | 介质 | 作用域 | 元数据精度 | 典型容量 |
|------|------|--------|-----------|---------|
| L1 | GPU HBM | 单实例私有 | 精确地址 | GB 级 |
| L2 | CPU DRAM | 单实例私有 | 精确地址 | 几十~几百 GB |
| L3 | Mooncake / 3FS / NIXL / AIBrix / file | **集群共享** | **不常驻，实时查询** | TB+ |

### 2.2 HiRadixTree：统一元数据组织

实现：[python/sglang/srt/mem_cache/hiradix_cache.py:72](python/sglang/srt/mem_cache/hiradix_cache.py#L72) `class HiRadixCache(RadixCache)`

关键扩展点（相对原始 RadixTree）：
- 每个节点除了记录 token span，还记录这段 KV **存在哪一层**：`device_indices`（L1）、`host_value`（L2）、是否已 `backuped` 到 L3。
- **L1/L2 元数据精确常驻**（节点里直接存地址 / index）；
- **L3 元数据不常驻、不持续同步**——访问 L3 时**实时向后端 query**（是否存在、在哪台机器、什么位置）。
  - 为什么这么设计？L3 是跨实例共享的、容量巨大且动态变化，如果在本地树里维护全量元数据，同步开销和内存开销都不可接受。

在 `match_prefix` 里能直接看到三层的边界判定（[hiradix_cache.py:1356](python/sglang/srt/mem_cache/hiradix_cache.py#L1356)）：匹配返回值里 `device_indices` 是 L1 命中、`host_hit_length` 是 L2 命中、`last_host_node` 是后续从 L2 load 回 L1 的起点。

---

## 3. 核心工作流（面试主线，一定讲透）

一个新请求进来，三个关键动作依次发生：**Local Match → Prefetch（L3→L2）→ Write-back（L1→L2→L3）**。

```
请求 tokens
   │
   ├─(1) Local Match：在 HiRadixTree 里匹配 L1+L2（纯内存遍历，无数据拷贝，极快）
   │
   ├─(2) 对没命中的部分，向 L3 query 元数据；命中长度 > 阈值(默认256 token) → Prefetch 到 L2
   │
   ├─ 把所需 KV 全部 load 回 GPU(L1) → 跑 prefill 计算
   │
   └─(3) prefill 完成后，把新生成的 KV 按写回策略落到 L2 / L3（供后续 & 跨实例复用）
```

### 3.1 Local Match（本地匹配）

- 从根遍历 HiRadixTree，按 token 序列前缀往下走。
- `page_size > 1` 时按 **page 粒度** 匹配（优化访存）。
- 匹配在某节点中途终止时，**split 节点**建立精确边界，利于后续匹配。
- 返回一个连续前缀：**前半在 L1、后半在 L2**。
- **关键卖点**：只遍历本地树、**不涉及任何数据拷贝**，所以极快。
- 代码：[`match_prefix` hiradix_cache.py:1356](python/sglang/srt/mem_cache/hiradix_cache.py#L1356)、`_match_prefix_helper`、`_split_node`。

### 3.2 Prefetch（从 L3 预取到 L2）—— HiCache 核心优化之一

实现：[`prefetch_from_storage` hiradix_cache.py:1389](python/sglang/srt/mem_cache/hiradix_cache.py#L1389)

**触发条件**（[hiradix_cache.py:1405](python/sglang/srt/mem_cache/hiradix_cache.py#L1405)）：
本地没命中的部分，向 L3 查询后续可连续匹配的 KV 长度；**L3 命中长度 > `prefetch_threshold`（默认 256 token，可配）** 才触发，避免小碎片预取得不偿失。还会检查 `prefetch_rate_limited()` 做限流，并在 L2 内存不足时先 `evict_host` 再尝试分配。

**三种终止策略**（latency vs 命中率的权衡，必背）：

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `best_effort` | GPU 一旦能开跑 prefill 就立刻停，不等 | 极度延迟敏感 |
| `wait_complete` | 必须等所有预取完成 | 追求最高命中率 |
| `timeout` | 超时或完成即停（**生产推荐**） | 平衡延迟和命中率、满足 SLO |

`timeout` 的超时公式（线性模型，区分固定开销和数据量开销）：
```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024
```
对应代码：`_prefetch_timeout_check_linear_func` / `can_terminate_prefetch` / `check_prefetch_progress`（[hiradix_cache.py:1243-1339](python/sglang/srt/mem_cache/hiradix_cache.py#L1243)）。

### 3.3 Write-back（写回 L1→L2→L3）

**三种写回策略**（必背，类比 CPU cache 的 write-through / write-back）：

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `write_through` | 每次访问立即写到下一级 | 带宽充足，缓存收益最强 |
| `write_through_selective` | 访问频次超阈值才写回 | 只备份热数据，省 IO |
| `write_back` | 仅当从上层被 evict 时才写回 | 存储容量紧张、最大化内存利用 |

代码里阈值逻辑：`write_through_threshold = 1 if policy=="write_through" else 2`（[hiradix_cache.py:179](python/sglang/srt/mem_cache/hiradix_cache.py#L179)），命中计数在 `_inc_hit_count`（[hiradix_cache.py:830](python/sglang/srt/mem_cache/hiradix_cache.py#L830)）里累加，达到阈值触发 `write_backup`。

**跨实例共享的精髓**：L2→L3 写回时**只传 L3 里还没有的部分**；一旦进了 L3，集群内**所有 SGLang 实例都能命中**（取决于 L3 后端实现）——这是 HiCache「在同样内存预算下大幅提升命中率」的关键。

---

## 4. 关键工程优化（体现深度，面试加分点）

### 4.1 多 Rank 同步（正确性，不是性能）

TP 多卡并行时，各 rank 必须对「是否预取够了」「成功取到的前缀长度」达成一致，否则状态错乱。
- 预取阶段：`all_reduce(op=MIN)` 保证所有 rank 看到相同的 L3 命中长度；
- 预取结束：再 `all_reduce(op=MIN)` 对成功取回的前缀长度取共识。
- 代码：`_all_reduce_attn_groups`（[hiradix_cache.py:191](python/sglang/srt/mem_cache/hiradix_cache.py#L191)），在 `query_storage_hit_length` / `check_prefetch_progress` 等处调用（用 MIN/MAX 取一致）。

### 4.2 零拷贝传输（Zero-Copy）

L2→L3 传输时直接传**内存地址 + 大小**，避免中间拷贝。设计文档 §Data Transfer Optimization。

### 4.3 面向「批」的数据布局（layout）—— 很能体现对硬件的理解

存储介质：[python/sglang/srt/mem_cache/memory_pool_host.py:228](python/sglang/srt/mem_cache/memory_pool_host.py#L228) `class HostKVCache`

三种 L2 内存布局（`--hicache-mem-layout`）：
- `layer_first`：GPU 计算天然布局（逐层算），但对 L3 的 IO 不友好。
- `page_first`：同一个 page 的所有 KV 连续 → 可作为**单个对象**零拷贝传给 L3，IO 效率高；但 L2→GPU 时要按「每层每 token」搬，碎。
- `page_first_direct`：折中——把一个 page 内**同一层的所有 token** 聚在一起，L2→GPU 传输能按 page-layer 粒度聚合。
- 代码里各布局分支：`MHATokenToKVPoolHost` / `MLATokenToKVPoolHost` 的 `load_to_device_per_layer` / `backup_from_device_all_layer`（[memory_pool_host.py:483/600/1035](python/sglang/srt/mem_cache/memory_pool_host.py#L483)）。

> 一句话讲清楚：**GPU 想要 layer-first，存储想要 page-first，`page_first_direct` 是两者的折中。**

### 4.4 CPU→GPU 传输优化（和预取同样关键）

- **计算-传输 overlap**：prefill 时，算第 N 层的同时把第 N+1 层的 KV 从 CPU 搬上 GPU，隐藏传输延迟。
  - 机制载体：`LayerDoneCounter` / `LayerLoadingEvent`（[managers/cache_controller.py:54-100](python/sglang/srt/managers/cache_controller.py#L54)），用逐层 event 做生产者-消费者同步。
- **GPU 辅助 IO kernel**（`--hicache-io-backend kernel`）：在 `cudaMemcpyAsync` 之上自研了一套专门搬 KV 的 kernel，相比 baseline **最高 3x** 传输速度。

### 4.5 MLA 的写回优化

- MHA 多 TP：每个 rank 只持有一个 token 的 `1/tp_size` KV。
- MLA：所有 rank 持有**完整且相同**的 KV。→ HiCache 让 **只有一个 rank 发起写回**，避免跨 rank 冗余存储。

### 4.6 与 PD 分离部署集成

- SGLang 支持 Prefill-Decode 分离（基于 Mooncake TransferEngine）。
- HiCache 可在 prefill 节点和 decode 节点**都开启**；decode 节点开启时，decode 产生的输出也会写回 L3。
- 代码：[python/sglang/srt/disaggregation/decode_hicache_mixin.py](python/sglang/srt/disaggregation/decode_hicache_mixin.py)

---

## 5. 统一接口与 L3 后端

抽象基类：[python/sglang/srt/mem_cache/hicache_storage.py:138](python/sglang/srt/mem_cache/hicache_storage.py#L138) `class HiCacheStorage(ABC)`，把 L3 的读/写/查询封装成简单一致接口（`get/set/batch_get/batch_set/exists/batch_exists`，以及 v2 的 `batch_*_v2`）。

内置后端（`--hicache-storage-backend`）：
- **Mooncake**：RDMA + 多网卡，零拷贝超快传输（KVCache-centric，源自 Kimi/Moonshot）。
- **DeepSeek 3FS (HF3FS)**：K8s 原生分布式存储。
- **NIXL**：统一 API，背后可接 3FS / GPUDirect Storage / S3 兼容对象存储。
- **AIBrix KVCache**：生产级 KV offloading 框架，跨引擎低开销复用。
- **file**：演示用的本地文件后端。
- **dynamic**：用户自定义后端（指定 module_path / class_name）。

> 另：**LMCache** 是 HiCache 的一个**替代方案**（`--enable-lmcache`），不是 L3 后端而是另一套分层缓存层。

后端目录：[python/sglang/srt/mem_cache/storage/](python/sglang/srt/mem_cache/storage/)

---

## 6. 关键参数速查（面试可能追问「怎么调」）

| 参数 | 作用 | 备注 |
|------|------|------|
| `--enable-hierarchical-cache` | 开启 HiCache | 必须 |
| `--hicache-ratio` | L2/L1 容量比 | 必须 > 1 |
| `--hicache-size` | L2 池大小(GB，**每 rank**) | 覆盖 ratio；8 rank × 30G = 240G |
| `--page-size` | 每 page token 数 | 大 page：IO 效率高、元数据少，但部分匹配时命中率降 |
| `--hicache-storage-prefetch-policy` | best_effort / wait_complete / **timeout(生产推荐)** | §3.2 |
| `--hicache-write-policy` | write_through / write_through_selective / write_back | §3.3 |
| `--hicache-io-backend` | direct / **kernel(推荐)** | kernel 最高 3x |
| `--hicache-mem-layout` | layer_first / page_first / page_first_direct | §4.3 |
| `--hicache-storage-backend` | file/mooncake/hf3fs/nixl/aibrix/dynamic | §5 |
| `--hicache-storage-backend-extra-config` | JSON/文件，传 prefetch_threshold、timeout 参数等 | |

> 调参直觉：**HiCache 越大→命中率越高→prefill 越快，但非线性**——热点 token 都缓存上之后，再加容量收益递减。

---

## 7. vLLM 及其他框架的对应机制对比

面试常问「别的框架有没有类似机制」。答案是**有，但拼图程度不同**。HiCache 的特点是把「GPU 前缀复用 + CPU offload + 分布式 KV 池 + 跨实例共享」**做成了一套内聚的分层系统**；其他框架早期是分散的功能点，近一年也在快速收敛到类似形态。

### 7.1 vLLM

| 能力 | vLLM 的对应 | 和 HiCache 对比 |
|------|------------|----------------|
| GPU 前缀复用 | **Automatic Prefix Caching (APC)**，v1 默认开启，基于 block hash | 对应 RadixAttention/L1。HiCache 用 radix tree，vLLM 用 hash-block；思路一致 |
| CPU offload (L2) | 早期 `--swap-space`（仅抢占时换出，非主动复用）；v1 的 **CPU offloading connector** 才真正把 KV 下沉 CPU 复用 | HiCache 的 L2 是一等公民、和预取/写回深度耦合 |
| L3 分布式/跨实例 | 通过 **KVConnector** 接口接 **LMCache**（CPU/磁盘/分布式分层 + 跨实例共享）；Production Stack 做 KV-aware 路由 + 分离式 prefill | 等价于 HiCache 的 L3；vLLM 把它做成**可插拔 connector**，HiCache 把整套分层**内建** |
| 数据布局/传输优化 | 也有 zero-copy、NIXL 传输等 | HiCache 显式提供 layer/page 布局选择 + 自研 GPU IO kernel |

**一句话总结 vLLM**：vLLM = **APC（≈L1）默认开** + **KVConnector/LMCache 插件**补齐 L2/L3。和 HiCache 终态相似，但 vLLM 走「核心精简 + 可插拔连接器」路线，SGLang 走「分层系统内建 + 统一 HiRadixTree」路线。值得一提：**LMCache 同时能接 vLLM 和 SGLang**，所以两边在 L3 上是会师的。

### 7.2 TensorRT-LLM

- **KV cache block reuse**（前缀复用）+ **host KV cache offloading**：paged KV 支持二级（host）内存池 + 基于优先级的 eviction。
- 对应 HiCache 的 L1+L2，但**缺乏开放的跨实例 L3 共享生态**（更偏单机/单实例分层）。

### 7.3 NVIDIA Dynamo

- **KVBM (KV Block Manager)** 提出 G1/G2/G3/G4 多级内存（GPU/CPU/本地 SSD/远程存储），思路与 HiCache 三级几乎同构。
- **KV-aware routing**：根据各实例已缓存的前缀，把请求路由到命中率最高的实例——这是 HiCache 本身没做、属于「上层调度」的互补能力。

### 7.4 Mooncake（Moonshot/Kimi）

- 不是「框架」而是**KVCache-centric 的分离式架构 + 存储引擎**，正是 SGLang HiCache 的**一个 L3 后端**。
- 提供 RDMA 零拷贝、多网卡、全局 KV 池。SGLang 与之集成而非竞争。

### 7.5 小结对比表

| 框架 | L1(GPU 复用) | L2(CPU offload) | L3(分布式/跨实例) | 整合形态 |
|------|:---:|:---:|:---:|------|
| **SGLang HiCache** | RadixAttention | ✅ 内建 | ✅ 内建多后端 | 统一 HiRadixTree 分层系统 |
| vLLM | APC(默认) | connector | LMCache/connector | 核心+可插拔连接器 |
| TensorRT-LLM | block reuse | ✅ host offload | ⚠️ 弱 | 单实例分层为主 |
| Dynamo | ✅ | ✅ KVBM | ✅ KVBM + KV路由 | 调度+块管理 |
| Mooncake | — | — | ✅(本身就是 L3) | 存储/分离式底座 |

> 安全表述（避免说错）：各框架迭代很快，"vLLM 没有 L2/L3" 是过时说法——更准确的是 **「HiCache 把分层做成内建一体化系统，vLLM 走可插拔 connector + LMCache 的路线，两者终态趋同」**。

---

## 8. 收益（Benefits）—— 用数字和场景说话

### 8.1 收益来自哪里（机制层面）

1. **避免重复 prefill 计算**：共享前缀只算一次，命中即取 KV。
2. **容量数量级扩张**：GPU 显存 → +CPU DRAM（几十~几百 GB/rank）→ +分布式存储（TB+），命中率显著提升。
3. **跨实例共享**：L3 让整个集群共享同一份热点 KV，同样内存预算下命中率更高。
4. **延迟可控**：prefetch timeout + write 策略可按 SLO 调，不让缓存拖慢尾延迟。

### 8.2 收益指标（面试要会说「优化了什么」）

- **TTFT（Time To First Token）↓**：prefill 命中即省去计算，首 token 更快。
- **Prefill 吞吐 / 整体 throughput ↑**：省下的算力服务更多请求。
- **有效 KV cache 命中率 ↑**：尤其多轮对话、长上下文、RAG、Agent 这类高复用场景。
- **GPU 利用率更优**：算力不再浪费在重复 prefill 上。

### 8.3 量化（引用官方 blog，记数量级即可）

- 自研 **GPU 辅助 IO kernel** 相比 baseline `cudaMemcpyAsync`：**最高 3x** CPU↔GPU 传输速度。
- 详细 benchmark 见官方 blog：<https://lmsys.org/blog/2025-09-10-sglang-hicache/>（多 QA / 长上下文场景下 TTFT 和吞吐的显著改善——面试时引用「数量级提升 KV 容量、显著降低 TTFT」并指向 blog 即可，不要硬背具体百分比以免说错）。

### 8.4 最适合的场景（什么时候开 HiCache 收益最大）

- ✅ **多轮对话**（历史前缀越来越长且被反复复用）
- ✅ **长上下文 / 长 system prompt / few-shot**
- ✅ **RAG**（相同文档块被多请求引用）
- ✅ **Agent / 工具调用**（固定 prompt 模板 + 长轨迹）
- ⚠️ 收益有限：前缀高度发散、几乎无复用的短请求（此时预取/写回的 IO 反而是开销）。

---

## 9. 面试 Q&A 预演（高频追问）

**Q: HiCache 和普通的 prefix caching 区别？**
A: 普通 prefix caching（含 vLLM APC、SGLang RadixAttention）只在 GPU 显存里复用，容量极小。HiCache 把它扩展成 **GPU/CPU/分布式三级**，并增加预取、写回、跨实例共享和一整套传输优化。

**Q: L3 元数据为什么不常驻本地？**
A: L3 跨实例共享、容量巨大、状态动态变化。本地维护全量元数据的内存和同步开销不可接受，所以**访问时实时 query 后端**（设计文档明确，见 §2.2）。

**Q: 怎么保证多卡一致性？**
A: 关键步骤用 `all_reduce`：判断 L3 命中长度和预取前缀长度时取 `MIN`，避免不同 rank 决策不一致（`_all_reduce_attn_groups`，[hiradix_cache.py:191](python/sglang/srt/mem_cache/hiradix_cache.py#L191)）。

**Q: prefetch 会不会拖慢延迟敏感请求？**
A: 三种策略可选；生产用 `timeout`，按 `base + per_ki_token*tokens/1024` 线性建模超时，平衡命中率与 SLO（§3.2）。

**Q: 为什么需要 page_first_direct 这种布局？**
A: GPU 逐层计算要 layer-first，存储零拷贝要 page-first，两者冲突。`page_first_direct` 把 page 内同一层的 token 聚在一起，让 L2→GPU 传输能按 page-layer 聚合，是两者折中（§4.3）。

**Q: MLA 模型有什么特别处理？**
A: MLA 各 rank 持有完整相同 KV，所以**只让一个 rank 写回**，避免冗余存储（§4.5）。

**Q: 和 vLLM 比谁更好？**
A: 不是非此即彼。HiCache 把分层做成内建一体化系统、用统一 HiRadixTree 管理；vLLM 走核心精简 + KVConnector/LMCache 可插拔路线，终态趋同。LMCache 还能同时接两者（§7）。

---

## 10. 源码导航（想深入时的入口）

| 模块 | 路径 |
|------|------|
| HiRadixTree / 主流程 | [python/sglang/srt/mem_cache/hiradix_cache.py](python/sglang/srt/mem_cache/hiradix_cache.py) |
| Cache 控制器 / 逐层 overlap / 预取写回执行 | [python/sglang/srt/managers/cache_controller.py](python/sglang/srt/managers/cache_controller.py) |
| L2 主机内存池 / 三种布局 | [python/sglang/srt/mem_cache/memory_pool_host.py](python/sglang/srt/mem_cache/memory_pool_host.py) |
| L3 统一存储抽象 | [python/sglang/srt/mem_cache/hicache_storage.py](python/sglang/srt/mem_cache/hicache_storage.py) |
| L3 后端实现 | [python/sglang/srt/mem_cache/storage/](python/sglang/srt/mem_cache/storage/) |
| PD 分离 + HiCache | [python/sglang/srt/disaggregation/decode_hicache_mixin.py](python/sglang/srt/disaggregation/decode_hicache_mixin.py) |
| 官方设计文档 | [docs/advanced_features/hicache_design.md](docs/advanced_features/hicache_design.md) |
| 最佳实践 | [docs/advanced_features/hicache_best_practices.md](docs/advanced_features/hicache_best_practices.md) |

---

### 一页纸记忆图

```
                      请求
                        │ Local Match (纯内存, 0 拷贝)
        ┌───────────────┼───────────────┐
   L1 GPU(私有)     L2 CPU(私有)     L3 分布式(共享)
   RadixAttention   主机内存池        Mooncake/3FS/NIXL/AIBrix
        ▲   │            ▲  │             ▲
        │   └─load_back──┘  └──prefetch───┘  (命中>256 token 触发)
        │                                   query 元数据(不常驻)
        └──────── write-back (through/selective/back) ──────►
   优化: zero-copy · page布局 · 计算-传输overlap · GPU IO kernel(3x) · MLA单rank写回 · all_reduce多卡一致
   收益: TTFT↓ · 吞吐↑ · KV容量↑数量级 · 跨实例命中率↑   (多轮/长上下文/RAG/Agent 最香)
```
