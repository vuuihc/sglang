# SGLang 分布式 KVCache 系统 Roadmap 深度问答

本文档针对 [Issue #21846](https://github.com/sgl-project/sglang/issues/21846) 路线图中的 15 个技术问题，从第一性原理出发进行深入浅出的解答。

---

## Q1: Mamba 模型的前世今生，DeepSeek V4 和 Qwen3.5 真的用了 Mamba 吗？

### 从 RNN 到 Mamba 的演进脉络

**第一性原理：序列建模的核心矛盾是"如何高效地记住历史信息"。**

#### RNN 时代（1990s）

循环神经网络（RNN）是最早的序列建模方案，核心思想是维护一个隐状态 h(t)：

```
h(t) = tanh(W·h(t-1) + U·x(t))
y(t) = V·h(t)
```

**致命缺陷**：梯度消失/爆炸。长序列中梯度要么衰减为零（学不到远距离依赖），要么指数增长（训练不稳定）。而且 RNN 的顺序计算本质导致**无法并行训练**。

#### LSTM（1997）

通过三个门控机制（遗忘门、输入门、输出门）解决梯度消失：

```
f(t) = σ(W_f·[h(t-1), x(t)])   ← 遗忘门：决定丢弃什么
i(t) = σ(W_i·[h(t-1), x(t)])   ← 输入门：决定存储什么
o(t) = σ(W_o·[h(t-1), x(t)])   ← 输出门：决定输出什么
c(t) = f(t)⊙c(t-1) + i(t)⊙tanh(W_c·[h(t-1), x(t)])
h(t) = o(t)⊙tanh(c(t))
```

> **关键洞察**：LSTM 的门控机制本质上是一种"动态对角 SSM"——状态转移矩阵变成了输入相关的对角矩阵。这为后来的 Mamba 埋下了伏笔。但 LSTM 仍然是顺序计算的，无法高效并行。

#### S4 — Structured State Space（2022）

Albert Gu 提出的现代 SSM 奠基之作。核心创新：

1. **从控制论借用状态空间方程**：
   ```
   连续：h'(t) = A·h(t) + B·x(t),  y(t) = C·h(t)
   离散：h(t) = Ā·h(t-1) + B̄·x(t),  y(t) = C·h(t)
   ```
2. **HiPPO 理论**：通过特殊初始化矩阵 A（HiPPO 矩阵），让状态能"记忆"历史输入的压缩表示
3. **结构化矩阵**：将 A 约束为对角加低秩（DPLR）结构，使 O(n) 复杂度的卷积运算成为可能
4. **双重计算模式**：训练时用 FFT 卷积并行计算，推理时用递归模式 O(1) 逐 token 生成

**局限**：S4 的参数 A、B、C 是**固定的**（输入无关），无法根据内容选择性记忆或遗忘。

#### Mamba — Selective SSM（2023）

Tri Dao 和 Albert Gu 的突破性工作。核心创新是**选择性状态空间**：

- 让 B、C、Δ 成为**输入相关的**：
  ```
  B(t) = Linear_B(x(t))
  C(t) = Linear_C(x(t))
  Δ(t) = softplus(Linear_Δ(x(t)))   ← 步长也动态化
  ```
- **选择性的直觉**：模型可以"选择"记住什么、忘记什么。比如遇到逗号时"忘记"（小 Δ），遇到关键名词时"记住"（大 Δ）
- **硬件感知并行扫描**：虽然选择性打破了 LTI 假设（无法再用 FFT 卷积），但通过 GPU 并行扫描算法，仍然实现了 O(n) 的并行训练

#### Mamba-2 — SSD 理论（2024）

证明了 Mamba 的选择性扫描与线性注意力之间存在**数学等价性**（Structured State Space Duality）。这为混合架构铺路——既然 SSM 和线性注意力在数学上等价，混合使用就变得自然。

### DeepSeek V4 用了 Mamba 吗？

**没有。** DeepSeek V4 的架构创新是：

1. **混合注意力架构（CSA + HCA）**：
   - **CSA（Compressed Sparse Attention）**：沿序列维度压缩 KV Cache + 滑动窗口局部注意力 + 稀疏网格长距离依赖
   - **HCA（Heavily Compressed Attention）**：更重度压缩 + 密集注意力
   - 两者交替堆叠，百万 token 上下文下 KV Cache 降至前代的 1/10
2. **流形约束超连接（mHC）**
3. **Muon 优化器**

DeepSeek 团队明确表示：**"没有跟风 SSM 或门控 DeltaNet 路线"**，而是通过压缩注意力解决长上下文效率问题。

### Qwen3.5 用了 Mamba 吗？

**不完全是 Mamba，但用了 Mamba 家族的"近亲"——Gated DeltaNet。**

Qwen3.5 的架构是 **Gated DeltaNet + Gated Attention + MoE 混合**：
- 75% 的层使用 Gated DeltaNet（线性注意力，O(n) 复杂度）
- 25% 的层使用传统 Gated Attention
- 3:1 的层比例

Gated DeltaNet 和 Mamba 的关系：两者都属于"线性复杂度序列建模"家族，在 Mamba-2 的 SSD 理论框架下被证明有数学等价性，但具体实现不同——Mamba 基于选择性 SSM，Gated DeltaNet 基于线性注意力 + delta 规则。

### 2025-2026 混合模型趋势

三个独立团队几乎同时收敛到相同的架构结论：

| 模型 | 团队 | 线性层类型 | 线性层占比 | 注意力层占比 |
|------|------|-----------|-----------|-------------|
| Qwen3.5 | 阿里 | Gated DeltaNet | 75% | 25% |
| Nemotron 3 Nano | NVIDIA | Mamba-2 | ~75% | ~25% |
| 混元 TurboS | 腾讯 | Mamba-2 | ~47% | ~5% |

**核心洞察**：纯 SSM/线性注意力存在"固定状态信息瓶颈"，纯 Transformer 的 O(n²) 复杂度在长上下文下不可接受。**75% 线性 + 25% 注意力**是帕累托最优的平衡点。

---

## Q2: Mooncake 的技术原理是什么？

### Mooncake 是什么？

Mooncake 是**月之暗面（Moonshot AI）**开发的 LLM 推理服务平台，也是 Kimi 的底层架构。它后来被开源到 [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)，成为 SGLang HiCache 的 L3 存储后端之一。

### 核心设计理念：以 KVCache 为中心的解耦架构

Mooncake 的核心洞察是：**LLM 推理中，KV Cache 的存储和传输是瓶颈，而不是计算本身。**

传统架构把 KV Cache 当作计算的附属品，Mooncake 把它提升为**一等公民**——整个系统围绕 KV Cache 的存储、传输和调度来设计。

### 架构组件

```
┌─────────────────────────────────────────────────────┐
│                   Conductor（全局调度器）              │
│  根据 KV Cache 分布和工作负载特征调度请求               │
└──────────┬──────────────────────────────┬───────────┘
           │                              │
    ┌──────▼──────┐               ┌───────▼──────┐
    │ Prefill Cluster │            │ Decode Cluster │
    │ (计算密集型)     │            │ (内存密集型)    │
    └──────┬──────┘               └───────┬──────┘
           │                              │
           └──────────┬───────────────────┘
                      │
           ┌──────────▼──────────┐
           │   Mooncake Store    │
           │  (分布式 KV Cache)   │
           │  CPU + DRAM + SSD   │
           │  RDMA 零拷贝传输     │
           └─────────────────────┘
```

#### 1. Mooncake Store（分布式 KV Cache 存储引擎）

- 利用 GPU 集群中**未被充分利用的 CPU、DRAM、SSD 和 NIC 资源**构建分布式缓存池
- 提供 Put/Get/Remove 等**对象级 API**，以分页块（page）形式管理 KV Cache
- 使用哈希键进行去重，根据访问频率动态调整缓存块的副本数量
- 支持 SSD Offload：当 DRAM 不足时，自动溢出到本地 SSD

#### 2. Transfer Engine（传输引擎）

- 基于 **RDMA 零拷贝**技术，直接在节点间传输数据，绕过 CPU
- 拓扑感知路径选择：根据网络拓扑选择最优传输路径
- 端点池化：复用连接，减少建立连接的开销
- 支持多 NIC 设备并行传输，充分利用网络带宽
- 支持 TCP 和 RDMA 两种协议

#### 3. Conductor（全局调度器）

- 根据 KV Cache 的分布情况和工作负载特征调度请求
- 目标：在满足 SLO（延迟约束）的前提下最大化吞吐量
- 会优先将请求调度到已有对应 KV Cache 的 Prefill 节点

### 在 SGLang 中的集成

在 SGLang 中，Mooncake 作为 HiCache 的 **L3 存储后端**：

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-storage-backend-extra-config '{
        "master_server_address": "127.0.0.1:50051",
        "protocol": "rdma",
        "global_segment_size": "4gb"
    }'
```

数据流：L1(GPU) ↔ L2(CPU) ↔ L3(Mooncake Store)

当 L1 和 L2 都 miss 时，HiCache 自动从 Mooncake 的分布式内存池获取 KV Cache，利用 RDMA 实现高带宽低延迟传输。

### 性能数据

- 在 A800 集群上，Kimi 处理的请求数增加了 115%
- 在 H800 集群上，增加了 107%
- 缓存命中率相比本地缓存最大提升 136%
- 传输引擎比 TCP 快 2.4-4.6 倍

---

## Q3: KeLing 是字节跳动的还是快手的？

**可灵（KeLing/Kling）是快手（Kuaishou）的产品，不是字节跳动的。**

- 2024年6月：快手正式上线可灵大模型
- 2025年4月：可灵 2.0 发布，全球用户突破 2200 万
- 可灵采用类 Sora 的 DiT（Diffusion Transformer）架构，与 Mamba/SSM 无关

**采用 MLA + Mamba/线性注意力混合架构的模型**实际上是：

| 模型 | 团队 | 架构 |
|------|------|------|
| Kimi Linear | 月之暗面 | KDA（Gated DeltaNet 改进版）+ MLA，3:1 混合 |
| AMD-HybridLM | AMD | MLA + Mamba2 混合 |
| 混元 TurboS | 腾讯 | GQA + Mamba2 + MoE FFN |

Roadmap 中提到的 "KeLing" 可能是指某个内部代号或模型名称，而非快手的可灵视频生成模型。具体是哪个模型需要看 SGLang 代码中的上下文。

---

## Q4: 3FS 是 SGLang 里面的，跟字节有啥关系？

### 3FS 是什么？

**3FS（Fire-Flyer File System）是 DeepSeek 开源的高性能分布式文件系统**，不是字节的。

- GitHub 仓库：[deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)
- 专为 AI 训练和推理工作负载设计
- DeepSeek 开源周的第五弹

### 3FS 的核心技术

1. **分离式架构**：计算节点与存储节点分离，存储资源可独立扩展
2. **CRAQ 一致性协议**：Chain Replication with Apportioned Queries，链式复制 + 分配查询，写全链、读任意，保证强一致性
3. **RDMA 网络**：利用 RDMA 实现存储和计算节点间的高效数据传输
4. **FoundationDB 元数据**：无状态元数据服务，由 FoundationDB（事务性 KV 存储）支撑
5. **POSIX 文件接口**：提供标准文件系统接口，降低使用门槛

### 性能数据

- 180 节点集群峰值读取吞吐量：**6.6 TiB/s**
- GraySort 基准：30 分钟排序 110.5 TiB 数据（3.66 TiB/min）
- KV Cache 操作峰值读取吞吐量：**40 GiB/s**

### 3FS 跟字节的关系

**3FS 跟字节跳动没有直接关系。** 它是 DeepSeek 的项目。但在 SGLang 中的集成路径涉及阿里云：

- SGLang 中的 3FS 后端代码位于 `python/sglang/srt/mem_cache/storage/hf3fs/`
- 3FS 的 Kubernetes 部署通过 [aliyun/kvc-3fs-operator](https://github.com/aliyun/kvc-3fs-operator) 实现
- 3FS 的 Python 客户端包名是 `hf3fs-py-usrbio`，源码构建来自 [novitalabs/3FS](https://github.com/novitalabs/3FS)

所以 3FS 是 DeepSeek 开源的，阿里云提供了 K8s 部署工具，SGLang 社区做了集成。跟字节跳动没有直接关系。

### 3FS vs Mooncake 对比

| 维度 | Mooncake | 3FS |
|------|----------|-----|
| 开发方 | 月之暗面（Moonshot AI） | DeepSeek |
| 核心定位 | KV Cache 专用分布式存储 | 通用高性能分布式文件系统 |
| 存储介质 | DRAM + SSD | NVMe SSD |
| 一致性协议 | 自研 | CRAQ（链式复制） |
| 传输方式 | RDMA 零拷贝 | RDMA + Usrbio |
| API 风格 | 对象级（Put/Get/Remove） | 文件级（POSIX） |
| KV Cache 支持 | 原生支持 | 通过专用接口支持 |

---

## Q5: RadixTree 和字典树（Trie）是一回事吗？

**不是一回事，但 RadixTree 是 Trie 的空间优化变体。**

### Trie（字典树/前缀树）

每条边代表**一个字符/token**，从根到某节点的路径代表一个字符串前缀。

```
存储 "hello", "help", "world"：

        root
       /    \
      h      w
      |      |
      e      o
      | \    |
      l  l   r
      |  |   |
      l  p   l
      |      |
      o      d
```

**问题**：如果一条路径上没有分叉，会产生大量只有一个子节点的中间节点，**空间浪费严重**。

### RadixTree（基数树/压缩前缀树）

核心改进：**压缩无分叉路径**——将连续无分叉的边合并为一条。

```
同样的数据：

           root
         /      \
      "hel"    "world"
       / \
    "lo"  "p"
```

"hello" 从 5 个节点压缩为 2 个节点（"hel" + "lo"）。

### 关键区别

| 特性 | Trie | RadixTree |
|------|------|-----------|
| 边的含义 | 单个字符/token | 任意长度的字符/token 序列 |
| 节点数量 | 多 | 少（压缩后） |
| 空间效率 | 低 | 高 |
| 查找复杂度 | O(L) | O(L)（相同） |
| 插入/删除 | 简单 | 需要 split/merge 操作 |

### SGLang 为什么用 RadixTree？

在 LLM 推理中，系统提示词通常有几百到几千个 token。如果用普通 Trie 每个 token 一个节点，会产生大量无分叉中间节点。RadixTree 将整段系统提示词压缩为一个节点，效率大幅提升。

SGLang 的 `TreeNode` 中 `key` 字段存储的就是一段**连续的 token 序列**（不是单个 token），这就是 RadixTree 压缩特性的体现。

---

## Q6: HiCache 是默认开启的吗？主要在哪些场景落地了？

### 默认关闭

HiCache **默认是关闭的**，需要通过 `--enable-hierarchical-cache` 显式开启：

```python
# server_args.py
enable_hierarchical_cache: bool = False
```

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-hierarchical-cache` | False | 总开关 |
| `--hicache-ratio` | 2.0 | Host/GPU 内存池比例 |
| `--hicache-size` | 0 | Host 内存池绝对大小(GB) |
| `--hicache-write-policy` | write_through | 写策略 |
| `--hicache-storage-backend` | None | L3 存储后端（mooncake/hf3fs/nixl 等） |

### 主要落地场景

1. **多轮对话**：不同轮次共享大量前缀（system prompt + 历史对话），HiCache 将历史 KV 缓存保存在 CPU 内存中
2. **多 QA 场景**：多个独立问题共享同一个长 system prompt 或文档上下文
3. **长上下文推理**：长文档 KV 缓存占用大量 GPU 显存，HiCache 将不活跃部分卸载到 CPU/外存
4. **PD 分离部署**：Prefill 节点启用 HiCache，实现跨实例 KV 缓存共享
5. **集群级 KV 共享**：通过 Mooncake/3FS 等分布式存储后端，实现跨推理实例的全局 KV 缓存共享

---

## Q7: SWA（Sliding Window Attention）具体是怎么实现的？窗口外的 KV 就扔掉了是吗？

### 从第一性原理理解 SWA

标准 Transformer 的注意力是全局的：每个 token 可以关注序列中的所有其他 token，KV Cache 必须保存所有历史 token，内存开销 O(L)。

SWA 的核心思想：**每个 token 只关注最近 W 个 token（W 为窗口大小）**，KV Cache 只需保留最近 W 个 token，内存开销从 O(L) 降为 O(W)。

### 窗口外的 KV 是不是直接扔掉了？

**不是简单地"扔掉"。SGLang 使用的是 Hybrid SWA 架构，模型中同时存在 Full Attention 层和 SWA 层。**

SGLang 维护了**两个独立的 KV 池**：

- **Full Attention 层**：保存所有 token 的 KV 数据（不丢弃）
- **SWA 层**：只保存最近 W 个 token 的 KV 数据

### RadixTree 中的 SWA 实现：Tombstone 机制

1. **SWA 节点使用 Tombstone（墓碑）**：当 SWA 数据从内部节点驱逐时，不是删除节点，而是将 `value` 设为 `None`（墓碑），但 Full Attention 的值保持完整。**树结构不变，只是 SWA 组件被标记为"已驱逐"**。

2. **匹配时的窗口验证**：`create_match_validator` 跟踪累积窗口长度，只有当连续窗口 ≥ `sliding_window_size` 时才返回 True。

3. **插入时的 Tombstone 恢复**：当新请求的 token 落在 SWA 窗口内时，可以**恢复之前被墓碑化的节点**——重新分配 SWA 池空间并写入新的 KV 数据。

4. **级联驱逐**：内部节点驱逐优先级为 Full(2) > SWA(1) > Mamba(0)。驱逐 Full 会级联驱逐 SWA；但驱逐 SWA 不会影响 Full。

### 哪些模型使用 SWA？

- **Mistral 系列**：窗口大小 4096
- **DeepSeek-V2/V3/R1**：Hybrid SWA 架构
- **Qwen2** 等部分模型

### 总结

| 问题 | 答案 |
|------|------|
| 窗口外的 KV 直接扔掉？ | ❌ 不是。Full Attention 层保留所有 KV，只有 SWA 层丢弃窗口外的 KV |
| SWA 层窗口外的 KV？ | ✅ 被释放了，但通过 Tombstone 机制保留树结构，窗口内的新请求可以恢复 |
| 为什么这样设计？ | Full 层保证全局信息传递，SWA 层减少计算和内存开销 |

---

## Q8: 支持 L2 RadixTree 提升缓存命中率——现在 L2 没支持 RadixTree 那它的作用是啥？

### 澄清前提

实际上，**SGLang 的 HiCache 从一开始就在 L2 层面维护了 RadixTree 结构**。HiRadixTree 的每个节点同时记录 GPU 和 CPU 的索引：

```python
class TreeNode:
    self.value: Optional[torch.Tensor] = None          # GPU 上的 KV 索引
    self.host_value: Optional[torch.Tensor] = None      # CPU 上的 KV 索引
```

所以"支持 L2 RadixTree"这个说法可能指的是**将 L2 的 RadixTree 能力进一步增强**，或者是在新的 Component 架构（UnifiedRadixCache）中为 L2 添加更完善的前缀匹配能力。

### 如果 L2 没有树结构（纯 KV 存储）vs 有树结构

| 维度 | L2 纯 KV 存储（无树） | L2 有 RadixTree |
|------|----------------------|-----------------|
| 查找方式 | 线性扫描或 hash 查找，无法做前缀匹配 | 沿树遍历，O(L) 时间找到最长共享前缀 |
| 缓存命中率 | 低——无法识别部分前缀匹配，只能整段命中/未命中 | 高——可以精确匹配到任意长度的共享前缀 |
| 内存效率 | 低——相同前缀的多个请求会重复存储 | 高——共享前缀只存一份 |
| 数据搬移 | 需要搬移整段 KV 数据 | 可以只搬移缺失的部分 |
| 驱逐策略 | 只能按 LRU 整段驱逐 | 可以按树节点粒度精确驱逐 |

### L2 RadixTree 带来的核心改进

1. **前缀感知的缓存匹配**：新请求到来时，可以在 L2 中找到最长匹配前缀，只将匹配到的部分从 CPU 搬到 GPU
2. **分层匹配，精确搬移**：`match_prefix` 返回 `host_hit_length`，精确告诉调度器有多少 token 在 CPU 上命中
3. **Write-through/Write-back 策略**：GPU 上新计算的 KV 数据可以通过树结构精确写入对应的 CPU 位置
4. **L3 预取的精确性**：有了树结构，从 L3 预取时可以精确知道需要哪些 token 的 KV 数据

**一句话总结**：L2 RadixTree 的本质是将"前缀感知"能力从 GPU 扩展到 CPU 内存层。没有树结构的 L2 只是一个"KV 数据桶"——你能存取数据，但不知道哪些数据可以复用；有了 RadixTree，L2 变成了一个"智能缓存"。

---

## Q9: KV Cache 实际上是成对的：Key 和 Value。而且在分布式场景下，不同 Rank 可能各存一部分。这个是啥意思？

### KV Cache 为什么是成对的？

在 Transformer 的注意力机制中：

```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

- **Q（Query）**：当前 token 的查询向量，"我在找什么"
- **K（Key）**：每个 token 的键向量，"我有什么"
- **V（Value）**：每个 token 的值向量，"我的内容是什么"

Q 是当前 token 动态生成的，不需要缓存。但 K 和 V 是之前所有 token 预计算好的，需要缓存下来供后续 token 使用。所以 KV Cache 总是成对的——**有 Key 就必须有对应的 Value**，缺一不可。

### 分布式场景下不同 Rank 各存一部分

在**张量并行（TP）**中，模型的注意力头被拆分到不同 GPU 上：

```
假设模型有 32 个注意力头，TP=4：
- Rank 0: 头 0-7 的 K 和 V
- Rank 1: 头 8-15 的 K 和 V
- Rank 2: 头 16-23 的 K 和 V
- Rank 3: 头 24-31 的 K 和 V
```

每个 Rank 只存自己负责的那部分注意力头的 KV Cache。一个完整的 KV Cache 条目被拆成了 4 份，分布在不同 GPU 上。

### 这跟 HiCache 的"组"概念有什么关系？

问题在于：当我们要把 KV Cache 存到 L3（远程存储）时，如果 Key 和 Value 分开存储、不同 Rank 的数据也分开存储，就会出现：

- **不完整的缓存**：只存了 Key 没存 Value，或者只存了 Rank 0 的数据没存 Rank 1 的
- **驱逐不一致**：删了 Key 但忘了删 Value，或者删了 Rank 0 的但 Rank 1 的还在

"组"概念就是把**同一个 token 的所有相关数据**（Key + Value，所有 Rank）打包成一个组，统一管理：
- 统一可见性：查一个组就能知道所有相关数据是否都在
- 统一驱逐：要删就整个组一起删，不会删一半

**类比**：之前你的书是散放的，现在按"套"放——比如《哈利波特》1-7 是一组，要借就借一套，要还就还一套。

---

## Q10: PD 分离中，P 和 D 难道不是都包含全部的模型吗？TP 也在 PD 之间分开？

### PD 分离的基本架构

**是的，每个 P 节点和 D 节点都包含完整的模型参数。** PD 分离是按**推理阶段**拆分，不是按模型参数拆分：

```
┌─────────────────┐         ┌─────────────────┐
│  Prefill 节点    │  KV Cache │  Decode 节点    │
│  (完整模型参数)   │ ────────▶ │  (完整模型参数)   │
│  专注: 处理输入   │   传输    │  专注: 逐字生成   │
│  需要大量计算     │          │  需要大量显存     │
└─────────────────┘         └─────────────────┘
```

### TP 在 PD 中的角色

TP（张量并行）是在**每个 P 或 D 节点内部**的并行方式，不是在 P 和 D 之间分开：

```
Prefill 节点（TP=4）:              Decode 节点（TP=2）:
┌───┬───┬───┬───┐                 ┌───┬───┐
│G0 │G1 │G2 │G3 │  ──KV Cache──▶ │G0 │G1 │
└───┴───┴───┴───┘                 └───┴───┘
4 张 GPU，各存 1/4 参数             2 张 GPU，各存 1/2 参数
```

### 异构 TP 是什么意思？

**异构 TP** 是指 Prefill 和 Decode 可以使用**不同数量的 GPU、不同型号的 GPU**：

- Prefill 用 4 张 A100（计算强，适合 prefill 的密集计算）
- Decode 用 2 张 H800（显存大，适合 decode 的大 batch）

**挑战**：不同 TP 度意味着 KV Cache 的切分方式不同。Prefill 端 4 个 Rank 各存 1/4 的 KV Cache，但 Decode 端只有 2 个 Rank，需要 1/2 的 KV Cache。HiCache 需要处理这种**KV Cache 重分片**的问题。

---

## Q11: "Prefill 直接把 KV Cache 传到 Decode 的 CPU 内存，绕过 Decode GPU"——这句话没说错吗？

### 你的理解是对的，但原文的表述也没错

原文是："Prefill transfers the KV cache directly to Decode's DRAM, **bypassing the GPU as an intermediary**"

这里的"绕过"是相对于**传统传输路径**而言的：

**传统路径**（GPU 作为中转站）：
```
Prefill GPU → 网络 → Decode CPU 内存 → Decode GPU 显存 → 使用
                                    ↑
                              GPU 是中转站
```

**Host 传输模式**（绕过 GPU 中转）：
```
Prefill GPU → 网络 → Decode CPU 内存（DRAM）→ 暂存
                                        ↓
                              需要时再搬到 Decode GPU
```

"绕过"的意思是：**KV Cache 不先送到 Decode GPU 再转存到 CPU，而是直接存到 CPU 内存**。GPU 不再是"中转站"。

### 为什么这样做？

1. **GPU 显存很宝贵**：长上下文场景下 KV Cache 很大（可能几 GB），如果先送到 GPU，会占用大量显存
2. **Decode 需要大 batch**：GPU 显存腾出来后，可以同时处理更多请求
3. **按需加载**：Decode 需要哪部分 KV Cache，再从 CPU 搬到 GPU，而不是一次性全部加载

**类比**：之前快递（KV Cache）直接送到你手里（GPU），但你手里东西太多拿不下。现在先送到你家楼下的储物柜（CPU 内存），你需要时再去拿。

---

## Q12: PP × HiCache 一致性修复——完全没看明白

### 什么是 PP（Pipeline Parallelism）？

PP 把模型按**层**拆分到不同 GPU 上：

```
GPU 0 (Rank 0): 层 1-10   ← 接收输入，处理前 10 层
GPU 1 (Rank 1): 层 11-20  ← 接收 GPU 0 的输出，处理中间 10 层
GPU 2 (Rank 2): 层 21-30  ← 接收 GPU 1 的输出，处理后 10 层，输出结果
```

### PP 中 KV Cache 怎么工作？

每个 Rank 只处理自己负责的层，因此每个 Rank 都有自己那部分层的 KV Cache：

```
Rank 0: 层 1-10 的 KV Cache
Rank 1: 层 11-20 的 KV Cache
Rank 2: 层 21-30 的 KV Cache
```

### 一致性问题是什么？

在 HiCache 中，**只有 Rank 0 维护 RadixTree**（因为只有 Rank 0 能看到完整的输入 token 序列）。其他 Rank 需要根据 Rank 0 的缓存决策来管理自己的 KV Cache。

**问题场景**：

```
时间线：
t1: Rank 0 决定 write-through（把 KV Cache 写到 L2/L3）
t2: Rank 1 还没收到通知，决定驱逐同一批 KV Cache
t3: Rank 0 收到新请求，认为 L2/L3 有缓存，但 Rank 1 已经删了
→ 数据不一致！
```

更具体地说：

1. **异步 write-through**：Rank 0 把 KV Cache 异步写入 L3 存储，但 Rank 1 和 Rank 2 的 write-through 可能还没完成
2. **异步 load-back**：Rank 0 从 L3 加载了 KV Cache，但其他 Rank 还没加载
3. **驱逐不同步**：Rank 0 驱逐了某个节点的 KV Cache，但其他 Rank 对应的 KV Cache 还在

### Event Numbering 方案

**核心思路**：给每个缓存操作编一个全局递增的序号（event number），所有 PP Rank 按序号顺序执行。

```
Rank 0 决策：
  Event #1: write-through token [1-100] 到 L3
  Event #2: 驱逐 token [1-50] 从 GPU
  Event #3: load-back token [1-50] 从 L3

同步给 Rank 1, Rank 2：
  "请按 Event #1, #2, #3 的顺序执行缓存操作"

结果：所有 Rank 的 RadixTree 状态在逻辑上一致
```

**类比**：工厂流水线，每个工人只做一道工序。之前工人各自为政（异步操作），现在给每个操作发一个序号，大家按序号来，确保步调一致。

---

## Q13: MTP 中"用大的目标模型验证这些预测"这一步骤怎么做的？

### Speculative Decoding / MTP 的基本流程

```
1. 草稿模型（小模型）快速生成 k 个候选 token：t1, t2, t3, t4, t5
2. 目标模型（大模型）一次性验证这 k 个 token
3. 接受匹配的 token，拒绝不匹配的，从拒绝点重新生成
```

### 验证的具体过程

**不是逐个检查 t1→t2 链条**，而是目标模型对全部 draft token 做**一次前向传播**（prefill），得到每个位置的概率分布，然后通过**rejection sampling** 决定接受/拒绝。

具体步骤：

```
1. 把 [原始上下文 + t1, t2, t3, t4, t5] 一起送给目标模型
2. 目标模型一次前向传播，得到每个位置的概率分布：
   - 位置 i: P_target(t_{i+1} | context, t1, ..., ti)
3. 对每个 draft token ti+1，计算接受概率：
   - r = P_target(ti+1) / P_draft(ti+1)   ← 两个概率的比值
   - 如果 r ≥ 1：P_target 认为这个 token 比草稿模型更可能，直接接受
   - 如果 r < 1：以概率 r 接受，概率 (1-r) 拒绝
4. 一旦某个 token 被拒绝，后续所有 draft token 都丢弃
5. 从拒绝点用目标模型的概率分布采样一个新 token，继续生成
```

### 举个例子

```
草稿模型生成：t1="你", t2="好", t3="吗", t4="，", t5="今"

目标模型验证：
- t1="你": P_target/P_draft = 1.2 ≥ 1 → 接受 ✓
- t2="好": P_target/P_draft = 0.8 → 以 80% 概率接受 ✓（假设随机数 < 0.8）
- t3="吗": P_target/P_draft = 0.3 → 以 30% 概率接受 ✗（随机数 > 0.3）→ 拒绝！

结果：接受 "你好"，丢弃 "吗，今"
从 t3 位置用目标模型重新采样，比如得到 "今天"
继续用草稿模型生成后续 token...
```

### 本质是提升 compute density 吗？

**是的，你的理解基本正确。** Speculative Decoding 的本质是：

- Decode 阶段是 **memory-bound**（受限于显存带宽），GPU 计算单元大量闲置
- 草稿模型利用这些闲置的计算能力，快速生成候选 token
- 目标模型一次 prefill 验证多个 token，把多次 decode 的 memory-bound 操作**合并成一次 compute-bound 的 prefill 操作**
- 这样就提高了**权重内存的计算密度**（每次加载权重矩阵时做更多计算）

### 和 KV Cache 的关系

- 草稿模型有自己的 KV Cache（draft KV cache）
- 目标模型也有自己的 KV Cache（target KV cache）
- HiCache 需要支持草稿模型的 KV Cache 也能缓存到 L2/L3
- 这样下次验证时，草稿模型的 KV Cache 可以从缓存加载，不用重新计算

---

## Q14: EP 不是只是把 FFN 这个步骤分成多个吗，和 KV Cache 有啥关系？

### EP 的基本概念

你说得对，**EP（Expert Parallelism）确实只是把 MoE 模型中的 FFN/专家部分拆分到不同 GPU 上**，注意力层是共享的（每个 GPU 都有完整的注意力层参数）。

```
MoE 模型的一层：

┌─────────────────────────────────────────┐
│  Attention 层（所有 GPU 共享，做 DP）      │  ← 每个 GPU 独立计算
├─────────────────────────────────────────┤
│  MoE FFN 层（按专家拆分，做 EP）           │
│  GPU 0: 专家 0, 1, 2                     │
│  GPU 1: 专家 3, 4, 5                     │  ← Token 路由到对应专家
│  GPU 2: 专家 6, 7, 8                     │
│  GPU 3: 专家 9, 10, 11                   │
└─────────────────────────────────────────┘
```

### EP 和 KV Cache 的直接关系

**EP 本身和 KV Cache 没有直接关系。** 注意力层是共享的，KV Cache 的生成和管理不受 EP 影响。

### 但 EP 使得 DPA 成为可能，间接影响 KV Cache

**DPA（Data Parallelism Attention）** 是 EP 带来的一个重要优化：

```
传统 MoE 部署（TP=4）：
┌──────────────────────────────┐
│ GPU 0: Attention(1/4头) + Expert(0-2) │  ← 注意力做 TP，KV Cache 1/4
│ GPU 1: Attention(1/4头) + Expert(3-5) │  ← 注意力做 TP，KV Cache 1/4
│ GPU 2: Attention(1/4头) + Expert(6-8) │  ← 注意力做 TP，KV Cache 1/4
│ GPU 3: Attention(1/4头) + Expert(9-11)│  ← 注意力做 TP，KV Cache 1/4
└──────────────────────────────┘
问题：每个 GPU 都存 1/4 的 KV Cache，4 个 GPU 存了 4 份（虽然各不同）
     但每个 GPU 只处理 1/4 的请求

DPA 部署（EP=4 + Attention DP=4）：
┌──────────────────────────────┐
│ GPU 0: Attention(全部头) + Expert(0-2) │  ← 注意力做 DP，处理不同请求
│ GPU 1: Attention(全部头) + Expert(3-5) │  ← 注意力做 DP，处理不同请求
│ GPU 2: Attention(全部头) + Expert(6-8) │  ← 注意力做 DP，处理不同请求
│ GPU 3: Attention(全部头) + Expert(9-11)│  ← 注意力做 DP，处理不同请求
└──────────────────────────────┘
优势：每个 GPU 独立处理完整请求，KV Cache 不需要跨 GPU 通信
     总 KV Cache 容量 = 4 × 单 GPU 容量（线性扩展）
```

**核心区别**：
- TP 模式下，4 个 GPU 协作处理 1 个请求，每个 GPU 存 1/4 的 KV Cache
- DPA 模式下，4 个 GPU 各自独立处理请求，每个 GPU 存完整的 KV Cache，但处理不同请求

DPA 的好处是**减少了 KV Cache 的跨 GPU 通信开销**，同时**总 KV Cache 容量线性扩展**。

### EP × HiCache 的具体问题

Roadmap 中提到 "EP & DP × HiCache"，可能涉及：

1. **DPA 模式下的 HiCache**：每个 GPU 独立管理自己的 KV Cache，HiCache 的 L2/L3 缓存如何协调？
2. **Token 分发与缓存一致性**：EP 中 token 需要路由到对应专家，如果 KV Cache 被缓存了，路由决策是否需要调整？
3. **Mooncake Token Dispatcher**：SGLang 中有 `mooncake.py` token 分发器，用于 MoE 架构下将 token 分发到不同专家，同时与 Mooncake 后端交互优化缓存

---

## Q15: CP（Context Parallelism）的概念是啥？怎么能并行起来？

### CP 解决什么问题？

当输入序列非常长（比如 100 万 token），即使单个 GPU 能放下模型参数，也放不下整个序列的 KV Cache 和中间激活。CP 把**长序列切分**到多个 GPU 上。

### CP 和其他并行方式的区别

| 并行方式 | 切分对象 | 切分维度 |
|---------|---------|---------|
| TP | 模型参数 | 注意力头/FFN 权重 |
| PP | 模型参数 | 层 |
| DP | 数据 | 不同请求 |
| **CP** | **序列** | **同一请求的 token 序列** |

### CP 怎么并行？

**CP 只在 Prefill 阶段使用**，因为 Prefill 需要处理整个长输入序列，计算量大。Decode 阶段每次只生成 1 个 token，不需要 CP。

#### Prefill 阶段的 CP

```
输入序列：[t1, t2, t3, ..., t1000000]（100 万 token）

CP=4 切分：
GPU 0: [t1, ..., t250000]
GPU 1: [t250001, ..., t500000]
GPU 2: [t500001, ..., t750000]
GPU 3: [t750001, ..., t1000000]
```

每个 GPU 只计算自己那段 token 的 Q、K、V。但注意力计算需要**看到所有 token 的 K 和 V**，所以需要 All-Gather 通信：

```
Step 1: 每个 GPU 计算自己那段的 Q, K,V
Step 2: All-Gather K 和 V → 每个 GPU 都有完整的 K, V
Step 3: 每个 GPU 用自己的 Q × 完整的 K,V 计算注意力
Step 4: 每个 GPU 得到自己那段 token 的输出
```

#### Zigzag 重排优化

简单的切分会导致**负载不均衡**：每个 GPU 计算注意力时，Q 只是自己那段，但 K/V 是全部。靠前的 GPU（Q 对应序列开头）计算量小（softmax 的分母小），靠后的 GPU 计算量大。

Zigzag 重排把序列**交错分配**：

```
简单切分：
GPU 0: [1-250K]     ← 计算量小
GPU 1: [250K-500K]
GPU 2: [500K-750K]
GPU 3: [750K-1M]    ← 计算量大

Zigzag 切分：
GPU 0: [1-125K] + [875K-1M]    ← 混合前后段，计算量均衡
GPU 1: [125K-250K] + [750K-875K]
GPU 2: [250K-375K] + [625K-750K]
GPU 3: [375K-500K] + [500K-625K]
```

#### CP 的通信模式

```
每个 Attention 层需要一次 All-Gather（获取完整 K/V）
→ CP 的通信开销与序列长度和 GPU 数量成正比
→ CP 适合序列很长、计算量远大于通信量的场景
```

### CP 和 KV Cache 的关系

1. **Prefill 阶段**：CP 切分序列，每个 GPU 只存自己那段 token 的 KV Cache。All-Gather 后每个 GPU 临时拥有完整 KV Cache，但只用来计算注意力，不持久化
2. **Prefill 结束后**：所有 GPU 的 KV Cache 合并成完整的 KV Cache，供 Decode 使用
3. **Decode 阶段**：不使用 CP，完整 KV Cache 由 Decode 节点管理

### CP × HiCache 的问题

- CP 模式下，KV Cache 分散在多个 GPU 上，HiCache 的 write-through/load-back 需要跨 GPU 协调
- 需要确保所有 CP Rank 的缓存操作同步，避免部分 GPU 写入缓存而其他 GPU 没有

---

## 总结

| 问题 | 核心答案 |
|------|---------|
| Q1 | DeepSeek V4 没用 Mamba，Qwen3.5 用了 Gated DeltaNet（Mamba 的数学等价物） |
| Q2 | Mooncake 是月之暗面的 KVCache 中心化解耦架构，利用 RDMA 零拷贝实现分布式 KV Cache |
| Q3 | KeLing（可灵）是快手的，不是字节的 |
| Q4 | 3FS 是 DeepSeek 开源的，跟字节没关系 |
| Q5 | RadixTree 是 Trie 的空间优化版，压缩无分叉路径 |
| Q6 | HiCache 默认关闭，主要用于多轮对话、长上下文、PD 分离场景 |
| Q7 | SWA 窗口外的 KV 不是简单扔掉，Full Attention 层保留所有 KV，SWA 层用 Tombstone 机制 |
| Q8 | L2 RadixTree 将"前缀感知"能力从 GPU 扩展到 CPU，实现智能缓存而非纯数据桶 |
| Q9 | TP 中不同 Rank 各存一部分注意力头的 KV Cache，"组"概念确保 Key+Value+所有 Rank 统一管理 |
| Q10 | P 和 D 都包含完整模型，TP 在各自内部，异构 TP 指 P 和 D 用不同 GPU 配置 |
| Q11 | "绕过 GPU"是相对传统路径而言，指 KV Cache 直接存到 CPU 内存而非先经 GPU 中转 |
| Q12 | PP 中不同 Rank 异步操作导致 RadixTree 状态不一致，Event Numbering 确保按序执行 |
| Q13 | 验证是目标模型一次 prefill + rejection sampling，本质是提升 memory-bound 的计算密度 |
| Q14 | EP 本身和 KV Cache 无直接关系，但 EP 使 DPA 成为可能，间接优化 KV Cache 管理 |
| Q15 | CP 在 Prefill 阶段切分长序列到多 GPU，通过 All-Gather 获取完整 K/V，Decode 不用 CP |
