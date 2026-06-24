# KV Cache Roadmap Q&A

这篇笔记回答关于 SGLang 分布式 KV Cache roadmap 的 15 个问题。整体视角是：

- 大模型推理的核心资源矛盾不是只有算力，还有 KV Cache 带来的显存、内存、网络和存储压力。
- Prefill 阶段偏计算密集，Decode 阶段偏带宽和延迟敏感。
- HiCache、PD disaggregation、SWA、MTP、TP、PP、EP、CP 这些技术，本质都在回答同一个问题：KV Cache 应该放在哪里、由谁拥有、如何复用、如何移动、如何保持语义一致。

## 先纠正几个点

上一轮回答里有几处表述需要修正：

- `DeepSeek V4`：目前没有公开证据表明它用了 Mamba。公开资料更像是 MoE + 混合注意力/压缩注意力路线，而不是 Mamba/SSM 混合模型。
- `Qwen3.5`：公开资料显示它采用混合架构，包含 Gated DeltaNet/线性注意力、Gated Attention、MoE 等。它和 Mamba 属于相近的“线性时间序列建模/状态更新”谱系，但严格说不等于用了 Mamba 原版模块。
- `KeLing`/`Kling`/`可灵`：公开归属是快手，不是字节。
- `3FS`：公开归属是 DeepSeek/Fire-Flyer 体系，不是字节。SGLang 支持 `hf3fs` 后端，不代表 3FS 是 SGLang 或字节的项目。

## 1. DeepSeek V4、Qwen3.5 用了 Mamba 吗？Mamba 是什么？

### 这两个模型到底是不是 Mamba

更准确的说法是：

- `DeepSeek V4`：目前应称为“混合注意力/MoE 架构”，不应称为 Mamba 混合架构。
- `Qwen3.5`：可称为“线性注意力/DeltaNet + 标准注意力 + MoE 的混合架构”，与 Mamba-like 路线接近，但不等同于 Mamba。

社区里经常把 Mamba、SSM、linear attention、DeltaNet、Gated DeltaNet 混在一起说，原因是它们都试图解决同一个问题：标准 Transformer attention 在长上下文下太贵。

### 从第一性原理看 Transformer 的问题

标准自回归 Transformer 生成第 `t` 个 token 时，每层 attention 会拿当前 query 去看历史所有 token 的 key/value：

```text
当前 token 的 Q
  attend
历史 token 1..t-1 的 K/V
```

这带来两个成本：

- 训练/Prefill 阶段，完整 attention 近似有 `O(L^2)` 的序列长度成本。
- Decode 阶段，每生成一个 token，都要读取历史 KV，KV Cache 随上下文长度线性增长。

长上下文、多轮 Agent、代码仓库问答会让这个问题更严重。

### SSM 和 Mamba 的前世今生

`SSM` 是 State Space Model，状态空间模型。它的核心思想是：不要显式保存所有历史 token，而是把历史压缩进一个状态。

一个极简抽象是：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

其中：

- `x_t` 是当前输入 token 的表示。
- `h_t` 是截至当前位置的历史状态。
- `y_t` 是输出。

这和 Transformer 很不一样：

- Transformer 像“每次都翻完整聊天记录”。
- SSM/Mamba 像“维护一份持续更新的会议纪要”。

Mamba 的关键进展不是简单地“有状态”，而是 selective SSM：

- 模型可以根据当前输入动态决定写入什么、遗忘什么、保留什么。
- 重要 token 对状态影响更大，不重要 token 可以被弱化。
- 用 selective scan 等 GPU 友好的实现，把递推模型做得足够快。

### 为什么不是所有模型都变成纯 Mamba

状态压缩有一个天然代价：它可能丢失精确细节。

例如：

- 查找上下文中某个精确字符串。
- 做 needle-in-a-haystack。
- 复杂代码引用、变量名、跨段复制。

这些任务中，完整 attention 仍然有优势，因为它可以直接回看具体 token。因此现在很多新模型走混合路线：

```text
多数层：Mamba/DeltaNet/linear attention，负责高效长程建模
少数层：full attention，负责精确召回
再叠加 MoE：降低每个 token 激活的计算量
```

所以“新模型往混合架构走”是对的，但“DeepSeek V4 和 Qwen3.5 都用了 Mamba”这个说法过强。

## 2. Mooncake 的技术原理是什么？3FS 呢？

### Mooncake 是什么

Mooncake 可以理解为一个以 KV Cache 为中心的 LLM serving 架构/系统。它的核心不是“存文件”，而是围绕长上下文推理做 KV Cache 的跨节点复用和传输。

第一性原理：

- Prefill 很贵，尤其是长 prompt。
- 多轮对话、Agent、共享 system prompt 会产生大量可复用前缀。
- 如果每次都重算前缀 KV，就是浪费 GPU。
- 如果把 KV Cache 放到远端池子里，并且能快速传回来，就可以用存储和网络换 GPU 计算。

Mooncake 常见心智模型：

```text
Prefill Pool
  生成 KV
  写入/传输
KVCache Pool
  远端内存/SSD/RDMA
  被多个实例共享
Decode Pool
  查询/读取 KV
  继续 decode
Scheduler
  根据缓存位置和负载做调度
```

它强调：

- RDMA、多 NIC、零拷贝或少拷贝传输。
- 跨实例的 KV Cache 共享。
- 让调度器知道“KV 在哪里”，而不是只看 GPU 是否空闲。

在 SGLang 里，Mooncake 主要作为 HiCache 的 L3 后端之一，也可用于 PD disaggregation 的 KV 传输。

### 3FS 是什么

`3FS` 是 `Fire-Flyer File System`，公开项目来自 DeepSeek/Fire-Flyer 体系。它更像一个面向 AI workloads 的高性能分布式文件系统。

第一性原理：

- AI 训练/推理需要极高吞吐的数据读写。
- 本地 SSD 容量有限，单机吞吐有限。
- 分布式 SSD + RDMA 可以把多机存储聚合成一个高吞吐共享文件系统。

典型结构：

```text
Client
  |
Metadata Service
  管理文件名、目录、chunk 元数据
  |
Storage Service
  管理 SSD 上的数据块
  |
RDMA / 高速网络
```

它可以服务：

- 训练数据读取。
- checkpoint。
- 推理 KV Cache 的 L3 存储。

在 SGLang 中，`hf3fs` 是 HiCache 的一种 L3 storage backend。也就是说，HiCache 负责 KV Cache 语义，3FS 负责底层高吞吐存储。

### Mooncake 和 3FS 的区别

一句话：

- Mooncake 更偏“KV Cache serving/传输/调度体系”。
- 3FS 更偏“高性能分布式文件系统”。

在 HiCache 里二者都可以作为 L3 后端，但抽象层次不同。

## 3. KeLing 是字节的吗？

公开信息看，`KeLing`/`Kling`/`可灵`是快手的产品，不是字节的。

上一轮把它说成字节是不严谨的。更准确表达应该是：

- 如果 roadmap 里写 `KeLing`，需要结合上下文确认它指的是快手可灵相关模型，还是某个内部代号。
- 不能默认说它是字节模型。

## 4. 3FS 是 SGLang 里面的，跟字节有什么关系？

关系应拆开看：

- 3FS 不是 SGLang 原生创造的文件系统。
- SGLang 支持把 3FS/HF3FS 作为 HiCache L3 storage backend。
- 公开信息不支持“3FS 属于字节”。
- 字节或火山生态可能有文章、实验、集成或分析 3FS，但这不等于项目归属。

SGLang 里的关系大概是：

```text
SGLang HiCache
  L1: GPU KV Cache
  L2: CPU/Host KV Cache
  L3: 可插拔存储后端
      - Mooncake
      - HF3FS / 3FS
      - NIXL
      - AIBrix
      - file
```

## 5. RadixTree 和字典树是一回事吗？

可以认为是亲戚，但不是完全一样。

`Trie`，也就是字典树，通常是一条边对应一个字符或 token：

```text
root
  你
    好
      吗
```

`Radix Tree`，也叫 compressed trie，压缩字典树，会把只有一个孩子的链压缩成一段：

```text
root
  "你好"
    "吗"
```

所以：

- Trie 更细粒度。
- RadixTree 是压缩后的 Trie。
- 二者都适合做前缀匹配。

SGLang 的 Radix Cache 用它来匹配 token 前缀。例如两个请求：

```text
请求 A: 你是一个代码助手，请解释 radix cache
请求 B: 你是一个代码助手，请解释 hicache
```

公共前缀“你是一个代码助手，请解释”对应的 KV Cache 可以复用。

## 6. HiCache 默认开启吗？主要在哪些场景落地？

HiCache 默认不开启。

SGLang 当前参数里：

```python
enable_hierarchical_cache: bool = False
hicache_ratio: float = 2.0
hicache_storage_backend: Optional[str] = None
```

需要显式传：

```bash
--enable-hierarchical-cache
```

如果要用 L3，还要配置：

```bash
--hicache-storage-backend mooncake
```

或：

```bash
--hicache-storage-backend hf3fs
```

### 主要落地场景

HiCache 最适合“前缀 KV 复用率高”的场景：

- 长 system prompt：比如固定角色、固定工具说明、固定代码规范。
- 多轮对话：历史上下文重复出现在下一轮请求中。
- Multi-QA：同一篇长文档上问多个问题。
- Agent workload：主 Agent 和子 Agent 共享大量上下文。
- PD disaggregation：prefill/decode 分离后，跨节点复用 KV。
- 多实例服务：多个 SGLang 实例共享 L3 中的 KV。

不适合的场景：

- 每个请求完全随机、几乎没有共同前缀。
- prompt 很短，重算比查缓存和搬数据还便宜。
- L3 网络/存储带宽不足，读写 KV 反而拖慢。

## 7. SWA 怎么实现？窗口外的 KV 就扔掉了吗？

`SWA` 是 Sliding Window Attention，滑动窗口注意力。

普通 causal attention 是：

```text
第 t 个 token 可以看 1..t-1 的所有历史 token
```

SWA 是：

```text
第 t 个 token 只看 max(1, t-W)..t-1 的最近 W 个 token
```

其中 `W` 是窗口大小。

### 窗口外 KV 是否扔掉

对 SWA 层来说，窗口外 KV 在后续 attention 中不会再被访问，因此可以不保留在 GPU 的 SWA KV pool 里，或者用循环 buffer 覆盖。

但要注意：

- 这只对模型架构本身设计成 SWA 的层成立。
- 不能拿一个 full attention 模型，推理时随便把窗口外 KV 扔掉，否则输出语义会变。
- 很多混合模型会同时有 full attention 层和 SWA 层。

SGLang 里有单独的 `SWAKVPool`，把 full attention layers 和 SWA layers 分开管理：

```text
full_kv_pool: 保存 full attention 层需要的 KV
swa_kv_pool: 保存 sliding window 层需要的 KV
```

也就是说，不是全模型统一把旧 KV 扔掉，而是按层区分。

## 8. L2 RadixTree 提升命中率到底是什么意思？

先拆开两个概念：

- `KV 数据本体`：真正的 key/value tensor，很大。
- `RadixTree 元数据`：描述哪些 token 前缀对应哪些 KV，这些 KV 在 L1、L2、L3 的哪里。

没有 RadixTree 的 L2 可以只是一个“KV 仓库”：

```text
我有一些 KV tensor
但很难高效回答：某个请求最长能复用多少前缀？
```

有 L2 RadixTree 后，L2 不只是存值，而是能做前缀语义匹配：

```text
请求 token 序列
  |
遍历 HiRadixTree
  |
前 2048 token 在 L1
后 4096 token 在 L2
剩下的去 L3 或重算
```

### 为什么会提升命中率

GPU L1 很小，CPU L2 大很多。如果只有 L1 做前缀树匹配，很多被 GPU evict 的前缀虽然还在 CPU，却不能被精确、高效地当成“前缀缓存命中”使用。

有 L2 RadixTree 后：

- 被 GPU 淘汰但还在 CPU 的 KV 仍可被前缀匹配命中。
- 可以知道命中长度，而不只是知道某个 page/key 是否存在。
- 可以把 L2 命中的 KV load back 到 GPU，避免重新 prefill。

SGLang HiCache 的设计里，HiRadixTree 节点会记录某段连续 token 的 KV 在 GPU、Host、Storage 哪些层存在。L2 的作用不是简单存 tensor，而是参与前缀复用决策。

## 9. “KV Cache 是成对的，不同 Rank 可能各存一部分”是什么意思？

每一层 attention 会把 hidden state 投影成：

```text
Q = query
K = key
V = value
```

生成下一个 token 时：

```text
当前 Q 和历史 K 做相似度
得到权重
用权重加权历史 V
```

所以历史缓存通常成对保存 K 和 V。缺 K 不行，缺 V 也不行。

### Rank 是什么

在多 GPU 推理里，每个 GPU 进程通常叫一个 rank。

以 Tensor Parallel 为例，模型的 attention heads 可能被切到不同 GPU：

```text
TP rank 0: head 0..7 的 K/V
TP rank 1: head 8..15 的 K/V
TP rank 2: head 16..23 的 K/V
TP rank 3: head 24..31 的 K/V
```

此时，一个 token 的完整 KV 逻辑上是全量的，但物理上分散在多个 rank 上。

这就是“不同 Rank 可能各存一部分”的意思。

### 为什么存储要有 group 概念

如果一个 page 的 K、V、多个 rank 分片分别作为独立对象存储，可能出现：

```text
K 在，V 不在
rank 0 在，rank 1 不在
```

这种半残状态对推理没用，还可能导致错误命中。

因此 roadmap 里说 storage 支持 group，是希望把这些相关对象作为一个逻辑整体：

```text
Group(page 123):
  rank0 K/V
  rank1 K/V
  rank2 K/V
  rank3 K/V
```

查询、可见性、驱逐都按 group 管理，避免只命中一半。

## 10. PD 分离中 P 和 D 不是都包含完整模型吗？TP 也在 PD 之间分开？

是的，通常 Prefill 节点和 Decode 节点都需要能够跑完整模型的 forward，只是它们负责的阶段不同：

- Prefill 节点：处理 prompt，生成初始 KV。
- Decode 节点：接收 KV，逐 token 生成输出。

它们不是“Prefill 只有前半个模型，Decode 只有后半个模型”。那是 PP 的切层概念，不是 PD 的概念。

### 那为什么还会有 PD + TP 异构

因为 P 和 D 虽然都跑完整模型，但可以用不同并行配置。

例如：

```text
Prefill side:
  4 张 GPU
  TP = 4
  适合大 batch、大 prompt 的 GEMM

Decode side:
  2 张 GPU
  TP = 2
  适合低延迟、持续 decode
```

这就是异构 TP：P 和 D 的 TP size 不一样。

难点是 KV 的布局转换。

如果 prefill 侧 `TP=4`，每个 rank 存 1/4 heads 的 KV；decode 侧 `TP=2`，每个 rank 需要 1/2 heads 的 KV。那么从 P 到 D 时要重分片：

```text
P rank0 + P rank1 的一部分 -> D rank0
P rank2 + P rank3 的一部分 -> D rank1
```

所以 PD 分离拆的是阶段，TP 拆的是模型内部张量。二者可以叠加，但需要跨阶段做 KV layout 对齐。

## 11. “Prefill 直接把 KV Cache 传到 Decode CPU 内存，绕过 Decode GPU”这句话对吗？

你的质疑是对的，“绕过 Decode GPU”容易引起误解。

更准确的说法是：

```text
Prefill 生成的 KV 不立即落到 Decode GPU 显存，
而是先传到 Decode 侧 CPU/Host DRAM。
Decode 真正需要时，再从 Host 加载到 GPU。
```

这里的“绕过”不是说永远不进 GPU，而是说传输落点不是 GPU：

```text
传统直觉:
Prefill GPU -> Decode GPU

Host transfer mode:
Prefill -> Decode Host DRAM -> 按需 -> Decode GPU
```

为什么这样做：

- Decode GPU 显存最稀缺。
- 长上下文 KV 很大，如果一到 decode 侧就全部塞 GPU，会降低 batch size。
- 先放 Host，可以让 decode 侧保留更多请求的 KV，再按需搬运。

所以建议避免说“绕过 GPU”，改成“先不落 Decode GPU，落到 Decode Host DRAM”。

## 12. PP x HiCache 一致性修复是什么？

`PP` 是 Pipeline Parallelism，按层切模型：

```text
PP stage 0: layer 0..9
PP stage 1: layer 10..19
PP stage 2: layer 20..29
```

每个 stage 只拥有自己那几层的 KV Cache。

### 为什么会有一致性问题

HiCache/RadixTree 要回答一个问题：

```text
这个请求的前多少 token 可以复用缓存？
```

在非 PP 场景，一个 rank 或一组 TP rank 管完整层，判断比较直接。

在 PP 场景，问题变成：

```text
stage 0 的前 4096 token KV 命中了
stage 1 的前 2048 token KV 命中了
stage 2 的前 4096 token KV 命中了

那整个模型到底能复用多少？
```

答案不能是 4096，因为 stage 1 只有 2048。如果不同 stage 用不同 prefix 长度继续跑，层与层之间的 hidden state、position、KV 边界会错位。

正确做法是取所有 stage 都一致可用的公共前缀：

```text
min(4096, 2048, 4096) = 2048
```

### 为什么提到 Rank 0 事件同步

HiCache 不只是读缓存，还会发生：

- 插入 RadixTree 节点。
- split 节点。
- evict 节点。
- write-back 到 L2/L3。
- load-back 到 GPU。

这些操作如果在不同 PP rank 上顺序不一致，就会导致每个 stage 的 RadixTree 状态不一样。

事件编号的思路是：

```text
Rank 0 决定缓存事件顺序:
  event 1: insert prefix A
  event 2: evict prefix B
  event 3: split node C

其他 PP ranks 按同样编号执行:
  event 1 -> event 2 -> event 3
```

类比流水线工厂：每个工位做不同工序，但大家必须对同一批订单按同一顺序操作。

## 13. MTP/Speculative Decoding 中目标模型如何验证多个 token？

你的理解方向基本对，但可以更精确一点。

假设 draft 模型猜了 5 个 token：

```text
x1, x2, x3, x4, x5
```

目标模型要验证这些 token 是否可以接受。它不是简单地发 5 个独立 query，而是把这段 draft token 当作一段候选 continuation，做一次 target forward。

这个 forward 会产生多个位置的预测分布：

```text
基于 prefix       -> 预测位置 x1 的分布
基于 prefix+x1    -> 预测位置 x2 的分布
基于 prefix+x1+x2 -> 预测位置 x3 的分布
...
```

由于 Transformer prefill 可以并行处理一整段序列，这些位置可以在一次 forward 里算出来。

### 接受链条怎么判断

如果是 greedy，可以直观理解为：

```text
target 认为第 1 个 token 也是 x1 -> 接受 x1
target 认为第 2 个 token 也是 x2 -> 接受 x2
target 认为第 3 个 token 不是 x3 -> 在这里断
```

如果是 sampling，验证会更复杂，通常用 speculative sampling 的接受/拒绝概率，保证最终分布仍等价于目标模型。

关键不变量是：

```text
最终进入输出序列和目标 KV Cache 的 token，必须由 target model 验证或采样决定。
```

draft 可以猜错，但不能污染 target KV。

### 加速本质是什么

你说“提升各层计算中权重内存的计算密度”是很接近的。

Decode 单 token 的问题是：

- 每步只处理一个 token。
- 读一遍大模型权重。
- 矩阵乘很瘦，GPU 利用率低。
- kernel launch 和调度开销也高。

MTP/speculative decoding 把多个 token 合到一次 target forward 验证：

```text
原来:
  读权重 5 次，每次算 1 token

现在:
  读权重 1 次，验证 5 token
```

收益来自：

- 更高的 GEMM 计算密度。
- 更少的逐 token 调度开销。
- 更少的权重读放大。

但它是否加速取决于 draft 命中率。如果 draft 经常错，验证后接受很短，收益会下降。

## 14. EP 只是把 FFN 分成多个 expert，和 KV Cache 有什么关系？

你的理解基本正确：EP 对 KV Cache 的直接关系不如 TP/CP/PP 强。

`EP` 是 Expert Parallelism，主要服务 MoE：

```text
Attention
  产生/读取 KV Cache
MLP/FFN
  token 被 router 分配给不同 expert
```

KV Cache 主要来自 attention 层，不是 FFN 层。所以从数学定义看：

- EP 不直接改变 KV Cache 是什么。
- EP 主要改变 FFN 的计算路径和通信。

### 那为什么 roadmap 还会提 EP x HiCache

因为真实系统里模块会互相影响：

- MoE token routing 会让不同 rank 的负载不均。
- Decode 阶段 batch 小，expert all-to-all 通信容易影响延迟。
- 如果同时有 EP + TP，attention KV 按 TP 分片，FFN experts 按 EP 分片，调度和通信更复杂。
- HiCache 的 prefetch/load-back/write-back 需要和 MoE forward 的执行顺序协调，避免缓存 I/O 阻塞关键路径。

所以 EP 不改变 KV 的定义，但会影响“生成 KV 的 forward 如何调度”和“缓存传输如何与计算重叠”。

## 15. CP 是什么？怎么并行？是在 decode 的时候吗？

`CP` 是 Context Parallelism，上下文并行。它按序列长度维度切分，而不是按层、head 或 expert 切分。

对比：

```text
TP: 切 hidden/head 维度
PP: 切 layer 维度
EP: 切 expert 维度
CP: 切 sequence/context 维度
```

### 为什么需要 CP

超长上下文时，单卡放不下全部 KV：

```text
token 1..1,000,000 的 KV
```

CP 把上下文分到多个 rank：

```text
CP rank 0: token 1..250k 的 KV
CP rank 1: token 250k..500k 的 KV
CP rank 2: token 500k..750k 的 KV
CP rank 3: token 750k..1M 的 KV
```

逻辑上模型还是 attend 全部上下文，只是物理上 KV 分散存放。

### Prefill 时怎么并行

Prefill 阶段，输入序列很长，可以把不同 token 段分给不同 rank。attention 需要跨段通信：

- 每个 rank 算自己 token 的 Q。
- K/V 分布在多个 rank。
- 通过 ring attention、all-gather K/V、reduce partial attention 等方式得到等价的完整 attention 结果。

核心原则：

```text
物理上分片，数学上等价于完整 attention。
```

### Decode 时怎么并行

Decode 时每步只有新 token 的 Q，但它要 attend 历史所有 KV。历史 KV 如果按 CP 分布在多个 rank：

```text
新 token Q
  attend
rank0 的历史 KV
rank1 的历史 KV
rank2 的历史 KV
rank3 的历史 KV
```

实现上可以：

- 把 Q 广播到各 CP rank。
- 每个 rank 对自己那段 KV 算局部 attention。
- 再把局部结果规约成完整 attention 输出。

所以你说“每次 decode，每层并行 attend 之前的 KV”是对的，不过要补一句：每个 rank attend 自己持有的 KV 分片，然后通过通信合并结果。

### CP 和 HiCache 的关系

CP 直接改变 KV 的归属：

```text
某个 token 的 KV 在哪个 CP rank 上？
某个 prefix 命中时，所有 CP rank 是否都命中对应分片？
load-back 时要搬到哪些 rank？
```

因此 CP x HiCache 需要更严格的命中长度同步和分片一致性。

## 总结

可以把这些技术统一成一张图：

```text
LLM 请求
  |
  |-- Prefill: 把 prompt 算成 KV
  |     |-- RadixTree: 找公共前缀
  |     |-- HiCache: L1 GPU + L2 Host + L3 Storage
  |     |-- PD: Prefill 节点专门负责这个阶段
  |
  |-- Decode: 每步读历史 KV 生成新 token
        |-- SWA: 某些层只保留窗口内 KV
        |-- MTP: 先草拟多个 token，再由 target model 验证
        |-- TP: KV 按 head/tensor 分片
        |-- PP: KV 按 layer stage 分片
        |-- EP: FFN expert 并行，间接影响调度
        |-- CP: KV 按 sequence/context 分片
```

最重要的不变量：

- 前缀一致：所有 rank/stage 对可复用 prefix 长度要达成一致。
- 位置一致：token position、page、window 映射不能错。
- 层一致：每层 KV 必须对应同一个请求状态。
- 分片一致：TP/PP/CP rank 只持有一部分，但合起来必须等价于完整模型。
- 验证一致：MTP 的 draft token 只有通过 target model 验证后才能进入最终 KV Cache。

## 相关 SGLang 代码和文档

- `docs/advanced_features/hicache_design.md`：HiCache 设计、L1/L2/L3、HiRadixTree、prefetch、write-back。
- `python/sglang/srt/server_args.py`：HiCache 默认参数，`enable_hierarchical_cache` 默认为 `False`。
- `python/sglang/srt/mem_cache/unified_radix_cache.py`：Unified Radix Cache 节点记录 device/host 状态。
- `python/sglang/srt/mem_cache/swa_memory_pool.py`：SWA 层和 full attention 层分池管理。
- `python/sglang/srt/mem_cache/storage/`：Mooncake、HF3FS、NIXL、AIBrix 等 L3 backend 实现。
