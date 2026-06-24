# 一、宏观视角：这段代码在 LLM 推理架构中的位置

## 1.1 LLM 推理的整体架构

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            用户发送请求                                    │
│                            "请帮我写一首诗..."                              │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        API Server (FastAPI)                                │
│  /v1/chat/completions                                                      │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        Tokenizer                                           │
│  "请帮我写一首诗..." → [19293, 28473, ...] (token IDs)                     │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        Scheduler (调度器) ★ 核心                            │
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │ Waiting Queue │───►│ Running Batch│───►│  Forward     │                 │
│  │ (等待的请求)   │    │ (当前批次)    │    │  Execution   │                 │
│  └──────────────┘    └──────┬───────┘    │  (GPU 计算)   │                 │
│                             │            └──────────────┘                 │
│                             ▼                                              │
│                    ┌────────────────┐                                     │
│                    │ RadixCache     │ ★ 我们修改的代码就在这里！            │
│                    │ (缓存管理器)    │                                     │
│                    │                │                                     │
│                    │  - 管理 KV cache │                                     │
│                    │  - 前缀匹配     │                                     │
│                    │  - 驱逐/加载    │                                     │
│                    └────────────────┘                                     │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        Model Executor (模型执行器)                          │
│  forward() → 调用 PyTorch/CuBLAS → GPU 上计算 attention                     │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        GPU                                                  │
│  ┌────────────────────┐  ┌────────────────────┐                           │
│  │  Model Weights     │  │  KV Cache          │                           │
│  │  (模型权重)         │  │  (注意力缓存)       │ ← 显存中的"记忆"           │
│  │  占用 20GB         │  │  动态增长/减少      │                           │
│  └────────────────────┘  └────────────────────┘                           │
└────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 为什么需要 KV Cache？

大模型生成文本是**自回归**的：生成第 N 个 token 时，需要用到前 N-1 个 token 的中间计算结果（KV 值）。

```
用户输入: "今天天气真好，我想去"

生成过程:
Step 1: K1, V1 = model("今天天气真好，我想去")        → 计算所有输入的 KV
Step 2: token_1 = model(K1, V1) → "公园"              → 追加 K2, V2
Step 3: token_2 = model(K1, V1, K2, V2) → "散步"      → 追加 K3, V3
        ↑ 如果每次都重新计算 K1,V1 就很浪费！
        
所以 KV Cache 把之前的 K,V 存起来，避免重复计算。
```

## 1.3 显存不够怎么办？Hierarchical Cache 登场！

```
┌──────────────────────────────────────────────────────┐
│              显存不够时的困境                           │
│                                                      │
│  GPU 显存只有 80GB                                    │
│  模型权重占用 40GB                                    │
│  剩下 40GB 给 KV Cache                                │
│                                                      │
│  但同时有 100 个并发请求...                            │
│  每个请求平均需要 2GB KV Cache                        │
│  需要 200GB > 40GB → 爆显存！                          │
└──────────────────────────────────────────────────────┘
```

**解决方案：把不常用的 KV 数据搬到 CPU 内存！**

```
┌──────────────────────────────────────────────────────────────────┐
│                   Hierarchical Cache 架构                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐         │
│  │                    GPU 显存 (80GB)                   │         │
│  │  ┌─────────────┐  ┌──────────────────────────────┐  │         │
│  │  │ Model       │  │  Hot KV Cache (活跃的)        │  │         │
│  │  │ Weights     │  │  正在生成的请求的 KV           │  │         │
│  │  │ 40GB        │  │  15GB                          │  │         │
│  │  └─────────────┘  └──────────────────────────────┘  │         │
│  │                         │ ▲                          │         │
│  │                         │ │ evict (驱逐)              │         │
│  │                         ▼ │ load_back (加载回来)      │         │
│  └─────────────────────────┼─┼──────────────────────────┘         │
│                            │ │                                     │
│              PCIe 带宽 ~32 GB/s                                    │
│                            │ │                                     │
│  ┌─────────────────────────┼─┼──────────────────────────┐         │
│  │                    CPU 内存 (512GB)  │                  │         │
│  │                         ▼ │                          │         │
│  │              ┌──────────────────────────────┐        │         │
│  │              │  Cold KV Cache (冷的)         │        │         │
│  │              │  暂时不生成的请求的 KV         │        │         │
│  │              │  300GB                        │        │         │
│  │              └──────────────────────────────┘        │         │
│  └──────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

## 1.4 代码中修改的部分在哪个环节？

```
HiCacheController (缓存控制器)
    │
    ├── start_loading()      ← Claude 修改：记录 start_event + token 数
    │       │
    │       │ 发起 DMA 传输
    │       ▼
    │   GPU Load Stream:  ──start_event──► DMA 传输 ──►──finish_event──►
    │
    └── loading_check()      ← Claude 修改：用 CUDA event 测量真实耗时
            │
            │ 检查 DMA 是否完成
            ▼
        if finish_event.query() == True:
            duration = start_event.elapsed_time(finish_event)
            上报到 Prometheus ──► Grafana 监控面板
```

---

# 二、中观视角：Radix Tree + Cache 的数据结构

## 2.1 Radix Tree（基数树）

KV Cache 不是简单的数组，而是用 **Radix Tree** 组织的，这样可以高效地做**前缀匹配**。

```
假设用户发了 3 个请求:
A: "今天天气真好，我想去公园散步"
B: "今天天气真好，我想去逛街"
C: "明天会下雨吗"

Radix Tree 的结构:

                     root
                    /    \
              "今天天气真好"  "明天会下雨吗" (C的完整路径)
                 /    \
        "，我想去"      │
               /       │
      "公园散步"(A)   "逛街"(B)

关键点:
- A 和 B 共享 "今天天气真好，我想去" 这段前缀
- 这段前缀的 KV Cache 只需要存一份！
- C 完全不共享，单独一条路径
```

## 2.2 每个树节点 (TreeNode) 的结构

```python
class TreeNode:
    key: List[int]           # token IDs，比如 [19293, 28473, ...]
    value: torch.Tensor      # GPU 上的 KV cache 数据 (如果有的话)
    host_value: torch.Tensor # CPU 上的 KV cache 数据 (如果被驱逐了)
    evicted: bool            # 标记是否被驱逐到 CPU
    lock_ref: int            # 引用计数，防止正在使用时被驱逐
    parent: TreeNode         # 父节点
    children: Dict           # 子节点
```

## 2.3 驱逐和加载的完整生命周期

```
初始状态 (都在 GPU):
┌────────────────────┐
│  GPU               │
│  TreeNode.value = [K,V data]  ← 在显存中
│  TreeNode.evicted = False
└────────────────────┘

触发驱逐 (显存不足):
HiCacheController.write_through()
├── 把 GPU 上的 KV 数据拷贝到 CPU
├── TreeNode.host_value = KV data on CPU
├── TreeNode.evicted = True
├── TreeNode.value = None (释放 GPU 显存)
└── 记录到 ack_write_queue (等待 DMA 完成确认)

后来需要这个节点 (用户请求了相同前缀):
init_load_back()
├── 发现 TreeNode.evicted = True
├── 收集需要加载的 device_indices (GPU 上要存放的位置)
└── 返回 device_indices 给 Scheduler

start_loading() ← Claude 修改的地方
├── 记录 start_event (CUDA event)
├── 在 GPU load_stream 上发起 DMA: CPU → GPU
├── DMA 完成后记录 finish_event
├── 把 (start_event, finish_event, node_ids) 存入 ack_load_queue
└── 记录 token 数量到 ack_load_num_tokens ← 新增

loading_check() ← Claude 修改的地方
├── 检查 finish_event.query()
├── 如果完成:
│   ├── duration = start_event.elapsed_time(finish_event)  ← 真实 DMA 耗时
│   ├── 上报 Prometheus
│   ├── TreeNode.evicted = False (标记回来了)
│   └── 从队列中删除
└── 如果没完成，等下次再检查
```

---

# 三、微观视角：CUDA Stream、Event、Queue 的底层机制

## 3.1 CUDA Stream（流）

CUDA Stream 是 GPU 上**命令的有序队列**。你可以把它理解成一个"工作台"。

```
概念比喻:
- CPU 是"老板"，发号施令
- GPU 是"工人"，执行任务
- Stream 是"任务清单"，工人按顺序执行

关键特性:
1. 同一个 Stream 内的操作严格按顺序执行
2. 不同 Stream 可以并行执行
3. CPU 往 Stream 里"投递"命令后立即返回（异步！）

在这个项目中的 Stream:
┌────────────────────────────────────────────────┐
│  compute_stream (计算流)                        │
│  ├── forward() 计算注意力                        │
│  └── 正常的模型推理                              │
├────────────────────────────────────────────────┤
│  load_stream (加载流)  ← DMA 传输在这里发生      │
│  ├── memcpy H2D (Host to Device)               │
│  └── 与 compute_stream 并行                    │
├────────────────────────────────────────────────┤
│  write_stream (写入流)                          │
│  ├── memcpy D2H (Device to Host)               │
│  └── 驱逐数据到 CPU                             │
└────────────────────────────────────────────────┘
```

## 3.2 CUDA Event（事件）

CUDA Event 是 Stream 上的**里程碑标记**。

```python
# 核心 API 详解:

# 1. 创建 Event
start_event = torch.cuda.Event()
finish_event = torch.cuda.Event()

# 2. 在 Stream 上记录 Event
start_event.record(stream)
#           │
#           └── 这个 Event 会被放到 stream 的命令队列中
#              当 stream 执行到这里时，Event 被"标记"

# 3. 等待 Event
event.wait(stream)
#     │
#     └── stream 会等待这个 Event 被标记后才继续执行

# 4. 查询 Event 是否已标记（非阻塞）
event.query()  # → True/False
#     │
#     └── 不等待，立即返回
#        True: Event 已被标记（之前的操作已完成）
#        False: 还没到那个里程碑

# 5. 同步等待 Event 完成（阻塞）
event.synchronize()
#     │
#     └── CPU 线程会阻塞，直到 Event 被标记

# 6. 计算两个 Event 之间的时间（核心！）
duration_ms = start_event.elapsed_time(finish_event)
#                │
#                └── 返回 GPU 上两个 Event 之间的实际耗时（毫秒）
#                   注意：不是 CPU 时间，是 GPU 硬件计时！
```

## 3.3 修改前后的执行流对比

### 修改前（错误的方式）:

```
CPU Timeline:
├── start_time = time.perf_counter()    ← 记录 CPU 时间
├── 调用 torch.cuda.memcpy(host, device) ← 提交 DMA 请求（~10μs）
├── 立即返回！DMA 还没开始呢！
├── duration = time.perf_counter() - start_time  ← ~10μs
└── 上报 Prometheus → 错误！只记录了提交开销！

GPU Load Stream Timeline:
├── (CPU 已经返回了)
├── ............. DMA 传输开始 .............  ← 真正的耗时在这里
├── 500μs 后才完成
└── 但 CPU 已经去干别的事了，没测量到！
```

### 修改后（正确的方式）:

```
CPU Timeline:
├── start_event.record(load_stream)     ← 在 GPU 流上放置起点标记
├── 调用 torch.cuda.memcpy(host, device) ← 提交 DMA 请求
├── finish_event.record(load_stream)    ← 在 GPU 流上放置终点标记
├── 把 (start_event, finish_event) 存入 ack_load_queue
└── CPU 去干别的事了... (异步！)

GPU Load Stream Timeline:
├── ▼ start_event (GPU 硬件计时开始)
├── ............. DMA 传输 .............  ← 真正的传输
├── ▼ finish_event (GPU 硬件计时结束)
└── 两个 Event 都被标记了

后来的某个时间点（下一个 scheduler step）:
CPU:
├── loading_check()
├── if finish_event.query():  ← 非阻塞检查
│   ├── True! DMA 已完成
│   ├── duration = start_event.elapsed_time(finish_event)  ← ~500ms ✓
│   └── 上报 Prometheus
└── else:
    └── 还没完成，等下次再检查
```

## 3.4 ack_load_queue 的 FIFO 设计

```python
ack_load_queue: List[HiCacheAck]

# HiCacheAck 的结构:
class HiCacheAck(NamedTuple):
    start_event: torch.cuda.Event   # DMA 开始标记
    finish_event: torch.cuda.Event  # DMA 完成标记
    node_ids: List[int]            # 涉及哪些树节点

# FIFO 队列的行为:
# 因为 DMA 是按顺序发起的，所以完成顺序也是 FIFO 的

索引:  0          1          2
      │          │          │
     [A] ─────► [B] ─────► [C]
      ↑ 最先发起   ↑ 第二个   ↑ 最后发起
      
loading_check() 从头开始检查:
- 如果 A 完成了 → A, B, C 可能都完成了
- 如果 B 没完成 → 停止检查（C 肯定也没完成）
- 删除已处理的条目
```

## 3.5 为什么需要 ack_load_num_tokens 这个平行列表？

```python
# 问题：每个 load_back 操作传输的 token 数量不同
# 需要在 loading_check() 中知道每次传输了多少 token，才能上报

# Claude 的方案：用索引平行对应

ack_load_queue:       [HiCacheAck_A, HiCacheAck_B, HiCacheAck_C]
ack_load_num_tokens:  [100,          250,          50]
                       ↑ 第0个       ↑ 第1个        ↑ 第2个

loading_check() 处理:
idx = 0: finish_event_A.query() → True
         token_count = ack_load_num_tokens[0] = 100
         上报: duration_A, 100 tokens

idx = 1: finish_event_B.query() → True
         token_count = ack_load_num_tokens[1] = 250
         上报: duration_B, 250 tokens

处理完后删除:
del ack_load_queue[:2]       → 剩下 [HiCacheAck_C]
del ack_load_num_tokens[:2]  → 剩下 [50]
```

---

# 四、传输内容详解

## 4.1 到底在传什么？

```
从 CPU 搬到 GPU 的数据是 KV Cache：

对于每个 token，需要保存:
- Key: shape = (num_heads, head_dim)   # 比如 (32, 128)
- Value: shape = (num_heads, head_dim) # 比如 (32, 128)

假设 100 个 token:
- Key: 100 × 32 × 128 × 2 bytes (fp16) = 8.2 MB
- Value: 100 × 32 × 128 × 2 bytes = 8.2 MB
- 总计: 16.4 MB

PCIe 带宽 ~32 GB/s:
- 传输 16.4 MB 需要: 16.4 / 32000 = ~0.5ms = 500μs

这就是为什么 load_back 的真实耗时是几百微秒到几毫秒！
```

## 4.2 indices 的作用

```python
host_indices: torch.Tensor   # CPU 内存中的位置索引
device_indices: torch.Tensor # GPU 显存中的目标位置索引

# 举例:
host_indices = [0, 5, 10, 15]    # CPU 上第 0, 5, 10, 15 页有数据
device_indices = [100, 101, 102, 103]  # 要放到 GPU 的第 100-103 页

# DMA 传输:
for i in range(num_pages):
    GPU[device_indices[i]] = CPU[host_indices[i]]
```

---

# 五、完整时序图

```
时间 →
─────────────────────────────────────────────────────────────────────────────

Scheduler Step N:
├── 发现需要从 CPU 加载 KV Cache
├── init_load_back() 收集 host_indices, device_indices
└── start_loading()
    ├── start_event.record(load_stream)          ← GPU 计时起点
    ├── 在 load_stream 上发起 DMA: CPU → GPU
    ├── DMA 完成后 finish_event 自动标记
    ├── ack_load_queue.append((start_event, finish_event, node_ids))
    └── ack_load_num_tokens.append(100)

Scheduler Step N+1 (CPU 干别的事，GPU DMA 在进行中):
├── 处理其他请求...
├── forward() 在 compute_stream 上运行
└── GPU 的 load_stream 独立传输数据

Scheduler Step N+2:
├── loading_check()
├── finish_event.query() → True (DMA 完成了!)
├── duration = start_event.elapsed_time(finish_event)  → 520ms ✓
├── metrics_collector.observe_load_back_duration(0.52)
├── metrics_collector.increment_load_back_num_tokens(100)
├── 更新 TreeNode.evicted = False
└── del ack_load_queue[:1], del ack_load_num_tokens[:1]

─────────────────────────────────────────────────────────────────────────────
Prometheus/Grafana 上看到:
sglang_load_back_duration_seconds{...} = 0.52  ← 真实的 DMA 耗时！
sglang_load_back_tokens_total{...} = 100       ← 传输了 100 个 token
```