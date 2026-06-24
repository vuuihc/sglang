# Load-Back Duration Metric 修复详解

## 一、问题背景

### 1.1 什么是 Load-Back？

在 SGLang 的 **Hybrid KV Cache** 架构中，GPU 显存有限时会把不常用的 KV cache **evict（驱逐）到 CPU 内存**。当后续请求需要这些 KV cache 时，需要从 CPU 内存**加载回 GPU 显存**，这个过程就叫 **load-back**。

```
┌─────────────┐      load-back      ┌─────────────┐
│   CPU       │ ──────────────────► │   GPU       │
│   Memory    │  (H2D DMA 拷贝)     │   VRAM      │
│  (Host)     │ ◄────────────────── │  (Device)   │
└─────────────┘      evict          └─────────────┘
```

### 1.2 为什么要监控 load-back 耗时？

- **性能诊断**：load-back 耗时直接影响请求的 **TTFT（Time To First Token）**
- **容量规划**：了解 H2D 带宽瓶颈，决定是否需要更多 GPU 显存
- **调度优化**：知道 load-back 有多慢，scheduler 可以更好地做 prefetch 决策

---

## 二、原来的问题

### 2.1 旧代码的计时方式

旧代码在 `hiradix_cache.py` 的 `load_back()` 函数中：

```python
def load_back(self, node: TreeNode, mem_quota: Optional[int] = None):
    start_time = time.perf_counter()  # ❌ 计时起点
    
    # ... 准备 host_indices, 调用 cache_controller.load() ...
    
    device_indices = self.cache_controller.load(host_indices=host_indices, ...)
    
    # ... 设置 node.value, 更新状态 ...
    
    if self.metrics_collector is not None:
        self.metrics_collector.observe_load_back_duration(
            time.perf_counter() - start_time  # ❌ 计时终点
        )
```

### 2.2 为什么这个计时是错的？

关键要理解 **CUDA 的异步执行模型**：

```
时间轴 ──────────────────────────────────────────────►

CPU 线程:
  [调用 load_back] ──────────────────────── [load_back 返回]
  │                                        │
  │ time.perf_counter() 测量的是这段 ─────────┤
  │                                        │
  ▼                                        ▼

GPU load_stream (异步执行):
                    [wait] ─── [H2D DMA 真正的数据传输] ─── [完成]
                              ▲
                    真正的耗时应该测量这段！！！
```

**详细分解 CPU 侧 `load_back()` 函数做了什么：**

```
CPU load_back() 函数执行流程:
┌─────────────────────────────────────────────────────────┐
│ 1. 遍历 evicted 节点，收集 host_indices      ~几微秒    │
│ 2. 调用 cache_controller.load()              ~几微秒    │
│    │                                                      │
│    ├─ 分配 GPU 内存 (alloc)                   ~几微秒    │
│    ├─ 把 CacheOperation 加入 load_queue       ~几微秒    │
│    └─ 返回 device_indices                    ~几微秒    │
│                                                         │
│ 3. 设置 node.value                         ~几微秒      │
│ 4. 更新各种状态                            ~几微秒      │
│                                                         │
│ ⏱️ time.perf_counter() 只测量了以上 CPU 工作！           │
│    总共大约几十微秒，完全不是 load-back 的真实耗时！      │
└─────────────────────────────────────────────────────────┘

⚠️ 注意：cache_controller.load() 只是把操作加入队列
        真正的 DMA 传输是异步的，发生在 GPU 的 load_stream 上
        CPU 函数返回时，DMA 还没开始呢！
```

### 2.3 后果

```
旧 metric 报告的值: ~50μs (0.00005 秒)  ← CPU 函数调用时间
实际 DMA 耗时:      ~5ms   (0.005 秒)    ← GPU 实际传输时间

差了 100 倍！这个 metric 完全不可信。
```

---

## 三、修复方案

### 3.1 核心思路

**不要测量 CPU 函数调用时间，改为测量 GPU 上 DMA 操作的真实耗时。**

使用 **CUDA Event** 机制：
- 在 DMA 开始前，在 `load_stream` 上 record 一个 **start event**
- 在 DMA 完成后，在 `load_stream` 上 record 一个 **finish event**
- 用 `elapsed_time(start, finish)` 计算两个 event 之间的毫秒数

### 3.2 修复后的代码架构

#### 3.2.1 新增工具函数

```python
# cache_controller.py

def timing_event_supported() -> bool:
    """探测当前后端是否支持 enable_timing=True 的 Event。
    
    - CUDA: 支持 ✓
    - 某些后端 (如 ROCm 旧版本): 可能不支持 ✗
    - 探测结果会缓存，只跑一次
    """
    # ... 探测逻辑 ...
```

```python
def make_event(enable_timing: bool = False):
    """创建事件，根据后端能力智能选择。
    
    enable_timing=True 时:
      - 后端支持 → 创建计时 event
      - 后端不支持 → 降级为普通同步 event
    
    这样代码不需要写 if/else，直接 make_event(enable_timing=True) 即可
    """
    if enable_timing and timing_event_supported():
        return device_module.Event(enable_timing=True)
    return device_module.Event()
```

#### 3.2.2 新增 dma_start_event

```python
# cache_controller.py - start_loading() 函数

def start_loading(self) -> int:
    # ... 准备阶段 ...
    
    producer_event = self.layer_done_counter.events[producer_id]
    producer_event.start_event.record()  # 用于跨 stream 同步
    
    # 🆕 新增：专门用于 metric 计时的 event
    dma_start_event = make_event(enable_timing=True)
    
    # ⚡ 关键点：切换到 load_stream 上执行
    with device_module.stream(self.load_stream):
        # 1. 等待 producer 完成（同步点）
        producer_event.start_event.wait(self.load_stream)
        
        # 2. 🆕 在 DMA 实际开始前 record
        dma_start_event.record(self.load_stream)
        
        # 3. 执行真正的 H2D DMA 拷贝
        for i in range(self.layer_num):
            self.mem_pool_host.load_to_device_per_layer(...)
        
        # 4. 最后一层完成后 record finish
        self.layer_done_counter.complete(self.layer_num - 1)
    
    # 5. 把 event 信息放入 ack 队列
    self.ack_load_queue.append(
        HiCacheAck(
            start_event=dma_start_event,  # 🆕 用 dma_start_event
            finish_event=producer_event.finish_event,
            node_ids=op.node_ids,
            num_tokens=len(op.host_indices),  # 🆕 新增 token 数量
        )
    )
```

#### 3.2.3 在 loading_check 中上报 metric

```python
# hiradix_cache.py - loading_check() 函数

def loading_check(self):
    for ack in self.cache_controller.ack_load_queue:
        if not ack.finish_event.query():
            break  # DMA 还没完成
        
        # DMA 完成了，处理后续...
        for ack_id in ack.node_ids:
            end_node = self.ongoing_load_back.pop(ack_id)
            self.dec_lock_ref(end_node)
        
        # 🆕 上报 metrics
        if self.metrics_collector is not None:
            # token 数量总是可以上报
            self.metrics_collector.increment_load_back_num_tokens(ack.num_tokens)
            
            # duration 需要后端支持 timing event
            if timing_event_supported():
                try:
                    # ⚡ elapsed_time 返回毫秒，转为秒
                    duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                    if duration_ms >= 0:
                        self.metrics_collector.observe_load_back_duration(
                            duration_ms / 1000.0
                        )
                except (RuntimeError, NotImplementedError):
                    # 防御性编程：即使探测通过，也可能偶发失败
                    pass
```

---

## 四、关键概念详解

### 4.1 CUDA Stream（流）

**概念**：GPU 上的工作队列。同一个 stream 内的操作按序执行，不同 stream 可以并发。

```
stream 1: [操作A] → [操作B] → [操作C]  (按序)
stream 2: [操作X] → [操作Y]            (可以和 stream 1 并发)
```

在这个修复中，涉及到的 streams：

| Stream | 用途 |
|--------|------|
| `load_stream` | 专门用于 H2D load-back 操作 |
| `write_stream` | 专门用于 D2H evict/write-back 操作 |
| default stream | CPU 调用默认使用的 stream |

### 4.2 CUDA Event（事件）

**概念**：标记 stream 执行到某个时间点的"书签"。

```python
# 创建 event
event = device_module.Event()

# record: 在指定 stream 上当前的位置打个标记
event.record(stream)

# wait: 让某个 stream 等待，直到 event 标记的位置执行完成
event.wait(other_stream)

# synchronize: CPU 等待，直到 event 标记的操作完成
event.synchronize()

# query: 非阻塞查询 event 是否已完成
is_done = event.query()  # True/False

# elapsed_time: 计算两个 event 之间的时间差（毫秒）
ms = start_event.elapsed_time(end_event)
```

**时序图**：

```
load_stream 的执行时间线:
─────────────────────────────────────────────────────────
          record(start_event)     record(finish_event)
               ↓                       ↓
               ╔═══════════════════╗
               ║   H2D DMA 传输    ║
               ╚═══════════════════╝

elapsed_time(start, finish) = 两个 ↓ 之间的时间
```

### 4.3 enable_timing 参数

**为什么不是所有 event 都 enable_timing=True？**

- **性能开销**：启用计时的 event 需要 GPU 记录精确时间戳，有额外开销
- **资源消耗**：某些 GPU 对计时 event 数量有限制

**这个 diff 的优化**：

```python
# ❌ 旧代码：所有 event 都是默认创建（行为因后端而异）
self.load_events = [device_module.Event() for _ in range(num_layers)]

# ✅ 新代码：只有需要的才启用 timing
self.load_events = [
    make_event(enable_timing=(i == num_layers - 1))  # 只有最后一个
    for i in range(num_layers)
]
```

为什么只有最后一个 layer 的 event 需要 timing？

```
Layer 0: [DMA] ─→ event[0] ─┐
Layer 1: [DMA] ─→ event[1] ─┤  这些只用于层间同步
Layer 2: [DMA] ─→ event[2] ─┤  
...                          │
Layer N: [DMA] ─→ event[N] ─┘  ← 只有这个用于 elapsed_time()
                                ← 因为它代表整个 load-back 的结束点

elapsed_time(dma_start_event, event[N]) = 总 DMA 时间
```

### 4.4 HiCacheAck 数据结构

**旧版**（三元组）：
```python
HiCacheAck(
    start_event=producer_event.start_event,
    finish_event=producer_event.finish_event,
    node_ids=[1, 2, 3]
)
```

**新版**（NamedTuple，带新字段）：
```python
class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]
    num_tokens: int = 0  # 🆕 新增：这次操作涉及多少个 token
```

**为什么要加 num_tokens？**

```python
# 旧代码：token 数在调用侧单独计算
device_indices = self.cache_controller.load(host_indices=host_indices, ...)
self.metrics_collector.increment_load_back_num_tokens(len(host_indices))

# 问题：调用侧和 ACK 处理侧分离，容易不一致

# 新代码：Ack 自带 token 数
ack = HiCacheAck(..., num_tokens=len(op.host_indices))
# 在 loading_check 中直接用 ack.num_tokens
self.metrics_collector.increment_load_back_num_tokens(ack.num_tokens)
```

### 4.5 异步 DMA 的完整生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPU (Scheduler 线程)                     │
│                                                                 │
│  1. load_back()                                                 │
│     ├─ 收集需要加载的节点                                        │
│     └─ 调用 cache_controller.load()                             │
│         └─ 返回 device_indices  ← CPU 函数在这里返回！           │
│                                                                 │
│  2. scheduler 继续做其他事情...                                 │
│                                                                 │
│  3. 稍后调用 loading_check()                                    │
│     └─ 检查 ack_load_queue 中的 DMA 是否完成                     │
│         └─ 完成 → 上报 metric → 清理状态                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    GPU (load_stream)                            │
│                                                                 │
│  [wait producer] ─→ [dma_start.record] ─→ [DMA 传输] ─→ [finish]│
│                         ↑                        ↑              │
│                    计时起点 ────────────── 计时终点              │
│                                                                 │
│  这是异步执行的！CPU 函数返回时，这里可能还没开始。               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 Merged Load-Back Operation

```python
# cache_controller.py
op = CacheOperation.merge_ops(self.load_queue)
```

**概念**：多个小的 load-back 请求会被**合并**成一次大的 DMA 操作，提高效率。

```
请求 A: 需要加载 100 个 token
请求 B: 需要加载 200 个 token
请求 C: 需要加载 150 个 token

旧方案：3 次独立的 DMA
新方案：merge_ops → 1 次 DMA 传输 450 个 token

对应的 metric 上报：
- 旧：observe_load_back_duration 被调用 3 次
- 新：observe_load_back_duration 被调用 1 次（包含 450 个 token）
```

这就是为什么 histogram 文档中提到：
> one observation may aggregate multiple requests fused by CacheOperation.merge_ops

---

## 五、优雅降级机制

### 5.1 为什么需要降级？

不是所有后端都支持 `Event(enable_timing=True)`：

| 后端 | 支持 timing |
|------|------------|
| CUDA (现代) | ✅ |
| ROCm (旧版本) | ❌ |
| Ascend | ❌ |

### 5.2 降级流程

```
timing_event_supported()
       │
       ├── 支持 ─→ 创建 timing event ─→ elapsed_time() 可用 ─→ 上报 duration metric
       │
       └── 不支持 ─→ 创建普通 event ─→ 跳过 duration 上报
                                             │
                                             └─→ 但 token 数量仍然上报！✓
```

### 5.3 防御性 try-except

```python
if timing_event_supported():
    try:
        duration_ms = ack.start_event.elapsed_time(ack.finish_event)
        if duration_ms >= 0:
            self.metrics_collector.observe_load_back_duration(duration_ms / 1000.0)
    except (RuntimeError, NotImplementedError):
        pass  # 静默跳过，不影响主流程
```

**为什么探测通过了还会失败？**

- 驱动 bug
- GPU 状态异常
- 极端并发场景下的竞态条件

**设计原则**：宁可少报一个数据点，也不要让整个 load-back 流程崩溃。

---

## 六、修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `cache_controller.py` | 新增 `timing_event_supported()`, `make_event()`；分离 `dma_start_event`；`HiCacheAck` 加 `num_tokens` |
| `hiradix_cache.py` | 删除旧的 `time.perf_counter()`；在 `loading_check()` 中用 CUDA event 上报 |
| `hi_mamba_radix_cache.py` | 同上 |
| `unified_radix_cache.py` | 同上 |
| `hybrid_cache_controller.py` | 同 `cache_controller.py` 的改动 |
| `metrics_collector.py` | 更新 histogram 文档说明 |

---

## 七、对比总结

### 7.1 计时范围对比

```
旧方案 (time.perf_counter):
┌────────────────────────────────────────┐
│ CPU load_back() 函数执行时间            │
│ - 遍历节点: ~10μs                       │
│ - 加入队列: ~5μs                        │
│ - 更新状态: ~5μs                        │
│ 总计: ~20-50μs                          │
│                                        │
│ ⚠️ 不包含 GPU DMA 时间！                │
└────────────────────────────────────────┘

新方案 (CUDA Event elapsed_time):
               ┌────────────────────────────────────┐
               │ GPU load_stream 上的 DMA 时间       │
               │ - 等待同步: ~0μs                   │
               │ - H2D 传输: ~1-10ms (取决于数据量)  │
               │ - 层间同步: ~几十μs                 │
               │ 总计: ~1-10ms                       │
               │                                    │
               │ ✅ 这才是用户真正感受到的延迟！      │
               └────────────────────────────────────┘
```

### 7.2 核心区别

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| 计时位置 | CPU | GPU |
| 计时 API | `time.perf_counter()` | `cudaEventElapsedTime()` |
| 测量对象 | 函数调用时间 | DMA 传输时间 |
| 是否包含排队时间 | ❌ | ❌ (只含 DMA，不含排队) |
| 异步感知 | ❌ 不感知 | ✅ 在 load_stream 上 record |
| 量级 | ~50μs | ~1-10ms |
| 准确性 | ❌ 完全不准 | ✅ 准确 |

---

## 九、深度解析：Merge Load-Back 与旧 Metric 设计的关系

### 9.1 你的问题非常关键

> Merge 之后，多个 load_back 的时间是怎么上报的？

**答案是：只上报一次！**

这是理解旧 metric 为什么"看似只能统计 scheduler 时间"的关键。

### 9.2 Merged Load-Back 的完整流程

```
时间线: CPU 侧 (Scheduler 线程)
─────────────────────────────────────────────────────────

[Request A 到达]
  └─ prefix match → 发现节点 evicted → 调用 load_back(node_A)
      └─ cache_controller.load(host_indices_A) → 加入 load_queue
      
[Request B 到达]
  └─ prefix match → 发现节点 evicted → 调用 load_back(node_B)
      └─ cache_controller.load(host_indices_B) → 加入 load_queue
      
[Request C 到达]
  └─ prefix match → 发现节点 evicted → 调用 load_back(node_C)
      └─ cache_controller.load(host_indices_C) → 加入 load_queue


[Scheduler 调用 start_loading()]  ← 只调用一次！
  │
  ├─ merge_ops(load_queue)  → 合并成一个大操作
  │   host_indices = cat([A, B, C])  # 3 个请求的 token 拼在一起
  │   node_ids = [id_A, id_B, id_C]
  │
  ├─ 一次 DMA 传输所有数据
  │
  └─ 一个 HiCacheAck 放入 ack_load_queue
      start_event=dma_start_event
      finish_event=finish_event
      node_ids=[id_A, id_B, id_C]
      num_tokens=len(A) + len(B) + len(C)


[后续 loading_check()]
  │
  └─ 这个 ACK 完成后:
      ├─ observe_load_back_duration(duration) ← 一次！
      └─ increment_load_back_num_tokens(num_tokens) ← 一次！
```

### 9.3 为什么旧代码用 time.perf_counter()？

**现在回头看旧代码的位置：**

```python
# 旧代码在 load_back() 函数中，不在 DMA 侧！

def load_back(self, node: TreeNode, ...) -> Optional[torch.Tensor]:
    start_time = time.perf_counter()  # ← 在这里计时
    
    # ... 准备工作 ...
    
    device_indices = self.cache_controller.load(...)  # ← 只是加入队列！
    
    # ... 设置状态 ...
    
    if self.metrics_collector is not None:
        self.metrics_collector.observe_load_back_duration(
            time.perf_counter() - start_time
        )  # ← 在这里上报
```

**关键发现：`load_back()` 被调用了 3 次（A/B/C 各一次），所以旧 metric 上报了 3 次！**

```
旧方案的时间线:

Request A: load_back(A) ─→ observe_load_back_duration(~50μs) ← 上报 1
Request B: load_back(B) ─→ observe_load_back_duration(~50μs) ← 上报 2
Request C: load_back(C) ─→ observe_load_back_duration(~50μs) ← 上报 3

然后: start_loading() ─→ merge_ops ─→ 一次 DMA (~5ms)
      loading_check() ─→ (旧代码这里没有 metric 上报！)
```

### 9.4 旧方案的致命问题

```
问题 1: 测量对象错误
  旧方案测量的是 load_back() 这个 CPU 函数的执行时间
  这个函数只是做了一些准备工作，然后加入队列就返回了
  真正的 DMA 还没开始呢！

问题 2: 与 merge 语义不匹配
  假设 merge 了 3 个请求:
  - 旧方案: 上报 3 次，每次 ~50μs，总共 150μs
  - 实际:   1 次 DMA，耗时 ~5ms
  
  用户看到 histogram 里都是 50μs 的数据点，以为 load-back 很快
  但实际上每次 DMA 都要 5ms！

问题 3: 数量失真
  histogram 里的 observation count = load_back() 调用次数
  而实际的 DMA batch count << load_back() 调用次数
```

### 9.5 新方案的正确做法

```python
# 新方案：在 loading_check() 中，等 DMA 完成后上报

def loading_check(self):
    for ack in self.cache_controller.ack_load_queue:
        if not ack.finish_event.query():
            break
        
        # DMA 真的完成了！
        for ack_id in ack.node_ids:
            # 处理 A, B, C 各自的状态...
        
        if self.metrics_collector is not None:
            # 上报 1 次！对应 1 次 DMA batch
            self.metrics_collector.increment_load_back_num_tokens(ack.num_tokens)
            if timing_event_supported():
                duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                self.metrics_collector.observe_load_back_duration(duration_ms / 1000.0)
```

```
新方案的时间线:

Request A: load_back(A) ─→ 加入队列 (无 metric)
Request B: load_back(B) ─→ 加入队列 (无 metric)
Request C: load_back(C) ─→ 加入队列 (无 metric)

然后: start_loading() ─→ merge_ops ─→ 一次 DMA (~5ms)
      loading_check() ─→ observe_load_back_duration(5ms) ← 上报 1 次！
```

### 9.6 Histogram 语义对比

**旧方案：**
```
sglang:load_back_duration_seconds histogram:
  observations: 3 次 (A, B, C 各一次)
  values:       [0.00005, 0.00005, 0.00005] 秒
  含义:         ❌ "load_back 函数调用耗时"
```

**新方案：**
```
sglang:load_back_duration_seconds histogram:
  observations: 1 次 (一次 DMA batch)
  values:       [0.005] 秒
  含义:         ✅ "一次 H2D DMA 传输耗时"
```

### 9.7 这是否是旧代码设计成在 scheduler 侧统计的原因？

**很可能！让我们分析：**

```
旧代码作者的思路可能是：

1. load_back() 在 scheduler 线程中被调用，是同步的
2. 在这里计时最简单，不需要跨线程/跨 stream
3. "反正只是要个大概的数字吧..."

但这个设计忽略了：
- CUDA 是异步的，函数返回 ≠ DMA 完成
- merge_ops 会把多个请求合并，语义变了
- 这个 metric 变得完全不可信
```

**更深层的原因可能是：**

```
历史演进推测：

Phase 1: 没有 merge，每次 load_back 对应一次 DMA
  → time.perf_counter() 虽然不精确，但至少量级差不多

Phase 2: 加了 merge_ops 优化
  → 多次 load_back → 一次 DMA
  → metric 彻底失真，但没人发现

Phase 3: 有人看 dashboard 发现 load-back 才 50μs
  → "哇好快！"
  → 实际是 5ms，差了 100 倍
```

### 9.8 Merge 后的 num_tokens 上报

```python
# 新方案中 num_tokens 的处理

# start_loading() 中：
op = CacheOperation.merge_ops(self.load_queue)
# op.host_indices = cat([A.host, B.host, C.host])
# len(op.host_indices) = len(A) + len(B) + len(C)

self.ack_load_queue.append(
    HiCacheAck(
        ...,
        num_tokens=len(op.host_indices),  # = 100 + 200 + 150 = 450
    )
)

# loading_check() 中：
self.metrics_collector.increment_load_back_num_tokens(ack.num_tokens)  # +450
```

**Counter 的语义：**
- 旧方案：每次 `load_back()` 调用上报 `len(host_indices)`，共 3 次
- 新方案：每次 DMA batch 上报 `sum(len)`，共 1 次
- **总 token 数是一样的！** Counter 是对的，Histogram 才是错的

---

## 十一、待讨论：是否应该包含排队时间？

### 11.1 什么是排队时间？

```
load_stream 上的操作队列:

[之前的 DMA-1] ─→ [之前的 DMA-2] ─→ [wait] ─→ [我的 DMA] ─→ [finish]
                                     ↑
                              dma_start_event.record()

排队时间 = 等待之前 DMA 完成的时间
DMA 时间 = 我自己的数据传输时间

当前方案只测量了 "DMA 时间"，不包含 "排队时间"
```

### 11.2 两种视角

**视角 1：用户感受的延迟**
- 用户从发起请求到数据准备好，感受的是 **排队 + DMA 的总时间**
- 如果要优化用户体验，应该关注总时间

**视角 2：DMA 带宽的纯粹指标**
- 如果只想了解 H2D 传输带宽是否正常，应该只测 DMA 时间
- 排队时间取决于系统负载，不是硬件带宽问题

### 8.3 可能的方案

```
方案 A（当前方案）：只测 DMA 时间
  优点：纯粹反映硬件带宽
  缺点：不包含排队延迟

方案 B：从 producer_event.start_event 到 finish_event
  优点：包含排队 + DMA，反映用户真实体验
  缺点：排队时间波动大，histogram 可能不够稳定

方案 C：两个 metric 都上报
  - sglang:load_back_dma_seconds (DMA 纯传输时间)
  - sglang:load_back_total_seconds (包含排队)
  优点：信息最全面
  缺点：需要额外的 event 和存储空间
```

这个取舍取决于 metric 的使用场景，后续讨论。
