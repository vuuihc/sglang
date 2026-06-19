# 模块一：进程模型与通信机制

> 基于 SGLang 代码库的面试备考精华笔记

---

## 架构拓扑

```
┌──────────────── 主进程 ────────────────┐
│  HTTP Server (FastAPI / uvicorn)       │
│  Engine                                │
│  TokenizerManager (asyncio + uvloop)   │
└────────────────┬───────────────────────┘
                 │ PUSH  scheduler_input_ipc
                 ▼
┌──────────── Scheduler 子进程 ──────────┐
│  PULL ← tokenizer                      │
│  CPU 调度主循环 (while True 同步)       │
│  TpWorker → GPU forward                │
│  PUSH → detokenizer                    │
└────────────────┬───────────────────────┘
                 │ PUSH  detokenizer_ipc
                 ▼
┌──────── DetokenizerManager 子进程 ──────┐
│  PULL ← scheduler                       │
│  token → 文字（增量 / 流式）             │
│  PUSH → tokenizer                       │
└────────────────┬────────────────────────┘
                 │ PUSH  tokenizer_ipc（环形回路）
                 ▼
         TokenizerManager (PULL)
         匹配 rid → HTTP streaming 输出
```

代码参考：`python/sglang/srt/entrypoints/engine.py:182-189`

---

## Q1：TokenizerManager 和 DetokenizerManager 为什么分进程？

**两个核心原因（按重要性排序）：**

**① 流水线并行，提升整体吞吐**

不同请求之间形成三级流水线：
```
Tokenizer    Detokenizer    GPU
req A: [tok]────────────────>[fwd]─>[detok]
req B:        [tok]─────>[fwd]─>[detok]
req C:              [tok]─>[fwd]─>[detok]
```
"A 在 detokenize"和"B 在 GPU forward"在时间上并行，降低每个请求的端到端延迟（减小 TPOT）。

**② 执行模型不兼容**

- `TokenizerManager` 用 asyncio + uvloop，需要同时服务大量并发 HTTP streaming 连接
- `Scheduler` 用 `while True` 同步死循环（`scheduler.py:1368`），目标是零开销地把 batch 送上 GPU
- `DetokenizerManager` 维护每个请求的增量解码状态（`DecodeStatus`：`surr_offset`、`read_offset` 等 UTF-8 边界信息），是独立的有状态同步循环

三者放在一个进程里会让 GIL 和执行模型相互干扰，让 GPU 产生 bubble。

---

## Q2：ZMQ 相比 gRPC 的真实优势

**不是优势的点（需要纠正的常见误区）：**
- gRPC 也支持 Unix Domain Socket（UDS），IPC 性能不是 ZMQ 独有优势
- gRPC bidirectional streaming 同样有内部 buffer，发送方无需等 ack，非阻塞不是 ZMQ 独有

**真实优势：**

| 对比点 | ZMQ + pickle | gRPC + protobuf |
|--------|-------------|-----------------|
| Schema 管理 | 零 schema，Python dataclass 直接 pickle，随时加字段 | 需要 `.proto` 文件 + codegen + 维护 stub |
| 序列化对象 | 任意 Python 对象（array、numpy、dataclass） | 必须是 protobuf message，传 numpy 要额外处理 |
| PUSH 语义 | 原生 round-robin 分发到多个 PULL 端 | 需要自行实现 load balancing |
| 工程迭代速度 | 改字段不用改 proto，适合快速研究迭代 | 改字段需重新 codegen |

**pickle vs protobuf 性能：** 没有 codebase 内的量化数据，不做无根据的声明。

**"零拷贝"的准确说法：**
- 普通路径（文本请求）：pickle 序列化，有拷贝
- 多模态大 tensor（图片 embedding）：通过 `shared_memory` + ZMQ 只传 handle，接收方 mmap 同一块内存，才是真正零拷贝（`managers/mm_utils.py: wrap_shm_features`）

**ZMQ socket 类型语义（不是 OS socket，是 ZMQ 自定义的消息路由模式）：**

| 类型 | 用途 | 位置 |
|------|------|------|
| `PUSH / PULL` | 单向流水线，主数据流 | Tokenizer→Scheduler→Detokenizer→Tokenizer |
| `DEALER / REP` | 异步双向 RPC，需要应答的控制命令 | `Engine.send_to_rpc`（update weights 等） |
| `REQ / REP` | 同步握手 | 多节点 DP 启动时分发 worker 端口 |

---

## Q3：高 QPS 下系统不卡死的三层机制

**误区：** 不是"每 50ms 凑满一个 batch"，而是以下三层机制组合。

### ① ZMQ 队列吸收突发流量

TokenizerManager PUSH 请求到 ZMQ socket，`recv_requests()` 是**非阻塞**的，每轮循环把当前积压的全部请求收走，ZMQ 在传输层做排队缓冲，不丢请求。

### ② Continuous Batching：每个 iteration 动态重组 batch

`event_loop_normal`（`scheduler.py:1368`）每轮：
1. 非阻塞收请求 → 加入 waiting_queue
2. 从 waiting + running 里选出新 batch
3. 执行 GPU forward
4. 已完成的请求退出，新请求插入

无需"等齐"，新请求平均等半个 iteration 就能上车。

### ③ CPU/GPU Overlap：调度和计算并行（`event_loop_overlap`）

`event_loop_overlap`（`scheduler.py:1393`）核心：

```python
while True:
    recv_reqs = self.request_receiver.recv_requests()   # 收新请求
    batch = self.get_next_batch_to_run()
    batch_result = self.run_batch(batch)                # 异步 launch GPU，不等结果
    self.result_queue.append((batch.copy(), batch_result))
    # 下一轮处理上轮结果，同时本轮 GPU 已在跑
```

CPU 在为 batch N+1 做调度时，GPU 在执行 batch N，这是 SGLang "Zero-Overhead Batch Scheduler" 的核心。

---

## Q4（延伸）：Backpressure 分层机制

三层门控，从外到内：

### 第一层：`max_queued_requests`（`scheduler.py:2142`）

waiting_queue 超长时直接 HTTP 503 拒绝新请求。启用优先级调度时可以踢低优先级的已排队请求给新请求让位。

### 第二层：`PrefillAdder.rem_total_tokens`（`schedule_policy.py:496`）

KV cache 预算门控，是 admission 的核心：

```
rem_total_tokens = (available_size + evictable_size) - rem_total_token_offset
```

`rem_total_token_offset` 的组成：

**初始化（每轮 `get_next_batch_to_run` 重建）：**
```
= num_mixed_decode_tokens                              # 当前 decode tokens
+ Σ min(max_new_tokens - len(output_ids), 4096) × ratio   # 每个 running req 剩余 decode 预估
```

**每加一个新 prefill 请求（`_update_prefill_budget`）：**
```
+= ceil_paged(extend_input_len)   # prefill 需要的 slot（page 对齐）
 + max_new_tokens                 # decode 最坏情况预留（不乘 ratio）
 + page_size                      # 每请求一个 page 的安全余量
```

注意：running 请求乘 `new_token_ratio`（< 1），新请求不乘（悲观全量预留）。

### 第三层：运行时 Retract（`scheduler.py:2770`）

decode 时真实分配失败则抢占：先驱逐 RadixCache 可驱逐节点，还不够则踢 output 最短的运行中请求回 waiting_queue。

---

## Q5（延伸）：new_token_ratio 自适应机制

**`NewTokenRatioTracker`（`scheduler_components/new_token_ratio_tracker.py`）：**

- per Scheduler 实例（per DP replica），batch 内所有请求共用
- 线性从 `init` 衰减到 `min`

**三个更新事件：**

| 事件 | 操作 | 含义 |
|------|------|------|
| 每次 decode 成功 | `decay_step()`：`current -= decay` | 系统运行正常，可以更激进 |
| OOM → retract | `current = (实际已生成 + buffer) / 声明上限` | 数据驱动重新校准 |
| 系统 idle | `reset()`：`current = init` | 清空历史，重新从保守值出发 |

**retract 后 ratio 计算（`estimate_new_token_ratio_after_retract`）：**
```python
ratio = (total_decoded_tokens + RETRACT_DECODE_STEPS * num_reqs) / (total_max_new_tokens + 1)
```

**设计批评：** 被 retract 的请求是被强制停止的（biased sample），其 `output_ids` 短不代表它们会自然提前结束。用这个估算未来利用率偏低，导致 ratio 设低后 admission 更激进，可能很快再次 OOM。系统将 retract 当作正常稳态控制手段（类似 TCP 拥塞，但没有 AIMD 的乘性减、加性增那么精确），在高吞吐场景下可接受，代价是 retract 会周期性出现。

---

## Q6（延伸）：OOM 时 GPU 显存的真实状态

**关键认知：KV cache 物理显存在启动时就一次性全部预分配好**（`memory_pool.py`，一个巨大的 `torch.zeros` tensor），`TokenToKVPoolAllocator` 只是一个管理 slot 索引的 CPU 侧数组（`free_pages`）。

OOM 时显存布局：
```
┌────────────────────────────────────────────┐  ← 启动时预分配（约 90% VRAM）
│  locked KV slots                           │  ← running req 已生成 token 的 K/V pair
│  RadixCache evictable prefix nodes         │  ← 历史前缀缓存，可驱逐
│  free slots                                │  ← 真正空闲
└────────────────────────────────────────────┘
```

**不存在"给未来 token 预留的显存"**，`rem_total_token_offset` 只是调度器软件侧的记账数字，没有任何物理显存对应。

OOM = `free slots + evictable slots < batch_size`（下一个 decode step 需要的 slot 数）。

`check_decode_mem` 的顺序：先驱逐 RadixCache → 再看 `available_size >= num_tokens` → 否则 retract。
