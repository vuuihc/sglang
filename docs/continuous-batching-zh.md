# SGLang Continuous Batching 源码梳理（面试讲解版）

> 目标：把 SGLang 里 continuous batching（连续批处理 / in-flight batching）从「请求进来」到「token 吐出去」的全链路讲清楚，每一步都能落到具体代码。
> 代码基于本仓库 `python/sglang/srt/managers/` 下的 `scheduler.py`、`schedule_policy.py`、`schedule_batch.py`。

---

## 0. 一句话先讲清楚「为什么要 continuous batching」

传统 static batching（静态批处理）：把 N 个请求凑成一个 batch，一起 prefill、一起 decode，**等最长的那个请求生成完，整个 batch 才释放**。短请求被长请求拖着，GPU 利用率低。

Continuous batching 的核心思想：**调度的粒度是「一次 forward(一步)」而不是「一整个请求」**。
- 每一步（iteration）结束后，已经生成完的请求立刻离开 batch、释放显存；
- 空出来的槽位立刻让等待队列里的新请求补进来（甚至插入正在 decode 的 batch 中一起跑）。

所以它也叫 **iteration-level scheduling**。SGLang 的实现就是围绕一个 `while True` 事件循环，每一轮决定「这一步跑哪些请求、跑 prefill 还是 decode」。

---

## 1. 整体架构：三个进程 / 角色

面试时先画这张图，说明请求的流转：

```
HTTP 请求
   │
   ▼
TokenizerManager        (进程1) 文本 → token id，做请求预处理，通过 ZMQ 发给 Scheduler
   │  (ZMQ IPC)
   ▼
Scheduler               (进程2) ★continuous batching 的大脑★
   │   - 维护 waiting_queue（等待队列）和 running_batch（正在 decode 的 batch）
   │   - 事件循环每轮组 batch → run_batch（调 GPU forward）→ 处理结果
   │  (ZMQ IPC)
   ▼
DetokenizerManager      (进程3) token id → 文本，流式返回给客户端
```

- 入口：[scheduler.py](../python/sglang/srt/managers/scheduler.py) 的 `Scheduler` 类。
- 三进程通过 ZMQ 解耦，是为了让 tokenize / detokenize 的 CPU 工作不阻塞 GPU 调度循环。

---

## 2. 主事件循环：continuous batching 的骨架

入口在 [scheduler.py:1426 `event_loop_normal`](../python/sglang/srt/managers/scheduler.py#L1426)：

```python
def event_loop_normal(self):
    while True:
        # 1) 收新请求，丢进 waiting_queue
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        # 2) 决定这一轮跑什么（核心！）
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # 3) 跑这一轮 forward + 处理结果
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()      # 没活干，做自检

        self.last_batch = batch
```

讲解要点（四步循环，背下来）：
1. **Receive** — `recv_requests()` 从 ZMQ 非阻塞收请求，`process_input_requests` 把每个请求变成 `Req` 对象放进 `waiting_queue`。
2. **Schedule** — `get_next_batch_to_run()` 是整个 continuous batching 的决策中心（下一节详讲）。
3. **Run** — `run_batch()` 真正调用 GPU 做一次前向。
4. **Postprocess** — `process_batch_result()` 取采样出的 token、判断哪些请求结束、流式输出、释放显存。

> 关键洞察：循环每转一圈只前进「一步」。一个生成 200 token 的请求会经过 1 次 prefill + ~200 次 decode，**每次 decode 之间这个循环都会重新评估要不要让新请求进来**——这就是 "continuous" 的来源。

### 2.1 进阶：overlap 事件循环（CPU/GPU 重叠）

[scheduler.py:1453 `event_loop_overlap`](../python/sglang/srt/managers/scheduler.py#L1453) 是优化版：
- 用一个 `result_queue` 把「处理上一轮结果（CPU）」和「发起这一轮 forward（GPU）」错开；
- 当前轮在 GPU 上算的时候，CPU 同时去组下一个 batch、处理上一轮采样结果，把 CPU 调度开销藏在 GPU 计算后面。
- 面试里可以提一句：「SGLang 默认开 overlap scheduler，把 Python 端的调度开销和 GPU forward 重叠，避免 GPU 等 CPU。」

---

## 3. `get_next_batch_to_run`：决策中心（最重要的一节）

代码在 [scheduler.py:2405](../python/sglang/srt/managers/scheduler.py#L2405)。它要回答两个问题：**这一步跑 prefill 还是 decode？跑哪些请求？**

核心调度策略是 **prefill 优先（prefill-first）**：

```python
def get_next_batch_to_run(self):
    # (A) 把上一轮的 prefill batch 合并进常驻的 running_batch
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.last_batch.filter_batch(...)        # 剔除已完成 / chunked 的请求
        self.running_batch.merge_batch(self.last_batch)

    # (B) 优先尝试组一个新的 prefill batch
    new_batch = self.get_new_batch_prefill()
    if new_batch is not None:
        ret = new_batch                          # 有新请求要 prefill → 先跑 prefill
    else:
        # (C) 没有要 prefill 的，就推进 running_batch 做一步 decode
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch
        else:
            ret = None
    return ret
```

讲解三个关键状态（面试官常追问的数据结构）：

| 状态 | 含义 | 位置 |
|------|------|------|
| `waiting_queue` | 还没开始算的请求队列 | `List[Req]` |
| `running_batch` | 正在 decode 的常驻 batch | `ScheduleBatch` |
| `last_batch` | 上一轮跑的 batch（可能是 prefill） | `ScheduleBatch` |

**(A) Prefill→Decode 的衔接**：prefill 和 decode 是分开的两次 forward。一个请求这一轮被 prefill，下一轮循环时它的 `last_batch` 会被 `merge_batch` 合并进 `running_batch`，从此进入逐 token 的 decode 流。这就是「新请求插进正在跑的 batch」的实现机制。

**(B)/(C) prefill 优先策略的取舍**：
- 优点：新请求能尽快拿到首 token（降低 TTFT，Time To First Token）。
- 代价：prefill 那一步 decode 请求会停一拍。SGLang 用 **chunked prefill**（见 §5）把长 prefill 切块，避免一次 prefill 占用过久、让 decode 请求饿死。
- 也支持 **mixed chunk**（`is_mixed_chunk`），把 prefill chunk 和 decode 请求拼在同一次 forward 里跑，进一步减少互相阻塞（见 [scheduler.py:2774](../python/sglang/srt/managers/scheduler.py#L2774)）。

---

## 4. `get_new_batch_prefill`：准入控制（谁能进 batch）

代码在 [scheduler.py:2533](../python/sglang/srt/managers/scheduler.py#L2533)，真正干活的是 `_get_new_batch_prefill_raw`。这是「显存够不够」的关键决策，面试常深挖。

### 4.1 先排序：调度策略（schedule policy）

[schedule_policy.py:166 `calc_priority`](../python/sglang/srt/managers/schedule_policy.py#L166) 决定 `waiting_queue` 的出队顺序：
- **FCFS**：先来先服务（最简单）。
- **LPM（Longest Prefix Match）**：默认 cache-aware 策略。按与 radix tree 前缀缓存的匹配长度排序，让**能复用最多 KV cache 前缀的请求先跑**，最大化 prefix cache 命中率。
- 队列很长时（>128）会退化成 FCFS，因为前缀匹配排序本身有开销（[schedule_policy.py:220](../python/sglang/srt/managers/schedule_policy.py#L220)）。
- 还有 in-batch prefix caching：同一批里多个请求共享前缀时，先只放一个进去，等它把前缀写进 cache，后面的就能命中（[schedule_policy.py:258](../python/sglang/srt/managers/schedule_policy.py#L258)）。

### 4.2 再准入：`PrefillAdder` 的预算模型

[schedule_policy.py:421 `PrefillAdder`](../python/sglang/srt/managers/schedule_policy.py#L421) 维护几个「预算」，逐个请求尝试加入，加不下就停：

- `rem_total_tokens`：KV pool 剩余可用 + 可驱逐（evictable）的 token 数。注意它**把 tree cache 里可被驱逐的缓存也算成可用空间**（[schedule_policy.py:514](../python/sglang/srt/managers/schedule_policy.py#L514)）。
- `rem_input_tokens`：本轮 prefill 总 token 上限（`max_prefill_tokens`），控制单次 prefill 的计算量。
- `rem_chunk_tokens`：chunked prefill 的单块大小（`chunked_prefill_size`）。

主循环 [scheduler.py:2646](../python/sglang/srt/managers/scheduler.py#L2646)：

```python
for req in self.waiting_queue:
    if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
        self.running_batch.batch_is_full = True
    if self.running_batch.batch_is_full:
        break
    req.init_next_round_input(self.tree_cache)   # 算前缀匹配，确定真正要算的 token
    res = adder.add_one_req(req, ...)             # 尝试塞进预算
    if res != AddReqResult.CONTINUE:
        break                                     # NO_TOKEN / 满了，停止收人
```

`add_one_req`（[schedule_policy.py:844](../python/sglang/srt/managers/schedule_policy.py#L844)）里每个请求要预留的显存是：

```python
total_tokens = req.extend_input_len + max_new_tokens + page_size
#              ↑prefill要算的输入   ↑给未来输出预留   ↑分页对齐余量
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN      # 显存不够，这个请求进不来
```

讲解要点：
- **为输出预留显存**：准入时不止看输入，还要按 `new_token_ratio` 估算未来要生成的 token 也占显存。这是防止 decode 中途 OOM 的「保守预估」。
- **`extend_input_len` = 输入长度 − 前缀缓存命中长度**：命中 radix cache 的前缀不用重算，直接复用已有 KV，这是 SGLang 的核心优化（RadixAttention）。
- 收人结束后，[scheduler.py:2736](../python/sglang/srt/managers/scheduler.py#L2736) `ScheduleBatch.init_new(...)` + `prepare_for_extend()` 把这批请求打包成一次 prefill forward。

---

## 5. Chunked Prefill：限制单步 prefill 的体量

问题：一个 32K token 的长 prompt 如果一次性 prefill，单个 forward 的**激活显存**和**计算时长**都会爆炸（激活显存 ∝ prefill token 数）。

解法：把长 prefill 切成多块，每块 ≤ `chunked_prefill_size`，分多轮循环算完。

- 当一个请求一次吃不下时，`add_one_req` 走 chunked 分支（[schedule_policy.py:962](../python/sglang/srt/managers/schedule_policy.py#L962)），按 `page_size` 对齐截断，标记为 `new_chunked_req`。
- Scheduler 用 `self.chunked_req` 跟踪「这个请求还有 prefill 没算完」（[scheduler.py:2722](../python/sglang/srt/managers/scheduler.py#L2722)）。
- 每轮循环它都被 `add_chunked_req` **优先继续算下一块**，直到最后一块（`contains_last_prefill_chunk`），才允许进入 decode。
- 还有 `enable_dynamic_chunking`：根据已算的历史长度动态预测下一块大小（[scheduler.py:2600](../python/sglang/srt/managers/scheduler.py#L2600)）。

> ⚠️ **常见误解纠正**：默认（非 mixed）模式下，chunked prefill **不会**让 decode 在块之间插进来。因为 `get_new_batch_prefill` 只有在 `chunked_req is None` 时才返回 `None`（[scheduler.py:2569](../python/sglang/srt/managers/scheduler.py#L2569)），所以长请求分块期间会**连续多轮跑 prefill**，running_batch 的 decode 全程等待，直到所有块算完。**「优先 prefill，没 prefill 才 decode」是严格成立的。**

chunked prefill 在默认模式的真正价值：
1. **限制单次 forward 的激活显存 / 计算量**，避免超长 prefill 撑爆显存或长时间独占 GPU。
2. **块之间调度器重获控制权**：可处理 abort、加入新 prefill 请求、做 priority 抢占（但不是 decode）。
3. **是 mixed chunk 的前提**：有了「块」，才能把块和 decode 缝进同一 forward。

**让 prefill 与 decode 真正交错，必须靠 `enable_mixed_chunk`（默认关，进程级）**：开启后，每次组 prefill batch 且存在 running decode 时，`mix_with_running`（[scheduler.py:2774](../python/sglang/srt/managers/scheduler.py#L2774)）把**整个 running decode batch** 缝进这次 prefill 的同一 forward，于是一次 forward 同时推进 prefill chunk 和 decode。例外（不 mix）：开了 `return_logprob` 或 embedding 输入。

> 一句话总结：**chunked prefill 限制单步 prefill 体量；decode 与 prefill 的真正交错由 mixed chunk（或 PD 分离）实现，而非 chunked prefill 本身。**

---

## 6. `update_running_batch` + retraction：显存兜底机制

代码在 [scheduler.py:2824](../python/sglang/srt/managers/scheduler.py#L2824)。每一步 decode 前要做的事：

```python
def update_running_batch(self, batch):
    batch.filter_batch(...)                      # 1) 剔除已完成的请求
    if not batch.check_decode_mem():             # 2) 显存够下一步 decode 吗？
        retracted_reqs, ... = batch.retract_decode(...)   # 3) 不够 → 回退部分请求
        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)  # 退回等待队列
    batch.prepare_for_decode()                   # 4) 准备 decode 的张量
    return batch
```

### 6.1 `check_decode_mem`（[schedule_batch.py:2275](../python/sglang/srt/managers/schedule_batch.py#L2275)）
下一步每个请求要 +1 个 token 的 KV 槽位。先尝试从 tree cache 驱逐（evict）腾空间，再看 allocator 够不够。

### 6.2 `retract_decode`（[schedule_batch.py:2288](../python/sglang/srt/managers/schedule_batch.py#L2288)）—— continuous batching 的安全网
当 KV pool 满了，必须从当前 batch「踢人」：
- 排序策略：优先踢**输出 token 最少、输入最长**的请求（`len(output_ids)` 小的留，损失最小）——因为它们已经生成的进度少，重算代价相对小。
- 被踢的请求 `reset_for_retract()` 后**退回 `waiting_queue`**，KV cache 释放，之后重新调度。
- 至少保留一个请求；如果连最后一个都放不下，优雅 abort 而不是 crash（[schedule_batch.py:2324](../python/sglang/srt/managers/schedule_batch.py#L2324)）。
- 同时调高 `new_token_ratio`：被迫 retract 说明之前对输出长度估计太乐观，于是临时变保守，少放点新请求；之后随时间 `decay_step()` 慢慢恢复（[scheduler.py:2898](../python/sglang/srt/managers/scheduler.py#L2898)）。

> 这是面试加分点：**continuous batching 因为是「乐观超额准入」，必须有 retraction 作为兜底**。准入时按估计的输出长度预留显存，估计不准就靠 retract 回退，再用动态 `new_token_ratio` 自适应。

---

## 7. `run_batch` + 处理结果：一步的闭环

- [scheduler.py:2966 `run_batch`](../python/sglang/srt/managers/scheduler.py#L2966)：调 `model_worker.forward_batch_generation`，做一次 GPU 前向 + 采样，拿到每个请求的 `next_token_ids`。可走 CUDA Graph 加速 decode。
- [batch_result_processor.py:588 `process_batch_result_decode`](../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L588)：
  - 把采样出的 token 追加到每个 `req.output_ids`；
  - `req.update_finish_state()` 判断是否结束（命中 EOS / 达到 `max_new_tokens` / stop string）；
  - `stream_output()` 把新 token 通过 ZMQ 发给 DetokenizerManager 流式返回；
  - 结束的请求会在下一轮 `filter_batch` 时离开 batch、释放 KV。

至此闭环：**结束的走人 → 显存释放 → 下一轮循环新请求补位**，continuous batching 持续滚动。

---

## 8. 面试讲解串词（30 秒 / 2 分钟 / 5 分钟）

**30 秒版**：
> SGLang 的 continuous batching 是 iteration-level 调度。一个 `while True` 事件循环，每轮决定跑 prefill 还是 decode。采用 prefill 优先策略：有新请求就先 prefill，然后把它合并进常驻的 running_batch 做逐 token decode。每步结束后已完成请求立刻离开、释放 KV cache，新请求随时补位。配合 RadixAttention 前缀缓存、chunked prefill 切长 prompt、以及显存不足时的 retraction 兜底。

**2 分钟版**：按 §2 四步循环 → §3 prefill 优先 → §4 准入预算 → §6 retraction 串。

**5 分钟版**：完整走 §1 架构 → §2 循环 → §3/§4 调度准入 → §5 chunked prefill → §6 retraction → §7 闭环，每段点一个代码位置。

### 高频追问准备
- **Q: prefill 和 decode 怎么区分调度？** 两次独立 forward，`forward_mode` 区分 `EXTEND`/`DECODE`；prefill 优先，prefill 后经 `merge_batch` 进 running_batch 转入 decode。
- **Q: 显存怎么管理？** KV cache 用分页 pool（`token_to_kv_pool_allocator`）+ radix tree 前缀缓存（可驱逐）。准入按 token 预算 + 预留输出，decode 前 `check_decode_mem`，不够就 evict / retract。
- **Q: 长请求会不会饿死短请求？** chunked prefill 切块；mixed chunk 让 prefill 和 decode 同 forward；retraction 防 OOM。
- **Q: 怎么提高吞吐？** LPM 调度最大化 prefix cache 命中、overlap scheduler 藏 CPU 开销、CUDA Graph 加速 decode、连续批处理本身提高 GPU 占用率。
- **Q: continuous batching vs static batching 的本质区别？** 调度粒度从「请求」降到「一步」，请求可在任意 iteration 进出 batch。

---

## 8.5 FAQ：PrefillAdder 预算模型 & 计算模型（易混点）

### Q1. PrefillAdder 的三个预算到底约束什么？

PrefillAdder 收 prefill 请求时，**同时**满足三个独立约束，任一用尽就停止收人：

| 变量 | 约束 | 初始值 | 物理意义 | 类型 |
|------|------|--------|---------|------|
| `rem_total_tokens` | **显存** | KV pool 可用 + 可驱逐 − 老请求预留 | 还能存多少 token 的 KV | property（动态算） |
| `rem_input_tokens` | **本轮 prefill 总计算量** | `max_prefill_tokens` | 这一步最多算多少输入 token | 计数器 |
| `rem_chunk_tokens` | **单块大小** | `chunked_prefill_size` | 单个请求一块最多多长 | 计数器 |

计数器在每收一个请求时由 [`_update_prefill_budget`](../python/sglang/srt/managers/schedule_policy.py#L600) 扣减。

**Q1a. `rem_total_tokens` 是「新 KV 没地方存」的意思吗？**
不是。它是准入前的**容量预检**，确保新请求的 KV（输入 + 预留输出）**将来有地方存**：
```python
rem_total_tokens = allocator.available_size()    # 当前空闲 KV 槽位
                 + tree_cache.evictable_size()    # 前缀缓存占着、但可丢弃回收的槽位
                 - rem_total_token_offset         # 减去：running 请求未来要生成的 token 预留
```
- 算上 `evictable_size`：radix tree 里没被锁住的前缀缓存可驱逐腾位，所以当「可用空间」——准入偏乐观。
- 减 `rem_total_token_offset`：老的 decode 请求未来每步还要 +1 token，必须先预留，否则被新请求挤占会 OOM。预留量 = `Σ min(max_new−已生成, CLIP) × new_token_ratio`。
- 每个新请求需要 `extend_input_len + max_new + page_size`，≥ `rem_total_tokens` 就 `NO_TOKEN` 停止收人。

**Q1b. `rem_input_tokens` 初始 = `max_prefill_tokens`？** 是（非 mixed 模式下）。`__init__` 里 `rem_input_tokens = max_prefill_tokens − num_mixed_decode_tokens`，约束**整批输入 token 总和**，防一步算太多。

**Q1c. `rem_chunk_tokens` 初始 = `chunked_prefill_size`？** 是。关闭 chunked prefill 时为 `None`（相关检查跳过）；开 `enable_dynamic_chunking` 时由 `predict_next_chunk_size` 动态覆盖。它约束**单个请求一块**的大小（vs `rem_input_tokens` 约束整批总和）。

### Q2. 计算模型 & chunked prefill

**Q2a. 一个 chunk 块和一个 page 一一对应吗？**
不。chunk 是 **page 对齐（page 整数倍）**，但通常远大于一个 page：
```python
trunc_len = rem_chunk_tokens // page_size * page_size   # 向下取整到 page 整数倍
```
典型 `chunked_prefill_size`≈8192，`page_size`≈1/16/32/64，所以**一块 = 很多 page**。page 是显存分配最小单位，chunk 是「一步算多长」的调度单位，只是要对齐到 page 边界。

**Q2b. 一次 run_batch，batch 里有什么？什么形状？**
SGLang forward 是 **ragged / varlen（变长不 padding）**，所有请求 token 拼成一维，用 `seq_lens` 记边界（类似 FlashAttention varlen）：

| `forward_mode` | 内容 | `input_ids` 形状 |
|------|------|------|
| **EXTEND**（prefill） | N 个请求，各贡献 `extend_input_len` 个输入 token | 1D，Σ extend_input_len（变长拼接） |
| **DECODE** | N 个请求，各贡献**正好 1 个** token | 1D，长度 = batch_size |
| **MIXED**（可选） | 一个 prefill chunk + 若干 decode 请求各 1 token 拼一起 | 1D，prefill 段 + decode 段 |

所以 prefill batch 不是矩形 `[B, L]`，而是不等长输入**首尾相接**成 `[total_tokens]`，attention 靠 cumulative seq lens 切分——这就是无需 padding、显存高效的原因。

**Q2c. 「每轮非 prefill 即 decode」和「chunked/mixed/PD 会组合」矛盾吗？**
不矛盾。基础模型是对的，后三个是**正交的细化**，构成从「完全隔离」到「完全融合」的光谱：

1. **基础（默认）**：每轮 batch 要么全 EXTEND 要么全 DECODE，二选一，prefill 优先。
2. **chunked prefill**：不改变「这轮是 prefill 还是 decode」。长请求 prefill 切多块，**每块仍是一次独立 EXTEND forward**，只是 prefill 轮变多变碎，让 decode 能在块之间穿插。→ 仍然二选一。
3. **mixed chunk**（`is_mixed_chunk`，默认关）：真正「同一次 forward 里既有 prefill 又有 decode」，把 prefill chunk 和 running decode 拼进同一 forward（`mix_with_running`）。
4. **PD 分离**（disaggregation）：prefill 和 decode **不在同一实例**。Prefill 实例只跑 EXTEND，Decode 实例只跑 DECODE，中间网络传 KV。不是「组合」，而是**物理拆开**（`disaggregation_mode` = PREFILL/DECODE/NULL，单机混合为 NULL）。

一句话：**默认二选一 → chunked 把 prefill 切碎（仍二选一）→ mixed 主动融合 → PD 彻底拆到两台机器**。

### Q3. `rem_input_tokens`(`max_prefill_tokens`) 和 `rem_chunk_tokens`(`chunked_prefill_size`) 啥关系？

两个都限制「每步 prefill 能算多少 token」，但范围和动作不同：

| | `rem_input_tokens` (`max_prefill_tokens=16384`) | `rem_chunk_tokens` (`chunked_prefill_size`) |
|---|---|---|
| 范围 | batch 所有请求输入 token 总和 | 也限 batch 总和 + **单请求截断阈值** |
| 生效 | 总生效 | 仅分块开启（`-1` 时为 None） |
| 超限动作 | 停止再收请求 | 把当前请求截成一块 |
| 取值来源 | 拍的经验默认值（粗兜底） | 按显存自动调（激活显存 ∝ 它） |

两者在 [`_update_prefill_budget`](../python/sglang/srt/managers/schedule_policy.py#L612) 里**都**按 `extend_input_len` 递减，谁先归零谁停。因 `chunked_prefill_size`(如 8192) ≤ `max_prefill_tokens`(16384)，**开分块时真正卡 batch 的是 `chunked_prefill_size`**，`max_prefill_tokens` 是外层兜底。分块关闭（`-1`）时只剩 `max_prefill_tokens`，长请求整块 prefill 不截断。

### Q4. mixed chunk 是进程级开关吗？开了每次都尽量 mix？

是。`enable_mixed_chunk`（默认 `False`）是启动参数，进程级固定。开启后，**每次**组 prefill batch 且 running decode 非空、且没开 `return_logprob`/embedding 输入时，就把**整个** running decode batch 缝进这次 prefill forward（[scheduler.py:2774](../python/sglang/srt/managers/scheduler.py#L2774)）——即「能 mix 就尽量 mix」。

### Q5. 既然 `get_next_batch_to_run` 严格「优先 prefill」，chunked prefill 怎么让 decode 插进来？

**默认模式下：插不进来**（这是上面 §5 的纠正）。`get_new_batch_prefill` 仅当 `chunked_req is None` 才返回 `None`，所以长请求分块期间连续多轮跑 prefill，decode 全程等待。
- 默认：chunked 的价值是**限制单步体量**（显存/计算）+ 块间让调度器重获控制权，**不是** decode 交错。
- decode 与 prefill 真正交错 = **mixed chunk**（同 forward 缝进 decode）**或 PD 分离**（物理拆开）。
- 三者都不改变「每轮 `get_next_batch_to_run` 优先 prefill」的主干——mixed 只是让「那个 prefill batch」内部已经带上了 decode。

### Q6. 为什么 `return_logprob` / embedding 输入时不能 mixed chunk？

- **`return_logprob`**：未实现的限制（代码挂 `TODO`）。prefill 要返回整段 prompt 每个位置的 input logprob，decode 只要最后 1 个；混在一条扁平变长 logits 里按位置切片提取太绕，暂未支持，故「开 logprob 就不 mix」。
- **`input_embeds`（embedding 输入）**：原理冲突。`mix_with_running` 拼的是 `input_ids`，而 embedding 请求走 `input_embeds`（`[N, hidden]` 向量 vs `[M]` 整数），形状对不上，拼不了。

### Q7. 部署形态 / GIL / 每次 run_batch 跑多少？（生产 sizing 手感）

**并行方式 × scheduler 进程：**

| 并行 | 切什么 | 用在哪 | scheduler |
|------|--------|--------|-----------|
| **TP** 张量并行 | 每层权重切片，每层 all-reduce | 节点内（NVLink） | tp 个进程，**SPMD 锁步同一 batch** |
| **PP** 流水线并行 | 按层切 stage | 跨节点 | pp 个 stage，micro-batch |
| **DP** 数据并行 | 整模型复制 | 扩吞吐 | dp 个**独立** scheduler |
| **EP** 专家并行 | MoE expert 分卡 | MoE | 配合 TP/DP |

总进程数 ≈ `dp × tp × pp`。默认 `tp=1,dp=1`（单卡单 scheduler）。

**sizing 决策逻辑**：① 单卡装得下？→ `tp=1`，用 `dp=N` 扩吞吐。② 装不下 → 节点内 `tp`（≤8，靠 NVLink）。③ 单节点装不下 → PP 跨节点 / 多节点 TP / (MoE) EP。④ 长上下文 → 留 KV 余量，可能多用卡。

**典型配置（fp16，8×80GB 节点）：**

| 模型 | 权重 | 部署 | 理由 |
|------|------|------|------|
| 7-8B | ~16GB | `tp=1,dp=8` | 单卡装得下，复制扩吞吐 |
| 30-34B | ~68GB | `tp=2~4` | 留 KV 余量 |
| 70B | ~140GB | `tp=4`（或 `tp=8` 低延迟，`tp=4,dp=2` 平衡） | 跨卡分权重 |
| 405B | ~810GB | **fp8** `tp=8`（~405GB）；fp16 需多节点 `tp=16` | 必须量化或跨节点 |
| DeepSeek V3/R1 671B MoE | fp8 ~671GB | **DP attention + EP**，8×H200 / 多节点 | SGLang 招牌 |

手感：**fp8 量化是大模型标配**（显存减半）；**TP 不跨节点**（每层 all-reduce，跨节点网络比 NVLink 慢一档，延迟爆炸）；小模型 `dp` 优先（无通信、吞吐线性），大模型被迫 `tp`，常组合 `dp×tp`；`chunked_prefill_size` 随显存自动调（小卡 2048 / 大卡 8192-16384）。

**GIL 不是瓶颈**：每个 scheduler 是独立**进程**（无跨进程 GIL）；进程内 forward 在 CUDA kernel 异步执行、**释放 GIL**，Python 只发起 kernel + 调度。真正的串行是「Python 调度 vs GPU 计算」，靠 overlap scheduler（§10.2）重叠消除。

**一次 run_batch = 一次 forward**：要么 prefill（总 token ≤ `chunked_prefill_size`，可能是长请求的块[独占]或多个短请求拼批），要么 decode（每请求 1 token，走 CUDA Graph）。不是固定 8192、也不是固定一个请求。

### Q8. 单个长请求分多块期间，中间块会塞别的请求 prefill 吗？

**分情况，关键看这个长请求还剩多少：**

| 阶段 | 长请求剩余 token | 这步 batch | 原因 |
|------|-----------------|-----------|------|
| **中间块** | > `chunked_prefill_size` | **独占**，别人进不来 | `add_chunked_req` 把整个预算吃满，`rem_chunk_tokens≈0`，后续 `add_one_req` 立即 `OTHER` break |
| **最后一块** | < `chunked_prefill_size` | 块 + 其他新请求拼批 | 块只占一部分，剩余预算可收新请求 |

`add_chunked_req`（[schedule_policy.py:695](../python/sglang/srt/managers/schedule_policy.py#L695)）里 `extend_input_len = min(剩余, rem_chunk_tokens)`：中间块剩余 > 预算 → 吃满 → 扣到 0。所以一个 100K token 的请求（~13 块）**前 12 块每块独占一步，running_batch 的 decode 全程停**，直到最后一块才可能和新请求拼批、之后才轮到 decode。这就是长 prompt「霸占」prefill 管线的来源，也是 mixed chunk / PD 分离要解决的问题。

（注：这和「多个**短**请求凑批」不矛盾——短请求各自一次就 prefill 完，自然能在一个 batch 里拼多个；「独占」只发生在长请求的**中间块**。）

---

## 9. 关键代码索引（速查）

| 主题 | 文件:行 |
|------|---------|
| 主事件循环 | [scheduler.py:1426](../python/sglang/srt/managers/scheduler.py#L1426) |
| overlap 循环 | [scheduler.py:1453](../python/sglang/srt/managers/scheduler.py#L1453) |
| 调度决策中心 | [scheduler.py:2405](../python/sglang/srt/managers/scheduler.py#L2405) |
| 组 prefill batch | [scheduler.py:2533](../python/sglang/srt/managers/scheduler.py#L2533) |
| decode + retraction | [scheduler.py:2824](../python/sglang/srt/managers/scheduler.py#L2824) |
| run_batch | [scheduler.py:2966](../python/sglang/srt/managers/scheduler.py#L2966) |
| 调度策略排序 | [schedule_policy.py:166](../python/sglang/srt/managers/schedule_policy.py#L166) |
| 准入预算 PrefillAdder | [schedule_policy.py:421](../python/sglang/srt/managers/schedule_policy.py#L421) |
| add_one_req 准入 | [schedule_policy.py:844](../python/sglang/srt/managers/schedule_policy.py#L844) |
| check_decode_mem | [schedule_batch.py:2275](../python/sglang/srt/managers/schedule_batch.py#L2275) |
| retract_decode | [schedule_batch.py:2288](../python/sglang/srt/managers/schedule_batch.py#L2288) |
| prepare_for_decode | [schedule_batch.py:2429](../python/sglang/srt/managers/schedule_batch.py#L2429) |
| decode 结果处理 | [batch_result_processor.py:588](../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L588) |
| merge_batch（prefill→decode） | [schedule_batch.py:2610](../python/sglang/srt/managers/schedule_batch.py#L2610) |
| FutureMap（overlap） | [overlap_utils.py:114](../python/sglang/srt/managers/overlap_utils.py#L114) |
| resolve_forward_inputs | [overlap_utils.py:81](../python/sglang/srt/managers/overlap_utils.py#L81) |

---

## 10. Prefill→Decode 衔接 & Overlap Scheduler（进阶机制）

### 10.1 `merge_batch`：prefill 完的请求怎么并入 decode 流

[schedule_batch.py:2610](../python/sglang/srt/managers/schedule_batch.py#L2610)。把 prefill batch 每个张量字段 **concat 到 running_batch 后面**：

```python
self.req_pool_indices = cat([self.req_pool_indices, other.req_pool_indices])  # 请求槽位id
self.seq_lens         = cat([self.seq_lens, other.seq_lens])                   # 各请求长度
self.input_ids        = cat([...])
self.reqs.extend(other.reqs)
self.sampling_info.merge_batch(other.sampling_info)  # 采样参数也合并
```

- **为什么 prefill 完不能立刻 decode？** prefill / decode 是两种 `forward_mode`，张量布局不同（prefill 变长拼接、decode 每请求 1 token）。所以请求先在自己的 prefill batch 跑完最后一块，**下一轮循环开头**才被 concat 进常驻 running_batch，从此逐 token decode。
- **`req_pool_indices` 是身份证**：每个请求在 `req_to_token_pool` 占一个槽位 id，KV cache 全靠它索引。merge 就是把新请求 id 加进 decode 大军的索引数组。
- 对应 §3 开头 `last_batch.is_extend() → running_batch.merge_batch(last_batch)`：上一轮 prefill 结果，这一轮开头并入 decode 流。

### 10.2 Overlap Scheduler（零开销调度）+ FutureMap

**问题**：朴素调度每步要等 GPU 算完、把采样 token 拷回 CPU 才能组下一个 decode batch，每步一次 GPU→CPU 同步，GPU 空转。decode 步很短，这开销占比极高。

**解法**：CPU 不等 GPU，提前组好并发出下一个 decode batch。矛盾是——下一步的输入 token 正是这一步 GPU **此刻在算**的采样结果，CPU 还不知道。用 [`FutureMap`](../python/sglang/srt/managers/overlap_utils.py#L114) 解决：一块**常驻 GPU、按 `req_pool_idx` 索引**的 buffer `output_tokens_buf`。

1. 调度时下一个 decode batch 的 `input_ids` 先留**空/占位**，不需要真值。
2. 上一步 forward 算完，用 `publish` 把采样 token **直接写进 GPU 的 `output_tokens_buf`**（不回 CPU）。
3. 下一步进 forward 时，[`resolve_forward_inputs`](../python/sglang/srt/managers/overlap_utils.py#L81) 用 `output_tokens_buf[req_pool_indices]` **在 GPU 上 gather** 出真实 token：

```python
batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
```

**效果**：token 全程待在 GPU，消除每步 GPU↔CPU 同步；CPU 组下一批的时间藏在 GPU forward 后面（对应 `event_loop_overlap` 的 `result_queue`）。这就是 "zero-overhead scheduler"。

> 面试金句：**SGLang 用 FutureMap 把「下一步输入依赖这一步输出」的数据依赖留在 GPU 上解决，避免 per-step device 同步，让 Python 调度开销与 GPU 计算重叠。**

---

## 11. SGLang vs vLLM（可迁移对比）

| 维度 | SGLang | vLLM |
|------|--------|------|
| 前缀缓存 | **RadixAttention**：radix tree 自动跨请求复用，LRU 驱逐，默认开 | Automatic Prefix Caching：hash-block |
| chunked prefill 默认 | 切块但**不与 decode 混**（prefill 优先）；混要 `enable_mixed_chunk` | 默认 chunked prefill **就 piggyback decode**（prefill+decode 同 batch） |
| 调度开销 | overlap scheduler + FutureMap，token 常驻 GPU | 异步 / multi-step，实现不同 |
| KV 显存 | 分页 pool + radix tree（可驱逐算可用空间） | PagedAttention（招牌） |
| 定位 | 复杂 agent / 结构化生成 / 多轮，前端 DSL | 通用高吞吐推理引擎 |

**最值得说的差异**：chunked prefill 默认行为**相反**——vLLM 默认 prefill+decode 同 batch 混跑（降 ITL），SGLang 默认 prefill 优先、不混（降 TTFT、实现更简单），要混需显式 `enable_mixed_chunk`。这正是 §5 / FAQ Q5 纠正的点，拿来对比很加分。

---

## 12. 显存预算：一张卡到底要多大（sizing 核心）

### 12.1 显存四块拆解（SGLang 官方公式，[server_args.py:1453](../python/sglang/srt/server_args.py#L1453)）

```
GPU 总显存 = 模型权重 + KV cache pool + 激活值 + CUDA Graph buffer
mem_fraction_static = (权重 + KV pool) / 总显存          # 静态部分，启动占好
剩下 (1 − mem_fraction_static) = 激活 + CUDA Graph        # 动态部分，运行时用
```

**核心**：权重固定，所以 **KV pool 和「激活+CUDA Graph」在「权重之外的剩余显存」里互相竞争**。`mem_fraction_static` 就是这两者之间的旋钮。

### 12.2 每块怎么算

- **权重** = 参数量 × 字节/参数（fp16=2，fp8=1，int4=0.5）。8B→16GB，70B→140GB。
- **KV per token** = `2 × num_layers × num_kv_heads × head_dim × dtype_bytes`
  - Llama3-8B（32 层，GQA 8 头，128，fp16）= **128 KB/token** → 8K 上下文 ≈ 1GB
  - Llama3-70B（80 层，8 头）= **320 KB/token** → 8K ≈ 2.5GB
  - GQA/MQA、MLA(DeepSeek 低秩压缩) 大幅降 KV
- **激活 + CUDA Graph**（[server_args.py:1584](../python/sglang/srt/server_args.py#L1584)，单位 MB）：
  ```python
  reserved_mem = 512 + max(chunked_prefill_size,2048)×1.5 + cuda_graph_max_bs×2 + ...
  mem_fraction_static = (gpu_mem − reserved_mem) / gpu_mem   # 未知默认 0.88，大卡 reserved 兜底≥10GB
  ```
  **激活 ∝ `chunked_prefill_size`（每步 token 数），不正比于单请求上下文长度**——长 prompt 被切成块，激活每步有界。

### 12.3 ⚠️ 纠正：激活小**不会**限制上下文/输出长度（方向相反）

| 限制谁 | 由哪块决定 |
|--------|-----------|
| 上下文 / 输出长度 / 并发数 | **KV pool 大小** |
| 单步 prefill 吞吐 / 最大 decode batch | 激活 + CUDA Graph（`chunked_prefill_size`/`cuda_graph_max_bs`） |

- 上下文/输出受限于 **KV pool**；输出变长 = 占更多 KV，pool 满了触发 retraction（§6），不是激活不够。
- **激活预留小 → KV pool 反而更大 → 上下文/并发能力更强**，代价是 prefill 切碎、吞吐低。方向和直觉相反。

### 12.4 `mem_fraction_static` trade-off

| | 调高(0.95) | 调低(0.7) |
|---|---|---|
| KV pool | 大→高并发/长上下文/少 retract | 小→低并发/短上下文/频繁 retract |
| 激活余量 | 小→大 batch/长 prefill 易 **OOM** | 大→大 chunk/大 batch，安全高吞吐 |

实践：大卡默认 ~0.85-0.9；OOM 就调低 `mem_fraction_static` 或调小 `chunked_prefill_size`/`cuda_graph_max_bs`。

### 12.5 估算「最少多大的卡」

```
卡显存 ≥ 权重 + reserved(激活+CUDA Graph，大卡≈10-15GB) + 目标 KV
目标 KV = 目标并发 × 平均(上下文+输出)长度 × kv_bytes/token
```

Llama3-8B（16GB 权重，128KB/token）：

| 卡 | 权重 | reserved | 剩 KV | KV token | 场景 |
|----|------|----------|-------|----------|------|
| 24GB(4090) | 16 | ~4 | ~4GB | ~32K | 能跑，低并发 |
| 40GB(A100) | 16 | ~6 | ~18GB | ~140K | 中并发 |
| 80GB(H100) | 16 | ~13 | ~51GB | ~400K | 高并发/长上下文 |

注：单请求最大上下文还受模型 `max_position_embeddings` 和 `req_to_token_pool` 上限约束。

> 面试版：**权重定下界，留 10-15GB 给激活+CUDA Graph，剩下全是 KV pool；KV pool 直接决定「并发 × 上下文长度」。`mem_fraction_static` 是 KV 与激活余量之间的旋钮，激活和上下文长度无关（被 chunked_prefill_size 限成每步有界）。**
