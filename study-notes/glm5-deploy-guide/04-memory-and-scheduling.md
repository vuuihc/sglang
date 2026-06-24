# 04 · `mem-fraction-static` 的说法,以及它和 `max-running-requests` 的区别

> 对应 **Q4**:"权重是可以提前算的,KV pool 大小决定能接多少请求,`max-running-requests` 也能控并发,
> 那 `mem-fraction-static` 到底有什么必要?"

这是个很好的问题,核心在于:**这两个参数控制的是完全不同的东西,一个管"显存怎么分",一个管"调度器放多少进来"。**

## 1. `mem-fraction-static` 的精确定义

直接看 [server_args.py:1961-1970](../../python/sglang/srt/server_args.py) 的官方注释:

```
GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers

mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity
                    = (capacity - activations - cuda graph buffers) / capacity
```

也就是说,一张卡的显存被分成**四块**:

| 块 | 性质 | 谁占的 |
|---|---|---|
| 模型权重 | **静态**,启动时定死 | 权重 |
| **KV cache pool** | **静态预分配**的一大块池子 | KV cache |
| 激活(activations) | **动态**,随 batch/序列长度涨落 | 前向中间结果 |
| CUDA Graph buffers | **动态**,随捕获的 max_bs 涨 | graph 重放缓冲 |

`mem_fraction_static` 就是**前两块(静态)占总显存的比例**。
剩下的 `1 - mem_fraction_static` 是**给动态部分留的安全余量**。

## 2. 为什么"权重能提前算"还需要这个参数?

你说得对,权重大小确实能提前算。但问题是 **KV pool 该开多大?**

KV pool 的大小 = `capacity × mem_fraction_static − 权重`。
而剩下的余量 `capacity × (1 − mem_fraction_static)` 必须够装 **激活 + CUDA Graph buffer**,
否则前向时 OOM。

**这里的难点:激活和 graph buffer 的大小事先不好精确算**,它取决于:
- `chunked-prefill-size`(一次最多算多少 prefill token → 激活越大)
- `cuda-graph-max-bs`(捕获多大的 decode batch → graph buffer 越大)
- 模型结构、是否开 MTP、是否开 TBO 等

所以 SGLang 用一个**启发式**估它(同段注释):

```
reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2     # 单位 GB
mem_fraction_static = (capacity - reserved_mem) / capacity
```

`mem-fraction-static` 就是你**手动覆盖这个估计**的旋钮:
- 调**高**(如 0.9):KV pool 更大、并发更高,但动态余量变小,prefill 太长/batch 太大时可能 OOM。
- 调**低**(如 0.8):更安全,但 KV pool 变小,能缓存的 token 变少。

> 一句话:**它决定的是"静态预分配 vs 给动态留多少余量"的切分点**,本质是一个**显存安全/激进的权衡旋钮**,不是并发开关。

## 3. 那 `max-running-requests` 管什么?

`max-running-requests` 是**调度器层面**的并发上限:同一时刻最多让多少请求处于运行态。
它**不预分配任何显存**,只是控制调度器一次往 batch 里塞多少请求。

## 4. 关键:两者是"硬墙"和"软闸"的关系

把 KV pool 想成一个**停车场**:

- **`mem-fraction-static` 决定停车场有多少车位**(KV pool 能放多少 token 的 KV)。这是**物理硬上限**。
- **`max-running-requests` 决定门口保安一次放几辆车进来**。这是**调度软上限**。

两者都会限制实际并发,但**机制和目的不同**,缺一不可:

### 场景 1:只有 `max-running-requests`,没有 `mem-fraction-static`?
不行。因为请求的**长度差异极大**:
- 1000 个各 100 token 的短请求 vs 8 个各 128k token 的长请求,
  KV 占用差几个数量级。
- `max-running-requests=64` 在短请求下绰绰有余,但 64 个 128k 长请求会**瞬间撑爆 KV pool**。
- KV pool 多大,根本由 `mem-fraction-static` 决定,**`max-running-requests` 管不到显存**。

当 KV pool 满了,SGLang 靠的是**抢占/排队**(基于 token 数),而不是 `max-running-requests`。

### 场景 2:只有 `mem-fraction-static`,没有 `max-running-requests`?
大多数情况够用(调度器会自动按 KV 剩余量收放)。
但 `max-running-requests` 仍有独立价值:
- **CUDA Graph 只捕获到 `cuda-graph-max-bs`**。如果实际并发超过它,decode 会退回 eager 模式变慢。
  所以常把 `max-running-requests` 和 `cuda-graph-max-bs` 对齐,保证大部分 batch 走 graph。
- **MTP / 投机解码**:开 MTP 时默认 `max-running-requests=48`(见 [deepseek_v32.md](../basic_usage/deepseek_v32.md)),
  因为投机解码每个请求的"等效 batch"被放大了(一次验证多个草稿 token),
  并发太高反而吞吐下降,需要显式限。
- 防止长尾:限制并发能让单请求延迟更稳定(避免一次塞太多导致每个都变慢)。

## 5. 数字推导(示意,8×H200 单卡 141GB)

假设单卡:
- 权重(FP8,EP=8 分摊后)≈ 80 GB
- `chunked-prefill-size=16384`,`cuda-graph-max-bs=256`
- 估计余量 `reserved = 16384/1024*1.5 + 256*2/1024 ≈ 24 + 0.5 ≈ 24.5 GB`(粗略示意)

则:
```
mem_fraction_static ≈ (141 - 24.5) / 141 ≈ 0.83
静态部分 = 141 × 0.83 ≈ 117 GB
KV pool  = 117 − 80(权重) ≈ 37 GB
```

37 GB KV pool 能装多少 token?(单 token KV ≈ 35 KB,见 [01-parallelism.md](01-parallelism.md))
```
37 GB / 35 KB ≈ 110 万 token
```

- 如果都是短请求(每个 ~1k token)→ 能同时缓存约 1000 个 → 这时 `max-running-requests` 才是有效约束。
- 如果是 128k 长请求 → 只能缓存约 8 个 → 这时 **KV pool(mem-fraction-static)** 才是有效约束,`max-running-requests=64` 根本碰不到。

**这就是为什么两个参数都要存在**:负载的形状决定了哪一个先成为瓶颈。

## 6. 一句面试话

> "`mem-fraction-static` 是**显存切分**:它定义(权重 + KV pool)占总显存的比例,反过来就是给激活和 CUDA Graph buffer 留多少安全余量,本质是个 OOM 安全 vs 并发激进的旋钮。`max-running-requests` 是**调度并发上限**,不分配显存。长请求场景由 KV pool(mem-fraction-static)卡住,短请求场景由 max-running-requests 卡住,二者机制不同,所以都需要;此外 max-running-requests 还要和 cuda-graph-max-bs、MTP 对齐以保证走 CUDA Graph。"
</content>
