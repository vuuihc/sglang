# 租一张 H20 抓 trace：一次诚实的 SGLang 推理 profiling 实战

我租 H20 的初衷很简单：想在 SGLang 这种成熟推理引擎里，找到一个“改几行代码就能提 PR”的性能优化点。

三天之后，我的结论是：**没有。**

更准确地说，不是在我这次实验覆盖的主流前向路径里没有。SGLang 的 decode 热路径已经被 cuda graph、算子融合、sync 消除等优化打磨得很充分。batch=1 的 decode，我原本以为会看到 GPU 大量空等 CPU；结果 trace 告诉我，cuda graph 打开后 GPU 利用率接近 **92%**，其中 **82%** 的 GPU 时间都花在 GEMM 上。

这不是一个令人沮丧的结果。相反，它是这次实验最有价值的地方：我没有靠直觉猜瓶颈，也没有靠静态扫描硬找 PR，而是用真实 trace 把一个假设证伪了。

这篇文章讲完整过程：怎么租卡、怎么抓 trace、怎么读 Perfetto、怎么量化 cuda graph 省下的开销，以及为什么“没找到 quick-win”本身也是一个高质量工程结论。

> 配图建议：开头放一张 Perfetto 对比图，左边 eager 模式 GPU 泳道全是气泡，右边 cuda graph 模式 kernel 排得很密。

## 1. 我到底想验证什么

目标不是“跑一个 benchmark 看吞吐”，而是回答三个问题：

1. SGLang 的小 batch decode 到底是不是 host-bound？
2. cuda graph 具体省了多少 host launch overhead？
3. 如果我要找一个可以提交的 perf PR，应该往模型前向、scheduler，还是 tokenizer 方向挖？

这三个问题不能只看最终 tok/s。吞吐只告诉你“快不快”，不告诉你“为什么快”。

所以我选择抓 trace。trace 的好处是它能把 CPU 和 GPU 的时间线摊开给你看：

- CPU 在什么时候发起 kernel；
- GPU 在什么时候真正执行 kernel；
- 两个 kernel 之间有没有空隙；
- 空隙到底是 GPU 在等 CPU，还是 GPU 已经被计算打满。

对推理引擎来说，trace 比单个 benchmark 数字更接近真相。

## 2. 为什么选 H20 + Qwen3-8B

这次实验环境是：

| 项 | 值 |
|---|---|
| GPU | NVIDIA H20 96GB |
| 架构 | Hopper / sm90 |
| 模型 | Qwen3-8B |
| 精度 | bf16 |
| 输入/输出 | input=512, output=64 |
| profiling 阶段 | decode |
| trace 窗口 | 稳态 5 步 |

H20 是一张很适合做这类实验的卡。

它有点“畸形”：显存带宽很高，大约 4TB/s；但 fp16/bf16 算力被砍得很厉害，约 148 TFLOPS。decode 阶段每次只生成一个 token，很多时候瓶颈是带宽和调度，而不是大 batch prefill 那种纯算力压榨。

同时，H20 是 sm90，能直接跑 SGLang 新版本依赖的 `sglang-kernel`。我一开始也尝试过消费级卡和旧架构卡，但新版 kernel 包对架构要求很硬，最后还是回到 H20。

Qwen3-8B 的好处是足够小，单卡能轻松跑 batch sweep；同时又不是玩具模型，decode 路径、attention、GEMM、cuda graph 都是真实路径。

## 3. 两种 profiling 入口：microbench 和 serving

这次主要用了两个入口。

第一个是 `bench_one_batch`，它是纯模型前向 microbench：

```bash
export SGLANG_TORCH_PROFILER_DIR=/path/to/profile_log

python -m sglang.bench_one_batch \
  --model-path Qwen/Qwen3-8B \
  --batch-size 1 8 32 64 \
  --input-len 512 \
  --output-len 64 \
  --mem-fraction-static 0.85 \
  --profile \
  --profile-stage decode \
  --profile-steps 5 \
  --disable-piecewise-cuda-graph
```

我把它封装在了 `study-notes/scripts/profile_decode_bottlenecks.sh` 里。它适合回答一个干净问题：**模型 forward 本身的瓶颈在哪里？**

但它也有明显缺点：它不经过真实 server，不经过 scheduler、tokenizer、detokenizer，也没有 HTTP 请求、排队、KV cache admission 这些逻辑。

所以第二个入口是真实 serving profile：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --disable-piecewise-cuda-graph

python -m sglang.bench_serving \
  --backend sglang \
  --model Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 300 \
  --max-concurrency 32 \
  --warmup-requests 16 \
  --profile \
  --profile-steps 20
```

这个入口适合回答另一个问题：**真实 serving 路径里，scheduler/tokenizer 是否还有 host-side 优化空间？**

这两个入口不能互相替代。microbench 干净，但看不到 server 侧；serving 真实，但噪声更多。一个成熟结论最好能被两边互相校验。

## 4. 读 trace 前先建立三个概念

### Prefill 和 decode

LLM 推理通常分两段。

Prefill 是把用户输入的一整段 prompt 一次性喂进模型。这一段 batch 和序列长度都大，GPU 通常被计算打满，属于 compute-bound。

Decode 是逐 token 生成。每一步只生成一个 token，但要读权重、读 KV cache、跑 attention 和 FFN。小 batch decode 很容易暴露固定开销，比如 CPU 发射 kernel 的延迟、Python scheduler 的调度、D2H sync 等。

很多推理引擎里的“小而美”性能 PR，都藏在 decode 阶段的气泡里。

### Host-bound 和 compute-bound

如果 GPU 时间线上两个 kernel 之间有明显空隙，说明 GPU 在等。它可能在等 CPU 发射下一个 kernel，也可能在等某个同步点。这类问题通常叫 host-bound。

如果 GPU 时间线排得很满，绝大多数时间都在执行 GEMM、attention、norm 这类 kernel，那就是 compute-bound 或 memory-bound。这个时候再去优化 Python 调度，收益就很小。

### Cuda graph 到底省了什么

不用 cuda graph 时，一次 forward 可能要由 CPU 发射几百个 GPU kernel：

```text
CPU: launch k1 -> launch k2 -> launch k3 -> ...
GPU:      [k1]      [k2]      [k3]
```

每次 launch 都有微秒级开销。单次看不大，几百次叠起来就是几毫秒。

cuda graph 的做法是先把整段 forward 录成一张图，之后每步只需要一次 `cudaGraphLaunch`：

```text
CPU: cudaGraphLaunch
GPU: [k1][k2][k3][k4][k5]...
```

所以 cuda graph 不是让 GEMM 本身更快，而是把“CPU 一个个发 kernel”这部分固定开销压掉。

这件事听起来很抽象，直到你把 graph 开关前后的 trace 放在一起看。

## 5. 实验一：batch 扫描，看 host 到 compute 的转折

先看 cuda graph 打开的情况下，batch 从 1 扫到 64 的 decode 稳态数据：

| batch | 每步时延 | 吞吐 | 每 token 时延 |
|---|---:|---:|---:|
| 1 | 6.08 ms | 164 tok/s | 6.08 ms |
| 8 | 6.65 ms | 1204 tok/s | 0.83 ms |
| 32 | 8.91 ms | 3590 tok/s | 0.28 ms |
| 64 | 13.39 ms | 4780 tok/s | 0.21 ms |

最关键的不是 batch=64 有多少吞吐，而是 batch=1 到 batch=8：

**工作量增加 8 倍，但每步时延只从 6.08ms 增加到 6.65ms，只涨了 9%。**

这说明 batch=1 时，整步时间里有相当一部分是固定开销。batch 增大后，这些固定开销被摊薄，所以吞吐接近线性上涨。

到了 batch=32、64，时延开始明显上涨，说明计算量终于开始主导。这就是小 batch 到大 batch 的转折。

如果只看这张表，我会自然猜测：batch=1 应该还有很多 host bubble，可以找优化点。

但下一步的 trace 分析推翻了这个直觉。

## 6. 实验二：batch=1 真的 host-bound 吗？

我把 batch=1、cuda graph 打开的 trace 丢进分析脚本，看 GPU kernel 时间分布。

结论是：

```text
GPU 真忙时间：约 28ms / 5 步 = 5.6ms/步
墙上时间：6.08ms/步
GPU 利用率：约 92%
```

也就是说，在 cuda graph 打开的真实配置下，batch=1 并没有大量空等。GPU 大部分时间都在干活。

再看 GPU 时间花在哪里：

| 类别 | 占比 |
|---|---:|
| `nvjet_*` GEMM | 约 82% |
| FlashAttention | 约 5.5% |
| fused Add + RMSNorm | 约 3.4% |
| cublasLt reduce / 其他 GEMM 辅助 | 约 2.9% |
| act / QK norm / RoPE 等 | 各 1% 左右 |

这就很尴尬了。

我原本以为小 batch decode 会是 host-bound，结果 trace 说：**batch=1 的主要时间在 GEMM 上。**

这不是 Python scheduler 的锅，也不是某个 `.item()` 同步点的锅。它更像 H20 上 bf16 GEMM 的硬件墙。

为什么 H20 上会这样？因为 H20 算力弱。GEMM 一旦占到 80% 以上，说明前向路径已经没有明显的“CPU 在拖 GPU 后腿”的空间了。你当然可以继续优化 GEMM，但那已经不是“改几行 Python”的 perf PR 了，而是 kernel/backend 级别的事情。

这里还有一个很好的假阳性教训。

分析工具把一个 `nvjet` 相关优化标成了高置信：PR #22392，用 CUTLASS FP8 scaled MM 替换 `nvjet`。但我回看 kernel dtype 之后发现，这次跑的是 bf16，不是 FP8。也就是说，这个匹配是按 kernel 家族名命中的，不代表当前实验真的能受益。

**工具说 confirmed，不等于结论 confirmed。必须回看 dtype、调用链和具体 PR 适用条件。**

## 7. 实验三：关掉 cuda graph，把气泡逼出来

为了确认 cuda graph 到底省了多少，我又跑了一组 eager 模式，也就是关掉 cuda graph：

```bash
EXTRA_ARGS="--disable-cuda-graph --disable-piecewise-cuda-graph" \
  bash study-notes/scripts/profile_decode_bottlenecks.sh
```

对比结果很直观：

| batch | cuda graph 开 | cuda graph 关 | 差值 |
|---|---:|---:|---:|
| 1 | 6.08 ms | 14.27 ms | 约 8.2 ms |
| 8 | 6.65 ms | 17.34 ms | 约 10.7 ms |

关图后，batch=1 的 GPU 真忙时间仍然约 5.4ms/步，但墙上时间变成了 14.27ms/步。

换句话说：

```text
eager 模式：
  5.4ms GPU 计算 + 8.6ms host 发射空隙 = 14.27ms/步

cuda graph 模式：
  forward 被录成一张图回放，host 发射空隙基本消失 = 6.08ms/步
```

这就是 cuda graph 的价值。它没有让矩阵乘变快，而是把几百个 kernel launch 之间的空隙压掉了。

这组数据也解释了为什么很多人说“小 batch decode 是 host-bound”，但我在 graph 打开后却看到 92% GPU 利用率。

两句话都可以是真的：

- **不用 cuda graph，小 batch decode 确实有大量 host launch bubble。**
- **用了 cuda graph 后，这部分 bubble 已经被 SGLang 消掉了。**

所以如果你的目标是给 SGLang 现在的主路径找 PR，就不能停留在“理论上小 batch 有气泡”这一层。你必须问：这条路径在当前配置下，气泡还在不在？

我的答案是：在这次实验里，基本不在了。

## 8. 实验四：为什么还要抓 eager trace

cuda graph 有一个副作用：trace 里的 Python location 会变差。

graph 打开时，很多 GPU kernel 的来源都被折叠到 `cudaGraphLaunch` 那一行。你能看到 GPU 在忙，但很难知道某个 kernel 具体是哪行 Python 触发的。

关掉 graph 后，trace 会变大很多，但源码归因更清楚：

| 项 | cuda graph 开 | cuda graph 关 |
|---|---|---|
| trace 大小 | 小，约 140KB | 大，约 1.9MB |
| 性能真实性 | 接近生产配置 | 不代表真实性能 |
| 源码归因 | 差，很多都指向 `cudaGraphLaunch` | 好，能看到具体 Python op |
| 适合用途 | 看真实性能和利用率 | 做 kernel 到源码的 mapping |

例如关图后可以看到：

- GEMM 主要来自 `layers/quantization/unquant.py` 里的 `aten::mm`；
- attention 来自 `jit_kernel/flash_attention_v3.py`；
- 一些 `Memset (Device)` 对应 cublasLt workspace 清零。

所以我的经验是：**正式性能判断看 graph trace，源码归因看 eager trace。**

只抓一种 trace，很容易误判。

## 9. 真实 serving 路径里还有机会吗？

到这里，microbench 已经说明：模型前向主路径没有明显 quick-win。

但 `bench_one_batch` 不经过 scheduler。真实服务里还有很多图外逻辑，比如：

- 请求排队；
- batch 组装；
- KV cache 分配；
- tokenizer / detokenizer；
- penalizer / sampling 信息准备；
- prefill 与 decode 的 overlap 调度。

所以我又抓了一次真实 serving 的 scheduler trace。

负载大致是：

| 项 | 值 |
|---|---|
| 模型 | Qwen3-8B |
| 数据 | random input=512 / output=128 |
| 请求数 | 300 |
| 并发 | 32 |
| profile 窗口 | 20 个 scheduler step |

`bench_serving` 结果里有一个很值得注意的尾部现象：

| 指标 | 值 |
|---|---:|
| 输出吞吐 | 1067 tok/s |
| TTFT 中位数 | 80 ms |
| ITL 中位数 | 8.05 ms |
| ITL 均值 | 26.68 ms |
| ITL P99 | 137 ms |
| ITL Max | 3356 ms |

ITL 均值远大于中位数，而且 Max 到了 3.3s，说明存在尾部停顿。它可能是 GC pause，也可能是周期性调度卡顿。不过这次 profiler 只抓了 20 步稳态窗口，没有覆盖到那次长尾，所以我没有在这篇里强行下结论。

再看 scheduler trace：

```text
GPU busy 1081ms / wall 1251ms = 86% 利用率
```

真实 serving 下 GPU 利用率仍然不低。这说明 SGLang 的 overlap scheduler 确实在发挥作用：CPU 侧调度和 GPU forward 被重叠起来，很多 host 开销被藏住了。

host 侧调用树里，最显眼的是这条：

```text
event_loop_overlap
└─ get_next_batch_to_run
   └─ get_new_batch_prefill
      └─ prepare_for_extend
         └─ alloc_for_extend
            └─ write_cache_indices
```

对应的 top host op 是：

```text
aten::copy_      约 834ms
aten::to         约 833ms
aten::_to_copy   约 832ms
```

这说明真正值得深挖的方向不是 `.item()` 这类简单 D2H sync，而是 prefill 阶段 KV cache index 写入、host 到 device 的索引拷贝、是否 pinned memory、是否能批量化。

但我继续查 open PR 后发现，相关路径已经有人覆盖过：PR #24734 做的就是 CPU-side vectorization for prefill prep + sampling info，覆盖的正是这类 `write_cache_indices` / `alloc_for_extend` 路径。正式发布前需要再确认这个 PR 的最新状态，但对这次实验来说，关键点已经成立：profiling 挖到的热点不是凭空猜的，它确实落在维护者也在关注的路径上。

这又一次印证了这次实验的结论：**profiling 定位是对的，但这个仓库确实很成熟。连我从 trace 里挖到的 host 热点，也已经有 maintainer 在优化。**

## 10. 这次实验真正教会我的东西

第一，别用经验替代 trace。

“小 batch decode 是 host-bound”这句话在很多场景下成立，但不能直接套到“当前 SGLang + cuda graph + H20 + Qwen3-8B bf16”这个具体配置上。graph 打开之后，host launch bubble 已经被压掉，剩下主要是 GEMM。

第二，别只看吞吐。

吞吐只能告诉你系统表现，不能告诉你瓶颈归因。batch=1 到 batch=8 吞吐接近线性上涨，看起来像固定开销很大；但 kernel 表告诉我，graph 打开后 GPU 利用率已经很高。没有 trace，这两个现象很容易被混在一起。

第三，工具输出必须回到代码和 dtype 验证。

这次 `nvjet` 匹配 FP8 CUTLASS PR 的例子就是典型假阳性。工具能帮你缩小范围，但不能替你判断适用条件。

第四，microbench 和 serving 要分开看。

`bench_one_batch` 很适合分析前向路径，但它看不到 scheduler。真实 serving profile 才能暴露 KV cache index、batch prep、sampling info 这些 host 侧问题。

第五，找不到 quick-win 不是失败。

如果三条路径都收敛到同一个结论：

- 静态扫描说主流热路径已经优化充分；
- microbench trace 说前向路径 GPU 利用率 92%，82% 时间在 GEMM；
- serving trace 说 scheduler 路径 GPU 利用率仍有 86%，主要热点已被在途 PR 覆盖；

那“没有躺在地上的一行 PR”就是一个有证据支撑的结论。

## 11. 如果你也想复现

核心脚本：

```bash
# decode microbench：扫 batch + 抓 trace
bash study-notes/scripts/profile_decode_bottlenecks.sh

# 关 cuda graph：抓 eager mapping trace
EXTRA_ARGS="--disable-cuda-graph --disable-piecewise-cuda-graph" \
  bash study-notes/scripts/profile_decode_bottlenecks.sh

# 真实 serving scheduler trace
bash study-notes/scripts/profile_serving_scheduler.sh
```

trace 打开方式：

```text
把 .trace.json.gz 直接拖进 https://ui.perfetto.dev
```

我建议按这个顺序看：

1. 先看 batch=1、cuda graph 开的 trace，确认 GPU 泳道是否紧密；
2. 再看 batch=1、cuda graph 关的 trace，观察 kernel 之间的气泡；
3. 对比两者，量化 graph 省下的 host launch overhead；
4. 最后看 serving scheduler trace，确认 microbench 没覆盖的 host 侧路径。

## 12. 下一步该往哪里挖

如果目标是继续找 SGLang 的 perf PR，我不会再在这条 bf16 dense forward 主路径上硬挖。

更合理的方向有三个：

1. **FP8 模型**：让 CUTLASS FP8 / DeepGEMM / FP8 scaled MM 这类优化真正进入实验范围。
2. **MoE 或 spec decode**：这些路径比 dense bf16 forward 更复杂，覆盖面也可能没那么充分。
3. **真实 serving 的尾部延迟**：这次看到 ITL Max 3.3s，但没有抓到现场。下一步应该拉长 profiler 窗口，加 GC 日志和周期性 scheduler trace，专门复现尾部停顿。

如果只是想学习推理引擎 profiling，我建议先复现这套最小闭环：

```text
bench_one_batch 抓 decode trace
→ Perfetto 看 GPU 气泡
→ 三表看 kernel 占比
→ 关 cuda graph 做源码归因
→ serving profile 看 scheduler
→ open PR / blame 去重
```

跑完这套，你对“推理引擎性能优化”这件事的理解，会比只看论文或架构图扎实很多。

## 结尾

我最开始租 H20，是想找一个便宜 PR。

最后我没有找到。

但我得到了一个更硬的结论：SGLang 的主流 decode 前向路径已经非常成熟，cuda graph 消掉了约 8.6ms/步的 host launch overhead，batch=1 下 GPU 利用率能到 92%，残余时间主要卡在 GEMM。真实 serving 路径里，scheduler 也通过 overlap 把很多 host 开销藏住了，剩下的 `write_cache_indices` / `alloc_for_extend` 线索也已经有在途 PR 覆盖。

这就是一次诚实 profiling 的价值：它不保证你一定找到能改的代码，但它能告诉你哪里不值得浪费时间，哪里才是真正值得继续挖的地方。
