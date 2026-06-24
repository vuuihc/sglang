# SGLang 推理 Profiling 学习笔记

> 一次完整的"租 GPU → 搭环境 → 抓 trace → 读 trace → 找优化点"实战记录。
> 目标:学会 LLM 推理性能 trace 的采集与分析方法,并据此寻找可提交的 perf PR。
> 日期:2026-06-17 · 硬件:AutoDL 单卡 NVIDIA H20(96GB, sm90)· 模型:Qwen3-8B(bf16)

## 这组文档怎么读

按顺序读:

1. [01-环境搭建.md](01-环境搭建.md) — 怎么把环境跑起来,以及踩到的三个坑(GPU 架构墙、torch2.11 inductor bug、可复用镜像)
2. [02-profiling方法论.md](02-profiling方法论.md) — 怎么抓 trace,怎么读三张表,eager vs cuda graph 的取舍
3. [03-实验数据与分析.md](03-实验数据与分析.md) — 实测数据 + GPU 利用率分析(host-bound→compute-bound、cuda graph 开/关对比)
4. [04-结论与下一步.md](04-结论与下一步.md) — 诚实的结论、假阳性教训、可提交 PR 真正藏在哪
5. [05-serving路径分析.md](05-serving路径分析.md) — 抓 scheduler 进程 trace,看 host 侧到底在干什么(本系列的实测高潮)

## 配套原始数据(在仓库 `.contribution-scan/` 下)

| 路径 | 内容 |
|---|---|
| `.contribution-scan/traces/dense_batch{1,8,32,64}_*.trace.json.gz` | cuda graph **开**,batch 扫描,decode 阶段 trace(每个 ~140KB) |
| `.contribution-scan/traces/nograph_batch{1,8}_*.trace.json.gz` | cuda graph **关**(eager),decode trace(每个 ~1.9MB,大 13 倍) |
| `.contribution-scan/logs/run_cudagraph_on.log` | graph 开的 bench 完整输出(含每 batch 时延/吞吐) |
| `.contribution-scan/logs/run_cudagraph_off.log` | graph 关的 bench 完整输出 |
| `.contribution-scan/traces/serving_scheduler_TP0.trace.json.gz` | **真实 serving 的 scheduler 进程 trace**(3.9MB,含 host 侧调度) |
| `.contribution-scan/analysis/triage_*.txt` | 三张表的分析结果(kernel / overlap / fuse) |
| `.contribution-scan/analysis/host_side_serving.txt` | scheduler 调用树 + host 算子 + sync-smell 统计 |
| `scripts/profile_decode_bottlenecks.sh` | decode microbench profiling(参数化 model/batch/spec/graph 开关) |
| `scripts/profile_serving_scheduler.sh` | serving 路径 profiling(起 server + bench_serving + 抓 scheduler trace) |

## trace 怎么打开看

把 `.trace.json.gz` 拖进 **https://ui.perfetto.dev**(无需解压)。
上方泳道是 CPU 线程(Python scheduler / kernel launch),下方是 CUDA stream(真正的 GPU kernel)。

## 一句话结论(剧透)

模型**前向路径**已经被 sglang 优化透了(cuda graph 消掉 host 发射开销 + 关键算子已融合);
H20 上 batch=1 decode 的瓶颈是**硬件算力**(82% 时间在 GEMM),不是软件可优化点。
**便宜的可提交 PR 不在前向路径,而在 `bench_one_batch` 没触发的真实 serving 路径(scheduler/tokenizer)**——下一步往那里挖。
