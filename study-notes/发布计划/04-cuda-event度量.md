# 04 · 《一个被漏掉的 CUDA event：如何正确度量 H2D 耗时》

- **源料**：`study/load-back-metric-fix.md`、`study/load_back_explanation.md`、
  `scripts/bench_load_back_event_overhead.py`，以及当前分支的实际代码改动
  （`fix/load-back-metric-accurate-dma-time` 上的 4 个 commit）
- **目标读者**：写 CUDA/异步流代码、需要给性能打点的工程师；HiCache/KV offload 感兴趣的人
- **一句话卖点**：一个 metric 报了 50μs、真实值 5ms——差 100 倍。因为它用 `time.time()` 测了一个**异步** GPU 操作
- **字数**：3–5k　**优先级**：★★（小而精，体现工程细节功力，最容易快速成稿）

## 钩子（前 200 字）

「有个性能指标报告 load-back 只花了 50 微秒。听起来很棒——直到我意识到，真实的 H2D DMA
传输要 5 毫秒。**这个 metric 差了整整 100 倍**，而且一直在骗所有看 dashboard 的人。
根因是一行再普通不过的 `time.perf_counter()`。」

## 结构（分节 → 源料 → 必引数字/事实）

| 节 | 内容 | 源料 | 必引硬料 |
|---|---|---|---|
| 1 | 背景：HiCache 里什么是 load-back（KV 从 CPU 搬回 GPU），为什么要监控它（影响 TTFT / 容量规划 / prefetch 决策） | fix §1 | H2D DMA 示意图 |
| 2 | 原来的错：`start=perf_counter()` … 调 `cache_controller.load()` … `observe(perf_counter()-start)` | fix §2.1 | 那段错误代码 |
| 3 | 为什么错：CUDA 异步执行模型——`load()` 只是把 op 入队，CPU 返回时 DMA **还没开始** | fix §2.2 | CPU/GPU 双时间轴图 |
| 4 | 后果量化：旧值 ~50μs（CPU 调用）vs 实际 ~5ms（GPU 传输），差 100× | fix §2.3 | **50μs vs 5ms** |
| 5 | 正确做法：CUDA Event——`load_stream` 上 record start/finish，`elapsed_time()` 取毫秒 | fix §3.1-3.2 | start/finish event 机制 |
| 6 | 工程细节：`timing_event_supported()` 优雅降级（后端不支持 event 时不崩） | fix §3.2、当前分支 commit | 兼容无 timing event 的后端 |
| 7 | 开销自证：event record 本身有多贵？我 bench 了 | `bench_load_back_event_overhead.py` | 跑出的开销数字 |
| 8 | 通用教训：**任何给异步 GPU 操作打点的地方，`time.time()` 都是错的**，一律换 CUDA event | fix 总结 | 可迁移结论 |

## 配图清单

- CPU/GPU 双时间轴图（fix §2.2 的 ASCII 重画）——本篇灵魂图
- 修复前后 metric 值对比（50μs → 真实毫秒）
- CUDA event record 位置示意（在 stream 上的两个点）

## 诚实时刻

- 这个 bug"看起来无害"——metric 又不影响推理结果，但它误导所有容量/调度决策
- 讲清 event 也有开销，我实测过、确认可接受（不是拍脑袋）

## 复现命令

```bash
python scripts/bench_load_back_event_overhead.py   # 量 CUDA event 打点开销
```

## 提示

- 本篇是**已落地的真实改动**（当前分支），可直接链 commit / PR，可信度最高
- 篇幅短、技术点单一，**最适合作为系列里的"快速成稿"补位**

## 状态：未开工 → 灵活插播（任何时候需要快速产一篇都可以先出它）
