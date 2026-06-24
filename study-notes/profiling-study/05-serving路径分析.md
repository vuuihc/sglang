# 05 · Serving 路径分析(scheduler 进程)

承接 04 文档的结论:可提交的 host 侧 PR 不在前向 microbench 里,而在真实 serving 路径。这一篇就去抓 scheduler 进程的 trace。

工具:`scripts/profile_serving_scheduler.sh` —— 起 server → `bench_serving --profile`(驱动 `/start_profile`+`/stop_profile`)→ 抓 scheduler(TP-0)进程 trace。
负载:Qwen3-8B,random 数据,input=512/output=128,300 请求,并发 32。
原始数据:`.contribution-scan/traces/serving_scheduler_TP0.trace.json.gz`、`analysis/triage_serving_scheduler.txt`、`analysis/host_side_serving.txt`。

---

## bench_serving 结果

| 指标 | 值 |
|---|---|
| 吞吐 | 16.3 req/s,输出 1067 tok/s,总 5233 tok/s |
| TTFT | 中位 80ms,均值 193ms,P99 1105ms |
| ITL | 中位 **8.05ms**,均值 **26.68ms**,P99 137ms,**Max 3356ms** |

> ⚠️ 注意 ITL 均值(26.68)远大于中位(8.05)、且 Max 3356ms —— 说明有**尾部停顿**(疑似 GC pause 或周期性调度卡顿)。本次 profiler 只抓了 20 步稳态窗口,大概率没覆盖到那次 3.3s 停顿,留作后续专项调查。

---

## scheduler 进程 GPU 利用率

20 个 scheduler 步的窗口里:**GPU busy 1081ms / 墙上 1251ms = 86% 利用率**。

即使在真实 serving 下,GPU 利用率依然很高 —— 因为 sglang 默认开了 **overlap scheduler**(`disable_overlap_schedule=False`):它把 CPU 侧调度工作和 GPU 计算重叠起来,藏住了 host 开销。这和 cuda graph 一样,是又一个**已经做好的优化**。

---

## host 侧到底在干什么(本篇核心)

torch profiler 带 CPU activities,所以能看到 scheduler 的 Python 调用树和 host 算子。

### scheduler 调用树(inclusive 时间)
```
event_loop_overlap                                    1265ms
└─ get_next_batch_to_run                  x74          948ms
   └─ get_new_batch_prefill               x74          941ms
      └─ _get_new_batch_prefill_raw       x74          941ms
         └─ prepare_for_extend            x4           936ms   (schedule_batch.py:1896)
            └─ alloc_for_extend           x4           933ms   (mem_cache/common.py:343)
               └─ write_cache_indices     x4           932ms   (mem_cache/common.py:64)
run_batch                                 x20          179ms
```

### top host 算子
```
aten::copy_   834ms  x388     ← 主要 host 开销
aten::to      833ms  x232     ┐ 同一批拷贝/dtype 转换
aten::_to_copy 832ms x142     ┘
aten::linear/matmul/mm  ~30ms each
```

### 关键:#28397 那类 `.item()` D2H sync —— 几乎不存在
```
aten::item              0.11ms  x49   ← 可忽略
aten::_local_scalar_dense 0.04ms x49  ← 可忽略
```
**这条非常重要**:per-step `.item()` 强制 D2H sync 的问题在这条 serving 热路径上**基本不存在**(总共 0.11ms)。说明 #28397 那一类已经被清理干净了 —— 又一次印证"这仓库热路径很成熟"。

---

## 这一篇的结论

1. **serving 路径的 GPU 利用率也很高(86%)** —— overlap scheduler 已经藏住了大部分 host 开销。
2. **`.item()` D2H sync 类的便宜 PR 在这里也挖不到**(已被清理)。
3. **唯一值得深挖的线索:`aten::copy_`/`aten::to`(~830ms)集中在 prefill 的 KV cache 索引路径** ——
   `prepare_for_extend → alloc_for_extend → write_cache_indices`([mem_cache/common.py:64](../../python/sglang/srt/mem_cache/common.py))。
   388 次 copy、平均 2.1ms/次(偏大,说明是**阻塞型拷贝**),其中是否有可合并/可缓存/可异步化的 host↔device 索引拷贝,需要逐调用点读 `write_cache_indices` 才能定论。这是一个**需要 `/investigate` 级别深挖**的方向,不是一行能改完的 quick win。
4. **尾部停顿(Max ITL 3.3s)**:疑似 GC pause,值得用更长的 profiler 窗口 + GC 日志专项复现。

---

## 诚实的总评(贯穿 03/04/05)

三条独立路径 —— 静态代码扫描、前向 microbench、serving scheduler profile —— **全部收敛到同一结论**:

> SGLang 的热路径(前向 + 调度)已经被 cuda graph、overlap scheduler、算子融合、sync 消除等优化覆盖得很充分。**没有躺在地上的 1 行 quick-win perf PR。** 真要做有价值的优化,得是 `write_cache_indices` 拷贝合并、尾部 GC 停顿这类**需要深挖和实测验证的中等工程量改动**,或者换 FP8/MoE/spec-decode 等覆盖更少的场景重新 profile。

这本身就是一个高质量的结论:你用 GPU 实测**证否**了"随便扫扫就能找到 perf PR"的乐观假设,并定位了少数几个真正值得投入的方向。

---

## 附:对 `write_cache_indices`/`alloc_for_extend` 线索的深挖(纯读代码)

顺着第 3 条线索读了 [mem_cache/common.py](../../python/sglang/srt/mem_cache/common.py) 的 `alloc_for_extend`(388 行)和 `write_cache_indices`(108 行):

**找到一个真实的反模式**:`alloc_for_extend` 在 405-415 行用 `torch.tensor(python_list)` 建小 host 张量,再 `.to(device, non_blocking=True)`。但源张量是 **pageable 内存**(`torch.tensor(list)` 不是 pinned),所以 **`non_blocking=True` 被静默忽略、拷贝实际是阻塞的**(PyTorch 经典坑)。而同文件 `write_cache_indices` 122-126 行对 `prefix_pointers` 却**正确地 pin 了** —— 前后不一致,后者甚至留了 TODO「some tensors can be reused」。

**然后去重(find-contribution-targets 的 Phase 5,最省时的一步),结论是 —— 已经在途**:

> **PR #24734「[srt] CPU-side vectorization for prefill prep + sampling info」(OPEN,2026-06-04 仍在更新)** 正在重写**完全相同**的路径:
> - `write_cache_indices`:`prefix_pointers` 改 pinned + async;
> - `alloc_for_extend` paged path:用 `cumsum`+索引+`torch.where` 向量化,干掉 per-req 切片 + `torch.cat`;
> - 顺带消掉 per-step **sampling info** 的 CPU→GPU sync(N 个 `torch.tensor(...,device=)` → 一次 pinned host 批量上传);
> - `memory_pool.HybridReqToTokenPool.alloc`:`select_index` 一次性 pinned + async。
> 核心思路就是把 per-req 的 Python 循环(`.item()` 读、逐个 device 张量分配、list-comp + `torch.cat`)换成「一次批量构建 + 一次 async pinned 上传」。

**这恰恰是教科书级的结论**:
1. **GPU profiling 是对的** —— 我们独立地、从实测数据定位到的唯一 host 热点,正是一位 maintainer 正在优化的地方。方法被验证了。
2. **去重救命** —— 如果不查 open PR 就动手改这个文件,必然和 #24734 冲突、白做。
3. 再次印证「仓库很成熟」:连这种中等工程量的 host 优化都已经有人在做。

**可做的事**:
- 想参与:去 review / 帮忙推动 #24734 落地(它最近还在更新,不算 stalled,所以是协作而非接管)。
- 想找新目标:换 **FP8 / MoE / spec-decode** 等覆盖更少的场景重新 profile(#24734 没覆盖到的路径),或专项复现 **尾部 GC 停顿**(Max ITL 3.3s)。
