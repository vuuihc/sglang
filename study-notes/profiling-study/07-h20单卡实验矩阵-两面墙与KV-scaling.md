# 07 · H20 单卡实验矩阵：两面墙、KV Scaling 与三个系统级发现

> 本篇承接 06。06 的 §七 列了一张"未做实验"清单，本篇把其中**单卡可行的全部做完**：
> ① 并发扫描 ② prefill/decode 撞墙验证（硬件计数器）③ 长上下文 KV scaling ④ 关图归因 trace。
> 多卡实验（TP/EP 扫描）和 bf16 对照（84.6GB 放不下 50G 数据盘）仍然做不了，诚实跳过。
>
> 日期：2026-07-07。机器：AutoDL 单卡 H20（96GB）。模型：Qwen3.6-35B-A3B-FP8（同 06）。
> **全部 4 组实验只用了 ~15 分钟 GPU 时间（≈¥2）**——方法见 §一。

---

## 〇、本篇回答了 06 答不上来的三个问题

06 §4.2 诚实承认过 torch profiler 的边界：它是"kernel 时间账本"，不是硬件计数器，回答不了：

1. **prefill 到底撞没撞算力墙、decode 到底撞没撞带宽墙？** → 本篇用 `nvidia-smi dmon` 实测回答（§三）。
2. **混合线性注意力的"亚线性 KV 增长"到底值多少钱？** → 本篇用 4K→120K 的 TTFT/ITL 曲线回答（§四）。
3. **开图 trace 里 kernel 归因不到源码行怎么办？** → 本篇补了关图 trace，每个 kernel 精确到 Python 行（§五）。

外加三个计划外的系统级发现（§六）和一桩悬案的关键线索（§七）。

---

## 一、方法：一个脚本跑完全部实验（可复用模板）

上一轮是"人肉逐条命令"，这一轮把整个实验矩阵写成**一个 battery 脚本**（`study-notes/scripts/` 可参考，实验产物在 `study-notes/contribution-scan/exp2-h20/`）：

```
启动 dmon 连续采集(1Hz, 带时间戳)            ← 全程后台记录硬件计数器
  └─ server A (--context-length 131072)
       ├─ 并发扫描: conc 1/8/32/64, in512/out128
       ├─ 墙探针:  prefill相(in8192/out8) + decode相(in128/out512)
       └─ KV scaling: L = 4K/8K/16K/32K/64K/120K, 单并发
  └─ server B (--disable-cuda-graph)
       └─ eager profile 10 步 → 源码归因 trace
每个阶段前后向 phases.log 写时间戳            ← 事后按时间窗切 dmon 数据
```

三个值得学的方法论点：

1. **dmon 全程连续采集 + phase 时间戳对齐**，而不是每个实验单独起 dmon。一个数据文件、事后任意切窗，还能看到阶段间的空闲基线。
2. **统计时只取"忙时"样本**（sm≥20% 的秒），否则 warmup/数据准备的空闲秒会稀释均值——第一版全窗口均值把 decode 的 mem% 从 64% 稀释到了 40%，差点得出错误结论。
3. **实验从便宜到贵排序**（先 512 token 的扫描、最后才是 120K 长文），前面的实验顺便把 kernel/自动调优都预热了，后面的数据更干净（这一点意外成为 §七 悬案的关键线索）。

启动命令沿用 06 的三个坑修复（`LD_LIBRARY_PATH` 补 cu13、`TORCHDYNAMO_DISABLE=1`、`--disable-piecewise-cuda-graph`），不再重复。

---

## 二、实验 ①：并发扫描——MoE 的批处理经济学

**负载**：in512/out128，并发 1/8/32/64（prompt 数 16/64/256/512），全部请求成功。

| 并发 | 输出吞吐 | 请求吞吐 | TPOT 中位 | ITL 中位 | ITL P99 | GPU 忙时 sm% / mem% |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 153 tok/s | 1.20 req/s | 5.56 ms | 5.55 ms | 5.98 ms | 76% / 19% (~0.77 TB/s) |
| 8 | 691 tok/s | 5.40 req/s | 8.16 ms | 8.14 ms | 9.07 ms | 86% / 35% (~1.4 TB/s) |
| 32 | 1676 tok/s | 13.1 req/s | 15.7 ms | 13.1 ms | 17.7 ms | 89% / 50% (~2.0 TB/s) |
| 64 | 2350 tok/s | 18.4 req/s | 22.1 ms | 17.4 ms | 27.3 ms | 86% / 50% (~2.0 TB/s) |

### 怎么读

**并发 ×64，每 token 时延只 ×3.1**（5.55→17.4ms），吞吐 ×15（153→2350 tok/s）。这就是 continuous batching 的核心价值：把"读一遍权重"的固定成本摊到更多 token 上。

但注意 **32→64 的边际收益骤降**（吞吐只 +40%，TPOT +41%）——收益开始被两个东西吃掉：
- 每步激活的**专家并集**变大（64 个 token × top-8 → 大概率覆盖大部分 256 专家），每步要搬的权重字节数逼近全量 34GB；
- 带宽利用率停在 ~50%（2 TB/s），没继续涨——说明此时不是纯带宽瓶颈，还有 kernel 发射/路由/量化等每步固定开销在分摊（H20 的 4 TB/s 没吃满）。

**和第一轮 dense Qwen3-8B 的对照**：dense 模型 batch 1→8 时延几乎不变（固定开销主导）；MoE 模型并发 1→8 时延就涨了 47%——因为 MoE 的"计算量"随 batch 涨得更快（激活专家变多），**host→compute 转折点比 dense 更靠前**。

---

## 三、实验 ②：两面墙——用硬件计数器坐实（本篇题眼）

torch profiler 说"时间花在哪个 kernel"，dmon 说"硬件哪个子系统在忙"。三档负载对照：

| 负载 | 特征 | 忙时 sm% | 忙时 mem%（≈带宽） | 功耗 |
|---|---|---:|---:|---:|
| **长文 prefill**（120K 单条） | 纯 prefill | **99%** | **7%**（0.28 TB/s） | 304W |
| **prefill 相**（in8192/out8, conc8） | prefill 主导 | **97%** | 23%（0.92 TB/s） | 360W |
| **decode 相**（in128/out512, conc64） | decode 主导 | 90% | **64%（2.6 TB/s）** | 438W |

（H20 参考值：FP8 算力 ~296 TFLOPS[被砍]，HBM3 带宽 ~4 TB/s[满血]。mem% 是 DRAM 活跃时间占比。）

### 怎么读

**prefill 撞算力墙**：SM 97-99% 忙、带宽只用 7-23%——计算单元满负荷，显存闲着。H20 被砍的正是算力，所以 prefill 是它的短板相：实测 prefill 吞吐 ~23-25K tok/s（同价位满血卡可达数倍）。

**decode 撞带宽墙（的方向）**：带宽拉到 2.6 TB/s，是 prefill 相的 2.8 倍；功耗 438W 也最高（数据搬运是耗电大户）。但注意**诚实的细节**：64% ≠ 100%，decode 并没有把 4 TB/s 吃满——剩下的时间花在专家路由、动态量化、kernel 间隙上。"decode 是带宽主导"成立，"decode 只受带宽限制"不成立。

### 最漂亮的一笔账：每步搬多少字节

用 `带宽 × 每步时延` 可以反推 decode 每步实际搬运的数据量：

- **并发 1**：0.77 TB/s × 5.55 ms ≈ **4.3 GB/步** —— 恰好 ≈ 3B 激活参数的 FP8 体积 + KV/激活值。**bs=1 的 decode 每步就是"把激活参数读一遍"**，教科书结论第一次被自己的乘法验证。
- **并发 64**：2.6 TB/s（wall probe 实测）× 17.2 ms ≈ **45 GB/步** —— 已经超过全部 34GB 权重，即 64 token 的专家并集基本覆盖全模型 + KV 读写。这就是 §二 里"32→64 收益骤降"的物理解释。

> 方法论沉淀：**dmon 的 mem% × 标称带宽 × 步时延 = 每步字节数**，这个三项乘法是把"kernel 占比"翻译成"物理瓶颈"的桥梁，成本为零，以后每次 profiling 都该顺手算。

---

## 四、实验 ③：KV Scaling——混合线性注意力的价值曲线（镇文之图）

**负载**：单并发，input 长度逐档翻倍，output 32，每档 2 条（Total input tokens 核验过 = 2×L，语料拼接路径真实有效）。

| 上下文 | TTFT 中位 | 隐含 prefill 吞吐 | ITL 中位 | ITL 相对 4K |
|---:|---:|---:|---:|---:|
| 4K | 159 ms | 25.7K tok/s | 5.56 ms | 1.00× |
| 8K | 228 ms | 36.0K tok/s* | 5.71 ms | 1.03× |
| 16K | 450 ms | 36.4K tok/s* | 5.95 ms | 1.07× |
| 32K | 819 ms | 40.0K tok/s* | 6.25 ms | 1.12× |
| 64K | 2022 ms | 32.4K tok/s* | 6.92 ms | 1.24× |
| **120K** | **5126 ms** | 24.0K tok/s | **8.15 ms** | **1.47×** |

（* 中段吞吐高于两端：小 L 有固定开销，大 L 的 full-attention O(n²) 项和 chunk 调度开始显形。）

### 两个可以当文章标题的结论

**1. 上下文 ×30，decode 时延只 +47%。**
标准全注意力模型 decode 每步要读全部历史 KV，时延随上下文近似线性涨；这个模型 40 层里只有 10 层 full attention（且 GQA 只有 2 个 KV head），其余 30 层是常数状态的 GDN——所以 120K 上下文的 ITL 只从 5.56 涨到 8.15ms。**在 H20 上，一个 35B MoE 模型带着 120K 上下文，decode 依然快过大多数 dense 8B 模型带 8K 上下文。**

**2. TTFT 随上下文基本线性（不是平方）。**
4K→120K 的 TTFT 曲线近似直线（prefill 吞吐稳定在 24-40K tok/s 区间）。注意力的 O(n²) 到 120K 都没能主导总成本，因为它只占 1/4 层数，prefill 的大头是随 token 数线性的 MoE FFN。**"长上下文很慢"的直觉在混合线性架构上需要重新校准。**

配合 06 的"单卡 KV 池 127 万 token"（本轮复核：1,277,404 token，且与 `--context-length` 无关，见 §六.3）：**混合线性 + 大显存 H20 = 单卡超长上下文的甜点组合**，现在有完整的实测曲线支撑。

---

## 五、实验 ④：关图归因——每个 kernel 的户口本

06 的开图 trace 里所有 kernel 都指向 `cudaGraphLaunch`，看不出谁发射的。本轮用 `--disable-cuda-graph` 的 server B 抓了 10 步 eager trace（22MB，`exp2-h20/eager-*.trace.json.gz`），三表输出在 `exp2-h20/eager_triage.txt`。

### decode 阶段归因表（核心行）

| Kernel | 占比 | 源码位置 |
|---|---:|---|
| `fused_moe_kernel` | **43.4%** | `layers/moe/moe_runner/triton_utils/fused_moe.py:218` |
| GDN decode 递归 | 9.8% | `layers/attention/fla/fused_recurrent.py:268` |
| deep_gemm FP8 GEMM ×3 | ~10% | `layers/quantization/fp8_kernel.py:1183` |
| FlashAttention-3 | 3.5% | `jit_kernel/flash_attention_v3.py:19` |
| **KV 写入 index/index_put** | **~6.5%** | **`mem_cache/memory_pool.py:396 copy_from`** |
| lm_head（bf16 nvjet） | 2.2% | `logits_processor.py:877`（lm_head 不量化） |
| FP8 动态量化 ×2 | 2.8% | `fp8_kernel.py:498 per_token_group_quant` |
| shared experts（bf16 mm） | 1.6%+1.1% | `models/qwen2_moe.py:392` |

### 怎么读

1. **开图/关图两份 trace 互相验证**：MoE 占 decode 43.4%（eager）vs 46.7%（06 的开图数据）——排名和量级一致，结论稳。
2. **`memory_pool.py:396 copy_from` 的 KV 写入路径占 decode ~6.5%**，第三次出现（05 的 `write_cache_indices`、06 的 ~4%、本轮 6.5%）。它与在途 PR #24734 同源，是这个仓库当前最值得跟的 host/内存优化线。
3. 两个小的"意料之外"：**lm_head 和 shared experts 仍是 bf16**（FP8 量化不覆盖它们，各占 1-2%）；GDN prefill 的 QKV 投影有专用融合 kernel（`gdn_fused_proj.py`）——说明这条新路径已有人做过一轮 kernel 工程。

---

## 六、三个计划外的系统级发现

### 6.1 DeepGEMM JIT 在启动期预编译（serverA.log 铁证）

启动日志显示：cuda graph capture 期间进入 "DeepGEMM JIT Pre-Compile session"，对每个 GEMM shape "with all Ms" 预编译（16384 个条目的 warmup 进度条）。**这意味着 FP8 模型的首次启动慢（本轮 capture 段 ~3 分钟）不是 bug，而是把 JIT 成本前置**；官方提供 `python -m sglang.compile_deep_gemm` 做持久化预编译，重复部署应该用它。

### 6.2 混合线性模型被迫放弃 overlap scheduler（代码证据链）

serverA.log 的 server_args 里 `disable_overlap_schedule=True`——**我们没传这个参数**。源码 `server_args.py` `_handle_mamba_radix_cache()`：mamba/hybrid 模型若架构不支持 `extra_buffer` 的 radix cache 策略，回落 `no_buffer` 并强制 `disable_overlap_schedule=True`（本模型正是 `mamba_scheduler_strategy='no_buffer'`）。

**含义**：第一轮 dense 模型里"overlap scheduler 把 host 开销藏进 GPU 计算"的优化，混合线性模型**享受不到**——线性层的递归状态使 radix cache 的推测性预取变复杂，工程上还没跟上。这是混合线性架构在 SGLang 里的**隐藏税**，也是一个潜在的上游贡献方向（关注 `extra_buffer` 策略对 qwen3_5 系的支持进度）。

### 6.3 KV 池容量与 `--context-length` 无关

context 8192（06）和 131072（本轮）的 KV 池同为 ~127.7 万 token（bf16, K+V 各 12.18GB）。池大小只由 `mem_fraction_static` 和权重体积决定；`--context-length` 只是单请求上限。**推论：单卡能同时服务「120K 上下文 × ~10 条并发」或「8K × ~150 条」，这是容量规划的直接依据。**

---

## 七、悬案进展：round 2 的 18.6s ITL 与 296 tok/s

06 留下的最大疑团：同模型同卡，round 2 输出吞吐仅 296 tok/s、Max ITL 18.6s。本轮同 conc32 负载测得 **1676 tok/s、Max ITL 1.1s**——差 5.7 倍，尾部差 17 倍。

**关键差异**：round 2 的 bench 是那台 server 冷启动后的**第一波真实负载**；本轮 conc32 之前已经跑过 conc1/conc8 两轮，kernel 编译、triton autotune、显存池都热了。**冷启动伪影（lazy JIT / autotune / 首轮显存整理）是当前最强假设**——JIT 预编译虽在启动期做了大头（§6.1），但 grouped GEMM 变体和 triton `fused_moe_kernel` 的 autotune 仍可能在首个真实 batch shape 上惰性触发。

**验证方法（留给下次开卡，10 分钟）**：冷启动 server → 立刻 conc32 bench（预期复现大 Max ITL）→ 原地重跑同一 bench（预期恢复 ~1676 tok/s）。若成立，结论改写为：**"round 2 测的是冷启动性能"，同时得到一条 benchmark 卫生铁律——永远不要用 server 起来后的第一轮压测数据。**

---

## 八、诚实的局限

1. **两面墙结论是"方向性"的**：decode mem% 64% 未打满，说明 conc64 decode 还有非带宽成分；要精确分解需 nsys + 更细的 kernel 级带宽计数器。
2. **KV scaling 每档只有 2 条请求**，TTFT/ITL 是小样本中位数；曲线形状可信（单调、平滑），单点精度 ±10% 内看待。
3. **dmon 1Hz 采样**对 <1s 的尖峰不敏感；sm% 是"有 kernel 在跑的时间占比"，不是 FLOP 利用率（MFU）。
4. 并发扫描与 06 不完全可比（context-length、range-ratio、预热状态不同）——本轮内部自洽，跨轮对比只看量级。

---

## 九、数据资产清单（全部在 `study-notes/contribution-scan/exp2-h20/`）

| 文件 | 内容 |
|---|---|
| `bench_conc{1,8,32,64}.log` | 并发扫描完整输出 |
| `bench_{prefill,decode}_wall.log` | 墙探针 |
| `bench_kv{4096..122880}.log` | KV scaling 六档 |
| `dmon.log` + `phases.log` + `analyze_dmon.py` | 硬件计数器 + 时间窗对齐脚本 |
| `eager-*.trace.json.gz`（22MB） | 关图归因 trace（拖进 ui.perfetto.dev） |
| `eager_triage.txt` | 三表分析输出 |
| `serverA_final.log` / `serverB.log` | 启动日志（JIT 证据、KV 池、server_args） |
| `env_metadata.txt` | 软件栈版本快照 |

## 十、一句话总结

> 用 15 分钟 GPU 时间的脚本化实验矩阵，把 06 留下的三个"回答不了"全部答掉：**prefill 撞算力墙（SM 99%/带宽 7%）、decode 带宽主导（2.6 TB/s，每步字节数与激活参数量精确对账）、混合线性架构让 120K 上下文的 decode 只比 4K 慢 47%**；顺手挖出三个系统级发现（JIT 前置、overlap scheduler 隐藏税、KV 池与 context 无关），并为 18.6s ITL 悬案锁定了"冷启动伪影"的头号嫌疑。
