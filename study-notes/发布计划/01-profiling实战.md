# 01 · 《租一张 H20 抓 trace：一次诚实的推理 profiling 实战》

- **源料**：`profiling-study/` 全套（00 导读 + 01~05 + `contribution-scan/traces|logs|analysis`）
- **目标读者**：想学会读 GPU trace、判断推理瓶颈在哪的推理/性能工程师
- **一句话卖点**：从"以为能找到 quick-win PR"到"诚实承认前向路径已被优化透"，附全程可复现数据
- **字数**：6–8k　**优先级**：★★★

## 钩子（前 200 字，二选一）

- A（反直觉）：「所有教程都说小 batch 推理是 host-bound。我抓完 trace 发现：batch=1 的 decode，82% 的 GPU 时间在算矩阵乘——它其实是 compute-bound。为什么？」
- B（诚实）：「我租 H20 的初衷是找一个'改几行就能提的 perf PR'。三天后我的结论是：**没有**。但这个'没有'本身，才是这篇文章值钱的地方。」

## 结构（分节 → 源料 → 必引数字）

| 节 | 内容 | 源料 | 必引硬料 |
|---|---|---|---|
| 1 | 我们在干什么 + 为什么选 H20/Qwen3-8B（H20 是"畸形卡"：算力砍到 ~148T、带宽满血 4TB/s） | 00 §2、01 | H20 vs H800/A100 对比表 |
| 2 | 两种 profiling 入口：microbench vs 真实 serving | 00 §3 | — |
| 3 | 读 trace 前的 5 个概念：host-bound/compute-bound/气泡/cuda graph/三张表 | 00 §4 | — |
| 4 | 实验一 · batch 扫描：host→compute 的转折点 | 03 实验一 | 6.08→6.65→8.91→13.39ms；batch1→8 工作量×8 时延仅+9% |
| 5 | 实验二 · batch=1 的真实 GPU 利用率（推翻直觉） | 03 实验二 | 利用率 ~92%、气泡仅 8%、82% 是 nvjet GEMM |
| 6 | 实验三 · 关掉 cuda graph 把气泡逼出来 | 03 实验三 | 关图利用率跌到 38%、cudagraph 省 ~8.6ms/步 |
| 7 | 实验四 · eager trace 的源码归因（每个 kernel 谁发的） | 03 实验四 | GEMM→`unquant.py:153`、attn→`flash_attention_v3.py:19`、memset 泡 |
| 8 | 诚实的结论 + 一个假阳性教训（#22392 FP8 匹配在 bf16 上不成立） | 04 | 三条独立路径收敛到同一结论 |
| 9 | 那便宜的 PR 到底在哪 → 引出下一篇（serving/scheduler 路径） | 04、05 | 引流到主线 03/02 |

## 配图清单

- H20 对比表（截 01）
- batch 扫描时延曲线（自己用 03 数据画一张折线图）
- Perfetto 泳道：graph 开 vs 关的对比截图（体现气泡）
- 三张表（kernel/overlap/fuse）截图

## 诚实时刻（本篇信任锚点）

- "小 batch 一定 host-bound" 被自己的数据推翻
- 工具报的 #22392 匹配是假阳性（dtype 不符）
- 最终没找到前向路径的 quick-win —— 但交叉验证让"仓库已成熟"成为硬结论

## 复现命令（贴给读者）

```bash
bash scripts/profile_decode_bottlenecks.sh   # 抓 baseline + 关图 mapping
# 三张表：analyze_llm_torch_profile.py --input <trace>
```

## 状态：未开工 → 建议作为"方法论硬核"篇，排在故事篇 02 之后
