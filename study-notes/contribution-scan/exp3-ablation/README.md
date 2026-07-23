# H20 单卡消融实验结果索引

> 采集时间：2026-07-15；GPU：NVIDIA H20 96GB；SGLang：0.5.13.post1。  
> 原始日志以本目录为准。下表不含 `_warmup_*`，吞吐均为 output token throughput。

## 第一批结果

| 实验 | 对照 | 结果 | 直接结论 |
|---|---|---|---|
| E1 Radix Cache | on → off | 996.98 → 393.98 tok/s；中位 TTFT 288.00 → 2540.44 ms | 共享前缀负载下，关缓存吞吐 -60.5%，TTFT 变为 8.82× |
| E2 FP8，Prefill 相 | BF16 → FP8 | 8.34 → 12.18 tok/s；中位 TTFT 3975.44 → 2740.74 ms | 吞吐 +46.0%，TTFT -31.1% |
| E2 FP8，Decode 相 | BF16 → FP8 | 3695.08 → 3919.27 tok/s；Mean ITL 8.04 → 7.67 ms | 吞吐只 +6.1%，符合 Decode 带宽受限预期 |
| E3 AWQ，concurrency=1 | BF16 → AWQ | 167.29 → 220.30 tok/s；Mean ITL 5.94 → 4.49 ms | 吞吐 +31.7%，ITL -24.4% |
| E3 AWQ，concurrency=32 | BF16 → AWQ | 3948.35 → 3186.65 tok/s；Mean ITL 7.98 → 9.92 ms | 吞吐 -19.3%，高并发时解量化开销反噬 |
| E4 EAGLE3 | concurrency 1/4/8/16/32 | 相对基线 1.87×/0.92×/0.59×/0.40×/0.29× | 仅单请求收益明显，交叉点在并发 1～4 之间 |
| E5 FP8 KV | BF16 → FP8 E4M3 | 474,376 → 948,752 token | 同显存 KV 容量恰好 2×；未提供 scale，不能据此推断精度无损 |
| E6 冷启动 | cold → warm | 3624.23 → 3951.82 tok/s；中位 TTFT 412.71 → 66.40 ms | 吞吐 +9.0%，首轮 TTFT 含 lazy 初始化税 |
| E7 全注意力长上下文 | 4K → 30K | Mean ITL 6.07 → 8.28 ms；中位 TTFT 417.83 → 2786.92 ms | ITL +36.4%，TTFT 变为 6.67× |
| E8 Chunked Prefill | off / 8192 / 2048 / 512 | 422.74 / 422.15 / 416.07 / 394.95 tok/s；P99 ITL 442.69 / 452.06 / 479.62 / 513.86 ms | 本负载下关闭切块没有恶化尾延迟；512 小块反而吞吐 -6.6%、P99 ITL +16.1% |
| E9 Overlap Scheduler | off → on | c8：1134.83 → 1255.21；c32：3573.12 → 3949.36 tok/s | 两档吞吐均约 +10.5% |
| E10 Attention backend | torch_native → FA3 | 4K：14.27 → 19.96；8K：8.45 → 10.40 tok/s | FA3 吞吐分别 +39.9%/+23.1%，中位 TTFT 均约 -31% |
| E11 MLA KV 池容量 | Qwen3-8B GQA BF16 vs DeepSeek-V2-Lite MLA BF16 | 474,376 → 1,759,061 token（同 mem-fraction） | 池容量约 **3.71×**（系统级对照，非纯结构压缩比） |
| E11 MLA 后端 A/B | flashinfer / flashmla / triton | 932.22 / 921.13 / 883.17 tok/s（并发 8 Decode） | FlashInfer 相对 Triton 吞吐 **+5.6%**，中位 ITL **-5.2%**；三后端差距为个位数 |
| E11 MLA 上下文曲线 | FlashInfer，输入 4K→30K | 中位 ITL 4.65 → 5.65 ms；Mean ITL 4.61 → 5.60 ms | 上下文约 7.32×，ITL 仅约 **+21.5%** |
| E12 torch.compile | off / on | off：167.12 tok/s；on：启动失败 | Torch 2.11 导入 Inductor 时触发 `AssertionError: duplicate template name`，本软件栈无有效 A/B 数据 |

## E4 完整曲线

| 并发 | 基线 tok/s | EAGLE3 tok/s | 加速比 |
|---:|---:|---:|---:|
| 1 | 168.90 | 316.56 | 1.87× |
| 4 | 631.38 | 583.44 | 0.92× |
| 8 | 1094.40 | 651.03 | 0.59× |
| 16 | 1789.69 | 713.70 | 0.40× |
| 32 | 2633.27 | 758.88 | 0.29× |

## E7 完整曲线

| 输入长度 | 吞吐 tok/s | 中位 TTFT ms | Mean ITL ms |
|---:|---:|---:|---:|
| 4,096 | 62.47 | 417.83 | 6.07 |
| 8,192 | 51.31 | 549.05 | 6.45 |
| 16,384 | 26.42 | 1289.29 | 7.12 |
| 30,000 | 13.47 | 2786.92 | 8.28 |

## E10 完整曲线

| 后端 | 4K tok/s | 4K 中位 TTFT ms | 8K tok/s | 8K 中位 TTFT ms |
|---|---:|---:|---:|---:|
| FA3 | 19.96 | 2277.36 | 10.40 | 3218.40 |
| FlashInfer | 19.27 | 2367.45 | 9.86 | 3559.40 |
| Triton | 17.67 | 2540.24 | 8.75 | 4057.33 |
| torch_native | 14.27 | 3313.93 | 8.45 | 4676.89 |

## E11 DeepSeek-V2-Lite-Chat（MLA 三连测）

对照模型：`deepseek-ai/DeepSeek-V2-Lite-Chat`（15.7B MoE，MLA，BF16）。与 Qwen3-8B 的参数量、层数、每 token KV 布局均不同；下列容量比为**同卡、同 mem-fraction 下可服务 token 池**的系统级对照，不能写成「MLA 相对 GQA 的纯结构压缩倍数」。

### KV 池容量

| 模型 / 结构 | 后端 | KV dtype | KV 池 token 数 | KV size |
|---|---|---|---:|---:|
| Qwen3-8B GQA（E5） | — | BF16 | 474,376 | 见 E5 server 日志 |
| DeepSeek-V2-Lite MLA | FlashInfer / Triton | BF16 | 1,759,061 | 50.96 GB |
| DeepSeek-V2-Lite MLA | FlashMLA | BF16 | 1,759,040 | 50.96 GB |

池容量比：`1,759,061 / 474,376 ≈ 3.71×`。

### MLA 后端 A/B（Decode，并发 8）

| 后端 | 输出吞吐 | 中位 TTFT | 中位 ITL | Mean ITL | P99 ITL |
|---|---:|---:|---:|---:|---:|
| FlashInfer | 932.22 tok/s | 84.29 ms | 8.44 ms | 8.43 ms | 9.04 ms |
| FlashMLA | 921.13 tok/s | 87.24 ms | 8.52 ms | 8.51 ms | 9.03 ms |
| Triton | 883.17 tok/s | 90.00 ms | 8.90 ms | 8.90 ms | 9.55 ms |

排序：FlashInfer ≳ FlashMLA > Triton。FlashInfer 相对 Triton 吞吐 +5.6%、中位 ITL -5.2%；FlashMLA 相对 Triton 吞吐 +4.3%。差距落在个位数，说明同一 MLA 模型下后端仍有差，但本环境未拉开到「结构红利完全被 kernel 浪费」的量级。

### 上下文长度曲线（FlashInfer，单并发，4 prompts，每条输出 32）

| 输入长度 | 输出吞吐 | 中位 TTFT | 中位 ITL | Mean ITL | P99 ITL |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 101.83 tok/s | 207.25 ms | 4.65 ms | 4.61 ms | 5.11 ms |
| 16,384 | 31.51 tok/s | 1107.93 ms | 5.09 ms | 5.05 ms | 5.67 ms |
| 30,000 | 27.64 tok/s | 1247.63 ms | 5.65 ms | 5.60 ms | 6.35 ms |

4K→30K 上下文约 7.32×：中位 ITL +21.5%（4.65→5.65 ms），Mean ITL +21.5%（4.61→5.60 ms）。同指标对照 Qwen3-8B 全注意力 E7 的 Mean ITL 6.07→8.28 ms（+36.4%）；两组模型不同，只能并列看曲线斜率，不能直接归因于「全是 MLA 的功劳」。

## 环境兼容性记录

- E12 若要得到可比数字，需要换用兼容的 Torch/SGLang 组合后单独复测；当前失败日志保留为环境兼容性记录。

