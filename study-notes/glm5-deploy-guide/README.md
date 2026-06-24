# SGLang 部署 GLM-5 满血版 —— 深度指南

> 目标读者:要在面试里把"用 SGLang 部署 GLM-5 满血版"这件事讲清楚的人。
> 每一篇都尽量做到:**先给结论 → 再讲原理 → 最后用具体数字手把手推导**。

## 0. 一句话题眼

在 SGLang 里,**GLM-5 不是一个独立模型,而是复用 DeepSeek-V3.2 的架构与代码路径**:

- 入口类 [`GlmMoeDsaForCausalLM`](../../python/sglang/srt/models/glm4_moe.py) 直接继承 `DeepseekV2ForCausalLM`。
- 官方文档 [deepseek_v32.md](../basic_usage/deepseek_v32.md) 标题就是 "DeepSeek V3.2/**GLM-5** Usage"。
- 架构 = **MLA(多头潜在注意力)+ DSA(DeepSeek 稀疏注意力)+ 大规模 MoE + MTP(多 token 预测)**。

所以部署套路 = DeepSeek-V3.2,只是 `--reasoning-parser glm45 --tool-call-parser glm47` 不同。

## 1. "满血版"指什么

| 权重 | 含义 | 部署门槛 |
|---|---|---|
| `zai-org/GLM-5` | BF16 全精度 | 显存最大,需多机 |
| `zai-org/GLM-5-FP8` | FP8 块量化的完整模型(**生产里说的"满血"通常是它**) | 单机 8×H200/B200 |
| `GLM-5-w4a8` | W4A8 低比特量化 | 单机最省,精度略降 |

本指南主线 = `zai-org/GLM-5-FP8`,单机 8 卡。

## 2. 最简启动(先跑通)

```bash
python3 -m sglang.launch_server \
  --model zai-org/GLM-5-FP8 \
  --tp 8 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

## 3. 生产配置(高吞吐满血版)

```bash
export SGLANG_ENABLE_SPEC_V2=1            # 投机解码 overlap 调度

python3 -m sglang.launch_server \
  --model zai-org/GLM-5-FP8 \
  --tp 8 --dp 8 --enable-dp-attention \   # 见 01-parallelism
  --moe-a2a-backend deepep --deepep-mode auto \   # 见 02-deepep
  --moe-runner-backend deep_gemm \        # 见 03-deepgemm
  --speculative-algorithm EAGLE \         # MTP，见 05-mtp
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
  --mem-fraction-static 0.85 \            # 见 04-memory
  --chunked-prefill-size 16384 \
  --context-length 131072 \
  --page-size 64 \
  --trust-remote-code \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name glm-5
```

## 4. 模块索引(你的 9 个问题落点)

| 文件 | 回答的问题 |
|---|---|
| [01-parallelism.md](01-parallelism.md) | **Q1** TP/EP/DP 是否同时存在?谁是 TP、谁是 DP、谁是 EP? |
| [02-deepep.md](02-deepep.md) | **Q2** `--deepep-mode auto` 两种模式区别;**Q3** DeepEP 到底干了啥 |
| [03-deepgemm.md](03-deepgemm.md) | **Q5** DeepGEMM 是什么优化 |
| [04-memory-and-scheduling.md](04-memory-and-scheduling.md) | **Q4** `mem-fraction-static` 的说法,和 `max-running-requests` 区别 |
| [05-mtp-and-speculative.md](05-mtp-and-speculative.md) | **Q6** MTP 和 EAGLE 一样吗?GLM-5 用哪个? |
| [06-dsa-indexer.md](06-dsa-indexer.md) | **Q9** indexer / IndexCache 的工作原理 |
| [07-pd-and-router.md](07-pd-and-router.md) | **Q7** 单机能跑为何还用 PP+CP;**Q8** sglang_router 怎么搞 |

## 5. 面试 30 秒收口

> "GLM-5 在 SGLang 里复用 DeepSeek-V3.2 架构:MLA + DSA 稀疏注意力 + 大规模 MoE + MTP。FP8 满血权重单机 8×H200 上,我用 **TP 切非注意力权重 + DP Attention 让每张卡独立算自己 token 的注意力**(因为 MLA 只有 1 个 KV head,纯 TP 会把 KV cache 复制 8 份),**MoE 走 EP + DeepEP all-to-all**,FP8 GEMM 用 DeepGEMM,小 batch 用 MTP(EAGLE 框架)提速,长上下文用 DSA 的 lightning indexer + IndexCache 降开销;再往上用 PD 分离把 prefill/decode 解耦,sglang_router 做 cache-aware 路由。"
</content>
</invoke>
