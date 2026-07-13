---
title: "大模型推理优化,拆到底只有五个动词"
topic: "technical / AI-infra"
data_type: "hierarchy + decision-tree"
complexity: "complex"
source_language: "zh"
user_language: "zh"
---

## Main Topic
用「两面墙(算力墙/访存墙)」作为物理地基,把大模型推理优化的所有技术收敛成五个动词:不算、少算、少搬、打满、破串行。给读者一个判断任何新技术归属的坐标系。

## Learning Objectives
After viewing this infographic, the viewer should understand:
1. 推理的两个阶段撞不同的墙:Prefill 撞算力墙、Decode 撞访存墙(算力利用率常<5%)。
2. 五个动词是从两面墙推导出来的优化优先级:不算→少算→少搬→打满→破串行。
3. 每个动词对应哪些具体技术,以及 MLA(少搬)、Spec Decoding(破串行)各归何处。

## Target Audience
- **Knowledge Level**: Intermediate → Expert(了解 LLM 推理但缺全局框架的工程师)
- **Context**: 读技术文章/建立知识地图
- **Expectations**: 一张图看懂推理优化全景与内在逻辑

## Content Type Analysis
- **Data Structure**: 顶部一个决策分叉(两面墙)→ 五层纵向递进的分类树
- **Key Relationships**: 撞哪面墙 → 决定优化方向;五动词按优先级排序(越靠前越便宜)
- **Visual Opportunities**: 两堵墙的意象;五个动词做成纵向递进的分层/台阶;每层挂具体技术标签;★标记 decode 主战场(少搬)与破串行

## Design Instructions (from user input)
- 竖版(portrait 9:16),适合知乎题图/公众号封面
- 顶部两面墙;中部 Prefill/Decode 对比;主体五动词全景树
- 五动词技术标签:
  1. 不算:KV Cache / PagedAttention / Radix Cache
  2. 少算(省FLOPs):量化 / 稀疏 / MoE
  3. 少搬(省带宽)★:MLA / GQA / KV量化 / FlashAttention / 融合
  4. 打满:Continuous Batching / 并行 / PD分离 / CUDA Graph / Overlap
  5. 破串行:Speculative Decoding / Medusa / EAGLE / MTP
