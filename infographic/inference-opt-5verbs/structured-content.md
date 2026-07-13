# 大模型推理优化,拆到底只有五个动词

## Overview
用「两面墙」立地基,把推理优化收敛成五个动词的全景框架。

## Learning Objectives
1. Prefill 撞算力墙,Decode 撞访存墙
2. 五个动词按优先级:不算→少算→少搬→打满→破串行
3. 每个动词对应的具体技术归位

---

## Section 0: 主标题
- Headline: "大模型推理优化 · 只有五个动词"
- Subhead: "一套框架看懂所有推理优化技术"

---

## Section 1: 两面墙(顶部地基)

**Key Concept**: 一个算子要么撞算力墙,要么撞访存墙。

**Text Labels**:
- 算力墙 Compute-bound —— 算不过来
- 访存墙 Memory-bound —— 数据搬不过来,算力干等

**对比条**:
- Prefill(处理 prompt,一次几千 token)→ 撞【算力墙】
- Decode(逐字生成,一次1个token却要搬整个模型)→ 撞【访存墙】,GPU 算力利用率常 < 5%

---

## Section 2: 五个动词全景树(主体,纵向从上到下,优先级递减)

**Key Concept**: 越靠前越便宜,省下来的永远比优化过的划算。
箭头链:不算 → 少算 → 少搬 → 打满 → 破串行

**1 · 不算 —— 能不算就不算(复用已有结果)**
- KV Cache · PagedAttention · Radix Cache · Prompt Caching

**2 · 少算 —— 省 FLOPs(治算力墙)**
- 量化 FP8/INT8/AWQ/GPTQ · 稀疏注意力 · MoE

**3 · 少搬 —— 省带宽(★ decode 主战场)**
- MLA / GQA / MQA · KV 量化 · KV 卸载(HiCache) · FlashAttention · 算子融合

**4 · 打满 —— 别让硬件空转**
- Continuous Batching · Chunked Prefill · TP/PP/EP · PD 分离 · CUDA Graph · Overlap

**5 · 破串行 —— 打掉自回归延迟地板(★ 用免费算力换延迟)**
- Speculative Decoding · Medusa / EAGLE · MTP · Lookahead

---

## Section 3: 底部金句
- "撞哪面墙?→ 该用哪个动词?—— 这就是判断任何新技术的坐标系。"

---

## Data Points (Verbatim)
- "Decode 阶段 GPU 算力利用率经常低于 5%"
- "MLA 把 KV Cache 体积砍掉一个数量级"
- "验证 K 个 token 和验证 1 个,权重都只搬一遍"

---

## Design Instructions
- Layout: 纵向分层递进(hierarchical-layers),顶部两面墙 + 五层动词台阶
- Aspect: portrait 9:16
- Language: 中文
- 视觉:两堵墙意象;五动词做成从上到下的彩色分层;★标记第3、5层;整体清爽有技术感,适合封面
