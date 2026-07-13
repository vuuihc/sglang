Create a professional infographic following these specifications:

## Image Specifications
- **Type**: Infographic (tech article cover / 知乎题图)
- **Layout**: hierarchical-layers (vertical pyramid/stacked tiers, top → bottom)
- **Style**: technical-schematic (Blueprint variant)
- **Aspect Ratio**: 9:16 portrait
- **Language**: Simplified Chinese (中文), with small English sub-labels where noted

## Core Principles
- Follow the layered/stacked structure precisely: a header "two walls" band on top, then FIVE clearly separated horizontal tiers stacked vertically, priority decreasing top→bottom.
- Blueprint aesthetic: deep navy background (#1E3A5F) with faint white technical grid, white/cyan line work, amber (#F59E0B) accent highlights, dimension-line and annotation styling, clean sans-serif + a few ALL-CAPS English tags.
- Concise: show keywords only, generous spacing, strong visual hierarchy. Not crowded.

## Layout & Content (top to bottom)

### HEADER BAND — 两面墙 (The Two Walls)
Two stylized brick/blueprint walls side by side:
- LEFT wall labeled: 算力墙 · COMPUTE-BOUND —— 算不过来
- RIGHT wall labeled: 访存墙 · MEMORY-BOUND —— 数据搬不过来,算力干等
Below the two walls, a thin comparison bar with two arrows:
- Prefill(一次处理几千 token)──▶ 撞【算力墙】
- Decode(一次1个token,却要搬整个模型)──▶ 撞【访存墙】,GPU 算力利用率常 < 5%

### TITLE (prominent, near top or overlapping header)
- Main title: 大模型推理优化 · 只有五个动词
- Subtitle: 一套框架看懂所有推理优化技术

### FIVE STACKED TIERS (each a distinct blueprint panel, distinct hue, numbered ①–⑤; a downward arrow chain on the side showing 不算→少算→少搬→打满→破串行; a side note "越靠前越便宜")

① 不算 —— 能不算就不算(复用已有结果)
   tags: KV Cache · PagedAttention · Radix Cache · Prompt Caching

② 少算 —— 省 FLOPs(治算力墙)
   tags: 量化 FP8/INT8/AWQ/GPTQ · 稀疏注意力 · MoE

③ 少搬 —— 省带宽 ★DECODE 主战场
   tags: MLA · GQA/MQA · KV 量化 · KV 卸载 HiCache · FlashAttention · 算子融合
   (highlight this tier with amber accent + a ★ badge)

④ 打满 —— 别让硬件空转
   tags: Continuous Batching · Chunked Prefill · TP/PP/EP · PD 分离 · CUDA Graph · Overlap

⑤ 破串行 —— 打掉自回归延迟地板 ★用免费算力换延迟
   tags: Speculative Decoding · Medusa/EAGLE · MTP · Lookahead
   (highlight with amber accent + a ★ badge)

### FOOTER STRIP
One-line takeaway in a blueprint annotation box:
撞哪面墙? → 该用哪个动词? —— 判断任何新技术的坐标系。

## Style Guidelines (technical-schematic / blueprint)
- Deep navy blueprint background with subtle white grid lines.
- White and cyan thin strokes, consistent stroke weight, geometric precision.
- Amber (#F59E0B) reserved for the two ★ tiers (③少搬, ⑤破串行) and key numbers.
- Dimension lines, small technical tick marks, corner registration marks for authenticity.
- Each tier in a slightly different tint (blue→teal→amber-tinted→blue→cyan) but all within the blueprint palette.
- Typography: clean Chinese sans-serif for labels; ALL-CAPS English for COMPUTE-BOUND / MEMORY-BOUND / DECODE and technique acronyms (MLA, KV Cache, CUDA Graph, EAGLE).
- Keep it readable at thumbnail size (cover use): big title, bold tier numbers.

## Critical
- All Chinese text rendered correctly and legibly.
- Do NOT invent techniques beyond those listed.
- Emphasize the causal spine: 两面墙 → 决定 → 五个动词(优先级 top→bottom).
