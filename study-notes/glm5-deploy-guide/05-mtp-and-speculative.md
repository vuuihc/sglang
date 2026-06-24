# 05 · MTP 和 EAGLE 不是一回事 —— GLM-5 到底用哪个

> 对应 **Q6**:"MTP 和 EAGLE 不一样吧,GLM-5 用的是哪个?"

## 1. 结论先行

你的直觉对:**MTP ≠ EAGLE**,它们是两个层面的东西。但 GLM-5 在 SGLang 上是**两者结合**:

> **GLM-5 自带 MTP(模型能力),SGLang 用 EAGLE(执行框架)来驱动这个 MTP。**
> 命令里写的是 `--speculative-algorithm EAGLE`,但它跑的是模型自带的 MTP 头。

## 2. 两个概念分别是什么

### MTP(Multi-Token Prediction)—— 是"模型自带的能力 / 权重"
- 训练时,GLM-5 / DeepSeek 除了主模型,还**多训练了一个(或几个)NextN 层**,
  专门用来**根据当前隐状态预测后面第 2、3… 个 token**。
- 这是**模型权重的一部分**,跟着 checkpoint 一起发布。
- 在 SGLang 代码里就是 [glm4_moe_nextn.py](../../python/sglang/srt/models/glm4_moe_nextn.py) 里的 `Glm4MoeModelNextN`,
  加载时打了 `is_nextn=True` 标记(见该文件 74、165 行)。它**复用主模型的一个 decoder 层结构**当草稿头。

### EAGLE —— 是"投机解码的执行算法 / 框架"
- EAGLE 是一种 **speculative decoding(投机解码)** 方法:用一个**轻量草稿模型**先连续猜若干 token,
  再让**大模型一次性并行验证**这些猜测,验证通过的就直接采纳,省去逐 token 解码。
- 它是 SGLang 里的**调度 + 验证机制**,本身不关心草稿模型是谁。

### 关系
> **MTP 提供"草稿头"(猜下一个 token 的权重),EAGLE 提供"怎么用草稿头去猜+验证"的引擎。**
> SGLang 的做法:把模型自带的 MTP/NextN 层**当作 EAGLE 的草稿模型**接进 EAGLE 流水线。
> 所以你不需要单独训一个 EAGLE 草稿模型——MTP 头就是现成的草稿模型。

文档 [deepseek_v32.md](../basic_usage/deepseek_v32.md) 原话:
> "SGLang implements **Multi-Token Prediction (MTP)** for DeepSeek V3.2 **based on EAGLE speculative decoding**."

## 3. GLM-5 的启用命令

```bash
export SGLANG_ENABLE_SPEC_V2=1      # 开启 draft/verify 的 overlap 调度

python -m sglang.launch_server --model zai-org/GLM-5-FP8 \
  --tp 8 --dp 8 --enable-dp-attention \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

注意:**算法名写 `EAGLE`,但实际用的草稿模型是 GLM-5 自带的 MTP 头**。这不矛盾——EAGLE 是执行框架,MTP 是它的草稿来源。

## 4. 三个参数手把手推导一次"投机一步"

设 `num-steps=3`, `eagle-topk=1`, `num-draft-tokens=4`。

| 参数 | 含义 |
|---|---|
| `--speculative-num-steps 3` | 草稿头**自回归地连续猜 3 步** |
| `--speculative-eagle-topk 1` | 每步保留 **top-1** 候选 → 草稿是一条**线性链**(不是树) |
| `--speculative-num-draft-tokens 4` | 一次提交给大模型验证的草稿 token 总数 = 4 |

**一轮流程**(已知当前真实 token 为 `t0`):

```
草稿头(MTP):  t0 → 猜 d1 → 猜 d2 → 猜 d3        (3 步,topk=1 → 链长 3)
组装验证序列:  [t0, d1, d2, d3]                  (4 个 token = num-draft-tokens)
大模型一次前向(并行验证这 4 个位置):
  位置 t0 的输出应是 d1? 是 → 接受 d1
  位置 d1 的输出应是 d2? 是 → 接受 d2
  位置 d2 的输出应是 d3? 否 → 在此截断，用大模型在该位置的真实输出 t' 替换
最终这一轮产出: t0 之后接受了 d1, d2, 再加 t'  → 一次前向吐出 3 个 token
```

**收益直觉**:
- 普通解码:1 次大模型前向 = 1 个 token。
- 投机解码:1 次大模型前向 ≈ 接受数 + 1 个 token(上例 ≈ 3 个)。
- **接受率越高、链越长,加速越大**。但草稿越长、错得越早就浪费越多,所以要 tune。

## 5. 为什么对"小 batch"特别有效?

- 小 batch 时,大模型一次前向**算力严重闲置**(GPU 没喂饱)。
  投机解码把"验证多个草稿 token"塞进这一次前向,**几乎不增加耗时却多产出 token** → 净赚。
- batch 很大时 GPU 本就喂饱了,投机反而可能挤占算力。
  所以文档建议:大 batch 用最小配置 `steps=1 topk=1 draft=2`,并调大 `--max-running-requests`(默认 48)。

## 6. 调优与 overlap

- 最优 `(num-steps, eagle-topk, num-draft-tokens)` 因 batch 而异,用
  [bench_speculative.py](../../scripts/playground/bench_speculative.py) 针对目标 batch 搜。
- `SGLANG_ENABLE_SPEC_V2=1`:开启 **draft 阶段和 verify 阶段的 overlap 调度**
  (草稿头算下一轮的同时,大模型在验证上一轮),进一步压低端到端延迟。

## 7. 一句面试话

> "MTP 是**模型自带的多 token 预测头**(权重的一部分),EAGLE 是 SGLang 里**投机解码的执行框架**。GLM-5 的做法是把它自带的 MTP/NextN 层当作 EAGLE 的草稿模型——所以命令里写 `--speculative-algorithm EAGLE`,跑的却是 MTP 头。草稿头连续猜几步、大模型一次并行验证,在小 batch 上几乎免费地把单次前向的产出从 1 个 token 提到 3 个左右。"
</content>
