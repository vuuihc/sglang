# 03 · DeepGEMM 是什么优化

> 对应 **Q5**:`--moe-runner-backend deep_gemm` 优化了什么。

## 1. 一句话

DeepGEMM 是一个**专为 FP8 块量化(block-wise FP8)MoE 设计的高性能 grouped GEMM 内核库**,
针对 Hopper/Blackwell 的 FP8 Tensor Core,用 JIT 编译生成贴合形状的内核,
分别为 **prefill(连续布局)** 和 **decode(masked 布局)** 优化。

它是 MoE 前向里 `grouped GEMM` 那一步的计算后端(DeepEP 管通信,它管算)。

## 2. 背景:MoE 的计算长什么样

EP 模式下,DeepEP 把 token 按专家聚好后(见 [02-deepep.md](02-deepep.md)),每张卡要算:

```
专家0:  [收到的 token_0..n0] × W_up0 → 激活 → × W_down0
专家1:  [收到的 token_0..n1] × W_up1 → 激活 → × W_down1
...
```

每个专家收到的 token 数 **n_i 不一样、运行时才知道**,而且每个 n_i 都不大。
这叫 **grouped GEMM**(一组形状不同的小矩阵乘),它有两个天然难点:

1. 每个专家的 M 维(token 数)是动态的、不规则的。
2. token 数小,普通 GEMM 内核启动开销占比高、Tensor Core 吃不饱。

DeepGEMM 就是把这组不规则小 GEMM 算得又快又稳。

## 3. DeepGEMM 的几个关键优化

### ① 原生支持 FP8 block-wise 量化
GLM-5-FP8 的权重是 **块量化**(每 128×128 块一个 scale),激活也分块量化。
DeepGEMM 的内核**直接吃 FP8 输入 + per-block scale**,在 Tensor Core 上做 FP8 累加、
内联反量化,不需要先 dequant 成 BF16 再算 → 省显存带宽、省一次 kernel。

### ② 两种内存布局,对应两个阶段
[expert_parallelism.md](../advanced_features/expert_parallelism.md) 原话:
"supporting **contiguous** layouts for prefill and **masked** layouts for decode"。

- **Prefill(contiguous / 连续布局)**:token 多,把各专家的 token 在内存里连续摆放,
  做一个大的分段 GEMM,带宽利用率最高。
- **Decode(masked / 掩码布局)**:token 极少(每请求 1 个),不重排内存,
  用 mask 标记每个专家实际有几个有效 token,配合固定形状 → **可被 CUDA Graph 捕获**,
  避开动态形状导致的重新捕获/重新编译。

### ③ JIT 编译贴合形状
DeepGEMM 经常 **运行时 JIT** 生成针对当前 (M, N, K, SM 数) 的最优内核,
而不是用一个通用内核硬扛所有形状 → 小矩阵也能把 SM 占满。
代码里也能看到它在协调 SM 资源,例如 [dsa_indexer.py:338](../../python/sglang/srt/layers/attention/dsa/dsa_indexer.py) 用
`deep_gemm.get_num_sms()` 拿 SM 数来分配 indexer 的算力。

## 4. 为什么不直接用 Triton / cuBLAS?

| 后端 | 问题 |
|---|---|
| cuBLAS | 不原生支持 FP8 block-wise + grouped + 动态 M |
| Triton(`--moe-runner-backend triton`) | 通用、可扩展,但要达到最优需手工 tune 配置;默认不一定打满 Hopper FP8 |
| **DeepGEMM** | 为这个特定场景(FP8 块量化 + grouped + prefill/decode 两布局)量身定做,开箱即最优 |

`--moe-runner-backend auto`(默认)会按硬件 + 量化方案自动挑;
对 **Hopper + FP8 block-wise** 的 GLM-5-FP8,挑的就是 `deep_gemm`,
所以显式写 `--moe-runner-backend deep_gemm` 等于"锁定这个最优解"。

## 5. 一句面试话

> "MoE 的计算是 grouped GEMM——一堆形状不规则的小矩阵乘。DeepGEMM 针对 FP8 块量化做了原生 Tensor Core 内核,
> prefill 用连续布局打满带宽、decode 用 masked 布局兼容 CUDA Graph,并用 JIT 生成贴合形状的内核让小矩阵也能占满 SM。
> DeepEP 负责把 token 搬到对应专家(通信),DeepGEMM 负责把专家算完(计算),两者配合构成 SGLang 的 EP MoE 引擎。"
</content>
