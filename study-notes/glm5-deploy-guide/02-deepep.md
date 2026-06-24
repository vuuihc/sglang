# 02 · DeepEP 到底干了啥 + `--deepep-mode auto` 两种模式

> 对应 **Q3**:DeepEP 具体做了什么;**Q2**:`auto` 的 normal / low_latency 有啥区别。

## 1. 问题:EP 模式下,token 怎么找到自己的专家?

回顾 [01-parallelism.md](01-parallelism.md):MoE 用 EP,把专家分散到各卡。比如:
- 256 个路由专家,EP=8 → 每张卡存 32 个专家。
- 每个 token 经过 gate(路由器)算出要去 **top-8** 个专家。

这 8 个专家**几乎一定散落在不同卡上**。于是产生一个棘手的通信问题:

> 卡 0 上有一批 token,它们要去的专家分散在卡 0~7;
> 同时卡 0 上的某些专家,也要接收来自卡 1~7 的 token。
> **每张卡既是发送方又是接收方,且每张卡发给每张卡的 token 数都不一样、运行时才知道。**

这就是经典的 **all-to-all(全交换)+ 变长(ragged)** 通信。朴素实现极慢。
**DeepEP 就是专门解决这个的高性能 all-to-all 通信库。**

## 2. DeepEP 具体做的三件事

[expert_parallelism.md](../advanced_features/expert_parallelism.md) 把 MoE 前向拆成:
`dispatch → pre-permute → grouped GEMM → post-permute → combine`。
DeepEP 负责其中的 **dispatch** 和 **combine** 两步通信:

### ① Dispatch(分发)
把每个 token 按 gate 选出的专家编号,**路由/打包**发送到对应专家所在的卡。
- 同一张卡上、要去同一个专家的 token 会被聚到一起(permute),
  这样下一步可以对每个专家做一次大的 grouped GEMM,而不是一个 token 一次小 GEMM。
- 利用 **NVLink(机内)+ RDMA(机间)** 做高带宽传输,并把通信和计算 overlap 起来。

### ② Combine(合并)
专家算完后,把每个 token 的结果**送回它原来所在的卡和位置**(逆 permute),
再按 top-k 权重加权求和。

### 为什么需要专门的库而不用 `all_to_all`?
- token→专家是**变长、不规则**的(每个专家收到的 token 数动态变化),通用 collective 不好用。
- MoE 解码时 batch 小、延迟敏感,DeepEP 做了 **低延迟内核 + CUDA Graph 兼容**(见下)。
- 它把 dispatch 的元数据计算、跨机 RDMA、permute 全部融合优化。

> 约束(面试加分点):**DeepEP / Mooncake / NIXL / MORI 都要求 `ep_size == tp_size`**。
> 想做 `ep_size < tp_size` 的混合 EP+TP,只能用 `--moe-a2a-backend none`(走 all-reduce/all-gather)。
> 见 [expert_parallelism.md](../advanced_features/expert_parallelism.md):"Currently, DeepEP ... only support cases where ep_size = tp_size"。

## 3. `--deepep-mode` 的两种模式

DeepEP 的 dispatch 有两套内核,对应推理的两个阶段:

| 模式 | 优化目标 | 适用阶段 | 特点 |
|---|---|---|---|
| **`normal`** | **高吞吐** | **Prefill** | 一次处理大量 token,带宽打满;**不兼容 CUDA Graph** |
| **`low_latency`** | **低延迟** | **Decode** | 每步只有少量 token(每请求 1 个),延迟优先;**兼容 CUDA Graph** |

为什么要分两套?因为两个阶段的通信形状完全不同:

- **Prefill**:一次进来成千上万个 prompt token,all-to-all 的数据量大 → 追求带宽利用率(normal)。
- **Decode**:每个请求每步只产 1 个 token,几十个请求也就几十个 token,数据量极小 → 此时**延迟由固定开销主导**,要用专门的低延迟内核,而且要能被 **CUDA Graph 捕获**(把整个 decode step 录成一张图重放,省掉 kernel launch 开销)。

### `--deepep-mode auto` 做的事
运行时**自动按当前 batch 是 prefill 还是 decode 切换** normal / low_latency,不用你手动管。
- 生产环境推荐 `auto`。
- `normal` / `low_latency` 写死,只用于调试或开发某一条路径。

> 文档原话([expert_parallelism.md](../advanced_features/expert_parallelism.md)):
> "`normal` mode (optimized for prefill, high throughput) and `low_latency` mode (optimized for decode, low latency and CUDA Graph compatibility). ... recommended to set `--deepep-mode auto`"。

## 4. 数字直觉

假设 decode 时 32 个并发请求,top-8 路由,EP=8:
- 每步只有 32 个 token,每个 token 复制 8 份(8 个专家)= 256 条小消息要 all-to-all。
- 这种"消息多但每条极小"的场景,瓶颈是**延迟/launch 开销**,不是带宽
  → 必须 `low_latency` + CUDA Graph,否则每步都被通信固定开销拖死。

而 prefill 一次 16k token × 8 = 128k 条,数据量巨大
  → `normal` 模式把带宽吃满才划算。

`auto` 就是让同一个服务在这两种极端之间自动选对内核。

## 5. 和 `--moe-runner-backend` 的分工

别混淆:
- `--moe-a2a-backend deepep` 管的是 **通信**(token 怎么在卡间搬)。
- `--moe-runner-backend deep_gemm` 管的是 **计算**(专家的矩阵乘怎么算)→ 见 [03-deepgemm.md](03-deepgemm.md)。

一条命令里两者配合:DeepEP 把 token 搬到位 → DeepGEMM 做 grouped GEMM → DeepEP 搬回去。
</content>
