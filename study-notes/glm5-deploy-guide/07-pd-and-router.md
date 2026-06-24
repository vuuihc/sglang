# 07 · PD 分离、PP+CP 为何存在,以及 sglang_router 怎么搞

> 对应 **Q7**:"单机就能跑完 prefill,为啥还用 PP?是因为 prefill 不用 EP 吗?"
> 对应 **Q8**:`sglang_router` 路由具体怎么搞。

---

## Part A — 先理解 PD 分离

### 为什么要把 Prefill 和 Decode 拆开
[pd_disaggregation.md](../advanced_features/pd_disaggregation.md) 给的两个理由:

1. **Prefill 打断 Decode**:prefill 是计算密集型(一次吞整个 prompt),decode 是访存密集型(每步 1 token)。
   混在一个引擎里,新来的 prefill batch 会**频繁打断**正在解码的 batch,decode 延迟抖动。
2. **DP attention 不均衡**:同一时刻一个 DP worker 在 prefill、另一个在 decode,decode 延迟被拖累。

**PD 分离 = 把 prefill 和 decode 放到不同实例(不同 GPU 组)**,各自用各自最优的并行策略,中间用 RDMA 传 KV。

> 关键收益:**两边可以用完全不同的并行配置**。这正是回答 Q7 的钥匙。

---

## Part B — Q7:单机能跑,为什么 Prefill 还用 PP+CP?

### 先纠正一个点
> ❌ "用 PP 是因为 prefill 不用 EP" —— 不对。

PP+CP 和 EP **不是替代关系**。prefill 实例里 MoE 照样可以用 EP/DeepEP 或 fused MoE。
PP+CP 解决的是**另一个维度的问题:超长序列(long context)**。

### CP(Context Parallel)解决什么:把"一条很长的序列"切开
- TP/EP 切的是**模型**(权重、专家)。CP 切的是**序列本身**——把一个长 prompt 的 token
  分到多张卡上,每张卡只持有序列的一段。
- 看 [deepseek_v32.md](../basic_usage/deepseek_v32.md) 的两种切法:
  - `in-seq-split`:序列均匀分给各 CP rank,attention 阶段各 rank 算自己段的 indexer,
    再 all-gather 拿到完整 KV。要求 `moe_a2a_backend=deepep`、prefill batch=1。
  - `round-robin-split`(默认):按 `token_idx % cp_size` 分配,**支持 fused MoE**
    (单机下 fused MoE 常比 DeepEP 还快)、支持 FP8 KV、支持多 batch。

### 为什么"单机能跑完"还需要 CP?
"能跑完"≠"跑得好"。对一个 128k+ 的超长 prompt:
- 单卡要独自承担这条序列的**全部注意力计算和激活显存** → **TTFT(首 token 延迟)非常高**,激活可能爆显存。
- CP 把这条序列摊到 8 张卡并行算 → **TTFT 大幅下降**,单卡激活压力减小。

所以 CP 不是为了"放得下",而是为了**把一条长序列的 prefill 算得更快、更省单卡显存**。

### 那为什么又叠 PP(Pipeline Parallel)?
- CP 在**一个节点内**用 8 张卡切序列(`--attn-cp-size 8`)。
- 想进一步 **scale 到多节点**拿到更多算力/显存、并用**流水线**重叠层间计算时,加 **PP**。
  PP 把层切成几段(`--pp-size 2`,配 `SGLANG_PP_LAYER_PARTITION=30,31`),跨 2 节点流水。
- [deepseek_v32.md](../basic_usage/deepseek_v32.md) 原话:PP+CP "combines PP and CP to scale across
  multiple nodes ... better throughput and TTFT"。即:**PP 跨节点扩规模 + 流水线提吞吐,CP 切序列降 TTFT**,
  两者叠加专治"超长上下文、prefill 很重"的场景。

### 一句话回答 Q7
> "PP 不是因为 prefill 不用 EP——EP/MoE 照常跑。PP+CP 是**长上下文专用的扩展手段**:
> CP 把一条超长序列切到多卡并行算,降 TTFT、省单卡激活;PP 再把层切到多节点流水,
> 跨节点扩算力、提吞吐。单机'能跑完'不等于'TTFT 可接受',这才是用它们的原因。
> 而且在 PD 分离下,prefill 端用 PP+CP、decode 端用 EP,各取所需。"

### Decode 端为什么推荐 EP?
decode 是访存密集、每步 token 少。EP + low_latency DeepEP + CUDA Graph 最适合
(见 [02-deepep.md](02-deepep.md))。所以 [deepseek_v32.md](../basic_usage/deepseek_v32.md) 写
"For the Decode nodes, it is recommended to use the **EP mode**"。

---

## Part C — Q8:sglang_router 怎么搞

### 它是什么
`sglang_router`(新名 SGLang Model Gateway)是一个**独立的路由进程**,坐在客户端和多个
SGLang 引擎实例之间,负责**负载均衡 + 缓存感知路由 + 容错**。两种用法:

### 用法 1:PD 分离路由(本指南场景)
来自 [deepseek_v32.md](../basic_usage/deepseek_v32.md) / [pd_disaggregation.md](../advanced_features/pd_disaggregation.md):

```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill  $PREFILL_ADDR 8998 \   # prefill 实例地址 + 它的 bootstrap 端口
  --decode   $DECODE_ADDR \         # decode 实例地址
  --host 127.0.0.1 --port 8000
```

请求的一生:
```
client → router(:8000)
   │  1. router 选 1 个 prefill 实例 + 1 个 decode 实例
   ▼
prefill 实例:算完整个 prompt 的 KV cache
   │  2. 通过 RDMA(Mooncake / NIXL 后端)把 KV 直接传给 decode 实例
   ▼  (prefill 侧 --disaggregation-bootstrap-port 8998 用于握手)
decode 实例:拿到 KV，开始逐 token 生成 → 流式返回给 client
```

- prefill 端启动时要带 `--disaggregation-mode prefill --disaggregation-bootstrap-port 8998`。
- decode 端 `--disaggregation-mode decode`。
- 两边 TP 不同(如 prefill TP=8、decode DP attention TP=1)时,KV 内存布局不同,
  用 **GPU staging buffer**(Mooncake)聚合后批量 RDMA,文档称高并发下 **2–5x** 吞吐提升。

### 用法 2:纯 DP 副本路由(非 PD)
如果你只是起了 N 个完整副本(DP,见 [dp_dpa_smg_guide.md](../advanced_features/dp_dpa_smg_guide.md)),
router 把请求分发到副本上。它的杀手锏是 **cache-aware 路由**:

> 优先把请求路由到**已经缓存了该 prompt 前缀(radix cache)**的副本上 → 命中 prefix cache,
> 省掉重复 prefill。这比朴素 round-robin 吞吐高得多,尤其多轮对话 / 共享 system prompt 场景。

其他策略:轮询、最少负载、power-of-two 等;还提供健康检查、故障实例自动摘除(容错)。

### 一句面试话(Q8)
> "sglang_router 是引擎前面的网关。PD 分离下,它给每个请求选一对 prefill+decode 实例,
> prefill 算完 KV 通过 RDMA 直传 decode,decode 出 token;它还做 cache-aware 路由——
> 优先把请求送到已缓存该 prefix 的实例命中 radix cache,加上健康检查和容错。
> 没有它,多实例就只能各自为政、prefix cache 命中率低、也没法做 PD 之间的 KV 编排。"

---

## 附:PD 分离最小命令(GLM-5,换 parser 即可)

Prefill:
```bash
python -m sglang.launch_server --model zai-org/GLM-5-FP8 \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
  --tp 8 --dp 8 --enable-dp-attention --mem-fraction-static 0.9 \
  --dist-init-addr $HOST:$DIST_PORT --trust-remote-code \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --host $LOCAL_IP --port $PORT
```
Decode:
```bash
python -m sglang.launch_server --model zai-org/GLM-5-FP8 \
  --disaggregation-mode decode \
  --tp 8 --ep 8 --moe-a2a-backend deepep --deepep-mode auto \
  --mem-fraction-static 0.9 \
  --dist-init-addr $HOST:$DIST_PORT --trust-remote-code \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --host $LOCAL_IP --port $PORT
```
Router:
```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill $PREFILL_ADDR 8998 --decode $DECODE_ADDR \
  --host 0.0.0.0 --port 8000
```
</content>
