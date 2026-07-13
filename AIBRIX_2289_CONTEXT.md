# AIBrix #2289 背景知识与上下文

> 目标 issue: [vllm-project/aibrix#2289](https://github.com/vllm-project/aibrix/issues/2289)
> "SGLang PD: prefill failure is silent and leaves the decode request hanging on bootstrap"
>
> 本文把修这个 bug 需要的上下文一次讲清:K8s 控制面/数据面划分 → Operator 全景 →
> AIBrix 定位 → PD 分离原理 → PD Router 请求流 → bug 精确定位与当前真实状态。

---

## 0. TL;DR

- **AIBrix** = 字节做的、现挂在 vllm-project 组织下的 **K8s 原生 LLM 推理基础设施**(控制面 operator + 数据面 gateway)。
- **#2289 在数据面 gateway**(Envoy ext_proc router)里,不在 operator 那半。修它靠的是 **SGLang 引擎知识**(PD 分离 + bootstrap 握手),不是 controller-runtime。
- bug 的核心:SGLang 的 prefill 在 goroutine 里异步发出,**失败被吞**,`Route()` 立刻返回、decode 照常派发 → 客户端挂到 SGLang 自己的 bootstrap 超时才拿到一个不透明错误。
- ⚠️ **关键现状(必须读现码,别信 issue 原文)**:代码在 issue 提交后被重构,原文说的"无条件报 success 200"**已被修**。**还没修**的是:(a) 异步失败**不发** `GatewayPrefillRequestFailTotal`(与 sync 路径不对称);(b) 失败**不 fail/retry 请求** → 仍会挂死。

---

## 1. K8s 控制面 vs 数据面

### 划分judgement:一句话判据
> **控制面管"应该是什么样"(desired state),不在请求路径上;数据面"实际承载流量/负载",每个请求都穿过它。**

### K8s 自身的控制面/数据面

| | 控制面 (Control Plane) | 数据面 (Data Plane) |
|---|---|---|
| 组件 | `kube-apiserver`、`etcd`、`kube-scheduler`、`kube-controller-manager`、`cloud-controller-manager` | `kubelet`、`kube-proxy`、CNI、**业务 Pod 本身** |
| 职责 | 存/调谐期望状态、调度决策 | 真正跑容器、转发网络流量 |
| 有流量穿过吗 | ❌ 声明式,管元数据 | ✅ 承载实际 workload/网络 |
| 挂了会怎样 | 集群"不能变更",但**已跑的负载继续跑** | 负载直接受影响 |

**记忆点**:控制面是"大脑"(决策、调谐),数据面是"手脚"(执行、扛流量)。控制面短暂宕机通常不影响存量服务,这正是二者解耦的价值。

### 映射到 LLM serving(也是 AIBrix 的分层)

| | 控制面 | 数据面 |
|---|---|---|
| 干啥 | 管 Deployment/CRD、扩缩容、LoRA 挂载、KV cache 编排 | **网关路由 + 推理引擎 Pod**(真正吐 token) |
| AIBrix 里 | `pkg/controller/*`(operator) | `pkg/plugins/gateway`(Envoy ext_proc)+ vLLM/SGLang pods |
| **#2289** | | ✅ **就在这:数据面 gateway 的 PD 路由** |

---

## 2. K8s Operator 全景

### 什么是 Operator
> **Operator = CRD(自定义资源)+ Controller(调谐循环)**,把"运维某个有状态应用的领域知识"写进代码。

核心是 **reconcile loop**:`观察实际状态 → 对比期望状态(spec)→ 采取动作收敛 → 重复`。框架通常是 `controller-runtime` / kubebuilder / Operator SDK。

⚠️ 术语细分:K8s 内置的叫 **controller**,社区把"CRD + controller 打包管理某个应用"的叫 **operator**——**本质是同一个 reconcile 模式**。

### K8s 内置 controllers(住在 `kube-controller-manager`)
Deployment、ReplicaSet、StatefulSet、DaemonSet、Job、CronJob、HPA(水平扩缩)、Node、Endpoints/EndpointSlice、Namespace、ServiceAccount、PV/PVC binder、Garbage Collector……

### 知名第三方 Operators(按领域)
- **可观测**:prometheus-operator、Grafana operator、OpenTelemetry operator
- **证书/安全**:cert-manager、External Secrets
- **GitOps/交付**:ArgoCD、Flux
- **数据库/中间件**:CloudNativePG / Zalando / CrunchyData(Postgres)、Strimzi(Kafka)、Elastic ECK、Redis、MongoDB
- **服务网格/网关**:Istio、Cilium、Gateway API 实现
- **弹性/事件驱动**:KEDA、Knative
- **集群生命周期**:Cluster API
- **AI/GPU**:NVIDIA GPU Operator、KubeRay、Kueue、Volcano、KServe、**AIBrix**

### AIBrix 自己的 operators(`pkg/controller/*`)
| controller | 管的 CRD / 职责 |
|---|---|
| `modeladapter` | LoRA adapter 热挂载 |
| `podautoscaler` | LLM 专用扩缩(KPA/APA,非默认 HPA) |
| `kvcache` | 分布式 KV cache 编排 |
| `roleset` / `podset` / `stormservice` | 工作负载编排(prefill/decode 角色组) |
| `rayclusterfleet` / `rayclusterreplicaset` | Ray 分布式推理 |
| `modelrouter` | 路由配置 |

---

## 3. AIBrix 是什么,在栈里的位置

**一句话:在 K8s 上把 vLLM/SGLang 这些引擎"管起来、调起来、省着用"的云原生控制面 + 数据面。它不做推理内核,只编排现有引擎。**

八大能力:高密度 LoRA、**LLM 网关路由**、LLM 专用 Autoscaler、Unified AI Runtime(sidecar)、分布式推理、**分布式 KV Cache**、异构 GPU 省钱服务、GPU 故障检测。

与 **Dynamo(NVIDIA)** 的关系:
- **Dynamo = 引擎层的分布式服务框架(数据面运行时,Rust 内核)**;
- **AIBrix = 引擎之上的 K8s 管理面(Go operator + Envoy 数据面)**;
- 两者只在 **"LLM 感知路由 + PD 分离"** 这一块重叠——**而 #2289 恰好落在这块最像 Dynamo、最吃引擎知识的地方**。

---

## 4. PD 分离(Prefill–Decode Disaggregation)

### 为什么要分
LLM 推理两阶段负载特征相反:
- **Prefill(处理 prompt)**:一次性算完整个上下文,**算力密集(compute-bound)**,吃 GPU 算力。
- **Decode(逐 token 生成)**:一次一个 token,**访存密集(memory-bound)**,吃显存带宽 + KV cache。

放同一张卡上会互相干扰(prefill 的大 batch 拖慢 decode 的 TTFT/ITL)。**分离**后各自用最合适的资源/并行度独立扩缩。

### 两个角色(AIBrix 用 K8s label 区分)
| label | 值 | 含义 |
|---|---|---|
| `roleset-name` | 任意 | 把一组 prefill+decode 绑成一个 "roleset" |
| `role-name` | `prefill` / `decode` | Pod 的角色 |
| `model.aibrix.ai/engine` | `vllm` / `sglang` / `trtllm` | 引擎类型 |

一个 roleset **必须同时有 ≥1 prefill 和 ≥1 decode** 才 eligible。

### 关键:KV 从 prefill 传到 decode
prefill 算出 KV cache 后,要把它交给 decode pod 用。**怎么交** = 三种引擎的根本差异:

| 引擎 | prefill 调用方式 | KV 传递机制 |
|---|---|---|
| **vLLM** | **同步**:等 prefill HTTP 返回,从响应里取 `kv_transfer_params` | 响应里带 KV 传输参数,gateway 转给 decode |
| **TRT-LLM** | **同步**:取 `disaggregated_params` | 同上 |
| **SGLang** | **异步**:fire-and-return,**不等** | **bootstrap 握手**:prefill/decode 通过一个带外(out-of-band)的 `bootstrap_room` 交会点(rendezvous)直接对传 KV |

### 为什么 SGLang 必须异步(collaborator varungup90 的原话核心)
> "Prefill and Decode request must run asynchronously. Sync request to prefill will just hung."

SGLang 的 PD 用 **bootstrap 握手**:prefill 和 decode 各自带同一个 `bootstrap_room` 号,在带外通道**互相等着对接**。如果 gateway 同步等 prefill HTTP 返回,而 prefill 又在等 decode 来对接——**双方互等,死锁**。所以 gateway 必须"发了 prefill 就走,立刻去派 decode",让两者在带外自己会合。

**⇒ 结论:#2289 的正确解不能照 vLLM 改成同步。必须保持异步,另想办法感知/处理失败。**

---

## 5. AIBrix PD Router 的请求流(数据面,`pd_disaggregation.go`)

Gateway 是一个 **Envoy External Processing (ext_proc)** 服务——Envoy 把每个请求经 gRPC 交给它做路由决策。核心 `Route()`:

```
Client → Envoy → ext_proc Route(ctx, readyPods)
  ├─ ValidateAndGetLLMEngine()      // 确认这批 pod 同引擎 (vllm/sglang/trtllm)
  ├─ filterPrefillDecodePods()      // 按 roleset 分组 + prompt 长度分桶
  ├─ 打分选出 (prefillPod, decodePod)
  ├─ AddPendingDecode()             // 记账,防惊群
  ├─ doPrefillRequest(ctx, prefillPod, engine)
  │     ├─ SGLang → 异步 goroutine(bootstrap 握手;Route 不等它)   ← bug 在这
  │     ├─ vLLM   → 同步,取 kv_transfer_params
  │     └─ TRT-LLM→ 同步,取 disaggregated_params
  └─ SetTargetPod(decodePod) → 返回 decode 地址(Envoy 把请求转给 decode)
```

相关指标(`pd_readme.md` 官方表):
| Metric | 何时发 |
|---|---|
| `GatewayPrefillRequestSuccessTotal` | prefill HTTP 成功。**SGLang 是异步、在后台 prefill 完成后才发**,不是 Route() 返回时 |
| `GatewayPrefillRequestFailTotal` | 引擎校验失败、pod 过滤失败、**prefill HTTP 错误** |

---

## 6. #2289 到底哪里错了(含当前真实状态)

### 代码位置(已重构)
- 原 issue 指向 `pd_prefill_request.go` 里的 goroutine —— **该文件已重构**。
- 现逻辑在 **`pkg/plugins/gateway/algorithms/pd/prefill/default.go`** 的 `Execute()` → `IsAsync()` 分支。

### 现码全文(SGLang 异步分支)
```go
if handler.IsAsync() {
    // SGLang uses a bootstrap handshake to coordinate KV transfer out-of-band;
    // fire asynchronously and return immediately.
    ...
    go func() {
        defer e.tracker.RemovePrefillRequest(requestID)
        if _, err := e.executeHTTP(apiURL, asyncCtx, payload); err != nil {
            klog.ErrorS(err, "prefill_request_failed", ...)
            return                       // ← 失败:只 log,然后 return
        }
        metrics.EmitMetricToPrometheus(..., GatewayPrefillRequestSuccessTotal, 1.0,
            {"status_code": "200"})       // ← 成功才发,已在 return 之后
        ...
    }()
    return nil                            // ← Route 立刻返回,decode 照发
}
```

### 逐条对账:issue 说的 vs 现码真相
| issue 原文声称 | 现码真相 |
|---|---|
| ❌ "无条件 emit `GatewayPrefillRequestSuccessTotal`" | ✅ **已修**:error 分支 `return` 在 Emit 之前,成功才发 |
| ✅ "失败被吞,只 log" | **仍是**:error 分支只 `klog.ErrorS` + `return` |
| ✅ "decode 照发 → 挂到 bootstrap 超时" | **仍是**:`return nil` 立即返回,decode 无条件派发 |

### 还没修、可做的两块
1. **【小、无争议】异步失败不发 fail 指标**
   `pd_readme.md` 明确写 `GatewayPrefillRequestFailTotal` 应在 "prefill HTTP error" 时发,但 **async 分支的 error 只 log、没发这个指标** → 与 sync 路径不对称、与文档不符。补一行 Emit 即可。**适合当第一个 PR。**
2. **【大、需与维护者对设计】失败不 fail 请求 → 仍挂死**
   核心功能 bug。保持异步的前提下的可选解:
   - 用 bootstrap ack / 短超时探测**门控 decode 派发**;
   - 失败**换一个 prefill pod 重试**;
   - 给客户端一个**快速 typed error**,而不是等 SGLang bootstrap 超时。
   这块要先在 issue 里和 varungup90 对齐方案(他强调"不能改同步")。

---

## 7. 建议的落地路径
1. **先在 issue 留言认领 + 亮方案**:开头点明"保持异步不动",直接回应"改同步会 hang"那条,证明读懂了 bootstrap 机制。同时确认维护者认这个 bug(issue 挂 `kind/bug`,但那条 "expected behavior" 评论有一点 wontfix 味)。
2. **PR-1(小):补 async 失败的 `GatewayPrefillRequestFailTotal`**,对齐文档与 sync 路径。低风险、快合、建立信任。
3. **PR-2(大):decode 门控 / 重试 / 快速 typed error**,借 PR-1 的信任推设计。

---

## 8. 术语表
- **Prefill / Decode**:LLM 推理两阶段(处理 prompt / 逐 token 生成),负载特征相反。
- **PD 分离**:把两阶段跑在不同 pod/卡上,各自扩缩。
- **KV cache**:注意力的 key/value 缓存;prefill 产出、decode 复用。
- **bootstrap_room / 握手**:SGLang PD 里 prefill 与 decode 带外对接 KV 的交会号/机制。
- **roleset**:AIBrix 里一组绑定的 prefill+decode 副本。
- **ext_proc**:Envoy External Processing,把路由决策外包给一个 gRPC 服务(即 AIBrix gateway)。
- **reconcile loop**:operator 的核心——持续把实际状态收敛到期望状态。
- **控制面/数据面**:管期望状态(不过流量)/ 承载实际流量。

## 9. 一手资料
- Issue: `vllm-project/aibrix#2289`
- 官方 PD 文档: `pkg/plugins/gateway/algorithms/pd_readme.md`
- Bug 代码: `pkg/plugins/gateway/algorithms/pd/prefill/default.go` → `Execute()`
- 路由主逻辑: `pkg/plugins/gateway/algorithms/pd_disaggregation.go`
- AIBrix 白皮书: arxiv 2504.03648
