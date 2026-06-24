# 06 · Qwen3.6-35B-A3B-FP8 单卡 H20 Serving Profiling 全程实录

> 本篇是一份**手把手的过程记录**,目标读者是想完整理解"从一台空机器到一份可分析的 serving trace"端到端链路的人。
> 每一步都尽量说清三件事:**做了什么 / 为什么这么做 / 结果如何**。
> 研究对象与上一轮(00–05,Qwen3-8B / bf16)不同,这次是 **Qwen3.6-35B-A3B-FP8** —— 一个**混合线性注意力 + MoE + FP8** 的多模态模型,单卡 H20。
>
> 日期:2026-06-18。机器:AutoDL 单卡 H20(96GB)。

---

## 〇、为什么做这个研究

上一轮(Qwen3-8B)的结论是:**SGLang 的热路径已经被优化得很充分,没有"躺在地上的一行 quick-win PR"**。doc 05 给出的出路是:

> 换 **FP8 / MoE / spec-decode** 等覆盖更少的场景重新 profile。

本轮就是沿着这条出路走的,而且一次性踩满三个"覆盖更少"的点:

1. **FP8**(block-wise e4m3)—— 量化路径(deep_gemm、per-token 量化)在 bf16 研究里完全没被触及。
2. **MoE**(256 专家 top-8)—— 专家路由、fused_moe kernel。
3. **混合线性注意力**(Gated Delta Net + 每 4 层一个 full attention)—— 这是 2026 年最前沿、且公网几乎没人 profile 过的架构。

附带的战略目的:**第一次真正在 GPU 上端到端跑通"起服务 → 压测 → 抓 trace → 分析"全链路**,把"我懂可观测性"从软主张变成有实测数据的硬证据。

---

## 一、环境与机器

### 1.1 硬件 / 计费模型(AutoDL)

| 项 | 值 |
|---|---|
| GPU | 1× NVIDIA **H20**,97871 MiB(~96GB),driver 580.65.06 |
| 系统盘 `/`(overlay) | 30G(仅 ~21G 可用)—— **放不下大模型** |
| 数据盘 `/root/autodl-tmp`(`/dev/vdb`) | **50G** —— 模型必须放这里 |
| 公共盘 `/autodl-pub/data` | 10T 但**只读**,不能写个人数据 |

**关键的计费认知(决定了后面所有省钱操作)**:

- AutoDL 按 **GPU-实例-小时**计费(H20 ~8 元/卡/h),**只要实例处于"有卡模式",哪怕 GPU 一直空闲也照扣**。
- AutoDL 提供 **「无卡模式开机」(~¥0.1/h)**:不占 GPU,但 CPU/网络/磁盘全在 —— **专门用来下载、装环境**。
- 数据盘 `autodl-tmp` 在关机/释放后**持久化**;系统盘也随实例保留。

👉 **铁律:下载、解压、写脚本这种 I/O 活,全在无卡模式做;只有"真要用 GPU"时才切有卡模式。**

### 1.2 软件栈(镜像里已装好)

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.3 | 在 `/root/miniconda3`,**非交互 SSH 不自动激活 conda**(坑见 §3) |
| PyTorch | **2.11.0+cu130** | CUDA **13.0** |
| SGLang | **0.5.13.post1** | pip 安装,无源码 checkout |
| ModelScope | 1.37.1 | 国内下模型用 |

> 注意 torch 是 **cu130(CUDA 13)**,而系统里同时有 CUDA 12.8 —— 这个"双 CUDA"是后面 deep_gemm 报错的根源(§3.1)。

---

## 二、模型选型与下载

### 2.1 为什么是 Qwen3.6-35B-A3B-FP8

选型经过了几轮筛(详见对话史),最终标准与结论:

| 要求 | 为什么 | 结论 |
|---|---|---|
| 够新 | 文章要有时效性,Qwen2.5 太老 | Qwen3.6(`model_type=qwen3_5_moe`) |
| 单卡能跑 TP=1 | 需要干净的基线原点 | **FP8 版 ~35.8G**,单卡 96G 装得下(bf16 版 84.6G 装不下) |
| MoE | 带出 EP 维度 + 贴合面试热点 | 256 专家 top-8 |
| 混合线性注意力 | 公网无人 profile + 直击面试缺口 | 3 线性 + 1 full(`full_attention_interval=4`),线性层是 Gated Delta Net |

**核实到的架构事实(来自 `config.json`)**:

```
architectures: ["Qwen3_5MoeForConditionalGeneration"]   # 多模态(含视觉塔)
text_config: 40 层, 16 attn heads / 2 KV heads, hidden 2048
            256 专家, 每 token 激活 8 个, max_position 262144 (256K)
            attention: hybrid, 3 linear + 1 full 交替
quantization_config: fp8, e4m3, block [128,128], activation dynamic
            modules_to_not_convert: 视觉塔(27 层)保持 bf16, 不量化
```

> ⚠️ **它是多模态模型**(有 `vision_config` + 视觉塔)。我们只做**文本** serving,视觉塔会被加载但不被触发 —— 白占一两 GB,无害。若要"纯文本研究"更干净,可换纯文本的混合线性 MoE(如 `qwen3_next` 系),但当时没找到去掉视觉塔的同名变体。

### 2.2 确认 SGLang 支持(下载前必做)

混合线性注意力模型在推理框架里的支持经常滞后,所以**花钱前先在本地源码确认**:

- `python/sglang/srt/models/qwen3_5.py` 注册了 `Qwen3_5MoeForConditionalGeneration`(EntryClass);
- 真实现了混合注意力:`Qwen3_5GatedDeltaNet` + `RadixLinearAttention` + 按 `full_attention_interval` 切层;
- server_args 里有针对它的 "mamba radix cache" 处理(线性层的状态缓存);
- 还自带 `qwen3_5_mtp.py`(MTP 投机解码)。

结论:**SGLang 0.5.13 原生支持,可以放心下。**

### 2.3 下载(无卡模式 + 数据盘)

**问题**:数据盘 50G,当时已用 16G(上一轮 Qwen3-8B 的 HF 缓存),只剩 35G,而模型要 35.8G —— **差一点点放不下**。

**处理**:删掉用不到的 Qwen3-8B 缓存(可逆,随时重下;上一轮成果已存 git),腾出 16G。

```bash
rm -rf /root/autodl-tmp/hf/hub/models--Qwen--Qwen3-8B \
       /root/autodl-tmp/hf/hub/.locks/models--Qwen--Qwen3-8B
# 数据盘从 35G 可用 → 50G 全空
```

**下载命令**(后台 + 国内 CDN + 断点续传):

```bash
pip install -U modelscope
nohup modelscope download \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --local_dir /root/autodl-tmp/Qwen3.6-35B-A3B-FP8 \
  > /root/autodl-tmp/download_qwen36.log 2>&1 &
```

**完整性校验**(下完必做,别假设它对):

```python
import json, glob, os
idx = json.load(open("model.safetensors.index.json"))
files = set(idx["weight_map"].values())
have  = {os.path.basename(p) for p in glob.glob("*.safetensors")}
print("缺失:", files - have)         # → NONE
print("零字节:", [f for f in have if os.path.getsize(f)==0])  # → NONE
```

结果:**42 个 safetensors 全到、无缺失、无零字节**,外加 `mtp.safetensors`(投机解码)、`outside.safetensors`。共 ~35G。

> **省钱复盘**:整个下载在无卡模式(~¥0.1/h)进行,花费可忽略;且一次性下完,后续所有 run(单卡/多卡)复用同一份。

---

## 三、启动服务 + Smoke Test(踩了三个真实环境坑)

下载完 **关机 → 有卡开机(1 卡)**。Smoke test 的意义:**用最小代价(1 卡、跑几个 token)验证"模型能不能加载、依赖齐不齐",绝不在多卡计费时才发现问题。**

事实证明这一步极其值得 —— 连踩三个坑,每个都得修。

### 3.1 坑一:deep_gemm 的 `libnvrtc.so.13` 缺失

第一次启动直接崩:

```
RuntimeError: Failed to load .../deep_gemm/_C.so
libnvrtc.so.13: cannot open shared object file
```

**根因**:`deep_gemm`(FP8 GEMM 后端)按 **CUDA 13** 编译,需要 `libnvrtc.so.13`;而它不在动态库搜索路径上。上一轮 Qwen3-8B 是 **bf16**,从不触发 deep_gemm,所以这个问题一直潜伏,**FP8 模型是第一个把它激活的**。

更隐蔽的一点:`configurer.py` 里 `_compute_enable_deep_gemm()` 只 `except ImportError`,**没接住 `.so` 加载的 `RuntimeError`** → 不会优雅降级,直接崩。所以即使设 `SGLANG_ENABLE_JIT_DEEPGEMM=0` 也没用(崩在那个 env 检查之前)。

**修复**:`libnvrtc.so.13` 其实存在于 pip 包里,只是没在路径上。把它加进 `LD_LIBRARY_PATH`(同时**保住 FP8 DeepGEMM 可用**,这正是我们要的):

```bash
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

> 💡 **这构成一个可提交的 SGLang PR(robustness)**。`configurer.py` 有两个逻辑缺陷:
> 1. **opt-out 顺序 bug**:`SGLANG_ENABLE_JIT_DEEPGEMM`(默认 True)在 `import deep_gemm` **之后**才检查,所以 `=0` 无法规避导入崩溃;
> 2. **降级不彻底**:只 `except ImportError`,接不住 `.so` 加载的 `RuntimeError`/`OSError` → 一个 CUDA 版本错配的 deep_gemm 会让**所有**(包括不用 FP8 的)启动崩溃,且报不可读的 traceback。
>
> 建议改法:**先查 env opt-out(在 import 前)→ 再 `except Exception` + warn + 优雅降级**。提 PR 前需在 GitHub 去重(搜 `deep_gemm` + `ImportError`/`libnvrtc`),并附"cu13 + FP8 模型"的复现。

### 3.2 坑二:torch 2.11 inductor "duplicate template name"

修了坑一,又崩在:

```
torch/_inductor/kernel/flex_attention.py ... TritonTemplate ...
AssertionError: duplicate template name
```

**根因**:torch 2.11 的 inductor 在注册 flex_attention 模板时的一个重复注册 bug,被 SGLang 的 piecewise cuda graph(走 torch.compile)路径触发。

**修复**(上一轮 doc 01 已记录的 "torch2.11 workaround"):

```bash
--disable-piecewise-cuda-graph      # 关 piecewise graph(常规 cuda graph 仍开)
export TORCHDYNAMO_DISABLE=1
```

### 3.3 坑三:bench_serving 拉不到 tokenizer / ShareGPT

服务起来后,压测脚本又崩 —— 它要从 HuggingFace 下东西:

- 不设 HF 环境 → 连真 `huggingface.co`(墙)→ httpx client closed;
- 设 `HF_HUB_OFFLINE=1` → 又下不了 `random` 数据集要的 **ShareGPT 语料**。

**根因**:这个版本的 `random` 数据集会**采样 ShareGPT 真实文本**再截断成目标长度(不是纯随机 token id),所以需要联网拿 ShareGPT json。

**修复**(匹配 `profile_serving_scheduler.sh` 的配置):用国内镜像、**不开 offline**、tokenizer 指向本地路径:

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 国内可达
# 不设 HF_HUB_OFFLINE
python -m sglang.bench_serving ... --tokenizer /root/autodl-tmp/Qwen3.6-35B-A3B-FP8
```

### 3.4 顺带学到的:三种"就绪信号"不能混用

启动等待时,我一开始用 `/health` 判断就绪 → **3 秒就返回 ready**,但模型其实还在加载。原因:

| 信号 | 含义 | 能否当"就绪" |
|---|---|---|
| `/health` | HTTP server 起来了(liveness) | ❌ 太早 |
| `/v1/models` 返回模型 | 模型真加载完(readiness) | ✅ |
| 日志 `The server is fired up and ready to roll!` | 最确定的就绪线 | ✅ 最稳 |

👉 教训:**等服务就绪要看 `/v1/models` 或日志的 "fired up",别用 `/health`。**

### 3.5 最终启动命令 + Smoke 结果

```bash
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export TORCHDYNAMO_DISABLE=1
python -m sglang.launch_server \
  --model-path /root/autodl-tmp/Qwen3.6-35B-A3B-FP8 \
  --tp 1 --trust-remote-code \           # 这个模型必须 trust-remote-code
  --watchdog-timeout 1200 \              # 大模型加载慢,调大
  --mem-fraction-static 0.85 \
  --context-length 8192 \
  --disable-piecewise-cuda-graph \       # torch2.11 workaround
  --host 127.0.0.1 --port 30000
```

**启动日志里的关键数据(本身就是文章素材)**:

| 阶段 | 数值 | 解读 |
|---|---|---|
| 权重加载 | **179.4s**,FP8 e4m3,**34.19 GB** | 加载完 avail 60.42 GB |
| **KV Cache 分配** | **1,275,284 tokens**(K 12.16GB + V 12.16GB) | 见下,核心发现 |
| CUDA graph capture | 166.4s,bs 覆盖 [1…121] | 之后 avail 13.7 GB |
| 就绪 | `The server is fired up and ready to roll!` | |
| 生成测试 | 连贯中文输出(带 `<think>`,是推理模型) | e2e 4.59s / 33 token |

🌟 **第一个独家发现 —— KV 容量 127 万 token**:
标准 35B Transformer 单卡 KV 撑死几万 token;而这个**混合线性架构只有 1/4 的层(full attention)持有随序列增长的 KV**,其余 3/4 是线性层(常数状态),所以单卡 H20 直接给到 **~127 万 token** 的 KV 容量。这就把"**混合线性 + 大显存 H20 = 超长上下文/超大 batch 的理想组合**"用实测坐实了。

---

## 四、实验:单卡 Serving Profiling

### 4.1 方法与参数(为什么这么设)

复用上一轮的 `profile_serving_scheduler.sh` 同款配置 —— 这样**数据格式与上一轮可比**:

```bash
python -m sglang.bench_serving --backend sglang \
  --model /root/autodl-tmp/Qwen3.6-35B-A3B-FP8 --tokenizer <同上> \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 300 --max-concurrency 32 --warmup-requests 16 \
  --profile --profile-steps 20 \
  --profile-output-dir <PROFILE_DIR> --profile-prefix serving
```

| 参数 | 为什么 |
|---|---|
| `random` in512/out128 | 中等长度,prefill 与 decode 都有量,贴近通用对话 |
| 300 prompts / 并发 32 | 让 scheduler 持续批处理,decode 阶段保持饱和(decode 是单卡瓶颈相) |
| `--profile --profile-steps 20` | 通过 `/start_profile`+`/stop_profile` 抓 **scheduler(TP-0)进程**稳态 20 步的 torch profiler trace —— 这是 `bench_one_batch` 抓不到的"真实 serving 路径"(含 scheduler / tokenizer host 开销) |

> 服务端必须用 `SGLANG_TORCH_PROFILER_DIR` 环境变量启动,`/start_profile` 才知道往哪写 trace。

### 4.2 Profiling 方法论详解(5W1H)—— 这次到底"怎么 profile"的

这一节专门讲清:**用了哪种 profiling 方法、抓到哪些数据、每类数据能说明什么问题**,以及——同样重要——**这套方法回答不了什么**。

#### Why · 为什么要 profile

bench_serving 的端到端指标(吞吐/TTFT/TPOT)只告诉你"**快不快**",不告诉你"**慢在哪**"。Profiling 的目的就是把"慢"拆开,回答三个层层递进的问题:

1. 时间花在**哪个阶段**(prefill vs decode)?
2. 花在**哪个 kernel**(MoE? 注意力? GEMM? 量化?)?
3. 是 **GPU 在算**,还是 **GPU 在等 CPU**(host-bound,trace 里表现为"气泡")?

#### What · 用什么方法 / 抓什么数据

本轮用的是 **PyTorch Profiler(CPU + GPU activities)**,经 SGLang 的 `/start_profile` + `/stop_profile` 接口在**真实运行的 server** 上触发。对比另一条入口:

| 入口 | 命令 | 抓什么 | 何时用 |
|---|---|---|---|
| **在线 serving**(本轮用) | `bench_serving --profile` 驱动 `/start_profile` | 真实 serving 路径:**scheduler + tokenizer 的 host 开销** + 模型前向 | 想看真实服务下的 host/调度瓶颈 |
| 离线 microbench(上一轮 doc 02) | `bench_one_batch --profile` | **纯模型前向**,确定性,干净 | 学读 trace、定位前向 kernel |

产出的数据有三层,从粗到细:

| 数据 | 形态 | 回答什么 |
|---|---|---|
| **trace.json.gz** | Chrome/Perfetto timeline(泳道) | 肉眼看"CPU 发起 → GPU 执行"的因果、气泡、同步点 |
| **Kernel 表** | 每个 GPU kernel 的时间占比 + 映射回 Python 行 + CPU op | "时间花在哪个 kernel"——本轮核心数据 |
| **CPU 调用树 / host 算子** | scheduler 的 Python inclusive 时间、`aten::*` 算子 | host 侧在干什么(调度、拷贝、sync) |

#### Where · 在哪个进程 / 位置抓

抓的是 **scheduler 进程(TP-0)** 的 trace(文件名 `...TP-0.trace.json.gz`)。
为什么是它:SGLang 的调度循环、tokenizer、内存管理、以及模型前向都在这个进程里,**这是真实 serving 的热路径**——`bench_one_batch` 那种纯前向 microbench 看不到调度/host 部分。trace 落在 `SGLANG_TORCH_PROFILER_DIR` 指定的目录。

#### When · 什么时候抓 / 抓多久

- **warmup 16 个请求之后**才开始 —— 跳过冷启动(JIT 编译、cache 预热),抓**稳态**。
- 抓 **20 个 scheduler 步**(`--profile-steps 20`)。够看清稳态结构,又不会让 trace 大到难分析。
- 这个窗口以 **decode 为主**(并发 32 持续批处理),所以 Kernel 表的 "decode" 分组最能代表瓶颈相。
- ⚠️ **局限**:20 步的窗口**大概率没覆盖到那次 18.6s 的 ITL 尾部停顿**(它是偶发的)。要抓尾部,得用更长窗口 + 专项触发 —— 这是 profiling "采样窗口"的固有取舍。

#### Who · 谁触发

`bench_serving` 作为**客户端**,在压测进行中通过 HTTP 调 server 的 `/start_profile`(带 `output_dir`、`num_steps`、`activities=[CPU,GPU]`)→ 跑若干步 → `/stop_profile`。即"**压测驱动 profiler**",保证抓的是有真实负载时的状态。

#### How · 怎么读这些数据

1. **Perfetto 肉眼看**(trace 拖进 https://ui.perfetto.dev):上排 = CPU 线程,下排 = CUDA stream;找**气泡**(GPU 空隙 = host-bound)和**同步点**(`cudaStreamSynchronize`、`cudaMemcpyAsync DtoH`)。
2. **自动三表**(`analyze_llm_torch_profile.py`):Kernel 表(谁占比高)、Overlap 表(可隐藏的气泡)、Fuse 表(可合并的 kernel 簇)。
3. **eager vs cuda graph 双 trace**(doc 02 的关键技巧):开图的 trace 反映真实性能但**源码归因差**(所有 kernel 都指向 `model_runner.py:3502` 的 `cudaGraphLaunch`);关图(`--disable-cuda-graph`)的 trace 慢但**能把 kernel 精确映射回 Python 行**。本轮是开图 trace,所以 Kernel 表里 CPU op 多为 `cudaGraphLaunch`、location 多指向 `_forward_raw` —— 想精确归因到某行,需要再抓一份关图的 mapping trace。

#### 这套方法能 / 不能回答什么(诚实边界)

| ✅ 能回答 | ❌ 不能回答(本轮没数据) |
|---|---|
| 时间花在哪个 kernel、占比多少 | **SM 利用率 / HBM 带宽利用率**(撞算力墙还是带宽墙) |
| prefill vs decode 各用哪些 kernel | **NVLink 流量**(多卡 TP 通信开销) |
| host vs device、有没有气泡 | 功耗、显存带宽是否打满 |
| kernel → Python 调用链(关图时) | 跨进程/跨卡的 timeline 对齐 |

👉 **关键认知**:torch profiler 给的是"**kernel 时间账本 + host 调用树**",**不是硬件计数器**。要回答"H20 上 decode 到底撞没撞 HBM 带宽墙""TP=4 的 all-reduce 吃了多少带宽",必须上 **nsys / DCGM / `nvidia-smi dmon`**——这些留到多卡阶段补(见 §七)。本轮的 kernel 占比结论(MoE 占 decode 46.7% 等)是扎实的;但"为什么慢=撞哪面墙"还需硬件层数据才能下定论。

### 4.3 采集到的数据(已拉回本地)

| 文件 | 内容 |
|---|---|
| `.contribution-scan/traces/qwen36_serving_TP0.trace.json.gz`(21MB) | scheduler trace,拖进 https://ui.perfetto.dev 看泳道/气泡 |
| `.contribution-scan/analysis/qwen36_serving_triage.txt` | `llm-torch-profiler-analysis` skill 出的三表 |
| `.contribution-scan/analysis/qwen36_serving_server.log` | 启动日志(KV 容量、加载耗时) |

分析命令(在**本地**跑,trace 已 scp 回来):

```bash
python3 .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --input .contribution-scan/traces/qwen36_serving_TP0.trace.json.gz
```

---

## 五、结果

### 5.1 bench_serving 端到端指标(对照上一轮 Qwen3-8B)

| 指标 | Qwen3-8B(bf16,doc 05) | **Qwen3.6-35B-A3B-FP8(本次)** |
|---|---|---|
| 吞吐 req/s | 16.3 | **4.51** |
| 输出吞吐 tok/s | 1067 | **296** |
| 总吞吐 tok/s | 5233 | 1451 |
| TTFT 中位 / 均值 / P99 (ms) | 80 / 193 / 1105 | 193 / 502 / 2059 |
| TPOT 中位 / 均值 / P99 (ms) | —— | 74.6 / 112 / 907 |
| ITL 中位 / 均值 / Max (ms) | 8.05 / 26.7 / 3356 | **14.0 / 100.9 / 18598** |

(本次:300 请求,66.5s,峰值并发 43,实际并发 31.66。)

### 5.2 Kernel 画像 —— decode 阶段(并发 32 的瓶颈相)

| Kernel | 类别 | 占比 | launches | 含义 |
|---|---|---:|---:|---|
| `fused_moe_kernel` | moe | **46.7%** | 1216 | MoE 专家计算,decode 头号开销 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | hybrid_linear | **10.7%** | 450 | GDN 线性注意力的 **decode 递归形式** |
| `deep_gemm::sm90_fp8_gemm...`(多条) | gemm | ~15% | | FP8 矩阵乘(靠 §3.1 修复才用得上) |
| `act_and_mul_kernel` | activation | 2.1% | 1216 | SwiGLU 激活 |
| `per_token_group_quant_8bit`(多条) | quantize | ~3% | | **FP8 动态激活量化**开销 |
| `topkGatingSoftmax` | softmax | 1.2% | 608 | MoE 门控 |
| `_causal_conv1d_update_kernel` | hybrid_linear | 1.2% | 450 | GDN 里的因果卷积 |

### 5.3 Kernel 画像 —— all(含 prefill)

| Kernel | 类别 | 占比 | 含义 |
|---|---|---:|---|
| `deep_gemm::sm90_fp8_gemm...`(大 tile) | gemm | **34.6%** | prefill 的大 FP8 矩阵乘 |
| `fused_moe_kernel` | moe | **28.3%** | MoE 专家 |
| `chunk_gated_delta_rule_fwd_*`(kkt_solve / h_blockdim64 / fwd_o / recompute_w_u) | hybrid_linear | ~11% 合计 | GDN 线性注意力的 **prefill 分块并行形式** |
| `_causal_conv1d_fwd_kernel` | hybrid_linear | 2.0% | GDN 卷积(prefill) |
| `index_put` / `index`(`mem_cache/memory_pool.py:396 copy_from`) | elementwise | ~4% 合计 | KV cache 写入路径(呼应 doc 05 的 `write_cache_indices` 线索) |
| `fused_qkvzba_split_reshape_cat` | other | 1.7% | GDN 的 QKV 投影融合 |

### 5.4 Overlap / Fuse 表

- **Overlap 表:空**(单 trace 模式偏保守,需双 trace 才能强归因 —— 与 doc 02 一致)。
- **Fuse 表:有假阳性**,例如把 GDN kernel(`chunk_gated_delta_rule`、`fused_recurrent_gated_delta_rule`)归到 "Fused MoE grouped-topk" 下 —— 明显错配。
  👉 **教训重申(doc 02)**:工具给的 "Confirmed" 必须回看 kernel 名 / dtype / 调用链验证后再信。

---

## 六、分析结论

1. **混合线性注意力的 "prefill/decode 二象性" 被实测看见了** —— 这是本轮最有价值的发现:
   - **prefill** 走 `chunk_gated_delta_rule_*`(**分块并行**形式,~11%),适合 compute-bound 的 prefill;
   - **decode** 走 `fused_recurrent_gated_delta_rule`(**O(1) 递归**形式,10.7%),适合逐 token 的 decode。
   同一个线性注意力层,两个阶段用**完全不同的 kernel** —— 教科书结论 + 自己的 kernel 数据。

2. **MoE FFN 主导 decode(46.7%),不是注意力** —— 与"注意力最贵"的直觉相反。即使只激活 3B 参数,`fused_moe_kernel` 仍是 decode 的最大头。

3. **FP8 的 trade-off 一正一反都有数据**:
   - 收益侧:权重 34GB(单卡装下)+ KV 127 万 token 容量;
   - 代价侧:`per_token_group_quant_8bit` 反复出现(动态量化激活的固定开销),且 deep_gemm 依赖正确的 CUDA 13 运行时(§3.1)。

4. **ITL 尾部爆炸(Max 18.6s,均值 100 >> 中位 14)** —— 比 Qwen3-8B 的 3.3s 尾巴严重得多。是 MoE 路由不均?线性状态缓存?调度?GC?**这是一个值得用更长 profiler 窗口 + 专项复现的深挖方向。**

5. **KV cache 写入路径**(`memory_pool.py copy_from` 的 index_put,~4%)再次浮现 —— 与 doc 05 指向的 `write_cache_indices` 同源,可与上游 #24734 联动关注。

---

## 七、未做 / 下一步(多卡矩阵)

本轮只做了**单卡 TP=1** 的 serving profiling。计划中的多卡实验(待有卡开机 4–5 卡):

| 实验 | 目的 | 预期结论 |
|---|---|---|
| **TP 扫描 1/2/4** | 量化 TP 通信开销、scaling 效率、NVLink 利用率 | H20 算力弱,TP 对 prefill 帮助 > decode |
| **EP 扫描 2/4** vs TP | MoE 专家并行 all-to-all vs TP all-reduce | 公网无人测,文章第二护城河 |
| **prefill vs decode 墙**(短/长两档负载 + dmon 抓 SM/HBM 利用率) | 证明 H20 prefill 撞算力墙、decode 撞带宽墙 | 题眼 |
| **长上下文 KV scaling**(4K→256K) | 混合线性的亚线性 KV 增长 | 全文最强的图 |
| **bf16 vs FP8 对照** | 需再下 84GB bf16 版(无卡模式) | FP8 在 H20 上对 prefill 提升不成比例 |
| **ITL 尾部专项** | 复现 18.6s 停顿根因 | 单独一篇 |

---

## 八、复现 Runbook(命令速查)

```bash
# ── 0. 无卡模式:下载模型到数据盘 ──
pip install -U modelscope
modelscope download --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --local_dir /root/autodl-tmp/Qwen3.6-35B-A3B-FP8

# ── 1. 有卡开机后,设环境(三个坑的修复) ──
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH  # 坑一
export TORCHDYNAMO_DISABLE=1                                                                            # 坑二
export HF_ENDPOINT=https://hf-mirror.com                                                               # 坑三

# ── 2. 起服务(带 profiler dir) ──
export SGLANG_TORCH_PROFILER_DIR=/root/autodl-tmp/profile_serving_qwen36
nohup python -m sglang.launch_server \
  --model-path /root/autodl-tmp/Qwen3.6-35B-A3B-FP8 --tp 1 --trust-remote-code \
  --watchdog-timeout 1200 --mem-fraction-static 0.85 --context-length 8192 \
  --disable-piecewise-cuda-graph --host 127.0.0.1 --port 30000 \
  > /root/autodl-tmp/serving_server.log 2>&1 &
# 等就绪:grep "fired up" 日志,别用 /health

# ── 3. 压测 + 抓 trace ──
python -m sglang.bench_serving --backend sglang \
  --model /root/autodl-tmp/Qwen3.6-35B-A3B-FP8 --tokenizer /root/autodl-tmp/Qwen3.6-35B-A3B-FP8 \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 300 --max-concurrency 32 --warmup-requests 16 \
  --profile --profile-steps 20 \
  --profile-output-dir "$SGLANG_TORCH_PROFILER_DIR" --profile-prefix serving

# ── 4. 拉回本地 + 分析 ──
# scp trace 回本地后:
python3 .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --input <trace.json.gz>

# ── 5. 用完立刻去 AutoDL 控制台:关机 / 切无卡模式(停止计费!) ──
```

---

## 九、一句话总结

> 用单卡 H20 + FP8 混合线性 MoE,**端到端跑通了"环境→下载→起服务→压测→抓 trace→分析"全链路**,踩平三个真实环境坑,并拿到两个公网稀缺的实测结论:**(1) 混合线性注意力 prefill/decode 用两套 kernel;(2) MoE FFN 而非注意力主导 decode**。同时坐实了"混合线性 + 大显存 H20 = 超长上下文(单卡 127 万 token KV)"的价值。多卡矩阵留待下一轮。
</content>
</invoke>
