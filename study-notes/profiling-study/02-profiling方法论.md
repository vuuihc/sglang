# 02 · Profiling 方法论

## 先建立心智模型:prefill vs decode

LLM 推理两个阶段,瓶颈完全不同:

- **Prefill / 大 batch decode** → **compute-bound**,GPU 算力打满,trace 里 kernel 排得密不透风。
- **小 batch decode**(batch=1~8)→ **memory-bandwidth + launch-overhead bound**,GPU 利用率常 <50%,trace 里全是**气泡(GPU 空等)**。

👉 sglang 那些小而美的 perf PR,大多在小 batch decode 的气泡里 —— GPU 在等 CPU(Python scheduler、kernel launch、D2H sync)。学会在 trace 里认出气泡是核心技能。

---

## 抓 trace:两种入口

### A. 离线 microbench(本次用的,最干净)
`sglang.bench_one_batch` 是**纯模型前向**的确定性 microbench,适合学读 trace:
```bash
export SGLANG_TORCH_PROFILER_DIR=/path/to/out
python -m sglang.bench_one_batch \
  --model-path Qwen/Qwen3-8B \
  --batch-size 1 8 32 64 \      # 一条命令扫 batch,看 host→compute 转折
  --input-len 512 --output-len 64 \
  --profile \
  --profile-stage decode \      # 只抓 decode 阶段(气泡在这);可选 prefill/all
  --profile-steps 5 \           # 抓稳态的几步(默认从 output_len//2 开始,跳过 warmup)
  --disable-piecewise-cuda-graph   # torch2.11 workaround
```
输出:`<prefix>_batch<B>_input<I>_output<O>.trace.json.gz`,每个 (batch,input,output) 一份。

> 封装好的脚本:`scripts/profile_decode_bottlenecks.sh`,参数化 `MODEL/BATCHES/SPEC/EXTRA_ARGS`。
> 关 cuda graph:`EXTRA_ARGS="--disable-cuda-graph --disable-piecewise-cuda-graph"`。

### B. 在线 serving(下一步要用的)
起真实 server + `/start_profile`+`/stop_profile`,能抓到 **scheduler/tokenizer** 的 host 开销 —— microbench 抓不到的部分(见 04 文档)。

---

## 读 trace:肉眼看 Perfetto

把 `.trace.json.gz` 拖进 https://ui.perfetto.dev。三件事:

1. **泳道结构**:上 = CPU 线程(Python 发起),下 = CUDA stream(GPU 执行)。对齐看"CPU 发起 → GPU 执行"的因果。
2. **气泡(GPU 空隙)**:两个 kernel 之间 GPU 没事干 = host-bound。气泡越宽,host 开销占比越大 = 越肥的优化点。
3. **同步点**:找 `cudaStreamSynchronize`、`cudaMemcpyAsync DtoH` —— 这些是 per-step D2H sync(#28397 那类)的现场。

---

## 读 trace:自动三表(本仓库 skill)

`llm-torch-profiler-analysis` skill 把肉眼活儿自动化:
```bash
python3 .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --input <trace.json.gz>
```
吐三张表:

| 表 | 含义 | 怎么用 |
|---|---|---|
| **Kernel 表** | GPU 时间花在哪,每个 kernel 映射回 Python 行 + CPU op | 看谁占比最高(本次:nvjet GEMM 占 82%) |
| **Overlap 表** | GPU 本可靠重叠隐藏的气泡 | 单 trace 模式偏保守,常为空;要强归因得用双 trace(mapping+formal) |
| **Fuse 表** | 可合并成一个的 kernel 簇,对照已知/在途融合 PR | 找可融合点,**但要警惕假阳性(见下)** |

默认只显示占比 ≥1% 的行。

---

## 关键技巧:eager vs cuda graph,两份 trace 各有用途

这是本次最实用的发现:

| | cuda graph **开** | cuda graph **关(eager)** |
|---|---|---|
| trace 大小 | 小(~140KB,一步一个 `cudaGraphLaunch`) | 大(~1.9MB,一步几百个 kernel launch) |
| 源码归因 | **差**:所有 kernel 都指向 `model_runner.py:3502`,CPU op 是 `cudaGraphLaunch` | **好**:GEMM 精确指到 `unquant.py:153 apply`,CPU op 是 `aten::mm` |
| 反映真实性能 | **是**(生产就是开图的) | 否(eager 比真实慢,但暴露 host 开销) |

👉 **想知道"某 kernel 是哪行 Python 发的" → 抓一份 `--disable-cuda-graph` 的 mapping trace;想看真实性能 → 用 graph 版。**
这正是 skill 文档里"双 trace(mapping + formal)"的用法。

---

## 警惕:别轻信 skill 的"Confirmed"(假阳性教训)

本次 fuse 表把 81.5% 标成 "Confirmed" 的 `PR #22392 CUTLASS FP8 scaled MM replacing nvjet`。
但我们跑的是 **bf16,不是 FP8**(kernel 名里是 `__nv_bfloat16`)。skill 是按 *nvjet 家族名*匹配的,不看 dtype → **假阳性**。

教训(与 find-contribution-targets 的"agent 过度自信"一致):**任何工具给的高置信结论,都要回看 kernel dtype / 调用链 / git blame 验证后再信。**
