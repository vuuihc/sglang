#!/usr/bin/env bash
# ablation_battery.sh — 六个动词消融实验矩阵(V2 数据篇)
# 计划文档: study-notes/profiling-study/08-消融实验计划-V2数据篇.md
#
# 用法(在租的 GPU 机器上, sglang 已装好):
#   bash ablation_battery.sh                # 全跑(P0 -> P1, 便宜在前)
#   bash ablation_battery.sh e1 e3 e4       # 只跑指定实验
#   ONLY_DOWNLOAD=1 bash ablation_battery.sh  # 只预下载模型
#
# 实验清单:
#   e1  不算·radix cache A/B         e2  少算·FP8 vs BF16
#   e3  少搬·AWQ 4bit vs BF16        e4  破串行·EAGLE3 并发扫描
#   e5  少占·KV fp8 容量             e9  打满·overlap A/B
#   e10 少搬·attention 后端 A/B      e12 少搬·torch.compile A/B
#   e7  结构·GQA KV-scaling 对照     e11 结构·MLA 三连测
#   e8  打满·chunked prefill 扫描    e6  冷启动悬案
set -o pipefail

########################  配置  ########################
PORT=${PORT:-30000}
URL="http://127.0.0.1:$PORT"
OUT=${OUT:-study-notes/contribution-scan/exp3-ablation}
MODEL_BF16=${MODEL_BF16:-Qwen/Qwen3-8B}
MODEL_FP8=${MODEL_FP8:-Qwen/Qwen3-8B-FP8}
MODEL_AWQ=${MODEL_AWQ:-Qwen/Qwen3-8B-AWQ}          # 若不存在改用 GPTQ-Int4
MODEL_MLA=${MODEL_MLA:-deepseek-ai/DeepSeek-V2-Lite-Chat}
EAGLE_DRAFT=${EAGLE_DRAFT:-Tengyunw/qwen3_8b_eagle3} # 开跑前在 HF 上核实一次
MEMFRAC=${MEMFRAC:-0.85}
# 07 的环境修复(cu13 LD_LIBRARY_PATH + HF_HOME + hf-mirror), AutoDL 镜像里已备好:
[ -f /root/sglang_env.sh ] && source /root/sglang_env.sh
set -u
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}   # e12 内部会临时打开

mkdir -p "$OUT"
PHASES="$OUT/phases.log"
phase() { echo "$(date +%s.%N) $*" | tee -a "$PHASES"; }

########################  基础设施  ########################
SERVER_PID=""
start_server() { # start_server <日志名> <模型> [额外参数...]
  local log="$OUT/$1"; shift
  local model="$1"; shift
  phase "SERVER_START $log $model $*"
  python -m sglang.launch_server --model-path "$model" --port "$PORT" \
    --mem-fraction-static "$MEMFRAC" --disable-piecewise-cuda-graph \
    "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    curl -sf "$URL/health" >/dev/null 2>&1 && { phase "SERVER_READY $log"; return 0; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "!! server died, tail $log:"; tail -20 "$log"; return 1; }
    sleep 5
  done
  echo "!! server timeout"; return 1
}
stop_server() { [ -n "$SERVER_PID" ] && { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; SERVER_PID=""; sleep 3; }; phase "SERVER_STOP"; }
trap 'stop_server; kill $(jobs -p) 2>/dev/null' EXIT

bench() { # bench <日志名> [bench_serving 参数...]
  local log="$OUT/$1"; shift
  phase "BENCH_START $log"
  python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" "$@" \
    > "$log" 2>&1
  phase "BENCH_END $log"
  grep -E "Output token throughput|Median TTFT|Median ITL|P99 ITL|Median TPOT|Request throughput" "$log" | sed "s/^/    /"
}
warmup() { # 卫生铁律: 每个 server 采数前先跑一轮丢弃的 warmup (07 §七)
  bench "_warmup_$1.log" --dataset-name random --random-input-len 512 --random-output-len 64 \
    --num-prompts 32 --max-concurrency 8 --random-range-ratio 1
}
decode_load() { bench "$1" --dataset-name random --random-input-len 128 --random-output-len 512 \
  --num-prompts $((32*8)) --max-concurrency "$2" --random-range-ratio 1; }
prefill_load() { bench "$1" --dataset-name random --random-input-len 8192 --random-output-len 8 \
  --num-prompts 64 --max-concurrency 8 --random-range-ratio 1; }

# dmon 全程采集(07 方法论: 一个文件, 事后按 phases 切窗)
nvidia-smi dmon -s pucm -d 1 -o T > "$OUT/dmon.log" 2>&1 &

########################  实验  ########################

e1() { # 不算·Radix Cache —— 前缀复用负载下关掉缓存
  for mode in on off; do
    extra=""; [ $mode = off ] && extra="--disable-radix-cache"
    start_server "e1_server_radix_$mode.log" "$MODEL_BF16" $extra || return 1
    warmup e1_$mode
    bench "e1_radix_${mode}.log" --dataset-name generated-shared-prefix \
      --gsp-num-groups 32 --gsp-prompts-per-group 8 \
      --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 128 \
      --max-concurrency 16
    grep -iE "cache.hit|token usage" "$OUT/e1_server_radix_$mode.log" | tail -3 || true
    stop_server
  done
}

e2() { # 少算·FP8 vs BF16 —— prefill 相收益大, decode 相收益小
  for m in "$MODEL_BF16:bf16" "$MODEL_FP8:fp8"; do
    model="${m%%:*}"; tag="${m##*:}"
    start_server "e2_server_$tag.log" "$model" || return 1
    warmup e2_$tag
    prefill_load "e2_prefill_$tag.log"
    decode_load  "e2_decode_$tag.log" 32
    stop_server
  done
}

e3() { # 少搬·AWQ 4bit vs BF16 —— bs=1 大赚 / conc32 缩水
  for m in "$MODEL_BF16:bf16" "$MODEL_AWQ:awq"; do
    model="${m%%:*}"; tag="${m##*:}"
    start_server "e3_server_$tag.log" "$model" || return 1
    warmup e3_$tag
    decode_load "e3_decode_${tag}_c1.log" 1
    decode_load "e3_decode_${tag}_c32.log" 32
    stop_server
  done
}

e4() { # 破串行·EAGLE3 —— 加速比 vs 并发曲线(文章的镇场图)
  for mode in off on; do
    extra=""
    [ $mode = on ] && extra="--speculative-algorithm EAGLE3 --speculative-draft-model-path $EAGLE_DRAFT --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32"
    start_server "e4_server_spec_$mode.log" "$MODEL_BF16" $extra || return 1
    warmup e4_$mode
    for c in 1 4 8 16 32; do
      bench "e4_spec_${mode}_c${c}.log" --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts $((c*8)) --max-concurrency $c --random-range-ratio 1
    done
    grep -iE "accept" "$OUT/e4_server_spec_$mode.log" | tail -3 || true
    stop_server
  done
}

e5() { # 少占·KV fp8 —— 只看启动日志的 KV 池容量
  for dtype in auto fp8_e4m3; do
    start_server "e5_server_kv_$dtype.log" "$MODEL_BF16" --kv-cache-dtype $dtype || return 1
    grep -iE "KV Cache|memory pool|#token" "$OUT/e5_server_kv_$dtype.log" | tail -5
    stop_server
  done
}

e9() { # 打满·Overlap Scheduler A/B(07 只有定性, 这里补定量)
  for mode in on off; do
    extra=""; [ $mode = off ] && extra="--disable-overlap-schedule"
    start_server "e9_server_overlap_$mode.log" "$MODEL_BF16" $extra || return 1
    warmup e9_$mode
    decode_load "e9_overlap_${mode}_c8.log" 8
    decode_load "e9_overlap_${mode}_c32.log" 32
    stop_server
  done
}

e10() { # 少搬·attention 后端 —— 朴素 vs FlashAttention 的直接价签
  for backend in fa3 flashinfer triton torch_native; do
    start_server "e10_server_$backend.log" "$MODEL_BF16" --attention-backend $backend || {
      echo "!! backend $backend 起不来(本身就是数据点), 跳过"; stop_server; continue; }
    warmup e10_$backend
    bench "e10_prefill4k_$backend.log" --dataset-name random \
      --random-input-len 4096 --random-output-len 8 --num-prompts 32 \
      --max-concurrency 8 --random-range-ratio 1
    prefill_load "e10_prefill8k_$backend.log" || echo "!! $backend 8K 失败(记录之)"
    stop_server
  done
}

e12() { # 少搬·torch.compile A/B(预期温和——本身是结论)
  for mode in off on; do
    extra=""; env_fix=""
    [ $mode = on ] && { extra="--enable-torch-compile"; env_fix=0; }
    TORCHDYNAMO_DISABLE=${env_fix:-1} start_server "e12_server_compile_$mode.log" "$MODEL_BF16" $extra || return 1
    warmup e12_$mode
    decode_load "e12_compile_${mode}_c1.log" 1
    stop_server
  done
}

e7() { # 结构·GQA 全注意力 KV-scaling(与 07 混合线性曲线同图对比)
  start_server "e7_server.log" "$MODEL_BF16" --context-length 32768 || return 1
  warmup e7
  for L in 4096 8192 16384 30000; do
    bench "e7_kv${L}.log" --dataset-name random \
      --random-input-len $L --random-output-len 32 --num-prompts 4 \
      --max-concurrency 1 --random-range-ratio 1
  done
  stop_server
}

e11() { # 结构·MLA 三连测(DeepSeek-V2-Lite)
  # (a) KV 池容量对账: 和 e5 的 Qwen3-8B 日志对比 每token字节
  # (b) 后端 A/B: 结构红利要 kernel 接住
  for backend in flashinfer triton flashmla; do
    start_server "e11_server_$backend.log" "$MODEL_MLA" --attention-backend $backend --trust-remote-code || {
      echo "!! MLA backend $backend 起不来(记录之)"; stop_server; continue; }
    grep -iE "KV Cache|#token" "$OUT/e11_server_$backend.log" | tail -3
    warmup e11_$backend
    decode_load "e11_decode_${backend}_c8.log" 8
    # (c) 只在第一个可用后端上跑 ITL vs 上下文
    if [ ! -f "$OUT/.e11_scaling_done" ]; then
      for L in 4096 16384 30000; do
        bench "e11_kv${L}_$backend.log" --dataset-name random \
          --random-input-len $L --random-output-len 32 --num-prompts 4 \
          --max-concurrency 1 --random-range-ratio 1
      done
      touch "$OUT/.e11_scaling_done"
    fi
    stop_server
  done
}

e8() { # 打满·chunked prefill 扫描 —— ITL P99 vs prefill 吞吐的权衡
  for size in 512 2048 8192 -1; do
    start_server "e8_server_cp$size.log" "$MODEL_BF16" --chunked-prefill-size $size || return 1
    warmup e8_$size
    # 混合负载: 长 prompt 和 decode 流一起, 看 decode 被 prefill 卡多惨
    bench "e8_mixed_cp$size.log" --dataset-name random \
      --random-input-len 4096 --random-output-len 256 --num-prompts 64 \
      --max-concurrency 16 --random-range-ratio 0.25   # 长短混合
    stop_server
  done
}

e6() { # 冷启动悬案(07 §七): 冷启动即压 -> 原地重跑
  start_server "e6_server.log" "$MODEL_BF16" || return 1
  # 注意: 故意不 warmup —— 这就是实验本身
  decode_load "e6_cold_c32.log" 32
  decode_load "e6_warm_c32.log" 32
  stop_server
}

########################  模型预下载  ########################
download_all() {
  for m in "$MODEL_BF16" "$MODEL_FP8" "$MODEL_AWQ" "$EAGLE_DRAFT" "$MODEL_MLA"; do
    huggingface-cli download "$m" --quiet || echo "!! 下载失败: $m (跑到对应实验前请手动解决)"
  done
}

########################  主流程  ########################
[ "${ONLY_DOWNLOAD:-0}" = "1" ] && { download_all; exit 0; }
nvidia-smi > "$OUT/env_gpu.txt"; pip list 2>/dev/null | grep -iE "sglang|torch|flashinfer" > "$OUT/env_pip.txt"

# 默认顺序: 便宜在前(同模型), 要下新模型的在后
DEFAULT="e5 e1 e9 e2 e7 e8 e10 e12 e3 e6 e4 e11"
TODO="${*:-$DEFAULT}"
echo "== 实验顺序: $TODO =="
for exp in $TODO; do
  echo; echo "======== $exp ========"; phase "EXP_BEGIN $exp"
  $exp || echo "!! $exp 未完整跑完, 继续下一个"
  phase "EXP_END $exp"
done
echo; echo "== 全部结束, 数据在 $OUT =="
