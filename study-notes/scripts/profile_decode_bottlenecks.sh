#!/usr/bin/env bash
#
# profile_decode_bottlenecks.sh — capture torch traces of SGLang decode for
# bottleneck hunting + learning trace analysis.
#
# Learning goal: sweep batch size on a single GPU and watch the decode loop go
# from HOST-BOUND (small batch → GPU full of bubbles, waiting on the Python
# scheduler / kernel launch / D2H syncs) to COMPUTE-BOUND (large batch → kernels
# packed back-to-back). The small-batch bubbles are where SGLang's small perf
# PRs live (cf. #28397 per-step D2H sync, #26397 topk==1 softmax skip).
#
# Each run drops a trace at:
#   $SGLANG_TORCH_PROFILER_DIR/<prefix>_batch<B>_input<I>_output<O>.trace.json.gz
# Open it in https://ui.perfetto.dev (top lane = CPU/Python, bottom = CUDA stream).
#
# Throwaway script — not part of the package. Tuned to fit a single 24GB card
# (RTX 4090D) by default; bump MODEL / MEM_FRACTION for H20 / H800.
#
# Usage:
#   bash scripts/profile_decode_bottlenecks.sh                 # plain decode sweep
#   SPEC=eagle3 bash scripts/profile_decode_bottlenecks.sh     # + EAGLE3 spec decode
#   MODEL=Qwen/Qwen3-32B-FP8 BATCHES="1 8" bash scripts/profile_decode_bottlenecks.sh
#
set -euo pipefail

# ---- knobs (override via env) ------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-8B}"          # fits 24GB in fp16 (~16GB weights)
BATCHES="${BATCHES:-1 8 32 64}"          # the host-bound -> compute-bound sweep
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
PROFILE_STAGE="${PROFILE_STAGE:-decode}" # decode | prefill | all  (decode = the bubble regime)
PROFILE_STEPS="${PROFILE_STEPS:-5}"      # capture a few steady-state steps, not just one
MEM_FRACTION="${MEM_FRACTION:-0.85}"     # lower if you OOM at the big batch
SPEC="${SPEC:-none}"                     # none | eagle3
PROFILE_DIR="${SGLANG_TORCH_PROFILER_DIR:-/root/profile_log}"
# torch 2.11.0's _inductor flex_attention double-registers a Triton template
# (AssertionError: duplicate template name). sglang's piecewise cuda graph pulls
# inductor in via torch.compile, so disable it (keeps the regular cuda graph).
# Drop this once on a torch build without the bug. Override with EXTRA_ARGS="".
EXTRA_ARGS="${EXTRA_ARGS---disable-piecewise-cuda-graph}"

# China HF mirror (harmless elsewhere; comment out if you use HF directly)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

echo "=========================================================================="
echo " model        : $MODEL"
echo " batches      : $BATCHES   input=$INPUT_LEN output=$OUTPUT_LEN"
echo " profile stage: $PROFILE_STAGE   steps=$PROFILE_STEPS   spec=$SPEC"
echo " traces -> $PROFILE_DIR"
echo "=========================================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo

# ---- spec-decode flags (only when SPEC=eagle3) -------------------------------
SPEC_ARGS=()
PREFIX="dense"
if [[ "$SPEC" == "eagle3" ]]; then
  # EAGLE3 draft head for Qwen3-8B. Swap to the matching head if you change MODEL.
  # topk=1 is the common config and is exactly the path the perf PRs target.
  SPEC_ARGS=(
    --speculative-algorithm EAGLE3
    --speculative-draft-model-path "${DRAFT_MODEL:-Tengyunw/qwen3_8b_eagle3}"
    --speculative-num-steps "${SPEC_STEPS:-3}"
    --speculative-eagle-topk "${SPEC_TOPK:-1}"
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-4}"
  )
  PREFIX="eagle3"
  echo " spec decode  : EAGLE3 draft=${DRAFT_MODEL:-Tengyunw/qwen3_8b_eagle3} topk=${SPEC_TOPK:-1}"
  echo
fi

# ---- run the sweep -----------------------------------------------------------
# One bench_one_batch invocation handles the whole batch list; it writes one
# trace per (batch,input,output) tuple. We keep input/output single-valued so
# the only axis that moves is batch size — clean for the learning comparison.
python -m sglang.bench_one_batch \
  --model-path "$MODEL" \
  --batch-size $BATCHES \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --mem-fraction-static "$MEM_FRACTION" \
  --profile \
  --profile-stage "$PROFILE_STAGE" \
  --profile-steps "$PROFILE_STEPS" \
  --profile-filename-prefix "$PREFIX" \
  $EXTRA_ARGS \
  "${SPEC_ARGS[@]}"

echo
echo "=========================================================================="
echo " Done. Traces in $PROFILE_DIR :"
ls -lh "$PROFILE_DIR"/*.trace.json.gz 2>/dev/null || echo "  (no traces found — check the log above for errors)"
echo
echo " Next:"
echo "   1) Open the batch=1 trace in https://ui.perfetto.dev — note the GPU bubbles."
echo "   2) Open the batch=64 trace — note the bubbles shrink (compute-bound)."
echo "   3) Send me the file paths (or scp them back) and I'll run"
echo "      llm-torch-profiler-analysis to produce the kernel / overlap / fuse tables."
echo "=========================================================================="
