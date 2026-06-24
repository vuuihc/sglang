#!/usr/bin/env bash
#
# profile_serving_scheduler.sh — profile the REAL serving path (scheduler +
# tokenizer + model forward), which bench_one_batch does NOT exercise.
#
# This is where the host-side per-step/per-request perf PRs live (e.g. #27965
# fill_ids reconstruction, #28397 penalizer D2H sync). cuda graph does not cover
# this code (it runs outside the captured graph), so the bubbles are real targets.
#
# Flow: launch server (with profiler dir) -> wait ready -> bench_serving --profile
# (drives /start_profile + /stop_profile) -> trace lands in PROFILE_DIR -> stop server.
#
# Throwaway script — not part of the package.
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
CONCURRENCY="${CONCURRENCY:-32}"           # keep the scheduler busy batching
RIN="${RIN:-512}"; ROUT="${ROUT:-128}"
PROFILE_STEPS="${PROFILE_STEPS:-20}"       # scheduler forward steps to capture
MEM_FRACTION="${MEM_FRACTION:-0.85}"
PROFILE_DIR="${SGLANG_TORCH_PROFILER_DIR:-/root/autodl-tmp/profile_serving}"
# torch2.11 workaround (see docs/profiling-study/01); keep regular cuda graph on.
SRV_EXTRA="${SRV_EXTRA---disable-piecewise-cuda-graph}"

export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "$PROFILE_DIR"
SRV_LOG=/root/autodl-tmp/serving_server.log

echo "=== launching server: $MODEL on :$PORT (traces -> $PROFILE_DIR) ==="
nohup python -m sglang.launch_server \
  --model-path "$MODEL" --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION" \
  $SRV_EXTRA > "$SRV_LOG" 2>&1 &
SRV_PID=$!
echo "server pid=$SRV_PID, log=$SRV_LOG"

cleanup() { echo "=== stopping server $SRV_PID ==="; kill "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT

echo "=== waiting for server ready (up to ~5 min) ==="
for i in $(seq 1 150); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "ready after ${i}x2s"; break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "SERVER DIED — tail log:"; tail -25 "$SRV_LOG"; exit 1; fi
  sleep 2
done
curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "server never became ready"; tail -25 "$SRV_LOG"; exit 1; }

echo "=== bench_serving with --profile (captures scheduler trace) ==="
python -m sglang.bench_serving \
  --backend sglang --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dataset-name random --random-input-len "$RIN" --random-output-len "$ROUT" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
  --warmup-requests 16 \
  --profile --profile-steps "$PROFILE_STEPS" \
  --profile-output-dir "$PROFILE_DIR" --profile-prefix serving

echo "=== traces ==="
ls -lh "$PROFILE_DIR"/*.trace.json.gz 2>/dev/null || echo "  (no trace — check $SRV_LOG and bench output)"
