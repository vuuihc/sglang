#!/bin/bash
# Launch a single-node PD-disaggregation cluster on 2 GPUs:
#   prefill server on GPU 0, decode server on GPU 1, mini load balancer.
# Usage: bash launch_pd.sh   (run from the repro dir; logs go to ./logs)
set -u
cd "$(dirname "$0")"
source ./env.sh

LOGDIR=./logs
mkdir -p "$LOGDIR"

HOST=127.0.0.1
LB_PORT=8000
PREFILL_PORT=8100
DECODE_PORT=8200
BOOTSTRAP_PORT=8500

# Issue #30233 window is max_req_input_len <= input_len < context_len. The
# scheduler computes max_req_input_len = min(context_len-1, kv_pool_size-1) - 5,
# while the tokenizer-manager gate rejects at context_len. To open a WIDE window
# (so the scheduler path, not the tokenizer gate, catches the over-long input),
# make the KV pool the limiter: large context_length + small max_total_tokens.
#   context_len = 8192, max_total_tokens = 2048
#   => max_req_input_len = min(8191, 2047) - 5 = 2042
#   => an input with 2042 <= tokens < 8192 hits set_finish_with_abort (our fix)
CTX=8192
MAXTOK=2048
COMMON="--model-path $MODEL --trust-remote-code --context-length $CTX \
  --max-total-tokens $MAXTOK \
  --disaggregation-transfer-backend mooncake --tp 1 --mem-fraction-static 0.75"

echo "[launch] prefill on GPU0 :$PREFILL_PORT"
CUDA_VISIBLE_DEVICES=0 $PY -m sglang.launch_server $COMMON \
  --disaggregation-mode prefill --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
  --host $HOST --port $PREFILL_PORT \
  > "$LOGDIR/prefill.log" 2>&1 &
echo $! > "$LOGDIR/prefill.pid"

echo "[launch] decode on GPU1 :$DECODE_PORT"
CUDA_VISIBLE_DEVICES=1 $PY -m sglang.launch_server $COMMON \
  --disaggregation-mode decode --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
  --base-gpu-id 0 --host $HOST --port $DECODE_PORT \
  > "$LOGDIR/decode.log" 2>&1 &
echo $! > "$LOGDIR/decode.pid"

echo "[launch] waiting for prefill+decode /health_generate ..."
for url in "http://$HOST:$PREFILL_PORT" "http://$HOST:$DECODE_PORT"; do
  for i in $(seq 1 120); do
    curl -sf "$url/health_generate" >/dev/null 2>&1 && { echo "  ready: $url"; break; }
    sleep 5
    if [ "$i" = 120 ]; then echo "  TIMEOUT: $url"; fi
  done
done

echo "[launch] mini load balancer :$LB_PORT"
$PY -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill "http://$HOST:$PREFILL_PORT" \
  --decode "http://$HOST:$DECODE_PORT" \
  --host $HOST --port $LB_PORT \
  > "$LOGDIR/lb.log" 2>&1 &
echo $! > "$LOGDIR/lb.pid"

for i in $(seq 1 60); do
  curl -sf "http://$HOST:$LB_PORT/health" >/dev/null 2>&1 && { echo "  LB ready"; break; }
  sleep 2
done
echo "[launch] cluster up. LB at http://$HOST:$LB_PORT"
