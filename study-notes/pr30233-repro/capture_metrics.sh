#!/bin/bash
# Snapshot prefill + decode server metrics and key server-info fields, so we can
# study PD-disaggregation behavior (KV pool size, running reqs, transfer stats,
# TTFT, throughput) after a repro run. Writes to ./metrics/<label>.
set -u
cd "$(dirname "$0")"
LABEL="${1:-snap}"
HOST=127.0.0.1
PREFILL=http://$HOST:8100
DECODE=http://$HOST:8200
OUT=./metrics/$LABEL
mkdir -p "$OUT"

for name in prefill decode; do
  url=$([ "$name" = prefill ] && echo $PREFILL || echo $DECODE)
  echo "[capture] $name /get_server_info"
  curl -s "$url/get_server_info" > "$OUT/$name.server_info.json" 2>/dev/null
  echo "[capture] $name /metrics"
  # keep only sglang: gauges/counters relevant to PD + KV
  curl -s "$url/metrics" 2>/dev/null \
    | grep -E "^sglang:(num_running_reqs|num_queue_reqs|token_usage|cache_hit_rate|num_used_tokens|gen_throughput|time_to_first_token|kv|transfer|bootstrap|prefill|decode)" \
    > "$OUT/$name.metrics.txt"
done
echo "[capture] saved to $OUT"
