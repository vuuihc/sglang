#!/usr/bin/env python3
"""Reproduce issue #30233 on a running PD cluster.

Sends two requests through the load balancer:
  1. a normal short request (sanity: cluster works end to end)
  2. an over-long request (> max_req_input_len, auto-truncate off)

Expected AFTER the fix: request 2 returns an "Input length ... exceeds" abort
error (HTTP 400) quickly, and the cluster stays healthy for request 3.

Buggy behavior (before the fix): request 2 either returns garbage decoded text
(decode ran on uninitialized KV) or hangs, because the aborted request was
enqueued for KV transfer.

Usage: python repro.py --lb http://127.0.0.1:8000 --ctx 4096
"""
import argparse
import json
import time

import requests


def gen(lb, prompt, max_new_tokens=16, timeout=60):
    t0 = time.time()
    try:
        r = requests.post(
            f"{lb}/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            },
            timeout=timeout,
        )
        dt = time.time() - t0
        return {"status": r.status_code, "elapsed": round(dt, 2), "body": r.text[:400]}
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "elapsed": round(time.time() - t0, 2), "body": ""}
    except Exception as e:  # noqa: BLE001
        return {"status": "ERROR", "elapsed": round(time.time() - t0, 2), "body": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", default="http://127.0.0.1:8000")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument(
        "--overlong-tokens",
        type=int,
        default=3000,
        help="target token count for the over-long prompt; must sit in "
        "[max_req_input_len, context_len) to hit the scheduler abort path",
    )
    args = ap.parse_args()

    results = {}

    # 1) sanity: short request
    print("=== [1] short request (sanity) ===")
    results["short"] = gen(args.lb, "The capital of France is")
    print(json.dumps(results["short"], indent=2))

    # 2) over-long request: input length in [max_req_input_len, context_len).
    # "word " tokenizes to ~1 token each, so N words ~= N tokens. This must be
    # >= max_req_input_len (2042 with our launch config) but < context_len
    # (8192), so the scheduler's validate_input_length aborts it (our fix path)
    # rather than the tokenizer-manager context_len gate.
    n_words = args.overlong_tokens
    long_prompt = "word " * n_words
    print(
        f"\n=== [2] over-long request (~{n_words} tokens; "
        f"target window [max_req_input_len, {args.ctx})) ==="
    )
    results["overlong"] = gen(args.lb, long_prompt, timeout=90)
    print(json.dumps(results["overlong"], indent=2))

    # 3) sanity again: cluster must still serve after the over-long request
    print("\n=== [3] short request again (cluster still healthy?) ===")
    results["after"] = gen(args.lb, "The capital of Japan is")
    print(json.dumps(results["after"], indent=2))

    # verdict
    print("\n=== VERDICT ===")
    ol = results["overlong"]
    body = ol["body"]
    # The scheduler-path abort (our fix) says "maximum allowed length"; the
    # tokenizer-manager gate says "model's context length". We want to prove the
    # SCHEDULER path returns a clean abort instead of garbage/hang.
    hit_scheduler_gate = "maximum allowed length" in body
    hit_tokenizer_gate = "context length" in body
    is_400 = ol["status"] == 400
    is_garbage_or_hang = ol["status"] in (200, "TIMEOUT")
    healthy_after = results["after"]["status"] == 200

    print(f"over-long HTTP status: {ol['status']} (elapsed {ol['elapsed']}s)")
    print(f"  hit scheduler validate_input_length gate (our fix path): {hit_scheduler_gate}")
    print(f"  hit tokenizer-manager context_len gate:                  {hit_tokenizer_gate}")
    print(f"  returned 200/garbage or timed out (buggy symptom):       {is_garbage_or_hang}")
    print(f"cluster healthy after over-long request:                   {healthy_after}")
    if is_400 and hit_scheduler_gate and healthy_after:
        verdict = "FIXED (scheduler aborts over-long PD req cleanly)"
    elif is_garbage_or_hang:
        verdict = "BUG REPRODUCED (over-long PD req produced output/hang instead of abort)"
    elif hit_tokenizer_gate:
        verdict = "INCONCLUSIVE (caught by tokenizer gate, not scheduler; widen window)"
    else:
        verdict = "NEEDS INSPECTION"
    print(f"RESULT: {verdict}")

    with open("repro_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote repro_result.json")


if __name__ == "__main__":
    main()
