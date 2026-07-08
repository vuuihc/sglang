# Handoff: PR #30233 — PD disaggregation garbage output on aborted over-long input

## TL;DR status

- **Code fix + unit tests: done and passing** (10/10 on a 4090 box).
- **End-to-end repro: NOT yet reproduced.** The naive single-node symmetric
  config does NOT trigger the bug because the decode server independently
  aborts the over-long request. The real trigger needs an **asymmetric**
  prefill/decode capacity. See "What's left" — this is the key open question.
- **Open design question:** the issue author's own fix is **Plan B**
  (`notify_decode_abort`, a cross-conn signal). Our fix is **Plan A** (drop the
  aborted req before it enters the prefill queue). Plan A may be incomplete for
  the real failure path. Must confirm with the asymmetric repro below.

## The bug (issue #30233)

PD disaggregation. A request whose input length is `>= max_req_input_len` but
`< context_len`, with `allow_auto_truncate=False` (default):

1. `validate_input_length` fails → `set_finish_with_abort` truncates
   `origin_input_ids` to a single token `[0]` ("skip the long prefill").
2. On the **prefill** scheduler, `_add_request_to_queue` (PREFILL branch)
   enqueues it into `disagg_prefill_bootstrap_queue` **with no aborted check**.
3. It runs a 1-token prefill and `send_kv_chunk(last_chunk=True)` reports a
   **successful** KV transfer of 1 token.
4. The **decode** worker preallocated KV for the *full* prompt length and now
   decodes from uninitialized slots → **garbage output**. Additionally
   `process_disagg_prefill_inflight_queue` can overwrite the `FINISH_ABORT` with
   `FINISH_LENGTH(length=0)` on `KVPoll.Success`.

Issue key params: `max_total_num_tokens=284928`, `max_req_input_len=284922`,
request of 284937 tokens. i.e. a **large** context / KV pool.

## Our fix (Plan A) — `python/sglang/srt/managers/scheduler.py`

New helper `_abort_disagg_request_before_queue(req)`: if `is_aborted(req)`,
promote `to_finish` → `finished_reason` and `stream_output([req], ...)`, return
True. Called at the top of both the PREFILL and DECODE branches of
`_add_request_to_queue`. Mirrors the existing invalid-disagg-request precedent
(`scheduler.py`, the `bootstrap_room is None` path that streams + returns).

Rationale for Plan A over Plan B: the aborted req has no valid KV, so it should
never enter the transfer pipeline at all — cleaner than letting it flow through
and signalling decode to abort afterwards.

## Unit tests — `test/registered/unit/managers/test_priority_scheduling_disaggregation.py`

Added `TestDisaggregationAbortedRequestQueueing` (3 tests):
- prefill-mode aborted req skips bootstrap queue + is streamed out
- decode-mode aborted req skips prealloc queue + is streamed out
- healthy req still enqueued (regression guard)

Run (on a GPU box with sglang installed):
```bash
python -m pytest test/registered/unit/managers/test_priority_scheduling_disaggregation.py -q
```
Result on the 4090 box: **10 passed** (7 pre-existing + 3 new).

## Why the naive e2e did NOT reproduce the bug

Config used: `--context-length 8192 --max-total-tokens 2048` on both servers,
3001-token input, mooncake backend, mini-LB.

Observation (`logs/decode.log`): `#prealloc-req: 0, #transfer-req: 0`, and the
over-long request's abort message appears in **both** prefill.log AND
decode.log. The mini-LB fans the request out to prefill AND decode; the decode
server has its **own** tokenizer_manager + scheduler and runs
`validate_input_length` with the **same** `max_req_input_len=2042`, so it aborts
the request itself and never preallocates. Both sides abort → clean 400 → bug
masked. Buggy and fixed trees produced byte-identical results.

## What's left (the important part)

**Reproduce with an ASYMMETRIC config** so prefill aborts but decode does not:
- Give **prefill** a small cap: `--max-total-tokens 2048` (→ prefill
  `max_req_input_len ≈ 2042`).
- Give **decode** a large cap: large `--max-total-tokens` (or none) and large
  `--context-length` so decode's `max_req_input_len` > the request length.
- Send an input with `2042 <= tokens < decode_max_req_input_len`.
- Expectation on **buggy** tree: decode preallocates full length, prefill sends
  1 token, decode emits garbage (or the inflight queue turns the abort into
  `FINISH_LENGTH(0)`). On **fixed** tree (Plan A): prefill streams the abort
  before enqueue — but VERIFY decode doesn't end up with an orphaned prealloc
  waiting for a transfer that never comes (this is exactly what Plan B guards).

If Plan A alone leaves decode hung/orphaned, the PR likely needs Plan B
(`notify_decode_abort` on the sender) in addition, per the issue author.

### How to run the repro harness (in this dir on the GPU box)

Edit `launch_pd.sh` to split the caps per side (currently symmetric):
- prefill: keep `--max-total-tokens 2048`
- decode: raise/remove it and set a large `--context-length`

Then:
```bash
cd <repro dir>
bash launch_pd.sh                                  # wait "cluster up"
python repro.py --lb http://127.0.0.1:8000 --ctx <decode_ctx> --overlong-tokens 3000
bash capture_metrics.sh after-overlong             # snapshots prealloc/transfer counts
bash teardown.sh
```
Watch `logs/decode.log` for `#prealloc-req` / `#transfer-req` > 0 and any
garbage/`FINISH_LENGTH`. To A/B: `scheduler.buggy.py` (origin/main, no fix) and
`scheduler.fixed.py` (our fix) are both in this dir — copy one over
`python/sglang/srt/managers/scheduler.py`, clear `__pycache__`, relaunch.

## Environment notes (machine-specific, NOT part of the PR)

The AutoDL 4090 box needed these workarounds, all baked into `env.sh`:
- `LD_LIBRARY_PATH` → `nvidia/cu13/lib` (sgl_kernel needs `libnvrtc.so.13`).
- `LD_PRELOAD` system `libstdc++.so.6` (mooncake needs `GLIBCXX_3.4.30`).
- `PATH` includes miniconda bin (flashinfer JIT needs the `ninja` executable).
- `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` (box has sgl-kernel 0.4.3, main wants
  0.4.4; our fix touches no kernels).
- `_sitefix/sitecustomize.py` stubs `torch._inductor.compile_fx` (torch 2.11
  inductor duplicate-template assert on import on this sm89 box).
- Model: `Qwen/Qwen2.5-0.5B-Instruct` via `HF_ENDPOINT=https://hf-mirror.com`.

On a different box these may not all be needed — start by trying a plain launch;
add workarounds only for the errors you actually hit.
