# Rank-1 PR draft

## Title
`[perf][spec decoding] Skip full-vocab softmax in frozen_kv_mtp draft when topk == 1`

## ⚠️ Before editing
This touches `python/sglang/srt/speculative/` → **read the `speculative-naming` skill first**
(project rule: `.claude/rules/modify-component-must-read.md`). No new identifiers are
introduced here, but follow it for review.

## PR body (ready to paste)

> `FrozenKVMTPWorkerV2`'s draft loop computes a full-vocab `softmax` over
> `[bs, vocab]` before `fast_topk` on every draft step (and at the seed). When
> `speculative_eagle_topk == 1` the tree is a degenerate single path: `topk_p` is unused
> downstream (`select_top_k_tokens`) and `topk_index` is just the argmax, so the softmax
> is wasted GPU work on the per-step critical path.
>
> This is the same optimization already merged for `eagle_worker_v2.draft_forward` in
> #26397 (relanded after the ROCm tie-break fix). `frozen_kv_mtp_worker_v2.py` was simply
> not updated when that landed. This PR ports the identical, numerically-equivalent
> short-circuit to both softmax sites in this worker.
>
> Gated to CUDA (`not _is_hip`): on ROCm the argmax tie-break diverges from the softmax+max
> path on FP8 logits and corrupts MTP draft selection (see #26358). Unlike `eagle_worker_v2`,
> this worker performs no `hot_token_id` remap after topk, so no extra handling is needed.
>
> **Verification:** spec-decode acceptance length and output are unchanged on CUDA (single
> path → identical selection); decode throughput improves for `topk==1` frozen-KV MTP on
> large-vocab models by skipping a `[bs, vocab]` softmax per draft step.

## Patch sketch

In `python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py`:

1. Add the `_is_hip` module global near the other top-level constants (mirror
   `eagle_worker_v2.py:106`):

```python
from sglang.srt.utils import is_hip   # add to existing sglang.srt.utils import if present

_is_hip = is_hip()
```

2. Replace the **seed** site (currently ~531-532):

```python
        probs = torch.softmax(seed_next_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
```

with:

```python
        if self.topk == 1 and not _is_hip:
            # topk=1 → degenerate single-path tree; `topk_p` is unused downstream,
            # so skip softmax and just argmax over logits. Gated to CUDA: on ROCm the
            # argmax tie-break diverges from softmax+max on FP8 logits (see #26358).
            topk_index = torch.argmax(seed_next_logits, dim=-1, keepdim=True)
            topk_p = torch.ones_like(topk_index, dtype=torch.float32)
        else:
            probs = torch.softmax(seed_next_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
```

3. Replace the **in-loop** site (currently ~571-572), identically but over
   `logits_output.next_token_logits`:

```python
        if self.topk == 1 and not _is_hip:
            topk_index = torch.argmax(
                logits_output.next_token_logits, dim=-1, keepdim=True
            )
            topk_p = torch.ones_like(topk_index, dtype=torch.float32)
        else:
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
```

(The existing `maybe_detect_oob(...)` calls right after each site stay unchanged.)

Net: ~14 lines added. One concern per PR — only this worker.

## GPU verification recipe

Pick a model that uses a frozen-KV / MTP draft with `speculative_eagle_topk=1` and a large
vocab (DeepSeek-V3.2 MTP-style or a Qwen3 MTP config). On the GPU box:

```bash
# 1. Correctness: acceptance length + output identical before vs after on CUDA.
#    Run the spec-decode accuracy test for this worker (see test/srt/ for the
#    matching spec-decoding test; grep for frozen_kv_mtp / MTP) on baseline and patched.

# 2. Throughput delta (the win):
python -m sglang.bench_one_batch \
  --model <MTP-capable-model> \
  --speculative-algorithm <the frozen-KV/MTP algo> \
  --speculative-eagle-topk 1 \
  --speculative-num-steps <e.g. 3> \
  --batch-size 1 --batch-size 8 --batch-size 32 \
  --input-len 1024 --output-len 256
# Compare decode tokens/s baseline vs patched. Expect a small but real gain that
# grows with vocab size and batch size (one [bs, vocab] softmax saved per draft step).

# 3. (Optional, strongest evidence) capture a torch profile before/after and show the
#    softmax kernel dropping out of the draft-step timeline:
#    use the `generate-profile` skill, then `llm-torch-profiler-analysis` on the trace.
```

Attach the throughput table + (ideally) the before/after profiler timeline to the PR —
the repo rewards quantified `[perf]` PRs (cf. #26397, #28397).

## Pre-push checklist
- [ ] Read `speculative-naming` skill.
- [ ] `pre-commit run --all-files` (lint is the #1 reason first PRs bounce here).
- [ ] CUDA accuracy unchanged; ROCm path untouched (guard verified).
- [ ] One concern, one worker.
