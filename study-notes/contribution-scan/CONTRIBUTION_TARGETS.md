# SGLang — GPU-verifiable inference perf contribution targets

Scan date: 2026-06-16 · Repo: sgl-project/sglang @ `25e696aa8d`
Focus: small (≤30 line), GPU-verifiable, numerically-equivalent perf PRs on the
per-step inference critical path.

## Headline

**This is a very well-maintained tree for this PR archetype.** The gold-standard
host-sync-elimination pattern (`logits[mask]` → `torch.where`) and the redundant-work
short-circuits (topk==1 fast paths, `seq_lens_cpu` mirrors, fused kernels) have already
been applied across the *mainstream* decode path — frequently with comments citing the
exact technique. Two independent deep-read scans (host-sync anti-patterns; redundant
compute) converged on the same conclusion: the high-traffic path is largely done.

The one genuinely actionable static target is a **direct port of already-merged work**
into a sibling speculative-decoding worker that was missed when the optimization landed.

## Ranked candidates

| Rank | Candidate | File | ~Lines | Risk | Dedup | Why it's a good PR |
|---|---|---|---|---|---|---|
| 1 | Skip full-vocab softmax in `frozen_kv_mtp` draft when `topk==1` | `python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py:531-532, 571-572` | ~14 | low | ✅ CLEAR | Exact numerically-equivalent port of merged **#26397**; sibling worker was never updated. Runs per draft step; saves a `[bs, vocab]` softmax kernel every step for the common `topk==1` config. GPU-verifiable via spec-decode throughput. |
| 2 | Same skip in `multi_layer_eagle` eager draft-extend path | `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py:591-595, 470-471` | ~10 | low | ✅ CLEAR | Same transform; lower impact (uncommon config + only the eager fallback; cuda-graph branch already precomputes). Good rank-2 follow-on. |
| — | (demoted) `torch.tensor(sliding_window_size)` → `torch.clamp` in Triton SWA decode | `python/sglang/srt/layers/attention/triton_backend.py:1545-1548` | ~2 | low | ✅ CLEAR | Numerically identical and per-step, **but it is not a real sync** — just a tiny CPU scalar alloc. Fails the "fixes something observable / GPU-measurable" filter. File as cleanup only if bundling. |

## Rank-1 verification trail (the part that matters)

- **It's an oversight, not a deliberate exclusion.** The `topk==1 and not _is_hip` argmax
  short-circuit was relanded in #26397 (`dd6f073377`) into `eagle_worker_v2.draft_forward`
  (lines 601-613). `frozen_kv_mtp_worker_v2.py`'s softmax sites were last touched by an
  unrelated refactor (#28093) and #23862 — neither ported the skip.
- **`topk==1` is a first-class config for this worker** — `frozen_kv_mtp` reads
  `self.topk = server_args.speculative_eagle_topk` and already branches on
  `if self.topk == 1:` (line 235).
- **Numerically equivalent.** Downstream consumer is the same `select_top_k_tokens`; for a
  degenerate single-path tree `topk_p` is unused, so argmax-over-logits ≡ argmax-over-softmax.
- **Simpler than the reference.** Unlike eagle, `frozen_kv_mtp` does **no** `hot_token_id[...]`
  remap after topk, so there's nothing extra to preserve.
- **ROCm guard required.** `not _is_hip` — on ROCm the argmax tie-break diverges from
  softmax+max on FP8 logits (DSV3.2 MTP GSM8K, see #26358). `frozen_kv_mtp` does **not**
  currently import `_is_hip`; the PR must add it.

## Why most candidates were dropped — see DROP_LIST.md

The two scans surfaced ~6 raw candidates; all but the above were rejected for being
gated-off (custom logit processors), per-request not per-step (eagle prefill extend),
already-optimized (penalizers, FA3/FlashInfer metadata, radix match, MoE upcasts), or
non-observable (the Triton clamp). This is the expected outcome for a mature repo.

## If rank-1/2 are too niche: Phase-8 dynamic pivot

Static scanning of the hot path is largely exhausted. Higher-yield next steps:
1. **Profile a live run** of the newest large-vocab model under spec decode and capture a
   real trace — crashes/regressions found this way are by definition not duplicated.
   (Repo has `generate-profile` and `llm-torch-profiler-analysis` skills for this.)
2. **Less-traveled attention backends** that likely missed the sync-elimination passes:
   `wave_backend.py`, `xpu_backend.py`, `dual_chunk_flashattention_backend.py`.
3. **Prefill/extend metadata prep** (explicitly out of this scan's per-step scope, but a
   real surface).
