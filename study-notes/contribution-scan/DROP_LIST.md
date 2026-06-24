# Drop list — evaluated and rejected (so future runs can skip)

Scan date: 2026-06-16 @ `25e696aa8d`

| Candidate | File | Reason dropped |
|---|---|---|
| Triton SWA `torch.tensor(sliding_window_size)` → `torch.clamp` | `layers/attention/triton_backend.py:1545` | CLEAR on dedup, numerically identical, per-step — but **not a real D2H sync** (CPU scalar vs CUDA tensor). Only a micro CPU alloc; not GPU-measurable. Fails "fixes something observable". |
| EAGLE `_draft_extend_for_prefill` topk==1 softmax skip | `speculative/eagle_worker_v2.py:719` | Same valid transform, but **per-request (prefill)**, not per decode step → out of the hot-path scope; low impact. |
| `apply_custom_logit_processor` nonzero + boolean-mask indexing | `layers/sampler.py:788,797` | Real syncs, but gated by `has_custom_logit_processor` (needs `--enable-custom-logit-processor` + per-req processors) → **off on the normal path**. Output shape data-dependent → no clean sync-free rewrite. |
| TRTLLM-MLA `seq_lens.max().item()` fallback | `layers/attention/trtllm_mla_backend.py:531` | The syncing `else` branch is **dead in practice** (`seq_lens_cpu` populated on real decode batches); no sync-free source exists when the mirror is absent. Not actionable. |
| Sampler `logprobs.clamp_(min=...)` | `layers/sampler.py:357` | Required for correctness of returned logprobs; cheap. Not a win. |
| `biased_grouped_topk_impl` sigmoid→add→topk→gather | `layers/moe/topk.py:1038` | Eager **fallback** only; GPU hot path uses fused `biased_grouped_topk_gpu`. No production impact. |
| `logits_processor._copy_logits_to_buffer` `.float()` | `layers/logits_processor.py:986` | Upcast to fp32 is intentional (sampler/logprob contract). Changing it changes numerics. |
| VocabParallelEmbedding post-embed all-reduce | `layers/vocab_parallel_embedding.py:528` | Already addressed by #26970 (replication). |

## Already-optimized (confirmed during scan, do not re-flag)
EAGLE `draft_forward` topk==1 (#26424/#26397) · `_draft_extend_for_decode` argmax ·
radix `match` gallop search (#27364) · `select_top_k_tokens` (`expand` not `repeat`) ·
`clamp_position` JIT kernel · trtllm MoE fp32 upcasts (#25189) · penalizers (`scatter_`/
`torch.where`) · min_new_tokens (#28397) · FlashInfer/FA3 `seq_lens_cpu` threading ·
rmsnorm/rotary `forward_cuda` (already fused).
