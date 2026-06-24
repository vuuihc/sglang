Triage View
Mode: single-trace
Framework: SGLang
Input traces: /Users/bytedance/works/sglang/.contribution-scan/traces/dense_batch1_input512_output64_decode.trace.json.gz

Kernel Table
##### decode
| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT | gemm | 12.30 ms | 43.9% | 360 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 5.39 ms | 19.3% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| nvjet_sm90_tst_256x8_64x6_2x1_v_bz_splitK_TNT | gemm | 3.09 ms | 11.0% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT | gemm | 1.72 ms | 6.1% | 5 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, 128, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<128>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true> > > > | gemm | 1.53 ms | 5.5% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign_0 | gemm | 0.96 ms | 3.4% | 360 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.81 ms | 2.9% | 360 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, cute::C<128> >, 5, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> > | gemm | 0.44 ms | 1.6% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::ActivationKind)0, true, false> | activation | 0.36 ms | 1.3% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16> | other | 0.35 ms | 1.2% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |
| void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u> | rope | 0.30 ms | 1.1% | 180 | python/sglang/srt/model_executor/model_runner.py:3502 _forward_raw | cudaGraphLaunch |

Overlap Opportunity Table
| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | No rows cleared the 1.0% reporting bar. Use mapping/formal mode for overlap attribution. | - | - | - | - |

Fuse Opportunity Table
##### decode
| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Confirmed | 22.82 ms | 81.5% | nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT (43.9%)<br>nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT (19.3%)<br>nvjet_sm90_tst_256x8_64x6_2x1_v_bz_splitK_TNT (11.0%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:3502<br>forward @ python/sglang/srt/layers/sampler.py:91 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | Matches an open upstream path (81.5% related GPU time). Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| Fused QK RoPE reshape + KV cache write | Candidate | 0.56 ms | 2.0% | void (anonymous namespace)::fused_rope_kernel<true, 128l, true, __nv_bfloat16, long, 16u> (1.1%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:3502 | python/sglang/srt/layers/attention/utils.py | Split kernels in this family take 2.0% of GPU time. This tree already has a matching path. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| Fused residual add + RMSNorm | Confirmed | 0.96 ms | 3.4% | kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign128o409640961_tensorptrbf16gmemalign_0 (3.4%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:3502 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | `Fused residual add + RMSNorm` is present in this trace (3.4% related GPU time). Residual add plus RMSNorm already has fused implementations across several backends. |
| In-place QK RMSNorm | Candidate | 0.35 ms | 1.2% | void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16> (1.2%) | _forward_raw @ python/sglang/srt/model_executor/model_runner.py:3502 | python/sglang/srt/models/utils.py<br>python/sglang/jit_kernel/norm.py | Split kernels in this family take 1.2% of GPU time. This tree already has a matching path. Q/K normalization already has in-place or model-specific fused implementations. |
