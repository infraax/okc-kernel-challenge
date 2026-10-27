# muse_decode_layer — judge job 0108f70b516d4ea7 (speedup 1.226x, score_us 18443)

Units µs (judge median, replay overhead already subtracted; p10–p90 overhead-corrected; sol floored at 2.5 µs). loss = weight × ln(ours/sol): what the board still
pays on this shape vs the floor; Σloss over shapes is the problem's total headroom in log units.

| # | shape | w | sglang | ours | sol | speedup | ours/sol | p10–p90 | kernels Σ | hidden | idle | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1 | 46 | 172.2 | 125.2 | 40.0 | 1.375 | 3.13 | 118.5–131.0 | 140.1 | 8.7 (6) | -14.9 | 52.45 (12%) |
| 1 | batch=1 ctx=80000 dist=soft layer=full load=decode 1 page=1 | 16 | 183.2 | 137.0 | 44.2 | 1.337 | 3.10 | 130.1–143.0 | 152.8 | 10.0 (6) | -15.8 | 18.11 (4%) |
| 2 | batch=16 ctx=80000 dist=soft layer=sliding load=decode 16 page=1 spread=agentic | 46 | 267.9 | 193.7 | 76.7 | 1.383 | 2.53 | 179.7–207.1 | 193.0 | 4.1 (3) | 0.7 | 42.61 (10%) |
| 3 | batch=16 ctx=80000 dist=soft layer=full load=decode 16 page=1 spread=agentic | 16 | 438.6 | 340.4 | 137.5 | 1.288 | 2.48 | 322.9–358.0 | 373.1 | 4.0 (3) | -32.7 | 14.51 (3%) |
| 4 | batch=32 ctx=80000 dist=soft layer=sliding load=decode 32 page=1 spread=agentic | 46 | 271.6 | 240.9 | 84.5 | 1.127 | 2.85 | 220.8–258.8 | 272.6 | 7.1 (5) | -31.7 | 48.19 (11%) |
| 5 | batch=32 ctx=80000 dist=soft layer=full load=decode 32 page=1 spread=agentic | 16 | 622.1 | 539.8 | 208.1 | 1.153 | 2.59 | 516.1–560.4 | 565.4 | 7.1 (5) | -25.6 | 15.25 (4%) |
| 6 | batch=1 ctx=80000 dist=soft layer=sliding load=verify 1 page=1 q_len=8 | 46 | 267.9 | 178.9 | 62.3 | 1.498 | 2.87 | 163.4–198.0 | 210.3 | 10.6 (6) | -31.4 | 48.54 (11%) |
| 7 | batch=1 ctx=80000 dist=soft layer=full load=verify 1 page=1 q_len=8 | 16 | 277.1 | 186.1 | 66.4 | 1.489 | 2.80 | 175.0–210.2 | 227.4 | 11.4 (6) | -41.3 | 16.48 (4%) |
| 8 | batch=16 ctx=80000 dist=soft layer=sliding load=verify 16 page=1 q_len=8 spread=agentic | 46 | 533.5 | 502.2 | 126.6 | 1.062 | 3.97 | 455.9–534.4 | 526.4 | 12.8 (6) | -24.2 | 63.39 (15%) |
| 9 | batch=16 ctx=80000 dist=soft layer=full load=verify 16 page=1 q_len=8 spread=agentic | 16 | 685.0 | 664.2 | 188.7 | 1.031 | 3.52 | 633.2–696.5 | 721.4 | 12.3 (6) | -57.2 | 20.14 (5%) |
| 10 | batch=32 ctx=80000 dist=soft layer=sliding load=verify 32 page=1 q_len=8 spread=agentic | 46 | 701.2 | 681.2 | 143.8 | 1.029 | 4.74 | 655.8–728.2 | 707.8 | 10.2 (5) | -26.6 | 71.55 (17%) |
| 11 | batch=32 ctx=80000 dist=soft layer=full load=verify 32 page=1 q_len=8 spread=agentic | 16 | 1059.6 | 1017.4 | 267.2 | 1.042 | 3.81 | 975.5–1057.1 | 1075.5 | 10.4 (5) | -58.1 | 21.39 (5%) |

## shape 0: batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1  (w 46, ours 125.2 µs, sol 40.0)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 0.0 | `_gemv_splitk` | dense_gemm | 28.7 | 22.9% | 2 | 14.3 |
| 0.1 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 26.7 | 21.3% | 1 | 26.7 |
| 0.2 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 21.8 | 17.4% | 1 | 21.8 |
| 0.3 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 13.5 | 10.8% | 1 | 13.5 |
| 0.4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 7.0 | 5.6% | 1 | 7.0 |
| 0.5 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.9 | 4.7% | 1 | 5.9 |
| 0.6 | `_epilogue_kernel` | moe_epilogue | 5.9 | 4.7% | 1 | 5.9 |
| 0.7 | `_attn_kernel` | attention | 5.4 | 4.3% | 1 | 5.4 |
| 0.8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 5.1 | 4.1% | 1 | 5.1 |
| 0.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.3 | 3.4% | 1 | 4.3 |
| 0.10 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.7 | 3.0% | 1 | 3.7 |
| 0.11 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 3.4 | 2.7% | 2 | 1.7 |
| 0.– | (hidden tail) | 6 kernels not listed by the judge report | 8.7 | 6.9% | | |
| 0.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -14.9 | -11.9% | | |

## shape 1: batch=1 ctx=80000 dist=soft layer=full load=decode 1 page=1  (w 16, ours 137.0 µs, sol 44.2)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 1.0 | `_gemv_splitk` | dense_gemm | 29.0 | 21.2% | 2 | 14.5 |
| 1.1 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 26.5 | 19.3% | 1 | 26.5 |
| 1.2 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 23.1 | 16.9% | 1 | 23.1 |
| 1.3 | `_attn_kernel` | attention | 16.0 | 11.7% | 1 | 16.0 |
| 1.4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 12.4 | 9.1% | 1 | 12.4 |
| 1.5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 6.8 | 5.0% | 1 | 6.8 |
| 1.6 | `_epilogue_kernel` | moe_epilogue | 6.2 | 4.5% | 1 | 6.2 |
| 1.7 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 6.0 | 4.4% | 1 | 6.0 |
| 1.8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 5.2 | 3.8% | 1 | 5.2 |
| 1.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.4 | 3.2% | 1 | 4.4 |
| 1.10 | `_reduce_kernel` | attention_reduce | 3.6 | 2.6% | 1 | 3.6 |
| 1.11 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.6 | 2.6% | 1 | 3.6 |
| 1.– | (hidden tail) | 6 kernels not listed by the judge report | 10.0 | 7.3% | | |
| 1.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -15.8 | -11.5% | | |

## shape 2: batch=16 ctx=80000 dist=soft layer=sliding load=decode 16 page=1 spread=agentic  (w 46, ours 193.7 µs, sol 76.7)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 2.0 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 49.2 | 25.4% | 1 | 49.2 |
| 2.1 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 39.1 | 20.2% | 1 | 39.1 |
| 2.2 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 27.2 | 14.0% | 2 | 13.6 |
| 2.3 | `_gemv_splitk` | dense_gemm | 26.3 | 13.6% | 2 | 13.2 |
| 2.4 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 12.3 | 6.3% | 1 | 12.3 |
| 2.5 | `_attn_kernel` | attention | 10.4 | 5.4% | 1 | 10.4 |
| 2.6 | `_epilogue_kernel` | moe_epilogue | 6.1 | 3.1% | 1 | 6.1 |
| 2.7 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.5 | 2.8% | 1 | 5.5 |
| 2.8 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.8 | 2.0% | 1 | 3.8 |
| 2.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.5 | 1.8% | 1 | 3.5 |
| 2.10 | `_reduce_kernel` | attention_reduce | 2.8 | 1.4% | 1 | 2.8 |
| 2.11 | `void sglang::muse_spark::routing_kernel<sglang::muse_spar...` | routing | 2.7 | 1.4% | 1 | 2.7 |
| 2.– | (hidden tail) | 3 kernels not listed by the judge report | 4.1 | 2.1% | | |
| 2.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.7 | 0.4% | | |

## shape 3: batch=16 ctx=80000 dist=soft layer=full load=decode 16 page=1 spread=agentic  (w 16, ours 340.4 µs, sol 137.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 3.0 | `_attn_kernel` | attention | 162.1 | 47.6% | 1 | 162.1 |
| 3.1 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 65.7 | 19.3% | 1 | 65.7 |
| 3.2 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 49.2 | 14.5% | 1 | 49.2 |
| 3.3 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 26.7 | 7.8% | 2 | 13.3 |
| 3.4 | `_gemv_splitk` | dense_gemm | 26.7 | 7.8% | 2 | 13.3 |
| 3.5 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 12.3 | 3.6% | 1 | 12.3 |
| 3.6 | `_epilogue_kernel` | moe_epilogue | 6.1 | 1.8% | 1 | 6.1 |
| 3.7 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.8 | 1.7% | 1 | 5.8 |
| 3.8 | `_reduce_kernel` | attention_reduce | 4.5 | 1.3% | 1 | 4.5 |
| 3.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.8 | 1.1% | 1 | 3.8 |
| 3.10 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.5 | 1.0% | 1 | 3.5 |
| 3.11 | `void sglang::muse_spark::routing_kernel<sglang::muse_spar...` | routing | 2.7 | 0.8% | 1 | 2.7 |
| 3.– | (hidden tail) | 3 kernels not listed by the judge report | 4.0 | 1.2% | | |
| 3.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -32.7 | -9.6% | | |

## shape 4: batch=32 ctx=80000 dist=soft layer=sliding load=decode 32 page=1 spread=agentic  (w 46, ours 240.9 µs, sol 84.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 4.0 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...` | moe_gemm(marlin) | 159.1 | 66.0% | 2 | 79.5 |
| 4.1 | `_gemv_splitk` | dense_gemm | 27.6 | 11.5% | 2 | 13.8 |
| 4.2 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 27.4 | 11.4% | 2 | 13.7 |
| 4.3 | `_attn_kernel` | attention | 13.4 | 5.6% | 1 | 13.4 |
| 4.4 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 12.4 | 5.1% | 1 | 12.4 |
| 4.5 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.7 | 2.4% | 1 | 5.7 |
| 4.6 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.1 | 1.7% | 1 | 4.1 |
| 4.7 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 3.9 | 1.6% | 1 | 3.9 |
| 4.8 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.8 | 1.6% | 1 | 3.8 |
| 4.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spar...` | routing | 2.7 | 1.1% | 1 | 2.7 |
| 4.10 | `void moe_align_block_size_kernel<int>(int const*, int*, i...` | routing | 2.7 | 1.1% | 1 | 2.7 |
| 4.11 | `_reduce_kernel` | attention_reduce | 2.7 | 1.1% | 1 | 2.7 |
| 4.– | (hidden tail) | 5 kernels not listed by the judge report | 7.1 | 2.9% | | |
| 4.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -31.7 | -13.2% | | |

## shape 5: batch=32 ctx=80000 dist=soft layer=full load=decode 32 page=1 spread=agentic  (w 16, ours 539.8 µs, sol 208.1)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 5.0 | `_attn_kernel` | attention | 316.7 | 58.7% | 1 | 316.7 |
| 5.1 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...` | moe_gemm(marlin) | 138.4 | 25.6% | 2 | 69.2 |
| 5.2 | `_gemv_splitk` | dense_gemm | 34.8 | 6.4% | 2 | 17.4 |
| 5.3 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 26.5 | 4.9% | 2 | 13.2 |
| 5.4 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 12.6 | 2.3% | 1 | 12.6 |
| 5.5 | `_reduce_kernel` | attention_reduce | 6.1 | 1.1% | 1 | 6.1 |
| 5.6 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.6 | 1.0% | 1 | 5.6 |
| 5.7 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.3 | 0.8% | 1 | 4.3 |
| 5.8 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 4.0 | 0.7% | 1 | 4.0 |
| 5.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.7 | 0.7% | 1 | 3.7 |
| 5.10 | `void sglang::muse_spark::routing_kernel<sglang::muse_spar...` | routing | 2.9 | 0.5% | 1 | 2.9 |
| 5.11 | `void moe_align_block_size_kernel<int>(int const*, int*, i...` | routing | 2.7 | 0.5% | 1 | 2.7 |
| 5.– | (hidden tail) | 5 kernels not listed by the judge report | 7.1 | 1.3% | | |
| 5.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -25.6 | -4.7% | | |

## shape 6: batch=1 ctx=80000 dist=soft layer=sliding load=verify 1 page=1 q_len=8  (w 46, ours 178.9 µs, sol 62.3)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 6.0 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 65.5 | 36.6% | 1 | 65.5 |
| 6.1 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 50.7 | 28.3% | 1 | 50.7 |
| 6.2 | `_gemv_splitk` | dense_gemm | 29.7 | 16.6% | 2 | 14.8 |
| 6.3 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 13.4 | 7.5% | 1 | 13.4 |
| 6.4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 7.6 | 4.2% | 1 | 7.6 |
| 6.5 | `_attn_kernel` | attention | 5.9 | 3.3% | 1 | 5.9 |
| 6.6 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.8 | 3.2% | 1 | 5.8 |
| 6.7 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 5.4 | 3.0% | 1 | 5.4 |
| 6.8 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.7 | 2.6% | 1 | 4.7 |
| 6.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 3.8 | 2.1% | 1 | 3.8 |
| 6.10 | `_epilogue_kernel` | moe_epilogue | 3.7 | 2.1% | 1 | 3.7 |
| 6.11 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 3.5 | 2.0% | 2 | 1.8 |
| 6.– | (hidden tail) | 6 kernels not listed by the judge report | 10.6 | 5.9% | | |
| 6.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -31.4 | -17.5% | | |

## shape 7: batch=1 ctx=80000 dist=soft layer=full load=verify 1 page=1 q_len=8  (w 16, ours 186.1 µs, sol 66.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 7.0 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 65.2 | 35.0% | 1 | 65.2 |
| 7.1 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 54.9 | 29.5% | 1 | 54.9 |
| 7.2 | `_gemv_splitk` | dense_gemm | 31.0 | 16.7% | 2 | 15.5 |
| 7.3 | `_attn_kernel` | attention | 17.3 | 9.3% | 1 | 17.3 |
| 7.4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 13.1 | 7.0% | 1 | 13.1 |
| 7.5 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 7.0 | 3.8% | 1 | 7.0 |
| 7.6 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.8 | 3.1% | 1 | 5.8 |
| 7.7 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 5.4 | 2.9% | 1 | 5.4 |
| 7.8 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.9 | 2.6% | 1 | 4.9 |
| 7.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.0 | 2.1% | 1 | 4.0 |
| 7.10 | `_epilogue_kernel` | moe_epilogue | 3.7 | 2.0% | 1 | 3.7 |
| 7.11 | `_reduce_kernel` | attention_reduce | 3.7 | 2.0% | 1 | 3.7 |
| 7.– | (hidden tail) | 6 kernels not listed by the judge report | 11.4 | 6.1% | | |
| 7.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -41.3 | -22.2% | | |

## shape 8: batch=16 ctx=80000 dist=soft layer=sliding load=verify 16 page=1 q_len=8 spread=agentic  (w 46, ours 502.2 µs, sol 126.6)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 8.0 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...` | moe_gemm(marlin) | 392.8 | 78.2% | 2 | 196.4 |
| 8.1 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 29.0 | 5.8% | 2 | 14.5 |
| 8.2 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | dense_gemm(cublas) | 25.2 | 5.0% | 1 | 25.2 |
| 8.3 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` | dense_gemm(cublas) | 16.4 | 3.3% | 1 | 16.4 |
| 8.4 | `_attn_kernel` | attention | 11.9 | 2.4% | 1 | 11.9 |
| 8.5 | `_gemv_splitk` | dense_gemm | 9.1 | 1.8% | 1 | 9.1 |
| 8.6 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 5.9 | 1.2% | 1 | 5.9 |
| 8.7 | `_reduce_kernel` | attention_reduce | 5.4 | 1.1% | 1 | 5.4 |
| 8.8 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 4.9 | 1.0% | 1 | 4.9 |
| 8.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.9 | 1.0% | 1 | 4.9 |
| 8.10 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.5 | 0.9% | 1 | 4.5 |
| 8.11 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...` | activation | 3.6 | 0.7% | 1 | 3.6 |
| 8.– | (hidden tail) | 6 kernels not listed by the judge report | 12.8 | 2.5% | | |
| 8.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -24.2 | -4.8% | | |

## shape 9: batch=16 ctx=80000 dist=soft layer=full load=verify 16 page=1 q_len=8 spread=agentic  (w 16, ours 664.2 µs, sol 188.7)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 9.0 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...` | moe_gemm(marlin) | 407.5 | 61.4% | 2 | 203.8 |
| 9.1 | `_attn_kernel` | attention | 184.7 | 27.8% | 1 | 184.7 |
| 9.2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 28.1 | 4.2% | 2 | 14.1 |
| 9.3 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | dense_gemm(cublas) | 24.1 | 3.6% | 1 | 24.1 |
| 9.4 | `_reduce_kernel` | attention_reduce | 16.0 | 2.4% | 1 | 16.0 |
| 9.5 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` | dense_gemm(cublas) | 15.3 | 2.3% | 1 | 15.3 |
| 9.6 | `_gemv_splitk` | dense_gemm | 9.0 | 1.4% | 1 | 9.0 |
| 9.7 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 6.2 | 0.9% | 1 | 6.2 |
| 9.8 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 5.2 | 0.8% | 1 | 5.2 |
| 9.9 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.8 | 0.7% | 1 | 4.8 |
| 9.10 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...` | norm/residual | 4.4 | 0.7% | 1 | 4.4 |
| 9.11 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...` | activation | 3.8 | 0.6% | 1 | 3.8 |
| 9.– | (hidden tail) | 6 kernels not listed by the judge report | 12.3 | 1.9% | | |
| 9.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -57.2 | -8.6% | | |

## shape 10: batch=32 ctx=80000 dist=soft layer=sliding load=verify 32 page=1 q_len=8 spread=agentic  (w 46, ours 681.2 µs, sol 143.8)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 10.0 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...` | moe_gemm(marlin) | 536.0 | 78.7% | 2 | 268.0 |
| 10.1 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | dense_gemm(cublas) | 34.4 | 5.0% | 2 | 17.2 |
| 10.2 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 31.6 | 4.6% | 1 | 31.6 |
| 10.3 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | dense_gemm(cublas) | 23.0 | 3.4% | 1 | 23.0 |
| 10.4 | `_attn_kernel` | attention | 18.6 | 2.7% | 1 | 18.6 |
| 10.5 | `_gemv_splitk` | dense_gemm | 11.7 | 1.7% | 1 | 11.7 |
| 10.6 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...` | norm/residual | 8.7 | 1.3% | 1 | 8.7 |
| 10.7 | `_reduce_kernel` | attention_reduce | 8.1 | 1.2% | 1 | 8.1 |
| 10.8 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...` | norm/residual | 7.7 | 1.1% | 1 | 7.7 |
| 10.9 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 6.0 | 0.9% | 1 | 6.0 |
| 10.10 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 5.9 | 0.9% | 1 | 5.9 |
| 10.11 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...` | activation | 5.9 | 0.9% | 1 | 5.9 |
| 10.– | (hidden tail) | 5 kernels not listed by the judge report | 10.2 | 1.5% | | |
| 10.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -26.6 | -3.9% | | |

## shape 11: batch=32 ctx=80000 dist=soft layer=full load=verify 32 page=1 q_len=8 spread=agentic  (w 16, ours 1017.4 µs, sol 267.2)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 11.0 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...` | moe_gemm(marlin) | 537.5 | 52.8% | 2 | 268.8 |
| 11.1 | `_attn_kernel` | attention | 342.4 | 33.7% | 1 | 342.4 |
| 11.2 | `_gemv_splitk` | dense_gemm | 43.0 | 4.2% | 1 | 43.0 |
| 11.3 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 35.1 | 3.4% | 1 | 35.1 |
| 11.4 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | dense_gemm(cublas) | 34.3 | 3.4% | 2 | 17.1 |
| 11.5 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | dense_gemm(cublas) | 23.0 | 2.3% | 1 | 23.0 |
| 11.6 | `_reduce_kernel` | attention_reduce | 16.4 | 1.6% | 1 | 16.4 |
| 11.7 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...` | norm/residual | 8.7 | 0.9% | 1 | 8.7 |
| 11.8 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...` | norm/residual | 7.8 | 0.8% | 1 | 7.8 |
| 11.9 | `void at::native::(anonymous namespace)::vectorized_layer_...` | norm/residual | 6.0 | 0.6% | 1 | 6.0 |
| 11.10 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...` | activation | 5.7 | 0.6% | 1 | 5.7 |
| 11.11 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 5.2 | 0.5% | 1 | 5.2 |
| 11.– | (hidden tail) | 5 kernels not listed by the judge report | 10.4 | 1.0% | | |
| 11.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -58.1 | -5.7% | | |

## families, weighted by each shape's loss share (where the board's headroom sits)

| family | weighted loss share |
|---|---|
| moe_gemm(marlin) | 36.3% |
| moe_gemm(cuda smallm) | 20.5% |
| dense_gemm(cublas) | 14.8% |
| attention | 9.8% |
| dense_gemm | 9.6% |
| norm/residual | 5.4% |
| moe_epilogue | 1.9% |
| attention_reduce | 1.0% |
| routing | 0.4% |
| activation | 0.3% |
