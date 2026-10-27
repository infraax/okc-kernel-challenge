# TRACE report

- problem=muse_decode_layer file=sonnet_v4.py torch=2.14.1+cu130 cuda=13.0 gpu=NVIDIA_H200 cupti=graph replays=20
- prepare time 47 s

## invariants

- all checks pass

## shape 0: batch=1,ctx=80000,dist=soft,layer=sliding,load=decode_1,page=1  (w 46, wall 129.63 us, span 125.28 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 0.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 0.00 | 12.59 | 0.00 | 0 | 9.7% |
| 0.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 13.42 | 1.79 | 0.83 | 0 | 1.4% |
| 0.2 | `_attn_kernel` | 15.60 | 4.96 | 0.42 | 0 | 3.8% |
| 0.3 | `_reduce_kernel` | 20.94 | 2.43 | 0.38 | 0 | 1.9% |
| 0.4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 23.95 | 6.62 | 0.42 | 0 | 5.1% |
| 0.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 30.82 | 4.69 | 0.07 | 0 | 3.6% |
| 0.6 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 36.03 | 3.44 | 0.64 | 1 | 2.7% |
| 0.7 | `_gemv_splitk` | 36.11 | 19.20 | 0.00 | 0 | 14.8% |
| 0.8 | `_gemv_splitk` | 40.82 | 22.02 | 0.00 | 1 | 17.0% |
| 0.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 62.75 | 2.88 | 0.06 | 1 | 2.2% |
| 0.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 66.11 | 6.26 | 0.42 | 0 | 4.8% |
| 0.11 | `smallm_route_kernel(int const*, int*, int*, int, int const*, int)` | 72.96 | 1.50 | 0.42 | 0 | 1.2% |
| 0.12 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 73.92 | 16.93 | 0.00 | 0 | 13.1% |
| 0.13 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const*, float cons` | 75.95 | 27.54 | 0.00 | 0 | 21.2% |
| 0.14 | `_epilogue_kernel` | 104.14 | 5.90 | 0.70 | 0 | 4.6% |
| 0.15 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 110.58 | 5.18 | 0.58 | 0 | 4.0% |
| 0.16 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 116.58 | 1.65 | 0.61 | 0 | 1.3% |
| 0.17 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunc` | 118.69 | 0.83 | 0.40 | 0 | 0.6% |
| 0.18 | `Memcpy DtoD (Device -> Device)` | 119.94 | 0.85 | 0.42 | 0 | 0.7% |
| 0.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 121.26 | 3.95 | 0.45 | 0 | 3.0% |

Totals: sum dur 151.21 | busy 117.92 | sum gaps 7.15 | head 4.13 | tail 3.98 | wall - span 4.35 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.83 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` (0.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (0.1) |
| 0.70 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint` (0.13) | `_epilogue_kernel` (0.14) |
| 0.64 | `void sglang::muse_spark_norm_residual_kernel<8192l` (0.5) | `void at::native::vectorized_elementwise_kernel<4, ` (0.6) |
| 0.61 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (0.15) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (0.16) |
| 0.58 | `_epilogue_kernel` (0.14) | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (0.15) |

Families (sum dur): moe_gemm(cuda smallm) 45.97, dense_gemm 41.22, dense_gemm(cublas) 27.83, norm/residual 14.90, moe_epilogue 5.90, attention 4.96, elementwise 4.27, routing 2.88, attention_reduce 2.43, other 0.85

## shape 1: batch=1,ctx=80000,dist=soft,layer=full,load=decode_1,page=1  (w 16, wall 151.52 us, span 147.30 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 1.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 0.00 | 12.00 | 0.00 | 0 | 7.9% |
| 1.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.78 | 1.76 | 0.72 | 0 | 1.2% |
| 1.2 | `_attn_kernel` | 14.94 | 15.76 | 0.38 | 0 | 10.4% |
| 1.3 | `_reduce_kernel` | 31.26 | 3.57 | 0.43 | 0 | 2.4% |
| 1.4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 35.25 | 6.62 | 0.42 | 0 | 4.4% |
| 1.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 42.00 | 4.70 | 0.13 | 0 | 3.1% |
| 1.6 | `_gemv_splitk` | 47.14 | 21.06 | 0.48 | 0 | 13.9% |
| 1.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 47.31 | 3.52 | 0.00 | 1 | 2.3% |
| 1.8 | `_gemv_splitk` | 51.54 | 23.65 | 0.00 | 1 | 15.6% |
| 1.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 75.33 | 2.85 | 0.10 | 1 | 1.9% |
| 1.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 78.59 | 6.16 | 0.42 | 0 | 4.1% |
| 1.11 | `smallm_route_kernel(int const*, int*, int*, int, int const*, int)` | 85.31 | 1.41 | 0.42 | 0 | 0.9% |
| 1.12 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 86.30 | 26.21 | 0.00 | 0 | 17.3% |
| 1.13 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const*, float cons` | 88.34 | 37.30 | 0.00 | 0 | 24.6% |
| 1.14 | `_epilogue_kernel` | 126.05 | 5.94 | 0.42 | 0 | 3.9% |
| 1.15 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 132.61 | 5.22 | 0.58 | 0 | 3.4% |
| 1.16 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 138.43 | 1.73 | 0.64 | 0 | 1.1% |
| 1.17 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunc` | 140.59 | 0.86 | 0.42 | 0 | 0.6% |
| 1.18 | `Memcpy DtoD (Device -> Device)` | 141.87 | 0.96 | 0.42 | 0 | 0.6% |
| 1.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 143.30 | 4.00 | 0.42 | 0 | 2.6% |

Totals: sum dur 185.28 | busy 140.59 | sum gaps 6.80 | head 4.02 | tail 3.98 | wall - span 4.22 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.72 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` (1.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (1.1) |
| 0.64 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (1.15) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (1.16) |
| 0.58 | `_epilogue_kernel` (1.14) | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (1.15) |
| 0.48 | `void sglang::muse_spark_norm_residual_kernel<8192l` (1.5) | `_gemv_splitk` (1.6) |
| 0.43 | `_attn_kernel` (1.2) | `_reduce_kernel` (1.3) |

Families (sum dur): moe_gemm(cuda smallm) 64.92, dense_gemm 44.71, dense_gemm(cublas) 27.33, attention 15.76, norm/residual 14.86, moe_epilogue 5.94, elementwise 4.38, attention_reduce 3.57, routing 2.85, other 0.96

## shape 2: batch=16,ctx=80000,dist=soft,layer=sliding,load=decode_16,page=1,spread=agentic  (w 46, wall 178.82 us, span 174.56 us, n 17, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 2.0 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 11.84 | 0.00 | 0 | 6.6% |
| 2.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.77 | 1.86 | 0.90 | 0 | 1.0% |
| 2.2 | `_attn_kernel` | 15.04 | 10.03 | 0.38 | 0 | 5.6% |
| 2.3 | `_reduce_kernel` | 25.49 | 2.64 | 0.39 | 0 | 1.5% |
| 2.4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | 28.58 | 6.53 | 0.45 | 0 | 3.7% |
| 2.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 35.39 | 3.78 | 0.13 | 0 | 2.1% |
| 2.6 | `_gemv_splitk` | 39.74 | 22.74 | 0.45 | 0 | 12.7% |
| 2.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 39.90 | 4.26 | 0.00 | 1 | 2.4% |
| 2.8 | `_gemv_splitk` | 45.17 | 14.35 | 0.00 | 1 | 8.0% |
| 2.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 59.52 | 3.10 | 0.00 | 1 | 1.7% |
| 2.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 63.28 | 5.62 | 0.42 | 0 | 3.1% |
| 2.11 | `smallm_route_kernel(int const*, int*, int*, int, int const*, int)` | 69.31 | 1.57 | 0.42 | 0 | 0.9% |
| 2.12 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 70.30 | 46.13 | 0.00 | 0 | 25.8% |
| 2.13 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*, float cons` | 109.18 | 34.22 | 0.00 | 0 | 19.1% |
| 2.14 | `_epilogue_kernel` | 144.08 | 5.70 | 0.42 | 0 | 3.2% |
| 2.15 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | 150.18 | 19.98 | 0.42 | 0 | 11.2% |
| 2.16 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 170.22 | 4.32 | 0.03 | 0 | 2.4% |

Totals: sum dur 198.67 | busy 169.65 | sum gaps 4.64 | head 4.06 | tail 3.94 | wall - span 4.26 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.90 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (2.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (2.1) |
| 0.45 | `_reduce_kernel` (2.3) | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` (2.4) |
| 0.45 | `void sglang::muse_spark_norm_residual_kernel<8192l` (2.5) | `_gemv_splitk` (2.6) |
| 0.42 | `void sglang::muse_spark::routing_kernel<sglang::mu` (2.9) | `void at::native::(anonymous namespace)::vectorized` (2.10) |
| 0.42 | `void at::native::(anonymous namespace)::vectorized` (2.10) | `smallm_route_kernel(int const*, int*, int*, int, i` (2.11) |

Families (sum dur): moe_gemm(cuda smallm) 81.92, dense_gemm(cublas) 40.21, dense_gemm 37.09, norm/residual 13.72, attention 10.03, moe_epilogue 5.70, elementwise 4.26, routing 3.10, attention_reduce 2.64

## shape 3: batch=16,ctx=80000,dist=soft,layer=full,load=decode_16,page=1,spread=agentic  (w 16, wall 340.56 us, span 336.05 us, n 17, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 3.0 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 11.87 | 0.00 | 0 | 3.5% |
| 3.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.74 | 1.82 | 0.80 | 0 | 0.5% |
| 3.2 | `_attn_kernel` | 14.99 | 160.27 | 0.38 | 0 | 47.1% |
| 3.3 | `_reduce_kernel` | 175.66 | 4.13 | 0.29 | 0 | 1.2% |
| 3.4 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | 180.27 | 6.82 | 0.45 | 0 | 2.0% |
| 3.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 187.34 | 3.73 | 0.16 | 0 | 1.1% |
| 3.6 | `_gemv_splitk` | 191.47 | 23.02 | 0.45 | 0 | 6.8% |
| 3.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 191.66 | 4.14 | 0.00 | 1 | 1.2% |
| 3.8 | `_gemv_splitk` | 196.94 | 14.61 | 0.00 | 1 | 4.3% |
| 3.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 211.20 | 3.07 | 0.00 | 1 | 0.9% |
| 3.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 214.90 | 5.73 | 0.42 | 0 | 1.7% |
| 3.11 | `smallm_route_kernel(int const*, int*, int*, int, int const*, int)` | 220.96 | 1.44 | 0.42 | 0 | 0.4% |
| 3.12 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 221.89 | 53.68 | 0.00 | 0 | 15.8% |
| 3.13 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*, float cons` | 266.94 | 38.06 | 0.00 | 0 | 11.2% |
| 3.14 | `_epilogue_kernel` | 305.60 | 5.66 | 0.42 | 0 | 1.7% |
| 3.15 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | 311.66 | 20.03 | 0.45 | 0 | 5.9% |
| 3.16 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 331.84 | 4.29 | 0.03 | 0 | 1.3% |

Totals: sum dur 362.37 | busy 331.68 | sum gaps 4.41 | head 3.98 | tail 3.89 | wall - span 4.51 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.80 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (3.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (3.1) |
| 0.45 | `_reduce_kernel` (3.3) | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` (3.4) |
| 0.45 | `void sglang::muse_spark_norm_residual_kernel<8192l` (3.5) | `_gemv_splitk` (3.6) |
| 0.45 | `_epilogue_kernel` (3.14) | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` (3.15) |
| 0.42 | `void sglang::muse_spark::routing_kernel<sglang::mu` (3.9) | `void at::native::(anonymous namespace)::vectorized` (3.10) |

Families (sum dur): attention 160.27, moe_gemm(cuda smallm) 93.18, dense_gemm(cublas) 40.54, dense_gemm 37.63, norm/residual 13.75, moe_epilogue 5.66, elementwise 4.14, attention_reduce 4.13, routing 3.07

## shape 4: batch=32,ctx=80000,dist=soft,layer=sliding,load=decode_32,page=1,spread=agentic  (w 46, wall 220.05 us, span 216.00 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 4.0 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 11.98 | 0.00 | 0 | 5.4% |
| 4.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.74 | 1.92 | 0.67 | 0 | 0.9% |
| 4.2 | `_attn_kernel` | 15.07 | 13.42 | 0.38 | 0 | 6.1% |
| 4.3 | `_reduce_kernel` | 29.01 | 2.64 | 0.45 | 0 | 1.2% |
| 4.4 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | 32.16 | 6.88 | 0.42 | 0 | 3.1% |
| 4.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 39.28 | 3.84 | 0.13 | 0 | 1.7% |
| 4.6 | `_gemv_splitk` | 43.60 | 23.71 | 0.42 | 0 | 10.8% |
| 4.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 43.76 | 2.91 | 0.00 | 1 | 1.3% |
| 4.8 | `_gemv_splitk` | 47.04 | 10.38 | 0.00 | 1 | 4.7% |
| 4.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 57.81 | 5.06 | 0.00 | 1 | 2.3% |
| 4.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 68.38 | 5.82 | 0.77 | 0 | 2.6% |
| 4.11 | `_local_ids_kernel` | 74.75 | 1.18 | 0.42 | 0 | 0.5% |
| 4.12 | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, in` | 76.35 | 2.82 | 0.39 | 0 | 1.3% |
| 4.13 | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, ` | 79.58 | 1.38 | 0.42 | 0 | 0.6% |
| 4.14 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 81.38 | 64.59 | 0.42 | 0 | 29.4% |
| 4.15 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind` | 147.06 | 2.21 | 0.93 | 0 | 1.0% |
| 4.16 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 149.78 | 37.01 | 0.42 | 0 | 16.8% |
| 4.17 | `_post_expert_norm_weighted_sum_kernel` | 187.31 | 4.00 | 0.53 | 0 | 1.8% |
| 4.18 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | 191.73 | 19.65 | 0.43 | 0 | 8.9% |
| 4.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 211.42 | 4.61 | 0.03 | 0 | 2.1% |

Totals: sum dur 226.01 | busy 208.32 | sum gaps 7.63 | head 3.84 | tail 3.95 | wall - span 4.05 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.93 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (4.14) | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (4.15) |
| 0.77 | `void sglang::muse_spark::routing_kernel<sglang::mu` (4.9) | `void at::native::(anonymous namespace)::vectorized` (4.10) |
| 0.67 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (4.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (4.1) |
| 0.53 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (4.16) | `_post_expert_norm_weighted_sum_kernel` (4.17) |
| 0.45 | `_attn_kernel` (4.2) | `_reduce_kernel` (4.3) |

Families (sum dur): moe_gemm(marlin) 101.60, dense_gemm(cublas) 40.43, dense_gemm 34.09, norm/residual 18.27, attention 13.42, routing 7.88, elementwise 2.91, attention_reduce 2.64, other 2.56, activation 2.21

## shape 5: batch=32,ctx=80000,dist=soft,layer=full,load=decode_32,page=1,spread=agentic  (w 16, wall 535.73 us, span 531.68 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 5.0 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 11.94 | 0.00 | 0 | 2.2% |
| 5.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.77 | 1.92 | 0.70 | 0 | 0.4% |
| 5.2 | `_attn_kernel` | 15.12 | 314.88 | 0.38 | 0 | 58.8% |
| 5.3 | `_reduce_kernel` | 330.70 | 5.95 | 0.26 | 0 | 1.1% |
| 5.4 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | 337.06 | 6.78 | 0.45 | 0 | 1.3% |
| 5.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 344.10 | 3.81 | 0.13 | 0 | 0.7% |
| 5.6 | `_gemv_splitk` | 348.50 | 23.71 | 0.48 | 0 | 4.4% |
| 5.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 348.67 | 2.94 | 0.00 | 1 | 0.5% |
| 5.8 | `_gemv_splitk` | 352.19 | 10.93 | 0.00 | 1 | 2.0% |
| 5.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 363.55 | 5.02 | 0.00 | 1 | 0.9% |
| 5.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 373.38 | 5.84 | 0.85 | 0 | 1.1% |
| 5.11 | `_local_ids_kernel` | 379.54 | 1.10 | 0.42 | 0 | 0.2% |
| 5.12 | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, in` | 381.04 | 2.82 | 0.42 | 0 | 0.5% |
| 5.13 | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, ` | 384.22 | 1.41 | 0.35 | 0 | 0.3% |
| 5.14 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 386.06 | 71.87 | 0.42 | 0 | 13.4% |
| 5.15 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind` | 458.96 | 2.24 | 0.86 | 0 | 0.4% |
| 5.16 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 461.65 | 40.59 | 0.40 | 0 | 7.6% |
| 5.17 | `_post_expert_norm_weighted_sum_kernel` | 502.67 | 4.02 | 0.54 | 0 | 0.8% |
| 5.18 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | 507.10 | 19.74 | 0.42 | 0 | 3.7% |
| 5.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 527.13 | 4.59 | 0.00 | 0 | 0.9% |

Totals: sum dur 542.10 | busy 524.05 | sum gaps 7.46 | head 3.94 | tail 3.90 | wall - span 4.05 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.86 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (5.14) | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (5.15) |
| 0.85 | `void sglang::muse_spark::routing_kernel<sglang::mu` (5.9) | `void at::native::(anonymous namespace)::vectorized` (5.10) |
| 0.70 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (5.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (5.1) |
| 0.54 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (5.16) | `_post_expert_norm_weighted_sum_kernel` (5.17) |
| 0.48 | `void sglang::muse_spark_norm_residual_kernel<8192l` (5.5) | `_gemv_splitk` (5.6) |

Families (sum dur): attention 314.88, moe_gemm(marlin) 112.46, dense_gemm(cublas) 40.38, dense_gemm 34.64, norm/residual 18.26, routing 7.84, attention_reduce 5.95, elementwise 2.94, other 2.51, activation 2.24

## shape 6: batch=1,ctx=80000,dist=soft,layer=sliding,load=verify_1,page=1,q_len=8  (w 46, wall 204.64 us, span 200.72 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 6.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 0.00 | 12.35 | 0.00 | 0 | 6.0% |
| 6.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 13.17 | 1.79 | 0.74 | 0 | 0.9% |
| 6.2 | `_attn_kernel` | 15.34 | 5.78 | 0.42 | 0 | 2.8% |
| 6.3 | `_reduce_kernel` | 21.95 | 2.43 | 0.74 | 0 | 1.2% |
| 6.4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 24.93 | 6.50 | 0.48 | 0 | 3.2% |
| 6.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 31.52 | 4.91 | 0.13 | 0 | 2.4% |
| 6.6 | `_gemv_splitk` | 36.93 | 20.37 | 0.48 | 0 | 10.0% |
| 6.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 37.12 | 4.24 | 0.00 | 1 | 2.1% |
| 6.8 | `_gemv_splitk` | 41.66 | 22.35 | 0.00 | 1 | 10.9% |
| 6.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 64.22 | 2.93 | 0.06 | 1 | 1.4% |
| 6.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 67.57 | 5.98 | 0.42 | 0 | 2.9% |
| 6.11 | `smallm_route_kernel(int const*, int*, int*, int, int const*, int)` | 73.98 | 1.58 | 0.42 | 0 | 0.8% |
| 6.12 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 75.01 | 64.43 | 0.00 | 0 | 31.5% |
| 6.13 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*, float cons` | 128.62 | 50.98 | 0.00 | 0 | 24.9% |
| 6.14 | `_epilogue_kernel` | 180.08 | 3.71 | 0.42 | 0 | 1.8% |
| 6.15 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 184.27 | 5.38 | 0.42 | 0 | 2.6% |
| 6.16 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 190.42 | 1.76 | 0.64 | 0 | 0.9% |
| 6.17 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunc` | 192.77 | 0.98 | 0.42 | 0 | 0.5% |
| 6.18 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp` | 194.08 | 2.11 | 0.35 | 0 | 1.0% |
| 6.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 196.38 | 4.18 | 0.19 | 0 | 2.0% |

Totals: sum dur 224.74 | busy 193.89 | sum gaps 6.74 | head 3.92 | tail 3.94 | wall - span 3.92 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.74 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` (6.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (6.1) |
| 0.74 | `_attn_kernel` (6.2) | `_reduce_kernel` (6.3) |
| 0.64 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (6.15) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (6.16) |
| 0.48 | `_reduce_kernel` (6.3) | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` (6.4) |
| 0.48 | `void sglang::muse_spark_norm_residual_kernel<8192l` (6.5) | `_gemv_splitk` (6.6) |

Families (sum dur): moe_gemm(cuda smallm) 116.99, dense_gemm 42.72, dense_gemm(cublas) 27.78, norm/residual 15.07, elementwise 7.33, attention 5.78, moe_epilogue 3.71, routing 2.93, attention_reduce 2.43

## shape 7: batch=1,ctx=80000,dist=soft,layer=full,load=verify_1,page=1,q_len=8  (w 16, wall 192.94 us, span 188.69 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 7.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 0.00 | 12.45 | 0.00 | 0 | 6.5% |
| 7.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 13.42 | 1.86 | 0.96 | 0 | 1.0% |
| 7.2 | `_attn_kernel` | 15.71 | 16.24 | 0.39 | 0 | 8.4% |
| 7.3 | `_reduce_kernel` | 32.42 | 3.62 | 0.42 | 0 | 1.9% |
| 7.4 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 36.59 | 6.85 | 0.43 | 0 | 3.6% |
| 7.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 43.58 | 4.90 | 0.13 | 0 | 2.5% |
| 7.6 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 49.22 | 4.19 | 0.67 | 1 | 2.2% |
| 7.7 | `_gemv_splitk` | 49.31 | 21.09 | 0.00 | 0 | 10.9% |
| 7.8 | `_gemv_splitk` | 54.13 | 23.17 | 0.00 | 1 | 12.0% |
| 7.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 77.50 | 2.93 | 0.10 | 1 | 1.5% |
| 7.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 80.85 | 5.95 | 0.42 | 0 | 3.1% |
| 7.11 | `smallm_route_kernel(int const*, int*, int*, int, int const*, int)` | 87.22 | 1.57 | 0.42 | 0 | 0.8% |
| 7.12 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 88.22 | 52.40 | 0.00 | 0 | 27.2% |
| 7.13 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*, float cons` | 131.41 | 36.64 | 0.00 | 0 | 19.0% |
| 7.14 | `_epilogue_kernel` | 168.53 | 3.60 | 0.35 | 0 | 1.9% |
| 7.15 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 172.56 | 5.28 | 0.43 | 0 | 2.7% |
| 7.16 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 178.45 | 1.73 | 0.61 | 0 | 0.9% |
| 7.17 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunc` | 180.80 | 0.93 | 0.45 | 0 | 0.5% |
| 7.18 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp` | 182.11 | 2.11 | 0.42 | 0 | 1.1% |
| 7.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 184.54 | 4.14 | 0.19 | 0 | 2.1% |

Totals: sum dur 211.65 | busy 181.76 | sum gaps 6.72 | head 4.06 | tail 3.90 | wall - span 4.25 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.96 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` (7.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (7.1) |
| 0.67 | `void sglang::muse_spark_norm_residual_kernel<8192l` (7.5) | `void at::native::vectorized_elementwise_kernel<4, ` (7.6) |
| 0.61 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (7.15) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (7.16) |
| 0.45 | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (7.16) | `void at::native::vectorized_elementwise_kernel<8, ` (7.17) |
| 0.43 | `_reduce_kernel` (7.3) | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` (7.4) |

Families (sum dur): moe_gemm(cuda smallm) 90.61, dense_gemm 44.26, dense_gemm(cublas) 28.17, attention 16.24, norm/residual 14.99, elementwise 7.23, attention_reduce 3.62, moe_epilogue 3.60, routing 2.93

## shape 8: batch=16,ctx=80000,dist=soft,layer=sliding,load=verify_16,page=1,q_len=8,spread=agentic  (w 46, wall 538.54 us, span 534.69 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 8.0 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` | 0.00 | 13.76 | 0.00 | 0 | 2.6% |
| 8.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 14.53 | 3.04 | 0.67 | 0 | 0.6% |
| 8.2 | `_attn_kernel` | 18.05 | 10.96 | 0.38 | 0 | 2.0% |
| 8.3 | `_reduce_kernel` | 29.70 | 5.42 | 0.67 | 0 | 1.0% |
| 8.4 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 35.57 | 7.07 | 0.42 | 0 | 1.3% |
| 8.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 42.98 | 4.43 | 0.16 | 0 | 0.8% |
| 8.6 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 47.86 | 26.53 | 0.42 | 0 | 4.9% |
| 8.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 48.08 | 3.49 | 0.00 | 1 | 0.6% |
| 8.8 | `_gemv_splitk` | 52.40 | 28.66 | 0.00 | 1 | 5.3% |
| 8.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 81.34 | 2.91 | 0.10 | 1 | 0.5% |
| 8.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 85.12 | 5.94 | 0.83 | 0 | 1.1% |
| 8.11 | `_local_ids_kernel` | 91.62 | 1.06 | 0.42 | 0 | 0.2% |
| 8.12 | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, in` | 93.14 | 2.88 | 0.48 | 0 | 0.5% |
| 8.13 | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, ` | 96.42 | 1.86 | 0.42 | 0 | 0.3% |
| 8.14 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 98.66 | 261.78 | 0.38 | 0 | 48.6% |
| 8.15 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind` | 361.30 | 3.84 | 0.93 | 0 | 0.7% |
| 8.16 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 365.65 | 136.69 | 0.38 | 0 | 25.4% |
| 8.17 | `_post_expert_norm_weighted_sum_kernel` | 502.99 | 5.10 | 0.56 | 0 | 0.9% |
| 8.18 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 508.49 | 20.54 | 0.42 | 0 | 3.8% |
| 8.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 529.41 | 5.28 | 0.03 | 0 | 1.0% |

Totals: sum dur 551.24 | busy 526.27 | sum gaps 8.18 | head 3.82 | tail 3.89 | wall - span 3.85 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.93 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (8.14) | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (8.15) |
| 0.83 | `void sglang::muse_spark::routing_kernel<sglang::mu` (8.9) | `void at::native::(anonymous namespace)::vectorized` (8.10) |
| 0.67 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` (8.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (8.1) |
| 0.67 | `_attn_kernel` (8.2) | `_reduce_kernel` (8.3) |
| 0.56 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (8.16) | `_post_expert_norm_weighted_sum_kernel` (8.17) |

Families (sum dur): moe_gemm(marlin) 398.47, dense_gemm(cublas) 70.94, dense_gemm 28.66, norm/residual 20.75, attention 10.96, routing 5.79, attention_reduce 5.42, activation 3.84, elementwise 3.49, other 2.92

## shape 9: batch=16,ctx=80000,dist=soft,layer=full,load=verify_16,page=1,q_len=8,spread=agentic  (w 16, wall 646.74 us, span 642.57 us, n 20, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 9.0 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` | 0.00 | 13.82 | 0.00 | 0 | 2.1% |
| 9.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 14.53 | 3.04 | 0.67 | 0 | 0.5% |
| 9.2 | `_attn_kernel` | 17.95 | 180.59 | 0.38 | 0 | 27.9% |
| 9.3 | `_reduce_kernel` | 198.90 | 14.96 | 0.26 | 0 | 2.3% |
| 9.4 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 214.18 | 7.58 | 0.32 | 0 | 1.2% |
| 9.5 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, f` | 221.92 | 4.53 | 0.16 | 0 | 0.7% |
| 9.6 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 226.88 | 26.21 | 0.45 | 0 | 4.1% |
| 9.7 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 227.12 | 3.34 | 0.00 | 1 | 0.5% |
| 9.8 | `_gemv_splitk` | 231.22 | 28.61 | 0.00 | 1 | 4.4% |
| 9.9 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 260.02 | 3.01 | 0.10 | 1 | 0.5% |
| 9.10 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 264.11 | 6.02 | 0.90 | 0 | 0.9% |
| 9.11 | `_local_ids_kernel` | 270.59 | 1.12 | 0.38 | 0 | 0.2% |
| 9.12 | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, in` | 272.24 | 2.85 | 0.42 | 0 | 0.4% |
| 9.13 | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, ` | 275.54 | 1.92 | 0.41 | 0 | 0.3% |
| 9.14 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 277.94 | 214.80 | 0.42 | 0 | 33.2% |
| 9.15 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind` | 493.77 | 3.86 | 0.93 | 0 | 0.6% |
| 9.16 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 498.10 | 112.82 | 0.38 | 0 | 17.4% |
| 9.17 | `_post_expert_norm_weighted_sum_kernel` | 611.29 | 4.90 | 0.32 | 0 | 0.8% |
| 9.18 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 616.61 | 20.53 | 0.45 | 0 | 3.2% |
| 9.19 | `void sglang::muse_spark_norm_residual_kernel<8192l, 256, true, true, t` | 637.37 | 5.22 | 0.03 | 0 | 0.8% |

Totals: sum dur 659.73 | busy 634.96 | sum gaps 7.46 | head 3.84 | tail 3.97 | wall - span 4.17 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.93 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (9.14) | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (9.15) |
| 0.90 | `void sglang::muse_spark::routing_kernel<sglang::mu` (9.9) | `void at::native::(anonymous namespace)::vectorized` (9.10) |
| 0.67 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` (9.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (9.1) |
| 0.45 | `void sglang::muse_spark_norm_residual_kernel<8192l` (9.5) | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` (9.6) |
| 0.45 | `_post_expert_norm_weighted_sum_kernel` (9.17) | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` (9.18) |

Families (sum dur): moe_gemm(marlin) 327.62, attention 180.59, dense_gemm(cublas) 71.18, dense_gemm 28.61, norm/residual 20.67, attention_reduce 14.96, routing 5.86, activation 3.86, elementwise 3.34, other 3.04

## shape 10: batch=32,ctx=80000,dist=soft,layer=sliding,load=verify_32,page=1,q_len=8,spread=agentic  (w 46, wall 690.11 us, span 682.96 us, n 19, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 10.0 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | 0.00 | 19.74 | 0.00 | 0 | 2.9% |
| 10.1 | `_attn_kernel` | 20.03 | 17.12 | 0.26 | 0 | 2.5% |
| 10.2 | `_reduce_kernel` | 38.03 | 7.84 | 0.70 | 0 | 1.1% |
| 10.3 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 46.32 | 9.12 | 0.43 | 0 | 1.3% |
| 10.4 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, true, false, ` | 56.02 | 8.45 | 0.10 | 0 | 1.2% |
| 10.5 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 64.99 | 33.52 | 0.45 | 0 | 4.9% |
| 10.6 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 65.22 | 3.33 | 0.00 | 1 | 0.5% |
| 10.7 | `_gemv_splitk` | 69.14 | 38.27 | 0.00 | 1 | 5.5% |
| 10.8 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 107.10 | 3.07 | 0.06 | 1 | 0.4% |
| 10.9 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 111.10 | 6.22 | 0.86 | 0 | 0.9% |
| 10.10 | `_local_ids_kernel` | 117.66 | 1.09 | 0.42 | 0 | 0.2% |
| 10.11 | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, in` | 119.20 | 3.22 | 0.41 | 0 | 0.5% |
| 10.12 | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, ` | 122.89 | 2.53 | 0.42 | 0 | 0.4% |
| 10.13 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 125.87 | 335.36 | 0.45 | 0 | 48.6% |
| 10.14 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind` | 462.80 | 5.73 | 1.01 | 0 | 0.8% |
| 10.15 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 468.96 | 174.53 | 0.39 | 0 | 25.3% |
| 10.16 | `_post_expert_norm_weighted_sum_kernel` | 644.25 | 5.41 | 0.54 | 0 | 0.8% |
| 10.17 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 650.01 | 25.20 | 0.42 | 0 | 3.7% |
| 10.18 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, true, false, ` | 675.36 | 7.54 | 0.00 | 0 | 1.1% |

Totals: sum dur 707.29 | busy 675.68 | sum gaps 7.15 | head 6.86 | tail 3.94 | wall - span 7.15 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 1.01 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (10.13) | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (10.14) |
| 0.86 | `void sglang::muse_spark::routing_kernel<sglang::mu` (10.8) | `void at::native::(anonymous namespace)::vectorized` (10.9) |
| 0.70 | `_attn_kernel` (10.1) | `_reduce_kernel` (10.2) |
| 0.54 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (10.15) | `_post_expert_norm_weighted_sum_kernel` (10.16) |
| 0.45 | `void sglang::muse_spark_norm_residual_kernel<8192l` (10.4) | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` (10.5) |

Families (sum dur): moe_gemm(marlin) 509.89, dense_gemm(cublas) 87.58, dense_gemm 38.27, norm/residual 27.62, attention 17.12, attention_reduce 7.84, routing 6.29, activation 5.73, other 3.62, elementwise 3.33

## shape 11: batch=32,ctx=80000,dist=soft,layer=full,load=verify_32,page=1,q_len=8,spread=agentic  (w 16, wall 1077.06 us, span 1070.27 us, n 19, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 11.0 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | 0.00 | 19.74 | 0.00 | 0 | 1.8% |
| 11.1 | `_attn_kernel` | 20.10 | 337.18 | 0.22 | 0 | 31.3% |
| 11.2 | `_reduce_kernel` | 357.42 | 16.11 | 0.26 | 0 | 1.5% |
| 11.3 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 374.18 | 9.06 | 0.32 | 0 | 0.8% |
| 11.4 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, true, false, ` | 383.58 | 8.37 | 0.08 | 0 | 0.8% |
| 11.5 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 392.45 | 33.47 | 0.45 | 0 | 3.1% |
| 11.6 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 392.69 | 3.36 | 0.00 | 1 | 0.3% |
| 11.7 | `_gemv_splitk` | 396.89 | 38.40 | 0.00 | 1 | 3.6% |
| 11.8 | `void sglang::muse_spark::routing_kernel<sglang::muse_spark::RoutingTra` | 435.30 | 3.04 | 0.06 | 1 | 0.3% |
| 11.9 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c` | 439.38 | 6.13 | 0.83 | 0 | 0.6% |
| 11.10 | `_local_ids_kernel` | 445.87 | 1.06 | 0.40 | 0 | 0.1% |
| 11.11 | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, in` | 447.39 | 3.17 | 0.42 | 0 | 0.3% |
| 11.12 | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, ` | 451.07 | 2.53 | 0.43 | 0 | 0.2% |
| 11.13 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 454.01 | 375.95 | 0.42 | 0 | 34.9% |
| 11.14 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind` | 830.72 | 5.76 | 0.86 | 0 | 0.5% |
| 11.15 | `void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 562949953487106` | 836.89 | 194.27 | 0.45 | 0 | 18.0% |
| 11.16 | `_post_expert_norm_weighted_sum_kernel` | 1031.65 | 5.14 | 0.54 | 0 | 0.5% |
| 11.17 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 1037.42 | 24.91 | 0.42 | 0 | 2.3% |
| 11.18 | `void sglang::muse_spark_norm_residual_kernel<8192l, 512, true, false, ` | 1062.69 | 7.65 | 0.00 | 0 | 0.7% |

Totals: sum dur 1095.30 | busy 1063.61 | sum gaps 6.51 | head 6.59 | tail 3.86 | wall - span 6.79 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.86 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (11.13) | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (11.14) |
| 0.83 | `void sglang::muse_spark::routing_kernel<sglang::mu` (11.8) | `void at::native::(anonymous namespace)::vectorized` (11.9) |
| 0.54 | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (11.15) | `_post_expert_norm_weighted_sum_kernel` (11.16) |
| 0.45 | `void sglang::muse_spark_norm_residual_kernel<8192l` (11.4) | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` (11.5) |
| 0.45 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sg` (11.14) | `void sglang::device::marlin_moe::Marlin<__nv_bfloa` (11.15) |

Families (sum dur): moe_gemm(marlin) 570.22, attention 337.18, dense_gemm(cublas) 87.18, dense_gemm 38.40, norm/residual 27.29, attention_reduce 16.11, routing 6.21, activation 5.76, other 3.59, elementwise 3.36

## families, weighted by shape weight (sum dur)

| family | w x us | share |
|---|---|---|
| moe_gemm(marlin) | 62623.0 | 43.6% |
| attention | 19263.1 | 13.4% |
| dense_gemm(cublas) | 18275.9 | 12.7% |
| moe_gemm(cuda smallm) | 15243.8 | 10.6% |
| dense_gemm | 13866.3 | 9.7% |
| norm/residual | 6832.3 | 4.8% |
| attention_reduce | 1849.8 | 1.3% |
| routing | 1788.2 | 1.2% |
| elementwise | 1583.4 | 1.1% |
| moe_epilogue | 947.5 | 0.7% |
| activation | 731.6 | 0.5% |
| other | 619.3 | 0.4% |
