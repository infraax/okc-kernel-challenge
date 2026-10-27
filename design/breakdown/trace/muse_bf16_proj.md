# TRACE report

- problem=muse_bf16_proj file=director_v1b.py torch=2.14.1+cu130 cuda=13.0 gpu=NVIDIA_H200 cupti=graph replays=20
- prepare time 4 s

## invariants

- all checks pass

## shape 0: layer=qkv_gate,load=decode_1,m=1  (w 62, wall 17.57 us, span 13.26 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 0.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 0.00 | 11.23 | 0.00 | 0 | 63.9% |
| 0.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 11.26 | 1.98 | 0.03 | 0 | 11.3% |

Totals: sum dur 13.21 | busy 13.23 | sum gaps 0.03 | head 3.95 | tail 3.87 | wall - span 4.31 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.03 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` (0.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (0.1) |

Families (sum dur): dense_gemm(cublas) 13.21

## shape 1: layer=o,load=decode_1,m=1  (w 62, wall 10.13 us, span 6.35 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 1.0 | `_gemv_splitk` | 0.00 | 6.35 | 0.00 | 0 | 62.7% |

Totals: sum dur 6.35 | busy 6.35 | sum gaps 0.00 | head 3.71 | tail 3.60 | wall - span 3.78 us

Families (sum dur): dense_gemm 6.35

## shape 2: layer=pooler,load=decode_1,m=1  (w 62, wall 22.21 us, span 18.53 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 2.0 | `_gemv_splitk` | 0.00 | 18.53 | 0.00 | 0 | 83.4% |

Totals: sum dur 18.53 | busy 18.53 | sum gaps 0.00 | head 3.63 | tail 3.57 | wall - span 3.68 us

Families (sum dur): dense_gemm 18.53

## shape 3: layer=unpooler,load=decode_1,m=1  (w 62, wall 10.78 us, span 6.78 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 3.0 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 4.93 | 0.00 | 0 | 45.7% |
| 3.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 4.96 | 1.79 | 0.03 | 0 | 16.6% |

Totals: sum dur 6.72 | busy 6.75 | sum gaps 0.03 | head 3.98 | tail 3.63 | wall - span 4.00 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.03 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (3.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (3.1) |

Families (sum dur): dense_gemm(cublas) 6.72

## shape 4: layer=router,load=decode_1,m=1  (w 62, wall 11.17 us, span 6.91 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 4.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 0.00 | 0.67 | 0.00 | 0 | 6.0% |
| 4.1 | `_gemv_splitk` | 1.38 | 5.50 | 0.64 | 0 | 49.2% |

Totals: sum dur 6.17 | busy 6.18 | sum gaps 0.64 | head 4.16 | tail 3.81 | wall - span 4.26 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.64 | `void at::native::vectorized_elementwise_kernel<4, ` (4.0) | `_gemv_splitk` (4.1) |

Families (sum dur): dense_gemm 5.50, elementwise 0.67

## shape 5: layer=qkv_gate,load=decode_16,m=16  (w 62, wall 18.74 us, span 14.40 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 5.0 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 11.98 | 0.00 | 0 | 63.9% |
| 5.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.37 | 2.02 | 0.35 | 0 | 10.8% |

Totals: sum dur 14.00 | busy 14.03 | sum gaps 0.35 | head 4.25 | tail 3.84 | wall - span 4.34 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.35 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (5.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (5.1) |

Families (sum dur): dense_gemm(cublas) 14.00

## shape 6: layer=o,load=decode_16,m=16  (w 62, wall 10.46 us, span 6.40 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 6.0 | `_gemv_splitk` | 0.00 | 6.40 | 0.00 | 0 | 61.2% |

Totals: sum dur 6.40 | busy 6.40 | sum gaps 0.00 | head 3.98 | tail 3.76 | wall - span 4.06 us

Families (sum dur): dense_gemm 6.40

## shape 7: layer=pooler,load=decode_16,m=16  (w 62, wall 22.88 us, span 18.75 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 7.0 | `_gemv_splitk` | 0.00 | 18.75 | 0.00 | 0 | 81.9% |

Totals: sum dur 18.75 | busy 18.75 | sum gaps 0.00 | head 4.03 | tail 3.73 | wall - span 4.13 us

Families (sum dur): dense_gemm 18.75

## shape 8: layer=unpooler,load=decode_16,m=16  (w 62, wall 21.78 us, span 17.49 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 8.0 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | 0.00 | 17.49 | 0.00 | 0 | 80.3% |

Totals: sum dur 17.49 | busy 17.49 | sum gaps 0.00 | head 4.16 | tail 3.78 | wall - span 4.29 us

Families (sum dur): dense_gemm(cublas) 17.49

## shape 9: layer=router,load=decode_16,m=16  (w 62, wall 11.84 us, span 7.52 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 9.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 0.00 | 0.64 | 0.00 | 0 | 5.4% |
| 9.1 | `_gemv_splitk` | 1.30 | 6.21 | 0.61 | 0 | 52.4% |

Totals: sum dur 6.85 | busy 6.88 | sum gaps 0.61 | head 4.19 | tail 3.84 | wall - span 4.32 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.61 | `void at::native::vectorized_elementwise_kernel<4, ` (9.0) | `_gemv_splitk` (9.1) |

Families (sum dur): dense_gemm 6.21, elementwise 0.64

## shape 10: layer=qkv_gate,load=decode_32,m=32  (w 62, wall 18.80 us, span 14.37 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 10.0 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 11.89 | 0.00 | 0 | 63.2% |
| 10.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.22 | 2.11 | 0.32 | 0 | 11.2% |

Totals: sum dur 14.00 | busy 14.03 | sum gaps 0.32 | head 4.34 | tail 3.82 | wall - span 4.43 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.32 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (10.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (10.1) |

Families (sum dur): dense_gemm(cublas) 14.00

## shape 11: layer=o,load=decode_32,m=32  (w 62, wall 10.93 us, span 6.80 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 11.0 | `_gemv_splitk` | 0.00 | 6.80 | 0.00 | 0 | 62.2% |

Totals: sum dur 6.80 | busy 6.80 | sum gaps 0.00 | head 4.02 | tail 3.79 | wall - span 4.13 us

Families (sum dur): dense_gemm 6.80

## shape 12: layer=pooler,load=decode_32,m=32  (w 62, wall 23.57 us, span 19.46 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 12.0 | `_gemv_splitk` | 0.00 | 19.46 | 0.00 | 0 | 82.6% |

Totals: sum dur 19.46 | busy 19.46 | sum gaps 0.00 | head 4.03 | tail 3.70 | wall - span 4.11 us

Families (sum dur): dense_gemm 19.46

## shape 13: layer=unpooler,load=decode_32,m=32  (w 62, wall 21.86 us, span 17.78 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 13.0 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | 0.00 | 17.78 | 0.00 | 0 | 81.3% |

Totals: sum dur 17.78 | busy 17.78 | sum gaps 0.00 | head 3.98 | tail 3.74 | wall - span 4.08 us

Families (sum dur): dense_gemm(cublas) 17.78

## shape 14: layer=router,load=decode_32,m=32  (w 62, wall 11.63 us, span 7.46 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 14.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 0.00 | 0.67 | 0.00 | 0 | 5.8% |
| 14.1 | `_gemv_splitk` | 1.31 | 6.11 | 0.61 | 0 | 52.5% |

Totals: sum dur 6.78 | busy 6.82 | sum gaps 0.61 | head 4.10 | tail 3.79 | wall - span 4.17 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.61 | `void at::native::vectorized_elementwise_kernel<4, ` (14.0) | `_gemv_splitk` (14.1) |

Families (sum dur): dense_gemm 6.11, elementwise 0.67

## shape 15: layer=qkv_gate,load=verify_1,m=8  (w 62, wall 19.12 us, span 14.78 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 15.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 0.00 | 12.38 | 0.00 | 0 | 64.7% |
| 15.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 12.72 | 2.02 | 0.32 | 0 | 10.6% |

Totals: sum dur 14.40 | busy 14.43 | sum gaps 0.32 | head 4.26 | tail 3.81 | wall - span 4.34 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.32 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` (15.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (15.1) |

Families (sum dur): dense_gemm(cublas) 14.40

## shape 16: layer=o,load=verify_1,m=8  (w 62, wall 10.48 us, span 6.42 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 16.0 | `_gemv_splitk` | 0.00 | 6.42 | 0.00 | 0 | 61.3% |

Totals: sum dur 6.42 | busy 6.42 | sum gaps 0.00 | head 3.98 | tail 3.79 | wall - span 4.06 us

Families (sum dur): dense_gemm 6.42

## shape 17: layer=pooler,load=verify_1,m=8  (w 62, wall 22.70 us, span 18.69 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 17.0 | `_gemv_splitk` | 0.00 | 18.69 | 0.00 | 0 | 82.3% |

Totals: sum dur 18.69 | busy 18.69 | sum gaps 0.00 | head 3.94 | tail 3.78 | wall - span 4.01 us

Families (sum dur): dense_gemm 18.69

## shape 18: layer=unpooler,load=verify_1,m=8  (w 62, wall 11.65 us, span 7.15 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 18.0 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 0.00 | 4.93 | 0.00 | 0 | 42.3% |
| 18.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 5.26 | 1.89 | 0.32 | 0 | 16.2% |

Totals: sum dur 6.82 | busy 6.80 | sum gaps 0.32 | head 4.26 | tail 3.95 | wall - span 4.50 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.32 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (18.0) | `void cublasLt::splitKreduce_kernel<32, 16, int, fl` (18.1) |

Families (sum dur): dense_gemm(cublas) 6.82

## shape 19: layer=router,load=verify_1,m=8  (w 62, wall 10.35 us, span 6.38 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 19.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 0.00 | 0.64 | 0.00 | 0 | 6.2% |
| 19.1 | `_gemv_splitk` | 0.93 | 5.42 | 0.29 | 0 | 52.4% |

Totals: sum dur 6.06 | busy 6.08 | sum gaps 0.29 | head 3.81 | tail 3.57 | wall - span 3.97 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.29 | `void at::native::vectorized_elementwise_kernel<4, ` (19.0) | `_gemv_splitk` (19.1) |

Families (sum dur): dense_gemm 5.42, elementwise 0.64

## shape 20: layer=qkv_gate,load=verify_16,m=128  (w 62, wall 20.93 us, span 16.93 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 20.0 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` | 0.00 | 13.62 | 0.00 | 0 | 65.1% |
| 20.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, ` | 13.57 | 3.33 | 0.00 | 0 | 15.9% |

Totals: sum dur 16.95 | busy 16.93 | sum gaps 0.00 | head 3.90 | tail 3.57 | wall - span 4.00 us

Families (sum dur): dense_gemm(cublas) 16.95

## shape 21: layer=o,load=verify_16,m=128  (w 62, wall 10.74 us, span 6.91 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 21.0 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 0.00 | 6.91 | 0.00 | 0 | 64.3% |

Totals: sum dur 6.91 | busy 6.91 | sum gaps 0.00 | head 3.70 | tail 3.65 | wall - span 3.83 us

Families (sum dur): dense_gemm(cublas) 6.91

## shape 22: layer=pooler,load=verify_16,m=128  (w 62, wall 28.21 us, span 24.48 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 22.0 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | 0.00 | 24.48 | 0.00 | 0 | 86.8% |

Totals: sum dur 24.48 | busy 24.48 | sum gaps 0.00 | head 3.71 | tail 3.62 | wall - span 3.73 us

Families (sum dur): dense_gemm(cublas) 24.48

## shape 23: layer=unpooler,load=verify_16,m=128  (w 62, wall 22.85 us, span 18.79 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 23.0 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 0.00 | 18.79 | 0.00 | 0 | 82.2% |

Totals: sum dur 18.79 | busy 18.79 | sum gaps 0.00 | head 3.92 | tail 3.81 | wall - span 4.06 us

Families (sum dur): dense_gemm(cublas) 18.79

## shape 24: layer=router,load=verify_16,m=128  (w 62, wall 14.29 us, span 10.06 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 24.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 0.00 | 0.70 | 0.00 | 0 | 4.9% |
| 24.1 | `_gemv_splitk` | 1.28 | 8.78 | 0.58 | 0 | 61.4% |

Totals: sum dur 9.48 | busy 9.47 | sum gaps 0.58 | head 4.03 | tail 3.78 | wall - span 4.23 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `void at::native::vectorized_elementwise_kernel<4, ` (24.0) | `_gemv_splitk` (24.1) |

Families (sum dur): dense_gemm 8.78, elementwise 0.70

## shape 25: layer=qkv_gate,load=verify_32,m=256  (w 62, wall 23.12 us, span 19.12 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 25.0 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | 0.00 | 19.12 | 0.00 | 0 | 82.7% |

Totals: sum dur 19.12 | busy 19.12 | sum gaps 0.00 | head 3.94 | tail 3.71 | wall - span 4.00 us

Families (sum dur): dense_gemm(cublas) 19.12

## shape 26: layer=o,load=verify_32,m=256  (w 62, wall 12.83 us, span 8.75 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 26.0 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 0.00 | 8.75 | 0.00 | 0 | 68.2% |

Totals: sum dur 8.75 | busy 8.75 | sum gaps 0.00 | head 4.02 | tail 3.81 | wall - span 4.08 us

Families (sum dur): dense_gemm(cublas) 8.75

## shape 27: layer=pooler,load=verify_32,m=256  (w 62, wall 33.41 us, span 29.26 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 27.0 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | 0.00 | 29.26 | 0.00 | 0 | 87.6% |

Totals: sum dur 29.26 | busy 29.26 | sum gaps 0.00 | head 4.08 | tail 3.73 | wall - span 4.15 us

Families (sum dur): dense_gemm(cublas) 29.26

## shape 28: layer=unpooler,load=verify_32,m=256  (w 62, wall 28.21 us, span 24.11 us, n 1, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 28.0 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | 0.00 | 24.11 | 0.00 | 0 | 85.5% |

Totals: sum dur 24.11 | busy 24.11 | sum gaps 0.00 | head 4.06 | tail 3.74 | wall - span 4.10 us

Families (sum dur): dense_gemm(cublas) 24.11

## shape 29: layer=router,load=verify_32,m=256  (w 62, wall 16.66 us, span 12.35 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 29.0 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc` | 0.00 | 0.74 | 0.00 | 0 | 4.4% |
| 29.1 | `_gemv_splitk` | 1.34 | 10.99 | 0.61 | 0 | 66.0% |

Totals: sum dur 11.73 | busy 11.74 | sum gaps 0.61 | head 4.19 | tail 3.87 | wall - span 4.31 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.61 | `void at::native::vectorized_elementwise_kernel<4, ` (29.0) | `_gemv_splitk` (29.1) |

Families (sum dur): dense_gemm 10.99, elementwise 0.74

## families, weighted by shape weight (sum dur)

| family | w x us | share |
|---|---|---|
| dense_gemm(cublas) | 15673.0 | 63.0% |
| dense_gemm | 8953.4 | 36.0% |
| elementwise | 251.7 | 1.0% |
