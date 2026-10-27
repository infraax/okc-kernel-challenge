# muse_bf16_proj — judge job c3b7c70b5edd43c5 (speedup 1.156x, score_us 4178)

Units µs (judge median, replay overhead already subtracted; p10–p90 overhead-corrected; sol floored at 2.5 µs). loss = weight × ln(ours/sol): what the board still
pays on this shape vs the floor; Σloss over shapes is the problem's total headroom in log units.

| # | shape | w | sglang | ours | sol | speedup | ours/sol | p10–p90 | kernels Σ | hidden | idle | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | layer=qkv_gate load=decode 1 m=1 | 62 | 13.8 | 14.0 | 7.9 | 0.989 | 1.78 | 13.5–14.6 | 11.2 | 0.0 (0) | 2.8 | 35.67 (3%) |
| 1 | layer=o load=decode 1 m=1 | 62 | 6.7 | 6.5 | 3.5 | 1.024 | 1.87 | 6.4–6.8 | 4.6 | 0.0 (0) | 1.9 | 38.82 (3%) |
| 2 | layer=pooler load=decode 1 m=1 | 62 | 21.7 | 19.2 | 14.0 | 1.130 | 1.37 | 18.8–19.9 | 19.4 | 0.0 (0) | -0.2 | 19.56 (2%) |
| 3 | layer=unpooler load=decode 1 m=1 | 62 | 6.7 | 6.8 | 2.5 | 0.995 | 2.70 | 6.6–6.9 | 5.5 | 0.0 (0) | 1.3 | 61.60 (5%) |
| 4 | layer=router load=decode 1 m=1 | 62 | 10.2 | 6.4 | 2.5 | 1.581 | 2.58 | 6.3–6.7 | 5.0 | 0.1 (0) | 1.4 | 58.73 (5%) |
| 5 | layer=qkv_gate load=decode 16 m=16 | 62 | 14.0 | 14.0 | 7.9 | 1.000 | 1.76 | 13.6–14.7 | 9.6 | 0.0 (0) | 4.4 | 35.04 (3%) |
| 6 | layer=o load=decode 16 m=16 | 62 | 6.8 | 6.7 | 3.6 | 1.020 | 1.88 | 6.5–7.0 | 5.5 | 0.0 (0) | 1.2 | 39.08 (3%) |
| 7 | layer=pooler load=decode 16 m=16 | 62 | 21.8 | 19.4 | 14.1 | 1.128 | 1.38 | 19.0–19.8 | 19.3 | 0.0 (0) | 0.1 | 19.81 (2%) |
| 8 | layer=unpooler load=decode 16 m=16 | 62 | 18.6 | 18.5 | 14.1 | 1.005 | 1.32 | 18.1–19.2 | 18.2 | 0.0 (0) | 0.3 | 17.01 (1%) |
| 9 | layer=router load=decode 16 m=16 | 62 | 15.0 | 7.4 | 2.5 | 2.033 | 2.95 | 7.2–7.6 | 5.6 | -0.0 (0) | 1.8 | 67.12 (5%) |
| 10 | layer=qkv_gate load=decode 32 m=32 | 62 | 14.2 | 14.0 | 8.0 | 1.014 | 1.75 | 13.6–14.6 | 10.3 | 0.0 (0) | 3.7 | 34.57 (3%) |
| 11 | layer=o load=decode 32 m=32 | 62 | 6.6 | 6.8 | 3.6 | 0.978 | 1.87 | 6.7–6.9 | 6.0 | 0.0 (0) | 0.8 | 38.85 (3%) |
| 12 | layer=pooler load=decode 32 m=32 | 62 | 22.2 | 20.1 | 14.1 | 1.104 | 1.42 | 19.8–20.9 | 19.6 | 0.0 (0) | 0.5 | 21.92 (2%) |
| 13 | layer=unpooler load=decode 32 m=32 | 62 | 19.1 | 18.9 | 14.1 | 1.012 | 1.33 | 18.6–19.6 | 18.5 | 0.0 (0) | 0.4 | 17.86 (1%) |
| 14 | layer=router load=decode 32 m=32 | 62 | 14.9 | 7.0 | 2.5 | 2.116 | 2.82 | 6.9–7.3 | 6.3 | 0.0 (0) | 0.7 | 64.24 (5%) |
| 15 | layer=qkv_gate load=verify 1 m=8 | 62 | 13.9 | 14.1 | 7.9 | 0.987 | 1.78 | 13.6–14.6 | 10.7 | 0.1 (0) | 3.4 | 35.92 (3%) |
| 16 | layer=o load=verify 1 m=8 | 62 | 6.8 | 6.8 | 3.5 | 1.010 | 1.92 | 6.6–7.0 | 5.2 | 0.0 (0) | 1.6 | 40.57 (3%) |
| 17 | layer=pooler load=verify 1 m=8 | 62 | 21.5 | 19.3 | 14.0 | 1.112 | 1.38 | 18.9–20.0 | 18.8 | 0.0 (0) | 0.5 | 19.92 (2%) |
| 18 | layer=unpooler load=verify 1 m=8 | 62 | 7.0 | 7.2 | 2.5 | 0.975 | 2.86 | 6.9–7.3 | 5.6 | 0.0 (0) | 1.6 | 65.18 (5%) |
| 19 | layer=router load=verify 1 m=8 | 62 | 26.2 | 6.7 | 2.5 | 3.894 | 2.69 | 6.6–6.9 | 5.1 | -0.0 (0) | 1.6 | 61.32 (5%) |
| 20 | layer=qkv_gate load=verify 16 m=128 | 62 | 17.1 | 17.0 | 8.4 | 1.009 | 2.02 | 16.8–17.4 | 16.4 | -0.0 (0) | 0.6 | 43.44 (3%) |
| 21 | layer=o load=verify 16 m=128 | 62 | 7.2 | 7.3 | 4.0 | 0.986 | 1.82 | 7.1–7.6 | 5.3 | 0.0 (0) | 2.0 | 37.24 (3%) |
| 22 | layer=pooler load=verify 16 m=128 | 62 | 24.7 | 24.5 | 14.6 | 1.009 | 1.67 | 23.8–25.1 | 24.9 | 0.0 (0) | -0.4 | 31.98 (3%) |
| 23 | layer=unpooler load=verify 16 m=128 | 62 | 19.7 | 19.6 | 14.6 | 1.005 | 1.34 | 19.3–20.6 | 20.5 | 0.0 (0) | -0.9 | 18.14 (1%) |
| 24 | layer=router load=verify 16 m=128 | 62 | 16.2 | 9.9 | 2.5 | 1.642 | 3.94 | 9.7–10.0 | 9.1 | -0.0 (0) | 0.8 | 85.07 (7%) |
| 25 | layer=qkv_gate load=verify 32 m=256 | 62 | 19.8 | 19.6 | 9.8 | 1.011 | 2.01 | 19.4–20.2 | 19.4 | 0.0 (0) | 0.2 | 43.18 (3%) |
| 26 | layer=o load=verify 32 m=256 | 62 | 9.2 | 9.2 | 4.5 | 1.002 | 2.06 | 9.0–9.4 | 8.1 | 0.0 (0) | 1.1 | 44.86 (4%) |
| 27 | layer=pooler load=verify 32 m=256 | 62 | 30.6 | 30.5 | 17.4 | 1.003 | 1.76 | 29.8–31.8 | 29.5 | 0.0 (0) | 1.0 | 34.96 (3%) |
| 28 | layer=unpooler load=verify 32 m=256 | 62 | 24.3 | 24.1 | 17.4 | 1.007 | 1.39 | 23.8–24.7 | 24.4 | 0.0 (0) | -0.3 | 20.37 (2%) |
| 29 | layer=router load=verify 32 m=256 | 62 | 17.4 | 12.3 | 2.7 | 1.418 | 4.59 | 12.0–12.6 | 11.1 | 0.0 (0) | 1.2 | 94.49 (8%) |

## shape 0: layer=qkv_gate load=decode 1 m=1  (w 62, ours 14.0 µs, sol 7.9)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 0.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 9.5 | 67.9% | 1 | 9.5 |
| 0.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 1.7 | 12.2% | 1 | 1.7 |
| 0.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 0.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 2.8 | 19.9% | | |

## shape 1: layer=o load=decode 1 m=1  (w 62, ours 6.5 µs, sol 3.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 1.0 | `_gemv_splitk` | dense_gemm | 4.6 | 70.3% | 1 | 4.6 |
| 1.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 1.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.9 | 29.7% | | |

## shape 2: layer=pooler load=decode 1 m=1  (w 62, ours 19.2 µs, sol 14.0)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 2.0 | `_gemv_splitk` | dense_gemm | 19.4 | 101.2% | 1 | 19.4 |
| 2.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 2.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -0.2 | -1.2% | | |

## shape 3: layer=unpooler load=decode 1 m=1  (w 62, ours 6.8 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 3.0 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 4.1 | 60.7% | 1 | 4.1 |
| 3.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 1.4 | 20.7% | 1 | 1.4 |
| 3.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 3.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.3 | 18.5% | | |

## shape 4: layer=router load=decode 1 m=1  (w 62, ours 6.4 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 4.0 | `_gemv_splitk` | dense_gemm | 4.2 | 65.1% | 1 | 4.2 |
| 4.1 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` | elementwise | 0.7 | 10.9% | 1 | 0.7 |
| 4.– | (hidden tail) | 0 kernels not listed by the judge report | 0.1 | 1.6% | | |
| 4.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.4 | 22.4% | | |

## shape 5: layer=qkv_gate load=decode 16 m=16  (w 62, ours 14.0 µs, sol 7.9)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 5.0 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 7.9 | 56.6% | 1 | 7.9 |
| 5.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 1.7 | 12.2% | 1 | 1.7 |
| 5.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 5.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 4.4 | 31.2% | | |

## shape 6: layer=o load=decode 16 m=16  (w 62, ours 6.7 µs, sol 3.6)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 6.0 | `_gemv_splitk` | dense_gemm | 5.5 | 82.3% | 1 | 5.5 |
| 6.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 6.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.2 | 17.7% | | |

## shape 7: layer=pooler load=decode 16 m=16  (w 62, ours 19.4 µs, sol 14.1)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 7.0 | `_gemv_splitk` | dense_gemm | 19.3 | 99.7% | 1 | 19.3 |
| 7.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 7.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.1 | 0.3% | | |

## shape 8: layer=unpooler load=decode 16 m=16  (w 62, ours 18.5 µs, sol 14.1)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 8.0 | `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 18.2 | 98.4% | 1 | 18.2 |
| 8.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 8.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.3 | 1.6% | | |

## shape 9: layer=router load=decode 16 m=16  (w 62, ours 7.4 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 9.0 | `_gemv_splitk` | dense_gemm | 4.9 | 66.4% | 1 | 4.9 |
| 9.1 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` | elementwise | 0.7 | 9.5% | 1 | 0.7 |
| 9.– | (hidden tail) | 0 kernels not listed by the judge report | -0.0 | -0.0% | | |
| 9.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.8 | 24.1% | | |

## shape 10: layer=qkv_gate load=decode 32 m=32  (w 62, ours 14.0 µs, sol 8.0)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 10.0 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 8.5 | 60.8% | 1 | 8.5 |
| 10.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 1.8 | 12.9% | 1 | 1.8 |
| 10.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 10.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 3.7 | 26.3% | | |

## shape 11: layer=o load=decode 32 m=32  (w 62, ours 6.8 µs, sol 3.6)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 11.0 | `_gemv_splitk` | dense_gemm | 6.0 | 88.6% | 1 | 6.0 |
| 11.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 11.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.8 | 11.4% | | |

## shape 12: layer=pooler load=decode 32 m=32  (w 62, ours 20.1 µs, sol 14.1)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 12.0 | `_gemv_splitk` | dense_gemm | 19.6 | 97.3% | 1 | 19.6 |
| 12.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 12.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.5 | 2.7% | | |

## shape 13: layer=unpooler load=decode 32 m=32  (w 62, ours 18.9 µs, sol 14.1)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 13.0 | `nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT` | dense_gemm(cublas) | 18.5 | 98.1% | 1 | 18.5 |
| 13.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 13.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.4 | 1.9% | | |

## shape 14: layer=router load=decode 32 m=32  (w 62, ours 7.0 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 14.0 | `_gemv_splitk` | dense_gemm | 5.6 | 79.5% | 1 | 5.6 |
| 14.1 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` | elementwise | 0.7 | 9.9% | 1 | 0.7 |
| 14.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 14.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.7 | 10.6% | | |

## shape 15: layer=qkv_gate load=verify 1 m=8  (w 62, ours 14.1 µs, sol 7.9)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 15.0 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 9.0 | 63.8% | 1 | 9.0 |
| 15.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 1.6 | 11.3% | 1 | 1.6 |
| 15.– | (hidden tail) | 0 kernels not listed by the judge report | 0.1 | 0.7% | | |
| 15.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 3.4 | 24.1% | | |

## shape 16: layer=o load=verify 1 m=8  (w 62, ours 6.8 µs, sol 3.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 16.0 | `_gemv_splitk` | dense_gemm | 5.2 | 76.7% | 1 | 5.2 |
| 16.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 16.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.6 | 23.3% | | |

## shape 17: layer=pooler load=verify 1 m=8  (w 62, ours 19.3 µs, sol 14.0)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 17.0 | `_gemv_splitk` | dense_gemm | 18.8 | 97.2% | 1 | 18.8 |
| 17.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 17.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.5 | 2.8% | | |

## shape 18: layer=unpooler load=verify 1 m=8  (w 62, ours 7.2 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 18.0 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | dense_gemm(cublas) | 4.1 | 57.3% | 1 | 4.1 |
| 18.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 1.5 | 21.0% | 1 | 1.5 |
| 18.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 18.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.6 | 21.7% | | |

## shape 19: layer=router load=verify 1 m=8  (w 62, ours 6.7 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 19.0 | `_gemv_splitk` | dense_gemm | 4.4 | 65.5% | 1 | 4.4 |
| 19.1 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` | elementwise | 0.7 | 10.4% | 1 | 0.7 |
| 19.– | (hidden tail) | 0 kernels not listed by the judge report | -0.0 | -0.0% | | |
| 19.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.6 | 24.1% | | |

## shape 20: layer=qkv_gate load=verify 16 m=128  (w 62, ours 17.0 µs, sol 8.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 20.0 | `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` | dense_gemm(cublas) | 13.3 | 78.4% | 1 | 13.3 |
| 20.1 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | dense_gemm(cublas) | 3.1 | 18.3% | 1 | 3.1 |
| 20.– | (hidden tail) | 0 kernels not listed by the judge report | -0.0 | -0.0% | | |
| 20.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.6 | 3.4% | | |

## shape 21: layer=o load=verify 16 m=128  (w 62, ours 7.3 µs, sol 4.0)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 21.0 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 5.3 | 72.9% | 1 | 5.3 |
| 21.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 21.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 2.0 | 27.1% | | |

## shape 22: layer=pooler load=verify 16 m=128  (w 62, ours 24.5 µs, sol 14.6)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 22.0 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` | dense_gemm(cublas) | 24.9 | 101.6% | 1 | 24.9 |
| 22.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 22.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -0.4 | -1.6% | | |

## shape 23: layer=unpooler load=verify 16 m=128  (w 62, ours 19.6 µs, sol 14.6)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 23.0 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 20.5 | 104.5% | 1 | 20.5 |
| 23.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 23.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -0.9 | -4.5% | | |

## shape 24: layer=router load=verify 16 m=128  (w 62, ours 9.9 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 24.0 | `_gemv_splitk` | dense_gemm | 8.3 | 84.2% | 1 | 8.3 |
| 24.1 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` | elementwise | 0.8 | 8.1% | 1 | 0.8 |
| 24.– | (hidden tail) | 0 kernels not listed by the judge report | -0.0 | -0.0% | | |
| 24.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.8 | 7.7% | | |

## shape 25: layer=qkv_gate load=verify 32 m=256  (w 62, ours 19.6 µs, sol 9.8)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 25.0 | `nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN` | dense_gemm(cublas) | 19.4 | 98.9% | 1 | 19.4 |
| 25.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 25.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.2 | 1.1% | | |

## shape 26: layer=o load=verify 32 m=256  (w 62, ours 9.2 µs, sol 4.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 26.0 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | dense_gemm(cublas) | 8.1 | 87.7% | 1 | 8.1 |
| 26.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 26.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.1 | 12.3% | | |

## shape 27: layer=pooler load=verify 32 m=256  (w 62, ours 30.5 µs, sol 17.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 27.0 | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT` | dense_gemm(cublas) | 29.5 | 96.6% | 1 | 29.5 |
| 27.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 27.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.0 | 3.4% | | |

## shape 28: layer=unpooler load=verify 32 m=256  (w 62, ours 24.1 µs, sol 17.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 28.0 | `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT` | dense_gemm(cublas) | 24.4 | 101.1% | 1 | 24.4 |
| 28.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 28.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -0.3 | -1.1% | | |

## shape 29: layer=router load=verify 32 m=256  (w 62, ours 12.3 µs, sol 2.7)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 29.0 | `_gemv_splitk` | dense_gemm | 10.4 | 84.7% | 1 | 10.4 |
| 29.1 | `void at::native::vectorized_elementwise_kernel<4, at::nat...` | elementwise | 0.7 | 5.7% | 1 | 0.7 |
| 29.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 29.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.2 | 9.6% | | |

## families, weighted by each shape's loss share (where the board's headroom sits)

| family | weighted loss share |
|---|---|
| dense_gemm | 49.8% |
| dense_gemm(cublas) | 46.7% |
| elementwise | 3.5% |
