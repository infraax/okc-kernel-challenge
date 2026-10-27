# muse_moe_w4a16 — judge job 76d9f4196733442d (speedup 1.436x, score_us 9368)

Units µs (judge median, replay overhead already subtracted; p10–p90 overhead-corrected; sol floored at 2.5 µs). loss = weight × ln(ours/sol): what the board still
pays on this shape vs the floor; Σloss over shapes is the problem's total headroom in log units.

| # | shape | w | sglang | ours | sol | speedup | ours/sol | p10–p90 | kernels Σ | hidden | idle | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bs=1 load=decode 1 sharding=ep8 | 60 | 76.8 | 41.4 | 11.4 | 1.856 | 3.62 | 36.6–48.0 | 48.3 | 0.0 (0) | -6.9 | 77.23 (19%) |
| 1 | bs=16 load=decode 16 sharding=ep8 | 60 | 194.6 | 140.1 | 52.5 | 1.389 | 2.67 | 112.0–172.9 | 93.5 | 0.0 (0) | 46.6 | 58.86 (15%) |
| 2 | bs=32 load=decode 32 sharding=ep8 | 60 | 204.1 | 186.2 | 70.8 | 1.096 | 2.63 | 156.0–220.6 | 238.3 | 0.0 (0) | -52.1 | 58.04 (14%) |
| 3 | bs=8 load=verify 1 sharding=ep8 | 60 | 142.3 | 97.6 | 37.3 | 1.457 | 2.62 | 77.4–124.4 | 99.1 | 0.0 (0) | -1.5 | 57.71 (14%) |
| 4 | bs=128 load=verify 16 sharding=ep8 | 60 | 465.9 | 314.0 | 101.3 | 1.484 | 3.10 | 276.6–352.3 | 362.3 | 0.0 (0) | -48.3 | 67.86 (17%) |
| 5 | bs=256 load=verify 32 sharding=ep8 | 60 | 622.1 | 434.2 | 108.9 | 1.433 | 3.99 | 383.0–478.6 | 491.4 | 0.0 (0) | -57.2 | 82.96 (21%) |

## shape 0: bs=1 load=decode 1 sharding=ep8  (w 60, ours 41.4 µs, sol 11.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 0.0 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 26.6 | 64.3% | 1 | 26.6 |
| 0.1 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 15.7 | 38.0% | 1 | 15.7 |
| 0.2 | `_epilogue_kernel` | moe_epilogue | 4.8 | 11.6% | 1 | 4.8 |
| 0.3 | `smallm_route_kernel(int const*, int*, int*, int, int, int)` | routing | 1.2 | 2.9% | 1 | 1.2 |
| 0.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 0.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -6.9 | -16.8% | | |

## shape 1: bs=16 load=decode 16 sharding=ep8  (w 60, ours 140.1 µs, sol 52.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 1.0 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 49.0 | 35.0% | 1 | 49.0 |
| 1.1 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 38.3 | 27.3% | 1 | 38.3 |
| 1.2 | `_epilogue_kernel` | moe_epilogue | 5.0 | 3.6% | 1 | 5.0 |
| 1.3 | `smallm_route_kernel(int const*, int*, int*, int, int, int)` | routing | 1.2 | 0.9% | 1 | 1.2 |
| 1.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 1.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 46.6 | 33.3% | | |

## shape 2: bs=32 load=decode 32 sharding=ep8  (w 60, ours 186.2 µs, sol 70.8)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 2.0 | `void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 con...` | moe_gemm(cuda largem) | 147.5 | 79.2% | 1 | 147.5 |
| 2.1 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 co...` | moe_gemm(cuda largem) | 84.2 | 45.2% | 1 | 84.2 |
| 2.2 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 3.6 | 1.9% | 1 | 3.6 |
| 2.3 | `largem_route_kernel(int const*, int*, int*, int*, int, in...` | routing | 3.0 | 1.6% | 1 | 3.0 |
| 2.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 2.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -52.1 | -28.0% | | |

## shape 3: bs=8 load=verify 1 sharding=ep8  (w 60, ours 97.6 µs, sol 37.3)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 3.0 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*...` | moe_gemm(cuda smallm) | 54.1 | 55.4% | 1 | 54.1 |
| 3.1 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const...` | moe_gemm(cuda smallm) | 40.4 | 41.4% | 1 | 40.4 |
| 3.2 | `_epilogue_kernel` | moe_epilogue | 3.4 | 3.5% | 1 | 3.4 |
| 3.3 | `smallm_route_kernel(int const*, int*, int*, int, int, int)` | routing | 1.2 | 1.2% | 1 | 1.2 |
| 3.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 3.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -1.5 | -1.5% | | |

## shape 4: bs=128 load=verify 16 sharding=ep8  (w 60, ours 314.0 µs, sol 101.3)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 4.0 | `void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 con...` | moe_gemm(cuda largem) | 231.5 | 73.7% | 1 | 231.5 |
| 4.1 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 co...` | moe_gemm(cuda largem) | 120.3 | 38.3% | 1 | 120.3 |
| 4.2 | `largem_route_kernel(int const*, int*, int*, int*, int, in...` | routing | 6.1 | 1.9% | 1 | 6.1 |
| 4.3 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 4.4 | 1.4% | 1 | 4.4 |
| 4.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 4.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -48.3 | -15.4% | | |

## shape 5: bs=256 load=verify 32 sharding=ep8  (w 60, ours 434.2 µs, sol 108.9)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 5.0 | `void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 con...` | moe_gemm(cuda largem) | 314.7 | 72.5% | 1 | 314.7 |
| 5.1 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 co...` | moe_gemm(cuda largem) | 160.6 | 37.0% | 1 | 160.6 |
| 5.2 | `largem_route_kernel(int const*, int*, int*, int*, int, in...` | routing | 10.6 | 2.4% | 1 | 10.6 |
| 5.3 | `_post_expert_norm_weighted_sum_kernel` | moe_epilogue | 5.5 | 1.3% | 1 | 5.5 |
| 5.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 5.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -57.2 | -13.2% | | |

## families, weighted by each shape's loss share (where the board's headroom sits)

| family | weighted loss share |
|---|---|
| moe_gemm(cuda largem) | 55.0% |
| moe_gemm(cuda smallm) | 39.5% |
| moe_epilogue | 3.7% |
| routing | 1.8% |
