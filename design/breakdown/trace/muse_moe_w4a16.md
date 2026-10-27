# TRACE report

- problem=muse_moe_w4a16 file=sonnet_v4.py torch=2.14.1+cu130 cuda=13.0 gpu=NVIDIA_H200 cupti=graph replays=20
- prepare time 93 s

## invariants

- all checks pass

## shape 0: bs=1,load=decode_1,sharding=ep8  (w 60, wall 52.18 us, span 47.42 us, n 4, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 0.0 | `smallm_route_kernel(int const*, int*, int*, int, int, int)` | 0.00 | 1.42 | 0.00 | 0 | 2.7% |
| 0.1 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 0.59 | 27.02 | 0.00 | 0 | 51.8% |
| 0.2 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const*, float cons` | 2.61 | 38.26 | 0.00 | 0 | 73.3% |
| 0.3 | `_epilogue_kernel` | 40.86 | 6.51 | 0.06 | 0 | 12.5% |

Totals: sum dur 73.21 | busy 47.34 | sum gaps 0.06 | head 4.56 | tail 3.89 | wall - span 4.76 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.06 | `void w4_smallm_kernel<false, 4>(uint4 const*, uint` (0.2) | `_epilogue_kernel` (0.3) |

Families (sum dur): moe_gemm(cuda smallm) 66.70, moe_epilogue 6.51

## shape 1: bs=16,load=decode_16,sharding=ep8  (w 60, wall 88.13 us, span 83.23 us, n 4, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 1.0 | `smallm_route_kernel(int const*, int*, int*, int, int, int)` | 0.00 | 1.34 | 0.00 | 0 | 1.5% |
| 1.1 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 1.01 | 48.45 | 0.00 | 0 | 55.0% |
| 1.2 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*, float cons` | 40.62 | 35.94 | 0.00 | 0 | 40.8% |
| 1.3 | `_epilogue_kernel` | 77.06 | 6.19 | 0.42 | 0 | 7.0% |

Totals: sum dur 91.92 | busy 82.80 | sum gaps 0.42 | head 4.72 | tail 4.00 | wall - span 4.90 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.42 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint` (1.2) | `_epilogue_kernel` (1.3) |

Families (sum dur): moe_gemm(cuda smallm) 85.73, moe_epilogue 6.19

## shape 2: bs=32,load=decode_32,sharding=ep8  (w 60, wall 228.54 us, span 223.86 us, n 4, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 2.0 | `largem_route_kernel(int const*, int*, int*, int*, int, int, int)` | 0.00 | 3.06 | 0.00 | 0 | 1.3% |
| 2.1 | `void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const*, float co` | 0.99 | 146.96 | 0.00 | 0 | 64.3% |
| 2.2 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const*, float c` | 137.70 | 81.42 | 0.00 | 0 | 35.6% |
| 2.3 | `_post_expert_norm_weighted_sum_kernel` | 219.65 | 4.19 | 0.42 | 0 | 1.8% |

Totals: sum dur 235.63 | busy 223.44 | sum gaps 0.42 | head 4.69 | tail 3.97 | wall - span 4.68 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.42 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, u` (2.2) | `_post_expert_norm_weighted_sum_kernel` (2.3) |

Families (sum dur): other 231.44, norm/residual 4.19

## shape 3: bs=8,load=verify_1,sharding=ep8  (w 60, wall 91.36 us, span 86.58 us, n 4, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 3.0 | `smallm_route_kernel(int const*, int*, int*, int, int, int)` | 0.00 | 1.31 | 0.00 | 0 | 1.4% |
| 3.1 | `void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, float const` | 0.90 | 53.23 | 0.00 | 0 | 58.3% |
| 3.2 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*, float cons` | 44.22 | 37.86 | 0.00 | 0 | 41.4% |
| 3.3 | `_epilogue_kernel` | 82.74 | 3.87 | 0.45 | 0 | 4.2% |

Totals: sum dur 96.27 | busy 85.97 | sum gaps 0.45 | head 4.67 | tail 4.13 | wall - span 4.78 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.45 | `void w4_smallm_kernel<false, 8>(uint4 const*, uint` (3.2) | `_epilogue_kernel` (3.3) |

Families (sum dur): moe_gemm(cuda smallm) 92.40, moe_epilogue 3.87

## shape 4: bs=128,load=verify_16,sharding=ep8  (w 60, wall 354.82 us, span 349.95 us, n 4, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 4.0 | `largem_route_kernel(int const*, int*, int*, int*, int, int, int)` | 0.00 | 6.26 | 0.00 | 0 | 1.8% |
| 4.1 | `void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const*, float co` | 1.02 | 230.05 | 0.00 | 0 | 64.8% |
| 4.2 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const*, float c` | 224.38 | 120.54 | 0.00 | 0 | 34.0% |
| 4.3 | `_post_expert_norm_weighted_sum_kernel` | 345.35 | 4.54 | 0.48 | 0 | 1.3% |

Totals: sum dur 361.39 | busy 349.50 | sum gaps 0.48 | head 4.70 | tail 4.06 | wall - span 4.87 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.48 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, u` (4.2) | `_post_expert_norm_weighted_sum_kernel` (4.3) |

Families (sum dur): other 356.85, norm/residual 4.54

## shape 5: bs=256,load=verify_32,sharding=ep8  (w 60, wall 493.39 us, span 488.18 us, n 4, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 5.0 | `largem_route_kernel(int const*, int*, int*, int*, int, int, int)` | 0.00 | 10.66 | 0.00 | 0 | 2.2% |
| 5.1 | `void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const*, float co` | 0.99 | 322.10 | 0.00 | 0 | 65.3% |
| 5.2 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const*, float c` | 320.91 | 161.82 | 0.00 | 0 | 32.8% |
| 5.3 | `_post_expert_norm_weighted_sum_kernel` | 482.85 | 5.22 | 0.26 | 0 | 1.1% |

Totals: sum dur 499.80 | busy 487.92 | sum gaps 0.26 | head 4.80 | tail 4.05 | wall - span 5.21 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.26 | `void w4_largem_kernel<false, 4, 2>(uint4 const*, u` (5.2) | `_post_expert_norm_weighted_sum_kernel` (5.3) |

Families (sum dur): other 494.58, norm/residual 5.22

## families, weighted by shape weight (sum dur)

| family | w x us | share |
|---|---|---|
| other | 64972.2 | 79.7% |
| moe_gemm(cuda smallm) | 14689.8 | 18.0% |
| moe_epilogue | 994.2 | 1.2% |
| norm/residual | 837.0 | 1.0% |
