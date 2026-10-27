# muse_attention_decode

## Shapes
| index | label | wall | &Sigma;dur | busy | &Sigma;gaps | head | tail | kernels |
|---|---|---|---|---|---|---|---|---|
| 6 | batch=1,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=verify_1,page=1,q_len=8 | 13.71 | 8.54 | 8.61 | 0.58 | 4.22 | 4.13 | 2 |
| 0 | batch=1,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=decode_1,page=1 | 12.11 | 7.39 | 7.41 | 0.29 | 4.14 | 4.07 | 2 |
| 8 | batch=16,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=verify_16,page=1,q_len=8,spread=agentic | 21.87 | 16.85 | 16.90 | 0.58 | 4.18 | 3.98 | 2 |
| 2 | batch=16,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=decode_16,page=1,spread=agentic | 17.20 | 12.86 | 12.83 | 0.29 | 4.02 | 3.82 | 2 |
| 10 | batch=32,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=verify_32,page=1,q_len=8,spread=agentic | 30.29 | 25.39 | 25.50 | 0.58 | 4.13 | 3.90 | 2 |
| 4 | batch=32,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=decode_32,page=1,spread=agentic | 21.47 | 16.42 | 16.40 | 0.61 | 4.40 | 3.95 | 2 |
| 7 | batch=1,ctx=80000,dist=coherent,heads=tp8,layer=full,load=verify_1,page=1,q_len=8 | 24.82 | 20.58 | 20.61 | 0.29 | 3.92 | 3.78 | 2 |
| 1 | batch=1,ctx=80000,dist=coherent,heads=tp8,layer=full,load=decode_1,page=1 | 19.23 | 14.90 | 14.88 | 0.29 | 3.98 | 3.84 | 2 |
| 9 | batch=16,ctx=80000,dist=coherent,heads=tp8,layer=full,load=verify_16,page=1,q_len=8,spread=agentic | 192.50 | 187.68 | 187.66 | 0.58 | 4.03 | 3.87 | 2 |
| 11 | batch=32,ctx=80000,dist=coherent,heads=tp8,layer=full,load=verify_32,page=1,q_len=8,spread=agentic | 369.87 | 364.38 | 364.53 | 0.61 | 4.29 | 4.00 | 2 |
| 3 | batch=16,ctx=80000,dist=coherent,heads=tp8,layer=full,load=decode_16,page=1,spread=agentic | 108.38 | 103.66 | 103.60 | 0.58 | 4.24 | 3.92 | 2 |
| 5 | batch=32,ctx=80000,dist=coherent,heads=tp8,layer=full,load=decode_32,page=1,spread=agentic | 195.42 | 190.62 | 190.69 | 0.58 | 4.08 | 3.87 | 2 |

## Top 15 items
| rank | kind | shape | name | stream | start &mu;s | dur &mu;s | weight &times; &mu;s |
|---|---|---|---|---|---|---|---|
| 1 | kernel | 11 | _full_kernel | 0 | 0.00 | 355.23 | 5683.68 |
| 2 | kernel | 5 | _full_cat_kernel | 0 | 0.00 | 187.04 | 2992.64 |
| 3 | kernel | 9 | _full_kernel | 0 | 0.00 | 181.28 | 2900.48 |
| 4 | kernel | 3 | _full_cat_kernel | 0 | 0.00 | 100.08 | 1601.28 |
| 5 | kernel | 10 | _attn_kernel | 0 | 0.00 | 17.26 | 793.96 |
| 6 | kernel | 4 | _attn_kernel | 0 | 0.00 | 13.44 | 618.24 |
| 7 | kernel | 8 | _attn_kernel | 0 | 0.00 | 11.12 | 511.52 |
| 8 | kernel | 2 | _attn_kernel | 0 | 0.00 | 9.82 | 451.72 |
| 9 | kernel | 10 | _reduce_kernel | 0 | 17.84 | 8.13 | 373.98 |
| 10 | kernel | 7 | _full_kernel | 0 | 0.00 | 16.77 | 268.32 |
| 11 | kernel | 6 | _attn_kernel | 0 | 0.00 | 5.76 | 264.96 |
| 12 | kernel | 8 | _reduce_kernel | 0 | 11.70 | 5.73 | 263.58 |
| 13 | kernel | 0 | _attn_kernel | 0 | 0.00 | 4.80 | 220.80 |
| 14 | head | 4 | (head) | - | 0.00 | 4.40 | 202.40 |
| 15 | head | 6 | (head) | - | 0.00 | 4.22 | 194.12 |

## Families
| family | w &mu;s | share | floor &mu;s | efficiency |
|---|---|---|---|---|
| other | 13633.60 | 75.2% | 7446.92 | 54.6% |
| attention | 2861.20 | 15.8% | 614.24 | 21.5% |
| attention_reduce | 1637.02 | 9.0% | 14.00 | 0.9% |

# muse_bf16_proj

## Shapes
| index | label | wall | &Sigma;dur | busy | &Sigma;gaps | head | tail | kernels |
|---|---|---|---|---|---|---|---|---|
| 18 | layer=unpooler,load=verify_1,m=8 | 11.65 | 6.82 | 6.80 | 0.32 | 4.26 | 3.95 | 2 |
| 9 | layer=router,load=decode_16,m=16 | 11.84 | 6.85 | 6.88 | 0.61 | 4.19 | 3.84 | 2 |
| 24 | layer=router,load=verify_16,m=128 | 14.29 | 9.48 | 9.47 | 0.58 | 4.03 | 3.78 | 2 |
| 4 | layer=router,load=decode_1,m=1 | 11.17 | 6.17 | 6.18 | 0.64 | 4.16 | 3.81 | 2 |
| 14 | layer=router,load=decode_32,m=32 | 11.63 | 6.78 | 6.82 | 0.61 | 4.10 | 3.79 | 2 |
| 29 | layer=router,load=verify_32,m=256 | 16.66 | 11.73 | 11.74 | 0.61 | 4.19 | 3.87 | 2 |
| 3 | layer=unpooler,load=decode_1,m=1 | 10.78 | 6.72 | 6.75 | 0.03 | 3.98 | 3.63 | 2 |
| 19 | layer=router,load=verify_1,m=8 | 10.35 | 6.06 | 6.08 | 0.29 | 3.81 | 3.57 | 2 |
| 11 | layer=o,load=decode_32,m=32 | 10.93 | 6.80 | 6.80 | 0.00 | 4.02 | 3.79 | 1 |
| 16 | layer=o,load=verify_1,m=8 | 10.48 | 6.42 | 6.42 | 0.00 | 3.98 | 3.79 | 1 |
| 6 | layer=o,load=decode_16,m=16 | 10.46 | 6.40 | 6.40 | 0.00 | 3.98 | 3.76 | 1 |
| 1 | layer=o,load=decode_1,m=1 | 10.13 | 6.35 | 6.35 | 0.00 | 3.71 | 3.60 | 1 |
| 26 | layer=o,load=verify_32,m=256 | 12.83 | 8.75 | 8.75 | 0.00 | 4.02 | 3.81 | 1 |
| 21 | layer=o,load=verify_16,m=128 | 10.74 | 6.91 | 6.91 | 0.00 | 3.70 | 3.65 | 1 |
| 20 | layer=qkv_gate,load=verify_16,m=128 | 20.93 | 16.95 | 16.93 | 0.00 | 3.90 | 3.57 | 2 |
| 15 | layer=qkv_gate,load=verify_1,m=8 | 19.12 | 14.40 | 14.43 | 0.32 | 4.26 | 3.81 | 2 |
| 25 | layer=qkv_gate,load=verify_32,m=256 | 23.12 | 19.12 | 19.12 | 0.00 | 3.94 | 3.71 | 1 |
| 5 | layer=qkv_gate,load=decode_16,m=16 | 18.74 | 14.00 | 14.03 | 0.35 | 4.25 | 3.84 | 2 |
| 10 | layer=qkv_gate,load=decode_32,m=32 | 18.80 | 14.00 | 14.03 | 0.32 | 4.34 | 3.82 | 2 |
| 0 | layer=qkv_gate,load=decode_1,m=1 | 17.57 | 13.21 | 13.23 | 0.03 | 3.95 | 3.87 | 2 |
| 22 | layer=pooler,load=verify_16,m=128 | 28.21 | 24.48 | 24.48 | 0.00 | 3.71 | 3.62 | 1 |
| 27 | layer=pooler,load=verify_32,m=256 | 33.41 | 29.26 | 29.26 | 0.00 | 4.08 | 3.73 | 1 |
| 12 | layer=pooler,load=decode_32,m=32 | 23.57 | 19.46 | 19.46 | 0.00 | 4.03 | 3.70 | 1 |
| 7 | layer=pooler,load=decode_16,m=16 | 22.88 | 18.75 | 18.75 | 0.00 | 4.03 | 3.73 | 1 |
| 28 | layer=unpooler,load=verify_32,m=256 | 28.21 | 24.11 | 24.11 | 0.00 | 4.06 | 3.74 | 1 |
| 17 | layer=pooler,load=verify_1,m=8 | 22.70 | 18.69 | 18.69 | 0.00 | 3.94 | 3.78 | 1 |
| 2 | layer=pooler,load=decode_1,m=1 | 22.21 | 18.53 | 18.53 | 0.00 | 3.63 | 3.57 | 1 |
| 23 | layer=unpooler,load=verify_16,m=128 | 22.85 | 18.79 | 18.79 | 0.00 | 3.92 | 3.81 | 1 |
| 8 | layer=unpooler,load=decode_16,m=16 | 21.78 | 17.49 | 17.49 | 0.00 | 4.16 | 3.78 | 1 |
| 13 | layer=unpooler,load=decode_32,m=32 | 21.86 | 17.78 | 17.78 | 0.00 | 3.98 | 3.74 | 1 |

## Top 15 items
| rank | kind | shape | name | stream | start &mu;s | dur &mu;s | weight &times; &mu;s |
|---|---|---|---|---|---|---|---|
| 1 | kernel | 27 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT | 0 | 0.00 | 29.26 | 1814.12 |
| 2 | kernel | 22 | nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT | 0 | 0.00 | 24.48 | 1517.76 |
| 3 | kernel | 28 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT | 0 | 0.00 | 24.11 | 1494.82 |
| 4 | kernel | 12 | _gemv_splitk | 0 | 0.00 | 19.46 | 1206.52 |
| 5 | kernel | 25 | nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN | 0 | 0.00 | 19.12 | 1185.44 |
| 6 | kernel | 23 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT | 0 | 0.00 | 18.79 | 1164.98 |
| 7 | kernel | 7 | _gemv_splitk | 0 | 0.00 | 18.75 | 1162.50 |
| 8 | kernel | 17 | _gemv_splitk | 0 | 0.00 | 18.69 | 1158.78 |
| 9 | kernel | 2 | _gemv_splitk | 0 | 0.00 | 18.53 | 1148.86 |
| 10 | kernel | 13 | nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT | 0 | 0.00 | 17.78 | 1102.36 |
| 11 | kernel | 8 | nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT | 0 | 0.00 | 17.49 | 1084.38 |
| 12 | kernel | 20 | nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN | 0 | 0.00 | 13.62 | 844.44 |
| 13 | kernel | 15 | nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT | 0 | 0.00 | 12.38 | 767.56 |
| 14 | kernel | 5 | nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT | 0 | 0.00 | 11.98 | 742.76 |
| 15 | kernel | 10 | nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT | 0 | 0.00 | 11.89 | 737.18 |

## Families
| family | w &mu;s | share | floor &mu;s | efficiency |
|---|---|---|---|---|
| dense_gemm(cublas) | 15672.98 | 63.0% | 11097.36 | 70.8% |
| dense_gemm | 8953.42 | 36.0% | 6138.07 | 68.6% |
| elementwise | 251.72 | 1.0% | 0.00 | 0.0% |

# muse_decode_layer

## Shapes
| index | label | wall | &Sigma;dur | busy | &Sigma;gaps | head | tail | kernels |
|---|---|---|---|---|---|---|---|---|
| 10 | batch=32,ctx=80000,dist=soft,layer=sliding,load=verify_32,page=1,q_len=8,spread=agentic | 690.11 | 707.29 | 675.68 | 7.15 | 6.86 | 3.94 | 19 |
| 8 | batch=16,ctx=80000,dist=soft,layer=sliding,load=verify_16,page=1,q_len=8,spread=agentic | 538.54 | 551.24 | 526.27 | 8.18 | 3.82 | 3.89 | 20 |
| 6 | batch=1,ctx=80000,dist=soft,layer=sliding,load=verify_1,page=1,q_len=8 | 204.64 | 224.74 | 193.89 | 6.74 | 3.92 | 3.94 | 20 |
| 0 | batch=1,ctx=80000,dist=soft,layer=sliding,load=decode_1,page=1 | 129.63 | 151.21 | 117.92 | 7.15 | 4.13 | 3.98 | 20 |
| 4 | batch=32,ctx=80000,dist=soft,layer=sliding,load=decode_32,page=1,spread=agentic | 220.05 | 226.01 | 208.32 | 7.63 | 3.84 | 3.95 | 20 |
| 2 | batch=16,ctx=80000,dist=soft,layer=sliding,load=decode_16,page=1,spread=agentic | 178.82 | 198.67 | 169.65 | 4.64 | 4.06 | 3.94 | 17 |
| 11 | batch=32,ctx=80000,dist=soft,layer=full,load=verify_32,page=1,q_len=8,spread=agentic | 1077.06 | 1095.30 | 1063.61 | 6.51 | 6.59 | 3.86 | 19 |
| 1 | batch=1,ctx=80000,dist=soft,layer=full,load=decode_1,page=1 | 151.52 | 185.28 | 140.59 | 6.80 | 4.02 | 3.98 | 20 |
| 9 | batch=16,ctx=80000,dist=soft,layer=full,load=verify_16,page=1,q_len=8,spread=agentic | 646.74 | 659.73 | 634.96 | 7.46 | 3.84 | 3.97 | 20 |
| 7 | batch=1,ctx=80000,dist=soft,layer=full,load=verify_1,page=1,q_len=8 | 192.94 | 211.65 | 181.76 | 6.72 | 4.06 | 3.90 | 20 |
| 5 | batch=32,ctx=80000,dist=soft,layer=full,load=decode_32,page=1,spread=agentic | 535.73 | 542.10 | 524.05 | 7.46 | 3.94 | 3.90 | 20 |
| 3 | batch=16,ctx=80000,dist=soft,layer=full,load=decode_16,page=1,spread=agentic | 340.56 | 362.37 | 331.68 | 4.41 | 3.98 | 3.89 | 17 |

## Top 15 items
| rank | kind | shape | name | stream | start &mu;s | dur &mu;s | weight &times; &mu;s |
|---|---|---|---|---|---|---|---|
| 1 | kernel | 10 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 125.87 | 335.36 | 15426.56 |
| 2 | kernel | 8 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 98.66 | 261.78 | 12041.88 |
| 3 | kernel | 10 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 468.96 | 174.53 | 8028.38 |
| 4 | kernel | 8 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 365.65 | 136.69 | 6287.74 |
| 5 | kernel | 11 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 454.01 | 375.95 | 6015.20 |
| 6 | kernel | 11 | _attn_kernel | 0 | 20.10 | 337.18 | 5394.88 |
| 7 | kernel | 5 | _attn_kernel | 0 | 15.12 | 314.88 | 5038.08 |
| 8 | kernel | 9 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 277.94 | 214.80 | 3436.80 |
| 9 | kernel | 11 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 836.89 | 194.27 | 3108.32 |
| 10 | kernel | 4 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 0 | 81.38 | 64.59 | 2971.14 |
| 11 | kernel | 6 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 0 | 75.01 | 64.43 | 2963.78 |
| 12 | kernel | 9 | _attn_kernel | 0 | 17.95 | 180.59 | 2889.44 |
| 13 | kernel | 3 | _attn_kernel | 0 | 14.99 | 160.27 | 2564.32 |
| 14 | kernel | 6 | void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*,  | 0 | 128.62 | 50.98 | 2345.08 |
| 15 | kernel | 2 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 0 | 70.30 | 46.13 | 2121.98 |

## Families
| family | w &mu;s | share | floor &mu;s | efficiency |
|---|---|---|---|---|
| moe_gemm(marlin) | 62622.96 | 43.6% | 29026.43 | 46.4% |
| attention | 19263.14 | 13.4% | 8029.66 | 41.7% |
| dense_gemm(cublas) | 18275.90 | 12.7% | 11684.08 | 63.9% |
| moe_gemm(cuda smallm) | 15243.84 | 10.6% | 0.00 | 0.0% |
| dense_gemm | 13866.30 | 9.7% | 4940.89 | 35.6% |
| norm/residual | 6832.30 | 4.8% | 560.73 | 8.2% |
| attention_reduce | 1849.84 | 1.3% | 0.00 | 0.0% |
| routing | 1788.18 | 1.2% | 0.05 | 0.0% |
| elementwise | 1583.38 | 1.1% | 0.00 | 0.0% |
| moe_epilogue | 947.46 | 0.7% | 0.39 | 0.0% |
| activation | 731.64 | 0.5% | 0.99 | 0.1% |
| other | 619.30 | 0.4% | 0.00 | 0.0% |

# muse_moe_w4a16

## Shapes
| index | label | wall | &Sigma;dur | busy | &Sigma;gaps | head | tail | kernels |
|---|---|---|---|---|---|---|---|---|
| 0 | bs=1,load=decode_1,sharding=ep8 | 52.18 | 73.21 | 47.34 | 0.06 | 4.56 | 3.89 | 4 |
| 5 | bs=256,load=verify_32,sharding=ep8 | 493.39 | 499.80 | 487.92 | 0.26 | 4.80 | 4.05 | 4 |
| 4 | bs=128,load=verify_16,sharding=ep8 | 354.82 | 361.39 | 349.50 | 0.48 | 4.70 | 4.06 | 4 |
| 2 | bs=32,load=decode_32,sharding=ep8 | 228.54 | 235.63 | 223.44 | 0.42 | 4.69 | 3.97 | 4 |
| 3 | bs=8,load=verify_1,sharding=ep8 | 91.36 | 96.27 | 85.97 | 0.45 | 4.67 | 4.13 | 4 |
| 1 | bs=16,load=decode_16,sharding=ep8 | 88.13 | 91.92 | 82.80 | 0.42 | 4.72 | 4.00 | 4 |

## Top 15 items
| rank | kind | shape | name | stream | start &mu;s | dur &mu;s | weight &times; &mu;s |
|---|---|---|---|---|---|---|---|
| 1 | kernel | 5 | void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const* | 0 | 0.99 | 322.10 | 19326.00 |
| 2 | kernel | 4 | void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const* | 0 | 1.02 | 230.05 | 13803.00 |
| 3 | kernel | 5 | void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const | 0 | 320.91 | 161.82 | 9709.20 |
| 4 | kernel | 2 | void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const* | 0 | 0.99 | 146.96 | 8817.60 |
| 5 | kernel | 4 | void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const | 0 | 224.38 | 120.54 | 7232.40 |
| 6 | kernel | 2 | void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const | 0 | 137.70 | 81.42 | 4885.20 |
| 7 | kernel | 3 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 0 | 0.90 | 53.23 | 3193.80 |
| 8 | kernel | 1 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 0 | 1.01 | 48.45 | 2907.00 |
| 9 | kernel | 0 | void w4_smallm_kernel<false, 4>(uint4 const*, uint2 const*,  | 0 | 2.61 | 38.26 | 2295.60 |
| 10 | kernel | 3 | void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*,  | 0 | 44.22 | 37.86 | 2271.60 |
| 11 | kernel | 1 | void w4_smallm_kernel<false, 8>(uint4 const*, uint2 const*,  | 0 | 40.62 | 35.94 | 2156.40 |
| 12 | kernel | 0 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 0 | 0.59 | 27.02 | 1621.20 |
| 13 | kernel | 5 | largem_route_kernel(int const*, int*, int*, int*, int, int,  | 0 | 0.00 | 10.66 | 639.60 |
| 14 | kernel | 0 | _epilogue_kernel | 0 | 40.86 | 6.51 | 390.60 |
| 15 | kernel | 4 | largem_route_kernel(int const*, int*, int*, int*, int, int,  | 0 | 0.00 | 6.26 | 375.60 |

## Families
| family | w &mu;s | share | floor &mu;s | efficiency |
|---|---|---|---|---|
| other | 64972.20 | 79.7% | 0.40 | 0.0% |
| moe_gemm(cuda smallm) | 14689.80 | 18.0% | 9715.35 | 66.1% |
| moe_epilogue | 994.20 | 1.2% | 3.46 | 0.3% |
| norm/residual | 837.00 | 1.0% | 51.49 | 6.2% |

# Top 20 items (all problems)
| rank | prob | kind | shape | name | dur &mu;s | weight &times; &mu;s |
|---|---|---|---|---|---|---|
| 1 | muse_moe_w4a16 | kernel | 5 | void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const* | 322.10 | 19326.00 |
| 2 | muse_decode_layer | kernel | 10 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 335.36 | 15426.56 |
| 3 | muse_moe_w4a16 | kernel | 4 | void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const* | 230.05 | 13803.00 |
| 4 | muse_decode_layer | kernel | 8 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 261.78 | 12041.88 |
| 5 | muse_moe_w4a16 | kernel | 5 | void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const | 161.82 | 9709.20 |
| 6 | muse_moe_w4a16 | kernel | 2 | void w4_largem_kernel<true, 4, 2>(uint4 const*, uint2 const* | 146.96 | 8817.60 |
| 7 | muse_decode_layer | kernel | 10 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 174.53 | 8028.38 |
| 8 | muse_moe_w4a16 | kernel | 4 | void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const | 120.54 | 7232.40 |
| 9 | muse_decode_layer | kernel | 8 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 136.69 | 6287.74 |
| 10 | muse_decode_layer | kernel | 11 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 375.95 | 6015.20 |
| 11 | muse_attention_decode | kernel | 11 | _full_kernel | 355.23 | 5683.68 |
| 12 | muse_decode_layer | kernel | 11 | _attn_kernel | 337.18 | 5394.88 |
| 13 | muse_decode_layer | kernel | 5 | _attn_kernel | 314.88 | 5038.08 |
| 14 | muse_moe_w4a16 | kernel | 2 | void w4_largem_kernel<false, 4, 2>(uint4 const*, uint2 const | 81.42 | 4885.20 |
| 15 | muse_decode_layer | kernel | 9 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 214.80 | 3436.80 |
| 16 | muse_moe_w4a16 | kernel | 3 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 53.23 | 3193.80 |
| 17 | muse_decode_layer | kernel | 11 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 194.27 | 3108.32 |
| 18 | muse_attention_decode | kernel | 5 | _full_cat_kernel | 187.04 | 2992.64 |
| 19 | muse_decode_layer | kernel | 4 | void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56294 | 64.59 | 2971.14 |
| 20 | muse_decode_layer | kernel | 6 | void w4_smallm_kernel<true, 8>(uint4 const*, uint2 const*, f | 64.43 | 2963.78 |