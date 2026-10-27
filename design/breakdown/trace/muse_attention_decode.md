# TRACE report

- problem=muse_attention_decode file=sonnet_v3.py torch=2.14.1+cu130 cuda=13.0 gpu=NVIDIA_H200 cupti=graph replays=20
- prepare time 16 s

## invariants

- all checks pass

## shape 0: batch=1,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=decode_1,page=1  (w 46, wall 12.11 us, span 7.70 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 0.0 | `_attn_kernel` | 0.00 | 4.80 | 0.00 | 0 | 39.6% |
| 0.1 | `_reduce_kernel` | 5.09 | 2.59 | 0.29 | 0 | 21.4% |

Totals: sum dur 7.39 | busy 7.41 | sum gaps 0.29 | head 4.14 | tail 4.07 | wall - span 4.41 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.29 | `_attn_kernel` (0.0) | `_reduce_kernel` (0.1) |

Families (sum dur): attention 4.80, attention_reduce 2.59

## shape 1: batch=1,ctx=80000,dist=coherent,heads=tp8,layer=full,load=decode_1,page=1  (w 16, wall 19.23 us, span 15.17 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 1.0 | `_full_cat_kernel` | 0.00 | 11.70 | 0.00 | 0 | 60.8% |
| 1.1 | `_reduce_kernel` | 11.97 | 3.20 | 0.29 | 0 | 16.6% |

Totals: sum dur 14.90 | busy 14.88 | sum gaps 0.29 | head 3.98 | tail 3.84 | wall - span 4.06 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.29 | `_full_cat_kernel` (1.0) | `_reduce_kernel` (1.1) |

Families (sum dur): other 11.70, attention_reduce 3.20

## shape 2: batch=16,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=decode_16,page=1,spread=agentic  (w 46, wall 17.20 us, span 13.12 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 2.0 | `_attn_kernel` | 0.00 | 9.82 | 0.00 | 0 | 57.1% |
| 2.1 | `_reduce_kernel` | 10.11 | 3.04 | 0.29 | 0 | 17.7% |

Totals: sum dur 12.86 | busy 12.83 | sum gaps 0.29 | head 4.02 | tail 3.82 | wall - span 4.08 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.29 | `_attn_kernel` (2.0) | `_reduce_kernel` (2.1) |

Families (sum dur): attention 9.82, attention_reduce 3.04

## shape 3: batch=16,ctx=80000,dist=coherent,heads=tp8,layer=full,load=decode_16,page=1,spread=agentic  (w 16, wall 108.38 us, span 104.18 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 3.0 | `_full_cat_kernel` | 0.00 | 100.08 | 0.00 | 0 | 92.3% |
| 3.1 | `_reduce_kernel` | 100.66 | 3.58 | 0.58 | 0 | 3.3% |

Totals: sum dur 103.66 | busy 103.60 | sum gaps 0.58 | head 4.24 | tail 3.92 | wall - span 4.20 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `_full_cat_kernel` (3.0) | `_reduce_kernel` (3.1) |

Families (sum dur): other 100.08, attention_reduce 3.58

## shape 4: batch=32,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=decode_32,page=1,spread=agentic  (w 46, wall 21.47 us, span 17.04 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 4.0 | `_attn_kernel` | 0.00 | 13.44 | 0.00 | 0 | 62.6% |
| 4.1 | `_reduce_kernel` | 14.08 | 2.98 | 0.61 | 0 | 13.9% |

Totals: sum dur 16.42 | busy 16.40 | sum gaps 0.61 | head 4.40 | tail 3.95 | wall - span 4.43 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.61 | `_attn_kernel` (4.0) | `_reduce_kernel` (4.1) |

Families (sum dur): attention 13.44, attention_reduce 2.98

## shape 5: batch=32,ctx=80000,dist=coherent,heads=tp8,layer=full,load=decode_32,page=1,spread=agentic  (w 16, wall 195.42 us, span 191.38 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 5.0 | `_full_cat_kernel` | 0.00 | 187.04 | 0.00 | 0 | 95.7% |
| 5.1 | `_reduce_kernel` | 187.73 | 3.58 | 0.58 | 0 | 1.8% |

Totals: sum dur 190.62 | busy 190.69 | sum gaps 0.58 | head 4.08 | tail 3.87 | wall - span 4.04 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `_full_cat_kernel` (5.0) | `_reduce_kernel` (5.1) |

Families (sum dur): other 187.04, attention_reduce 3.58

## shape 6: batch=1,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=verify_1,page=1,q_len=8  (w 46, wall 13.71 us, span 9.23 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 6.0 | `_attn_kernel` | 0.00 | 5.76 | 0.00 | 0 | 42.0% |
| 6.1 | `_reduce_kernel` | 6.34 | 2.78 | 0.58 | 0 | 20.3% |

Totals: sum dur 8.54 | busy 8.61 | sum gaps 0.58 | head 4.22 | tail 4.13 | wall - span 4.48 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `_attn_kernel` (6.0) | `_reduce_kernel` (6.1) |

Families (sum dur): attention 5.76, attention_reduce 2.78

## shape 7: batch=1,ctx=80000,dist=coherent,heads=tp8,layer=full,load=verify_1,page=1,q_len=8  (w 16, wall 24.82 us, span 20.90 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 7.0 | `_full_kernel` | 0.00 | 16.77 | 0.00 | 0 | 67.6% |
| 7.1 | `_reduce_kernel` | 17.06 | 3.81 | 0.29 | 0 | 15.4% |

Totals: sum dur 20.58 | busy 20.61 | sum gaps 0.29 | head 3.92 | tail 3.78 | wall - span 3.92 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.29 | `_full_kernel` (7.0) | `_reduce_kernel` (7.1) |

Families (sum dur): other 16.77, attention_reduce 3.81

## shape 8: batch=16,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=verify_16,page=1,q_len=8,spread=agentic  (w 46, wall 21.87 us, span 17.47 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 8.0 | `_attn_kernel` | 0.00 | 11.12 | 0.00 | 0 | 50.8% |
| 8.1 | `_reduce_kernel` | 11.70 | 5.73 | 0.58 | 0 | 26.2% |

Totals: sum dur 16.85 | busy 16.90 | sum gaps 0.58 | head 4.18 | tail 3.98 | wall - span 4.40 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `_attn_kernel` (8.0) | `_reduce_kernel` (8.1) |

Families (sum dur): attention 11.12, attention_reduce 5.73

## shape 9: batch=16,ctx=80000,dist=coherent,heads=tp8,layer=full,load=verify_16,page=1,q_len=8,spread=agentic  (w 16, wall 192.50 us, span 188.24 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 9.0 | `_full_kernel` | 0.00 | 181.28 | 0.00 | 0 | 94.2% |
| 9.1 | `_reduce_kernel` | 181.86 | 6.40 | 0.58 | 0 | 3.3% |

Totals: sum dur 187.68 | busy 187.66 | sum gaps 0.58 | head 4.03 | tail 3.87 | wall - span 4.26 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `_full_kernel` (9.0) | `_reduce_kernel` (9.1) |

Families (sum dur): other 181.28, attention_reduce 6.40

## shape 10: batch=32,ctx=80000,dist=soft,heads=tp8,layer=sliding,load=verify_32,page=1,q_len=8,spread=agentic  (w 46, wall 30.29 us, span 26.08 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 10.0 | `_attn_kernel` | 0.00 | 17.26 | 0.00 | 0 | 57.0% |
| 10.1 | `_reduce_kernel` | 17.84 | 8.13 | 0.58 | 0 | 26.8% |

Totals: sum dur 25.39 | busy 25.50 | sum gaps 0.58 | head 4.13 | tail 3.90 | wall - span 4.21 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.58 | `_attn_kernel` (10.0) | `_reduce_kernel` (10.1) |

Families (sum dur): attention 17.26, attention_reduce 8.13

## shape 11: batch=32,ctx=80000,dist=coherent,heads=tp8,layer=full,load=verify_32,page=1,q_len=8,spread=agentic  (w 16, wall 369.87 us, span 365.14 us, n 2, replays 20/20)

| pos | kernel | start | dur | gap | stream | % of wall |
|---|---|---|---|---|---|---|
| 11.0 | `_full_kernel` | 0.00 | 355.23 | 0.00 | 0 | 96.0% |
| 11.1 | `_reduce_kernel` | 355.84 | 9.15 | 0.61 | 0 | 2.5% |

Totals: sum dur 364.38 | busy 364.53 | sum gaps 0.61 | head 4.29 | tail 4.00 | wall - span 4.73 us

Largest gaps:

| gap us | after | before the next |
|---|---|---|
| 0.61 | `_full_kernel` (11.0) | `_reduce_kernel` (11.1) |

Families (sum dur): other 355.23, attention_reduce 9.15

## families, weighted by shape weight (sum dur)

| family | w x us | share |
|---|---|---|
| other | 13633.6 | 75.2% |
| attention | 2861.2 | 15.8% |
| attention_reduce | 1637.0 | 9.0% |
