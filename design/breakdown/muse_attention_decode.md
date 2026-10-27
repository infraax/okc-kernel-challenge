# muse_attention_decode — judge job 4b047c1bf5354162 (speedup 1.689x, score_us 2229)

Units µs (judge median, replay overhead already subtracted; p10–p90 overhead-corrected; sol floored at 2.5 µs). loss = weight × ln(ours/sol): what the board still
pays on this shape vs the floor; Σloss over shapes is the problem's total headroom in log units.

| # | shape | w | sglang | ours | sol | speedup | ours/sol | p10–p90 | kernels Σ | hidden | idle | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 1 page=1 | 46 | 20.3 | 8.0 | 2.5 | 2.540 | 3.19 | 7.7–8.4 | 7.0 | 0.0 (0) | 1.0 | 53.39 (10%) |
| 1 | batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 1 page=1 | 16 | 30.6 | 15.3 | 4.3 | 1.999 | 3.59 | 15.0–15.6 | 13.0 | 0.0 (0) | 2.3 | 20.45 (4%) |
| 2 | batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 16 page=1 spread=agentic | 46 | 27.0 | 13.7 | 2.5 | 1.979 | 5.46 | 13.4–14.0 | 12.3 | 0.0 (0) | 1.4 | 78.09 (14%) |
| 3 | batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 16 page=1 spread=agentic | 16 | 200.1 | 106.9 | 63.2 | 1.872 | 1.69 | 106.0–107.8 | 107.1 | 0.0 (0) | -0.2 | 8.42 (2%) |
| 4 | batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 32 page=1 spread=agentic | 46 | 30.9 | 16.4 | 3.3 | 1.881 | 4.99 | 16.1–16.9 | 14.3 | 0.1 (0) | 2.1 | 73.92 (14%) |
| 5 | batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 32 page=1 spread=agentic | 16 | 383.0 | 195.0 | 126.4 | 1.965 | 1.54 | 193.6–196.2 | 196.7 | -0.0 (0) | -1.7 | 6.94 (1%) |
| 6 | batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 1 page=1 q_len=8 | 46 | 22.7 | 9.5 | 2.5 | 2.388 | 3.79 | 9.2–9.9 | 7.5 | 0.0 (0) | 2.0 | 61.34 (11%) |
| 7 | batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 1 page=1 q_len=8 | 16 | 35.3 | 21.5 | 4.3 | 1.645 | 5.02 | 21.1–21.9 | 19.6 | -0.1 (0) | 1.9 | 25.81 (5%) |
| 8 | batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 16 page=1 q_len=8 spread=agentic | 46 | 28.8 | 17.3 | 2.5 | 1.664 | 6.92 | 16.8–17.7 | 16.4 | -0.1 (0) | 0.9 | 88.99 (16%) |
| 9 | batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 16 page=1 q_len=8 spread=agentic | 16 | 204.9 | 192.1 | 63.4 | 1.067 | 3.03 | 190.7–193.0 | 190.4 | 0.0 (0) | 1.7 | 17.74 (3%) |
| 10 | batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 32 page=1 q_len=8 spread=agentic | 46 | 33.9 | 25.4 | 3.7 | 1.337 | 6.81 | 25.0–25.7 | 23.3 | 0.1 (0) | 2.1 | 88.26 (16%) |
| 11 | batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 32 page=1 q_len=8 spread=agentic | 16 | 388.8 | 368.3 | 126.8 | 1.056 | 2.91 | 366.6–370.3 | 367.4 | 0.1 (0) | 0.9 | 17.06 (3%) |

## shape 0: batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 1 page=1  (w 46, ours 8.0 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 0.0 | `_attn_kernel` | attention | 4.7 | 58.9% | 1 | 4.7 |
| 0.1 | `_reduce_kernel` | attention_reduce | 2.3 | 28.8% | 1 | 2.3 |
| 0.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 0.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.0 | 12.3% | | |

## shape 1: batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 1 page=1  (w 16, ours 15.3 µs, sol 4.3)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 1.0 | `_full_cat_kernel` | attention | 10.4 | 67.8% | 1 | 10.4 |
| 1.1 | `_reduce_kernel` | attention_reduce | 2.6 | 17.0% | 1 | 2.6 |
| 1.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 1.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 2.3 | 15.2% | | |

## shape 2: batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 16 page=1 spread=agentic  (w 46, ours 13.7 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 2.0 | `_attn_kernel` | attention | 9.9 | 72.5% | 1 | 9.9 |
| 2.1 | `_reduce_kernel` | attention_reduce | 2.4 | 17.6% | 1 | 2.4 |
| 2.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 2.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.4 | 9.9% | | |

## shape 3: batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 16 page=1 spread=agentic  (w 16, ours 106.9 µs, sol 63.2)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 3.0 | `_full_cat_kernel` | attention | 103.8 | 97.1% | 1 | 103.8 |
| 3.1 | `_reduce_kernel` | attention_reduce | 3.3 | 3.1% | 1 | 3.3 |
| 3.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 3.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -0.2 | -0.2% | | |

## shape 4: batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 32 page=1 spread=agentic  (w 46, ours 16.4 µs, sol 3.3)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 4.0 | `_attn_kernel` | attention | 11.7 | 71.2% | 1 | 11.7 |
| 4.1 | `_reduce_kernel` | attention_reduce | 2.5 | 15.2% | 1 | 2.5 |
| 4.– | (hidden tail) | 0 kernels not listed by the judge report | 0.1 | 0.6% | | |
| 4.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 2.1 | 13.0% | | |

## shape 5: batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 32 page=1 spread=agentic  (w 16, ours 195.0 µs, sol 126.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 5.0 | `_full_cat_kernel` | attention | 193.3 | 99.1% | 1 | 193.3 |
| 5.1 | `_reduce_kernel` | attention_reduce | 3.4 | 1.7% | 1 | 3.4 |
| 5.– | (hidden tail) | 0 kernels not listed by the judge report | -0.0 | -0.0% | | |
| 5.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | -1.7 | -0.9% | | |

## shape 6: batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 1 page=1 q_len=8  (w 46, ours 9.5 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 6.0 | `_attn_kernel` | attention | 5.0 | 52.7% | 1 | 5.0 |
| 6.1 | `_reduce_kernel` | attention_reduce | 2.5 | 26.4% | 1 | 2.5 |
| 6.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 6.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 2.0 | 20.9% | | |

## shape 7: batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 1 page=1 q_len=8  (w 16, ours 21.5 µs, sol 4.3)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 7.0 | `_full_kernel` | attention | 16.1 | 74.9% | 1 | 16.1 |
| 7.1 | `_reduce_kernel` | attention_reduce | 3.6 | 16.8% | 1 | 3.6 |
| 7.– | (hidden tail) | 0 kernels not listed by the judge report | -0.1 | -0.5% | | |
| 7.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.9 | 8.8% | | |

## shape 8: batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 16 page=1 q_len=8 spread=agentic  (w 46, ours 17.3 µs, sol 2.5)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 8.0 | `_attn_kernel` | attention | 11.0 | 63.6% | 1 | 11.0 |
| 8.1 | `_reduce_kernel` | attention_reduce | 5.5 | 31.8% | 1 | 5.5 |
| 8.– | (hidden tail) | 0 kernels not listed by the judge report | -0.1 | -0.6% | | |
| 8.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.9 | 5.2% | | |

## shape 9: batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 16 page=1 q_len=8 spread=agentic  (w 16, ours 192.1 µs, sol 63.4)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 9.0 | `_full_kernel` | attention | 184.4 | 96.0% | 1 | 184.4 |
| 9.1 | `_reduce_kernel` | attention_reduce | 6.0 | 3.1% | 1 | 6.0 |
| 9.– | (hidden tail) | 0 kernels not listed by the judge report | 0.0 | 0.0% | | |
| 9.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 1.7 | 0.9% | | |

## shape 10: batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 32 page=1 q_len=8 spread=agentic  (w 46, ours 25.4 µs, sol 3.7)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 10.0 | `_attn_kernel` | attention | 15.4 | 60.7% | 1 | 15.4 |
| 10.1 | `_reduce_kernel` | attention_reduce | 7.8 | 30.7% | 1 | 7.8 |
| 10.– | (hidden tail) | 0 kernels not listed by the judge report | 0.1 | 0.4% | | |
| 10.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 2.1 | 8.2% | | |

## shape 11: batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 32 page=1 q_len=8 spread=agentic  (w 16, ours 368.3 µs, sol 126.8)

| k# | kernel | family | GPU µs | share of wall | calls | µs/call |
|---|---|---|---|---|---|---|
| 11.0 | `_full_kernel` | attention | 358.3 | 97.3% | 1 | 358.3 |
| 11.1 | `_reduce_kernel` | attention_reduce | 9.0 | 2.4% | 1 | 9.0 |
| 11.– | (hidden tail) | 0 kernels not listed by the judge report | 0.1 | 0.0% | | |
| 11.– | (idle / gaps) | wall − Σkernels: launch gaps, bubbles, graph overhead | 0.9 | 0.3% | | |

## families, weighted by each shape's loss share (where the board's headroom sits)

| family | weighted loss share |
|---|---|
| attention | 75.1% |
| attention_reduce | 24.9% |
