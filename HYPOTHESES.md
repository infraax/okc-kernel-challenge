# Hypotheses, tests and verdicts
Every idea that reached the judge, with the reading that decided it. "Twin" = a copy of the kernel with one cost removed (no exp,
no convert, no compute, …), timed side by side inside one measurement-only judge run. Result write-ups: `design/breakdown/`.

## Confirmed (shipped)
| idea | test | reading | result |
|---|---|---|---|
| Fused split-KV attention in 2 launches instead of SGLang's ~6 | submit | 1.072× → after a profile-driven rebuild 1.484× | shipped (attention v1/v2) |
| Full-attention layers: two KV heads as one 16-row wgmma problem | submit | 1.484 → 1.689× | shipped (attention v3) |
| **Bigger attention tiles (BN 128) for verify at batch ≥ 16** | measurement C26' (job 9206dc7a…), sweep C27 (2d42052a…) | ×1.16 / ×1.13 on the two weakest shapes, 3 seeds ±1 µs; BN128/3 stages/4 warps is the local optimum | attention 1.689 → **1.770** |
| Custom CUDA NVFP4 W4A16 expert GEMMs (small-M streaming split-K, large-M) | submits | MoE 1.175 → 1.255 → 1.436 | shipped |
| Unified expert kernel, 3 CTAs/SM, stage activations once per CTA | measurement C20 (a78c1b9c…) | occupancy alone +1–3 %, stage-once +8–13 % | MoE **1.486** |
| Router/pooler run side by side only if their shared memory lets them co-reside | measurement C22a' | the pooler's 147 KB/CTA blocked the router; 74 KB config fixed it at T ≤ 8 | decode_layer 1.271 → 1.387 |
| Port the attention verify path into the full layer | check 62a4b559… | verify-16 full 579 → 534 µs, verify-32 883 → 847 µs | decode_layer **1.433** |

## Refuted or parked (with the killing reading)
| idea | test | reading | verdict |
|---|---|---|---|
| Attention verify is bound by exp2 (MUFU) + FP8 convert | twins NOEXP / NOCVT (9206dc7a…) | exp 7 %, convert 1–4 % of the kernel | refuted |
| Split exp2 between MUFU and an FMA polynomial (FA4 style) | twins EXPSPLIT25/50 | −12 % / −21 % | killed |
| Deeper prefetch (per-warp cp.async ring) in the MoE GEMM | measurement C21b (698aa839…) | 2.7× bytes in flight → +11 % per CTA but residency 3 → 2 CTAs/SM: net −22 % | killed |
| **Warp-specialised MoE GEMM** (1 bulk-copy producer warp, 12 mma consumers, mbarrier ring) — designed by an AI reviewer, predicted ×1.3 | 3 measurement runs | C30b: copy-issue bound (80 copies/stage) → 1.7–3× slower; C30d after fix (2 copies/stage): 0.82× of the old kernel, consumers capped ≈ 2 TB/s; C31 bisect: the ring alone (no math) ≈ old kernel's rate | parked after 3 readings |
| PDL (early launch) on Triton kernels with the wait first | submit v3_s13 | noise | closed |
| "Tools before the wait": light L2 weight prefetch before the PDL wait | measurement C33 (fcbc85a4…) | +2.0 µs at T = 1 (light), −3.9 µs (full prefetch → bandwidth congestion), stream priority 0 | below the 3 µs bar; congestion effect confirmed |
| T = 1 quick wins proposed by a secondary AI (grid width, late trigger, smaller staging, route on side stream) | claims checked in code first, then C37 (f31c9839…) | one claim refuted in code; measured: −1.4, −0.5, −6.0, +1.1 µs | no winner |
| WSTREAM-style tensor-core GEMV for BF16 projections (16 B loads, many in flight) | measurement C39 (e2a5099d…), 20 cells | 0.2–2.5 TB/s vs 2.6–3.7 TB/s for the existing kernels | killed (prediction missed by ~2×) |
| Fuse the post-FFN norm into the unpooler kernel (grid barrier) | promote check v5g | merged kernel 10.0 µs = the two kernels it replaced | ≈ 0 |
| Same fusion without the barrier (last-CTA scalars + small apply kernel) | check v5h (ce45f6ae…) | 1.425 vs board 1.433; fused kernel 10.7 µs | parked (2 strikes) |
| Triton GEMV configs for qkv_gate | measurement sweeps (26664259…, eb563a62…), ~160 configs | ≤ 3 % vs cuBLAS | closed |
| TMA-descriptor GEMV, one-shot BK 1024 GEMV, FP8 operands, single-launch megakernel, fused routing kernel | earlier checks / rules | slower, illegal or not buildable in the window | closed |

## Predictions vs readings (calibration)
| source | prediction | reading |
|---|---|---|
| AI design reviewer, MoE v5 | 1.72× | 1.486× |
| AI design reviewer, MoE ring v6 | 1.70× | 1.213× (check) |
| AI design reviewer, warp-specialised v7 | GEMM1 234 → 160 µs | 267–285 µs (slower) |
| builder agents using the judge's own load formula on measured medians | v5 1.266, v5b 1.39, v5c 1.44, attention v5 1.75 | 1.263, 1.386, 1.429, 1.770 |
| builder agent, tensor-core GEMV | ~4 TB/s | 0.2–2.5 TB/s |
Lesson: predictions anchored in the judge's own measured medians were accurate; first-principles Hopper performance models were not.
