# C26' result: attention verify-full cost split (judge check job 9206dc7a1a3340bc, 2026-10-26 08:16Z)
File: kernels/muse_attention_decode/measure_verify_v1.py (fd8d7d8). Check 1.699x [1.673, 1.724] = board (kernel_fn = sonnet_v3).
Raw log: c26-measure-verify-v1.9206dc7a1a3340bc.log (S/C/B lines; 3 seeds × 5 reps, overhead 4.61 µs subtracted; 31 s of 240 s).

| variant | #9 verify b16 µs | #11 verify b32 µs | #7 verify b1 µs | reading |
|---|---|---|---|---|
| base (= board) | 189.1 | 364.1 | 20.9 | |
| GATHER (load floor, no cp.async) | 100.9 | 193.9 | 14.8 | loads alone ≈ 53 % of base |
| MMAONLY (loads + dots) | 120.2 | 256.0 | 15.9 | |
| NOEXP | 175.4 (−7 %) | 338.3 (−7 %) | 20.1 | exp2 is NOT the bottleneck |
| NOCVT | 181.7 (−4 %) | 361.5 (−1 %) | 18.2 (−13 %) | convert small except b1 |
| EXPSPLIT25 / 50 | 211.2 / 227.3 | 416.8 / 442.3 | 21.7 / 22.5 | **slower 12–21 %: killed** |
| **BN128** (stages 3, 255 regs, 96 KB smem) | **163.5 (x1.16)** | **322.2 (x1.13)** | 24.8 (x0.84) | **win at b ≥ 16; seeds agree ±1 µs; err 7.6e-5** |
| PROGx2 / PROGd2 | 194.2 / 357.1 | 367.8 / 705.7 | 20.9 / 23.6 | grid is right |
| ST3 / ST2 | 192.7 / 223.6 | 373.8 / 435.4 | 21.2 / 23.0 | 4 stages is right |
| W8 | 248.7 | 486.0 | 24.8 | worse |
| reducer (RED) | 6.5 | 9.0 | 4.6 | 2–3 % (22 % at b1) |

## Verdict
1. The MUFU/exp2 hypothesis (fable-08 D5, builder idea 2) is **refuted**: exp is 7 %, convert 1–4 % on the big shapes. The FMA polynomial
   costs more than it saves (degree-6 poly + registers). Do not retry exp-split without a new reading.
2. The real cost: 47 % of base is "compute other than loads" that does not overlap with the loads, but exp+cvt explain only ~11 %.
   The rest is online-softmax bookkeeping (row max, rescale, masking), the V smem round trip and barriers, done in series per tile.
   Larger tiles (BN128) amortise that per-tile overhead → +13–16 %. Structural fix = overlap softmax of one tile/warpgroup with the
   loads + MMA of the next (FA3 ping-pong / producer warps) — the same theme as the MoE warp-specialisation brief.
3. Next build (cheap, low risk): sonnet_v4 = sonnet_v3 with BN128 for verify-full batch ≥ 16 (keep BN64 for batch 1); then sweep
   BN128 neighbours (num_warps, stages 2..4) and check whether decode-full (_full_cat_kernel) also gains from a bigger BN.
   Board estimate: the two weakest shapes (1.07x / 1.06x) → ~1.21x / ~1.19x; problem 1.689 → ~1.72–1.76 (estimate).
