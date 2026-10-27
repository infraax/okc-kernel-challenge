# Results: every board submission, in order
Speedup = the judge's paired weighted-geomean speedup vs SGLang (higher is better). Job ids refer to `data/judge-runs/log.jsonl` and
`data/judge-runs/profiles/<problem>.<job>.{txt,json}`. 86 judge runs in total: 22 scored submissions, 64 checks (most of them
measurement-only experiments).

| time (UTC) | problem | file | speedup | job | what changed |
|---|---|---|---|---|---|
| 10-24 12:37 | bf16_proj | director_v1b.py | 1.156 | c3b7c70b5edd43c5 | Triton split-K GEMV for the router (BF16 hi+lo) and pooler/o at m ≤ 32, cuBLAS elsewhere; configs from judge sweeps |
| 10-24 13:13 | attention | sonnet_v1.py | 1.072 | 668f4376a75245b5 | fused split-KV flash decoding in Triton: 2 launches instead of SGLang's ~6 |
| 10-24 13:20 | decode_layer | sonnet_v1.py | 1.067 | 1e672a4ec9184ebd | SGLang's layer with three judged kernels spliced in |
| 10-24 13:37 | attention | sonnet_v2.py | 1.484 | f6f326822293406f | v1 rebuilt from the judge's profile and an sm_90a SASS read of v1 |
| 10-24 14:22 | moe | sonnet_v1.py | 1.152 | 786d9deec42c4f89 | SGLang's Marlin expert path + one fused Triton epilogue at M < 32 |
| 10-24 14:42 | decode_layer | sonnet_v2.py | 1.177 | 24563c15645245d3 | three more judged pieces (our attention, router/pooler GEMVs) |
| 10-24 15:21 | attention | sonnet_v3.py | 1.689 | 4b047c1bf5354162 | full-attention layers: KV-head pairs as one 16-row wgmma problem |
| 10-24 16:27 | moe | sonnet_v2b.py | 1.175 | d6de8485a100465d | per-M config table for Marlin's knobs + epilogue, from a judge sweep |
| 10-24 17:28 | decode_layer | sonnet_v3_s1.py | 1.196 | 2cb883c4c64c45a2 | judge-confirmed MoE knobs in the layer |
| 10-24 17:34 | decode_layer | sonnet_v3_s13.py | 1.193 | a1dca6eae5f9461e | + PDL on Triton launches (noise → later closed) |
| 10-24 17:43 | moe | sonnet_v3.py | 1.241 | a248091002d140e6 | custom CUDA small-M streaming split-K W4A16 path (M ≤ 16) |
| 10-24 17:51 | moe | sonnet_v3b.py | 1.255 | bda0751de2ad420a | tuned small-M config table |
| 10-24 19:32 | moe | sonnet_v4.py | 1.436 | 76d9f4196733442d | custom CUDA large-M path (17–256 tokens), per-bucket fallback to Marlin |
| 10-24 19:32 | decode_layer | sonnet_v4.py | 1.226 | 0108f70b516d4ea7 | MoE small-M CUDA path in the layer |
| 10-25 14:28 | decode_layer | sonnet_v5.py | 1.271 | cafdb9a48bb0400e | attention v3 full-layer kernel in the layer |
| 10-25 18:15 | decode_layer | sonnet_v5b.py | 1.387 | e9464cee59d849eb | router fixup kernel at T=1, pooler smem halved (co-residency), large-M experts |
| 10-25 18:59 | moe | sonnet_v5.py | **1.486** | 99f12b5d253b4707 | unified kernel, 3 CTAs/SM, stage-once activations (bitwise = v4) |
| 10-26 05:52 | decode_layer | sonnet_v5c.py | 1.423 | 3b00183396934c33 | early pre-norm, unpooler GEMV T≤8, epilogue grid |
| 10-26 05:58 | bf16_proj | director_v2.py | **1.170** | 7506407220b54361 | judged config winners per cell |
| 10-26 08:54 | attention | sonnet_v5.py | **1.770** | 1f0a0c8f29bb4a7b | BN 128 tiles for verify-full at batch ≥ 16 (found by a measurement run) |
| 10-26 15:37 | decode_layer | sonnet_v5f.py | **1.433** | efc7af96cfa347ca | attention v5 verify-full path in the layer |
| 10-26 17:59 | decode_layer | sonnet_v5g.py | 1.431 | 12f46f38e9ef48fb | glue fusion + prefetch riders (noise; board keeps 1.433) |

## Leaderboard context (2026-10-27 07:30 UTC)
| problem | ours | rank | #10 | #1 |
|---|---|---|---|---|
| attention | 1.770 | 313 | 2.624 | 3.032 |
| decode_layer | 1.433 | 427 | 2.226 | 2.380 |
| moe | 1.486 | 525 | 2.453 | 2.620 |
| bf16_proj | 1.170 | 799 | 1.563 | 1.637 |
| overall | — | 437 | | |
Rank trajectory overall: 374 (10-26 06:38) → 396 (10-26 18:00) → 421 (10-27 00:08) → 437 (10-27 07:30).

## Where the time goes (decode_layer, batch 1, last board entry)
101–108 µs vs a 40 µs memory floor; 16 kernels: our own kernels 79 µs, library kernels (cuBLAS projections, SGLang norm/routing) 27 µs.
Per-kernel tables: `design/breakdown/muse_decode_layer.md`, `tools/chain.py data/judge-runs/profiles/muse_decode_layer.efc7af96cfa347ca.json`.
