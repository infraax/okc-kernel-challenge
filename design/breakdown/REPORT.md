# okc time breakdown — REPORT (2026-10-24 ~20:00Z, Dex: "every step indexed, counted, timed; whole fleet")

## 1. What was measured, how, by whom
| level | what | source | who |
|---|---|---|---|
| L0 | per shape × per kernel table from the judge's own profile JSONs (top-12 kernels, hidden tail) | `tools/breakdown.py` → `design/breakdown/<p>.md`, `kernels.csv`, `INDEX.md` | director |
| L1 | **exact kernel timeline on the judge H200**: every kernel/memset/memcpy of every judged shape, start, duration, gap, stream; 20 profiled graph replays, cold L2, judge's own case seeds | `tools/make_trace.py` + `trace_runtime.py` (Sonnet C17) → 4 checks (jobs c2820491, 9b13f66a, dc147ca3, be92d308); `tools/parse_trace.py` → `trace/<p>.md`; `tools/trace_report.py` (Jules C18) → `trace/timeline_<p>.html` Gantt, `trace/SUMMARY.md`, `trace/items.csv` | Sonnet, Jules, director |
| bytes | minimum HBM bytes + FLOPs per shape × kernel family → floor µs and efficiency | `bytes_model.csv` (180 rows), `grokb-bytes.md` | Grok Build |
| why | per-problem bottleneck mechanisms, anomalies, level-2 stamp plan | `argon-deep.md` | Argon |
| check | recomputed 12 random rows from raw JSON (all match), 5 definition issues + 7 trace-method issues | ledger <ledger-msg> | Grok Bot |
| digest | per-problem digests | **failed**: NVIDIA endpoint returned HTTP 504 / empty for every model today | interns |

**Validation.** The judge subtracts its graph-launch overhead (~4.1 µs/shape) and its scored median equals our L1 *span*
(first kernel start → last kernel end) within 0.1–0.9 µs (bf16 #1: 6.44 vs 6.35; #2: 18.91 vs 18.53). So: head/tail launch
time is NOT scored; **scored time = kernel durations + the gaps between them.** CUPTI sees kernels inside graph replay on the
judge (cupti=graph on all 4). 323 kernel records, all parser invariants pass.

Red-team fixes applied (Grok Bot): family weighting no longer double-counts weight (I5), family names fixed for truncated
names and our kernels (I5), sol floored at 2.5 µs so 0.1 µs attention floors don't dominate (I4), p10–p90 overhead-corrected
(I1), trace/measure files hard-refused by okc_run.py and promote.py submit paths (T6). Open: I2 — attention's board speedup
(1.689) is not the weighted geo-mean of its shapes (1.816); the exact board formula is unknown, so rank attention by gains.

## 2. Where the scored time goes (L1, weighted by shape weight)
| problem | shapes | launches (avg/shape) | gaps | glue kernels (main stream) | dominant family |
|---|---|---|---|---|---|
| muse_moe_w4a16 (1.436x) | 6 | 24 (4.0) | 0.2% | 1.9% | **cuda largem GEMMs** 55% of loss, smallm 39% |
| muse_decode_layer (1.226x) | 12 | 232 (19.3) | 1.9% (5.7% at decode-1) | 6.2% (~12% at decode-1) | **Marlin MoE** 44% of Σdur, attention 13%, cuBLAS 13% |
| muse_attention_decode (1.689x) | 12 | 24 (2.0) | 1.0% | 0 | full-layer kernels 75% (`_full_kernel` verify 16/32 = 181 + 355 µs) |
| muse_bf16_proj (1.156x) | 30 | 43 (1.4) | 1.2% | 1.0% (+3.7% cuBLAS splitK-reduce) | cuBLAS 63% / our Triton GEMV 36%, at 69–71% of the bytes floor |

Gaps between kernels are small everywhere (0.3–0.8 µs each); **the time is inside kernels**, except decode_layer's small
shapes where 20 launches × ~0.45 µs gaps + glue kernels cost ~15 µs of ~125 µs.

## 3. Top bottlenecks across all problems (weight × µs above the bytes floor)
| # | problem / shape | kernel | measured | floor | note |
|---|---|---|---|---|---|
| 1 | decode_layer #10/#8/#11/#9 (verify 32/16) | **Marlin MoE ×2** | 510 / 345 / 636 / 351 µs | 130–151 | layer still on Marlin above 16 tokens; MoE problem's new largem path is 1.25–1.5x faster than Marlin |
| 2 | moe #5 / #4 (bs 256 / 128) | **w4_largem GEMM1 + GEMM2** | 322 + 162 µs / ~300 µs | 109 / 101 (sol), 173 / 162 (bytes) | ~1.4 TB/s ≈ 35% of HBM; hot experts (77 tokens) split into 32-token chunks → weights re-read per chunk |
| 3 | attention #11 / #9 (verify 32/16 full) | `_full_kernel` | 355 / 181 µs | 152 / ~76 | same KV as decode's `_full_cat_kernel` (187 / 100 µs) but 1.85x slower (Argon: occupancy, smem staging) |
| 4 | decode_layer #1/#3/#5 (full decode) | v2 `_attn_kernel` in the layer | 15.8 / ~158 / ~319 µs | | **the layer never got attention v3's `_full_cat_kernel`** (100 / 187 µs on the same work) |
| 5 | moe #0–#3 (bs 1–16) | w4_smallm GEMM1 + GEMM2 | 17+28 µs (decode-1 in the layer) | 11.4 sol | ~1.7 TB/s; GEMM2 > GEMM1 at M=1 is suspicious (Argon: z-loop wave + split-K fixup) |
| 6 | decode_layer all small shapes | glue: layer_norm 6.3, 2× splitK-reduce 1.7, fill+memcpy 1.7, route 1.5, + 20 gaps | ~15–18 µs per shape | 0 | fusion candidates (pre-expert norm into pooler, drop F.pad/memcpy, Triton `o`/`unpooler` without splitK) |
| 7 | bf16 pooler/unpooler m=128/256 | cuBLAS nvjet | 24–29 µs | 17 | wave quantisation (128 CTAs on 132 SMs) |

## 4. What this says to do next (ranked by expected board gain per check)
1. **C19 decode_layer v5** = v4 + attention v3's `_full_cat_kernel` (full decode shapes) + MoE v4's largem path (>16 tokens).
   Expected: #5 −130 µs, #3 −55 µs, verify 16/32 −15..25%. Both parts already judged in their own problems. Predicted 1.23 → ~1.35x.
2. **C20 largem efficiency (level 2)**: `%globaltimer` stamps in `w4_largem_kernel` (load / dequant / mma / split-K fixup) +
   chunking that keeps a hot expert's weights in one CTA group (no re-read per 32-token chunk). Target 1.4 → ≥ 2.5 TB/s.
3. **C21 decode_layer glue fusion**: pre-expert norm into the pooler GEMV, Triton `o`/`unpooler` without splitK-reduce, drop
   fill/memcpy. ~10–15 µs on every small shape (≈ +5–8% on 6 of 12 shapes).
4. attention verify-full `_full_kernel` → reuse `_full_cat_kernel` structure (parked lane; ask Dex).
5. bf16: Argon's `_CFG` entries for unpooler m=1/8 and counter-based split-K (−2..4 µs on 8 shapes).

## 5. Files
`design/breakdown/INDEX.md`, `<p>.md` (L0) · `trace/<p>.md` (L1 per shape, indexed tables, largest gaps) ·
`trace/timeline_<p>.html` (Gantt, open in a browser) · `trace/SUMMARY.md`, `trace/items.csv` · `bytes_model.csv` ·
`argon-deep.md` · `grokb-bytes.md` · `TRACE-FORMAT.md`. Regenerate: `tools/breakdown.py`; `tools/make_trace.py <p> <file>` →
`okc check` → `tools/parse_trace.py` + `tools/trace_report.py`.
