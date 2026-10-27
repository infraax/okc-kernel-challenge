# The Hotz playbook for okc (synthesis of 3 research agents, 2026-10-25)
Sources (full analyses next to this file): `streams-ANALYSIS.md` (32 stream transcripts, 21 episodes with timestamps, 38 practices),
`tinygrad-ANALYSIS.md` (tinygrad @ d16046f: architecture, DEBUG ladder, BEAM, 12 verified perf-bug fixes, okcdbg design),
`philosophy-ANALYSIS.md` (40 blog posts, Lex/Latent Space transcripts, 111 HN comments, 34 PR reviews, 36 rules). Quotes and links
are in those files; this page is the director's synthesis mapped to our kernels. Where his method has no evidence for Hopper-specific
kernel work (wgmma/TMA/W4A16/decode attention), the streams agent says so — the method transfers, his kernels do not.

## 1. The loop (ten lines)
1. **Never optimise without three numbers:** reference (SGLang), ceiling (bytes ÷ bandwidth, FLOPs ÷ peak), ours. "Where am I between the two" decides the move.
2. **Is the time inside the kernels or around them?** (host/launch/gaps vs scheduler/fusion vs kernel). Different fixes, different bounds.
3. **Count kernels and bytes.** A higher kernel count or bytes-per-step than expected is a speed bug by itself (150 vs 61 kernels; 83,525 vs 3,412 MB/token in a tinygrad MoE PR).
4. **Read the generated code** (PTX/SASS, register/smem usage, instruction histograms) — uncoalesced loads, gated loads + add-zero, extra converts.
5. **Bisect by deleting:** disable global loads, only-A, only-B, no-compute "fake" kernel; each cut says which part owns the time. Watch for the compiler deleting what you removed.
6. **Copy from a faster implementation one trick at a time** (he disassembled Apple's kernel: 3.7 → 8.1 TFLOPS).
7. **Say the expected number before running**; a surprise is information. "If you aren't within a couple percent of the memory bandwidth there's something wrong."
8. **Correctness in tiny pieces first** (zeros, a single 1, ones, random; per-kernel diff vs reference), then speed.
9. **Stuck → minimal repro at the lowest level, ask the hardware a yes/no question; "if we're brute forcing, we've lost"** → change tools.
10. **Prefer deleting code; pin every win with a test** (kernel-count / bytes / tok-s ratchet) and diff generated code on refactors.

## 2. Where we already do this, and where we don't
| Hotz move | okc today | gap |
|---|---|---|
| ceiling first | bytes_model.csv (Grok Build), sol from judge | not printed per kernel per run; no % of bound in our logs |
| inside vs around | L1 trace: gaps 0.2–1.9%; decode_layer small shapes 14% waiting on the side stream | fix = router/pooler co-residency (Fable G1) |
| count kernels/bytes | trace counts kernels; no bytes-per-step counter, no ratchet tests | **build** (okcdbg L1/L3 + tests) |
| read generated code | never looked at PTX/SASS, regs, spills for our CUDA kernels on sm_90 | **build** (L7: -Xptxas -v, instruction histogram) |
| bisect by deleting | Fable's NOCOMPUTE "pure stream" twin (C20) is exactly this | run it; add only-weights / only-activations / no-fixup twins |
| copy from the faster one | Marlin reaches ~2.5 TB/s with a 4-stage cp.async pipeline; ours 1.6–2.1 | Fable's ring = taking Marlin's trick into our kernel |
| say the number first | okc_run requires --predict, verdicts logged | keep; add per-kernel predictions |
| tiny correctness pieces | tests vs reference, NVRTC on the 2060 | add per-kernel CPU/torch diff mode (VALIDATE) in measurement files |
| record what failed | experiments/log.jsonl, AUTONOMY §2 | keep |

## 3. "Computed twice / read twice" candidates in OUR stack (from the measurements; each needs one check to confirm)
1. **bf16 router GEMV at m=128/256:** BM=64 → 2–4 M-blocks each stream the full 8+8 MB hi/lo weights (Argon: 32 MB traffic vs 2.7 µs floor on #29). Read once with BM=m or a persistent M loop.
2. **MoE largem chunking:** 27 chunks for 24 experts at bs256 (hot experts re-read per 32-token chunk), +12% bytes.
3. **decode_layer glue that only moves data:** fill + memcpy + F.pad around the unpooler, cuBLAS split-K partial write + reduce read (×2 per step), eager layer_norm writing an intermediate that GEMM1 re-reads.
4. **MoE epilogue at small M:** one CTA does two passes over 8×4096 (sum-of-squares, then write).
5. **attention verify:** RoPE recomputed per split (Argon), 128 exp2 + 128 FP8→FP16 converts per key (Fable: the real cap).
6. **Structural:** pooler occupies the GPU so the router (independent work) runs 4× slower — "work that serves other work".

## 4. okcdbg — our DEBUG ladder (spec in tinygrad-ANALYSIS §5; build order)
- **L1 run** (exists as the trace): kernels, busy%, gaps, floor, x_floor per shape.
- **L2 kern** (next): per kernel the tinygrad DEBUG=2 line — µs, GB/s, % of achievable BW (calibrated by a copy kernel in prepare()), TFLOPS, floor µs, x_floor; shape-encoded kernel names (`moe_g1_T32_N8192_K4096_e24`).
- **L3 bytes**: declared unique vs traffic bytes per tensor (passes), argument table (stride, contiguity, ptr%16).
- **L5 dups**: D1 identical launches, D2 value numbering, D3 passes > expected (catches §3 items 1–2), D4 copies/memsets/dead writes.
- **L7 code**: regs/spills/smem per kernel, PTX/SASS histogram (ld.global, cp.async, mma, cvt, ex2), source hash; "process replay" = diff of hashes between board and candidate.
- **Ratchets in tests/**: kernel count per shape, bytes per step, and the board µs per shape (fail if a candidate is worse) — tinygrad's `check_schedule(x, n)` idea.
- **BEAM harness inside prepare()** (search space = our configs, L2 flush, min of 3, early stop) — already our measurement-file practice; make it a library.

## 5. What not to take
His lazy graph compiler, rewrite-rule fusion and own driver stack (no time, wrong layer for a 10-day kernel race); his schedule
optimism (critics are right about it); brute-force sweeps when the design is the limit (our largem sweep ≤ 2% is exactly that).
