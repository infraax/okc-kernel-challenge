# fable-07-okc-gaps output
executor: Fable (one-shot) · 2026-10-24 · inputs: the 17-file pack + public web (URLs cited) · design only, nothing run.
Assumptions are tagged [An] and listed in D7 with the cheapest test. Numbers without a tag come from the pack (02, 03-profiles, 13).

## 0. Headline (read this if nothing else)
1. **MoE is the lever on two boards.** Marlin W4A16 moves 0.8–2.0 TB/s of a 4.8 TB/s H200 and sits at ~15% of tensor peak on
   verify-32; it is 82–97% of moe_w4a16 time and 25–77% of decode_layer time. A Hopper-shaped grouped W4A16 GEMM (large-M)
   plus a split-K streaming kernel (small-M) is worth moe 1.15 → ~2.0x and decode_layer 1.18 → ~1.6x on its own.
2. **The attention "2 TB/s ceiling" is refuted by our own run.** v3 full-decode moves 606 MB in 195 µs = 3.1 TB/s (b32) and
   2.8 TB/s (b16) of *useful* bytes. What is slow is (a) full **verify** (1.6 TB/s on the same bytes: it is MMA/exp/issue
   bound, not HBM bound) and (b) the **sliding** shapes (8–25 µs for ≤ 4 MB of KV = pure launch+reducer cost, weight 46 vs 16).
   A single-launch sliding kernel alone is 1.69 → ~1.90; verify-full warp specialisation adds → ~2.3. Leaders' 2.7 ≈ both + near-floor full decode.
3. **decode_layer**: no megakernel in 10 days; a 6-launch layer (fusion map D3) + PDL + D1's GEMMs → ~1.7–1.9x.
4. **Stop**: more programs/deeper cp.async pipelines on full decode; any Marlin-wrapper polishing; bf16 (parked, correct);
   Humming (not in the image); the literal routing hint; engram beyond the serving draft controller; sub-CI wins.
5. **Serving**: 220-token outputs on multi-turn coding = prefill (prefix cache) + 220 decode steps; at c=16/32 the verify
   batch *is* our verify shapes. Build order D5; the three daily checks are an A/B/noise triplet, never three new ideas.

## D1. moe_w4a16: beating Marlin W4A16 on H200
Facts from the pack. One local expert = w13 16.8 MB + w2 8.4 MB + scales 3.1 MB ≈ 28 MB (H = I = 4096, E2M1 + E4M3/16).
Floors → active experts per shape: decode-1 ≈ 2, decode-16 ≈ 9, decode-32 ≈ 12, verify-1 ≈ 6, verify-16 ≈ 17, verify-32 ≈ 19
(verify-32 is also near the math floor: 108.9 µs ≈ 108 GFLOP at 989 TFLOP/s → ~1,080 local rows [A11]).
Marlin's effective weight bandwidth from the profiles: decode-1 1.3 TB/s, decode-16 1.4, decode-32 2.0, verify-1 1.9,
verify-16 1.0, verify-32 0.76 TB/s. It gets *worse* with M. Why (public code, same lineage as SGLang's copy [A2]):
- vLLM/SGLang MoE-Marlin picks the thread config itself; `thread_k/thread_n` are hard-coded −1 in the entry point, so the only
  Python-reachable knobs are `moe_block_size ∈ {8,16,32,48,64}`, `use_atomic_add`, `use_fp32_reduce`, workspace size
  (https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/csrc/moe/marlin_moe_wna16/ops.cu;
  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_wna16_marlin.py).
- For `moe_block_size > 16` (thread_m_blocks > 1) it switches to the "large batch" config {thread_k 64, thread_n 256, 256
  threads} with `blocks_per_sm = 1`: a persistent grid of 132 CTAs × 8 warps doing dequant + `mma.sync` (same ops.cu). The
  baseline's heuristic gives block 8 (bs 1/8/16), 16 (bs 32), 48 (bs 128), 64 (bs 256): exactly the shapes where it collapses.
- Marlin is an Ampere design; on Hopper `mma.sync` alone forfeits ~37% of peak and Machete exists because "Marlin … struggles
  on Hopper" (https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel). Marlin's own
  claim is near-ideal speedup "up to batchsizes of 16-32" (https://github.com/IST-DASLab/marlin).
Dequant can be exact: E2M1 has ≤ 2 significant bits, E4M3 ≤ 4, so value × block-scale has ≤ 6 significant bits and fits
BF16 (8) exactly; the FP32 per-expert global scale goes in the FP32 epilogue [A3]. So a custom kernel reproduces Marlin's
numerics (BF16 operands, FP32 accumulate); the 1.5% L2 bound is not at risk.

| # | option | shapes it moves | expected (µs now → target; load speedup vs SGLang) | effort | risk | first measurement check |
|---|---|---|---|---|---|---|
| 1 | Marlin knob sweep in `prepare()`: block size per shape (incl. forcing 16 at bs 128/256 so 4 CTAs/SM run; weights then re-read per M-block, L2 50 MB absorbs ~1–2 experts), atomic-add vs fp32-reduce, workspace `max_blocks_per_sm` | verify-16/32, decode-32 | 476/691/170 → 380/520/150 (1.2/1.2/1.35x) | 0.5 d (reuse O2 sweep v3) | low | 1 check, all 6 shapes, per-variant table; kill if best < CI |
| 2 | **Triton grouped W4A16 GEMM, large-M path**: tiles (expert, 64/128-row M-block from `sorted_token_ids`, 128–256 N) flattened into one persistent grid with an atomic tile counter (stream-K style, https://arxiv.org/abs/2301.03598) so one hot expert does not serialise; in-kernel nibble-LUT dequant to BF16 × E4M3 scale, `tl.dot` BF16 (wgmma at M ≥ 64), FP32 acc; gate/up rows **interleaved in prepare()** (lossless) so SiLU·up is GEMM1's epilogue and `cache1` halves | verify-16/32, decode-32 (+decode_layer's same shapes) | 476/691/170 → 250/330/110 (1.85/1.9/1.8x) | 2–3 d Sonnet-A + Argon review | medium: Triton dequant throughput, compile time, skew | GEMM1-only vs Marlin GEMM1 at bs 128/256 over {BM, BN, BK, warps, stages} inside prepare(); kill if < 1.3x Marlin after 2 checks |
| 3 | **Small-M streaming split-K kernel** (CUDA `load_inline` or Triton): CTA = (expert, 128-row N slab, K-slice); contiguous 16-B loads (or `cp.async.bulk`, sm_90), LUT dequant, `mma.sync m16n8k16` BF16 with M padded to 16, FP32 partials, last-CTA fixup (your bf16 design); grid ≥ 2 CTAs/SM even for 2 experts | decode-1/16, verify-1 | 54.7/154/112 → 25/95/65 (3.2/2.0/2.2x) | 2 d Argon draft + Sonnet port | medium: judge-only debug | decode-1 GEMM1 only; predict 54 MB / 4 TB/s + 3 µs |
| 4 | Fuse GEMM1 → SiLU → GEMM2 in one persistent kernel | all, small | ≤ 67 MB of cache traffic at bs 256 (≈ 14 µs) + 1 launch | 2 d | medium | only after 2+3 |
| 5 | Routing-hint L2 prefetch | — | 0: ids are inputs and the GEMM is their first consumer | — | — | not applicable here |
| 6 | Humming / other GEMM libraries | — | not in the judge image (no network) | — | — | no |

Order: 1 (day 1, cheap, feeds the sweep harness), 2 (days 1–3), 3 (days 2–3), 4 later. Predicted moe overall: 1 alone
1.15 → ~1.25; +2 → ~1.65; +3 → ~2.2 (range 1.8–2.5; leader ~2.35). Every µs here also lands in decode_layer (D3-F1).
Design notes for 2/3: E2M1 → BF16 via a 16-entry LUT on the nibble (sign folded), scales loaded as [BN, BK/16] E4M3 and
broadcast; for `is_ep` keep the local-id kernel; skewed routing means the tile list is value-dependent in *count* only — launch
the grid for the worst case (all M-blocks of `sorted_token_ids`) and let empty tiles exit (shape-only grid, graph-safe).

## D2. attention_decode full layers (and the sliding layers that pay more)
Per-load decomposition of the current 1.689x (SGLang µs vs ours, weights 46 sliding : 16 full):
decode-1 1422/612 = 2.32x · decode-16 4445/2338 = 1.90 · decode-32 7551/3876 = 1.95 · verify-1 1607/781 = 2.06 ·
verify-16 4604/3870 = 1.19 · verify-32 7781/7060 = 1.10. Three observations:
- **Full decode is not at a 2 TB/s wall.** b32: floor 126.35 µs × 4.8 TB/s = 606 MB; ours 195 µs → 3.1 TB/s; b16 2.8 TB/s.
  Random 128-B lines ≈ 256 B per key [A1] already stream at 60–65% of peak. Headroom is 1.5x, not 3x. Little's law: 4.8 TB/s
  × ~0.7 µs DRAM latency (L2 far-miss ~620–740 cycles, https://arxiv.org/html/2501.12084v1) ≈ 3.4 MB in flight ≈ 26 KB/SM;
  v3's 4 programs × 2 pending tiles × 64 keys × 256 B = 128 KB/SM is already plenty → bytes-in-flight is *not* the limiter;
  DRAM row/bank locality and TLB reach on a ~0.65 GB random footprint are [A7]. Not TMA: `.tile::gather4` needs sm_100
  (https://github.com/triton-lang/triton/issues/8335); H200 has no gather TMA.
- **Full verify is compute/issue-bound.** Same 606 MB in 368 µs (1.65 TB/s). Per key and KV head: QK 64×64 + PV 64×64 MACs →
  84 GFLOP at b32 (85 µs at BF16 peak, ~64 µs with FP8 QK), **plus 328 M exp2 → 78 µs on MUFU** (16 ex2/clk/SM on Hopper,
  https://arxiv.org/abs/2603.05451) plus FP8→FP16 conversion of V (~40 µs of cvt). With 2 programs × 4 warps per SM nothing
  overlaps; FA3 gets 75–85% utilisation only with producer/consumer warpgroups and ping-pong (https://arxiv.org/abs/2407.08608).
- **Sliding shapes are fixed cost.** ≤ 4 MB of KV takes 8–25 µs: two launches, a 2.3–7.8 µs reducer, ~3 µs startup. With
  weight 46 they are 27% of decode-16 and 17% of verify-32 load time. A 2.7x leader must sit near 5–12 µs here (table below).

Leader anatomy (inferred [A10]): sliding 5/8/10 (decode b1/16/32), 6/9/12 (verify); full decode 10/80/160; full verify
10/110/210 → loads 3.65/2.70/2.50/3.69/2.12/1.99 → geo 2.69. Our path to ~2.3:
**Move A — single-launch sliding (cheapest, surest).** Fold the reducer into the split kernel: each program stores its
partial, `tl.atomic_add(counter[b,h,rowblock], 1, sem="acq_rel")`, the last arriver merges, applies v_scale/RMSNorm/gate and
writes BF16. Counters: `torch.zeros` in `kernel_fn` (a memset node; attention has no prepare()) [Q5]. Also use the window to
size splits (2048 keys → ~8 splits × 128 keys per (b,h)); drop EDGE masking work where the window covers the split.
Predict sliding 7.98/13.65/16.43/9.49/17.3/25.37 → 5.5/9/11/6.5/11/15 → **overall 1.69 → ~1.90**. One measurement check (split
counts, import-time sweep) + one full check. Verify-full's 15 µs reducer disappears the same way (−4% there).
**Move B — verify-full warp specialisation.** 8 warps per program = 2 warpgroups, each owning one KV head on the *same* K/V
tiles (no cross-warpgroup reductions: v3's 32-instr/key problem came from splitting one head across groups). Triton routes:
`num_warps=8` with two independent dot chains, or the automatic warp-specialisation path (Triton ≥ 3.2 on Hopper,
https://pytorch.org/blog/warp-specialization) [A5]. Relieve MUFU FA4-style: compute 25–50% of each row's exp2 with a
Cody-Waite + degree-3 polynomial on FMA units (https://arxiv.org/abs/2603.05451; it is exact to FP32 ulp-level, not a
precision cut). Keep P in FP16 unless the organisers say P may be FP8 [Q1]. Predict 192/368 → 140/270, b1 21.5 → 14 →
**A+B overall ~2.33**. 3–4 days, high risk; kill if no ≥ 15% gain on verify-full b16 after 3 checks (keep A).
**Move C — sort the gather by cache row (full decode, optional).** Attention is permutation-invariant over keys, so a split
may own any subset. Two small kernels: histogram the request's page ids into 256 address buckets, scatter ids bucket-wise
(~20 MB traffic, ~8 µs), then the attention kernel walks ids in address order → DRAM page and TLB locality. Gain only if cache
rows are dense in the allocated region [A7]: 3.1 → ~3.8 TB/s (b32 195 → 165, b16 107 → 92). Gate by shape (batch × ctx ≥ 1 M
keys). First test is a measurement-only gather kernel (no math): random order vs sorted order, same bytes: it yields the true
H200 random-row ceiling in one check and decides C. Every key is still attended; outputs bit-identical up to summation order.
**Move D (defer)**: `cp.async.bulk` 1-D (sm_90, 16-B multiples) lets one thread fetch a 128-B row → fewer issue slots per
key for the decode kernel; needs CUDA `load_inline`, so only after A–C.
Predictions per load after A+B (+C in brackets): decode-1 3.44x, decode-16 2.40 (2.6), decode-32 2.40 (2.6), verify-1 3.07,
verify-16 1.68, verify-32 1.55 → ~2.33 (~2.45). Stop: "more programs / deeper pipelines" for full decode (refuted by 3.1 TB/s).

## D3. decode_layer fusion map
Budget at decode-1 sliding (profile 0; 139 µs GPU, 135.5 judged, floor 40): Marlin 45.0 · `_gemv_splitk` 28.0 (pooler ~19 +
o/router) · cuBLAS nvjet 25.2 + splitK-reduce 3.4 (qkv_gate, unpooler) · layer_norm 6.5 · epilogue 5.9 · attention 5.2 +
reduce 2.3 · norm_residual 4.5 + 3.5 · routing 2.8 · ~6 small kernels (align, local ids, act_mul, count/sort, elementwise)
≈ 8 → ~20 launches. At verify-16 sliding Marlin is 436 of 570 µs; at full b16/b32 attention is 46–56%.
| step | fuse | launches | µs saved (decode-1 / verify-16) | legal note |
|---|---|---|---|---|
| F1 | D1 GEMMs (large-M + small-M) | 0 | −20 / −230 | same numerics as Marlin |
| F2 | router GEMV hi+lo → sigmoid+bias → exact top-8 → local ids → `moe_align` (+count/sort) in one kernel, grid = tokens | −4 | −5 / −6 | exact FP32 routing; sort is value-dependent in content, not shape |
| F3 | o-proj split-K GEMV + rmsnorm(a) + α1/β1 residual + rmsnorm(s)(1+w) in one last-CTA-fixup kernel; pooler GEMV with the router hi/lo as extra N columns (same input h), post-RMSNorm in its fixup | −4 | −7 / −8 | our bf16 fixup design; no new precision |
| F4 | tail: MoE epilogue (norm-weighted sum) → unpooler GEMV (T ≤ 8: 1024 cols, 8 MB) → rmsnorm(f(1+w)) → α2/β2 → final rmsnorm in one fixup kernel | −3 | −6 / −6 | row-complete epilogue needs all N tiles → fixup pattern |
| F5 | attention: v3 already fuses QK-norm/RoPE/FP8 write/gate; take D2-A single launch | −1 | −2.5 / −5 | — |
| F6 | PDL on every Triton launch: `launch_pdl=True`, `tl.extra.cuda.gdc_wait()` before the first dependent load, `gdc_launch_dependents()` after the main loop (Triton PR #6394; https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py). Works under graph capture via programmatic edges (https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) [A4]. Beware pipelined loads hoisted above `gdc_wait` (triton issue #11905): put the wait first, no loads before it | 0 | −8–12 / −10 | timing only, no numerics |
Keep cuBLAS for qkv_gate and unpooler at m ≥ 128 (your sweep showed it wins). Resulting layer = 6 launches: [attention+own
keys] → [o+norms+residual] → [router+pooler+align] → [GEMM1+SiLU] → [GEMM2] → [epilogue+unpooler+norms].
Prediction: decode-1 135 → 85–90 µs (load 1.26 → ~1.9x), verify-16 sliding 513 → ~300, full shapes follow D2 → overall
**1.18 → 1.7–1.9x** (leader 2.15). Order: F6 probe first (one check on the existing v2: measures the real gap per boundary,
zero numerics risk), F2, F1 as D1 lands, F3/F4, F5.
Megakernel for T ≤ 8: **no.** Precedents (Hazy "No Bubbles", 78% of H100 bandwidth for Llama-1B,
https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles; MPK, https://github.com/mirage-project/mirage) are
interpreter-style persistent CUDA kernels with in-kernel schedulers: weeks of work, and our only Hopper is the judge
(15-min runs, no profiler). Triton has no inter-CTA scheduling beyond atomics. The 6-launch layer + PDL captures most of
the bubble savings (launch gaps ≈ 1–2 µs each at ~20 boundaries [A4]) at a tenth of the risk.

## D4. Engram ideas and the routing hint, improved
1. **Draft cascade (serving, request-local, legal)** — sharpen the mapping in 10 §5:
   - Why it can be big here: outputs in multi-turn coding re-emit input spans (file edits, tool JSON, repeated code). Prompt
     lookup decoding reports ~2.4x on input-grounded tasks with no quality change (https://github.com/apoorvumang/prompt-lookup-decoding;
     SpecBench: PLD best where input/output overlap, https://sites.google.com/view/spec-bench). DFlash is a *block* drafter
     (8 tokens, one pass, https://proceedings.mlr.press/v306/chen26bv.html); n-gram drafts can be 16–32 long when the match is unique.
   - Cascade = S1 n-gram lookup on this request's prompt+output (suffix 3–5 tokens); S2 margin = longest *unique* match
     length L → draft length `k = min(32, L + 4)` if L ≥ 6, else DFlash's 8-block; S3 confusion pairs = after a rejection at
     position p, blacklist (context n-gram → continuation) within the request; S4 priors = offline calibration table
     P(accept ≥ j | L bucket, source) as a static JSON constant, used only to choose k. Nothing persists across requests.
   - SGLang already ships `--speculative-algorithm NGRAM` and `DFLASH` (https://docs.sglang.io/docs/advanced_features/speculative_decoding);
     adaptive draft length exists only for EAGLE/EAGLE3 (https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding),
     so the hybrid worker + request-local adaptivity is our code, python-only (allowed).
   - **Test first (CPU, free)**: `tools/ngram_sim.py` on public agentic-coding trajectories (SWE-agent/OpenHands dumps):
     % of output tokens copyable with L ≥ 6, accepted tokens/step for fixed-8 vs cascade; train/test split per engram's rule.
     Go if copyable ≥ 30% (PLAN A7) and cascade ≥ +15% accepted tokens/step at c=1 and c=16 batch sizes.
2. **DC-removal methodology → our prepare() sweeps.** Interleave variants round-robin with a fixed pivot variant every
   round; score = median over rounds of (pivot time / variant time) — the judge's own paired-ratio form; subtract the
   per-round pivot deviation (the "shared component") before ranking; bootstrap a CI over rounds and refuse to rank
   differences inside it. Jules can do this in a day; it protects every check budget below.
3. **Dismissed too fast**: (a) *block pool / larger pages*: not for the kernel track, but in serving the page size is a
   server argument; page_size > 1 turns the scored random-row gather into contiguous rows and speeds every attention
   kernel — worth asking whether it is allowed ("model unchanged" is satisfied) [Q3]. (b) *harness validation*: the mutant
   suite (skip-one-key, FP8-coarser Q, swapped top-8 tie, dropped split-K tail) stays worth its hour. (c) *inverse-RoPE
   exactness*: use as the bitwise test of the fused RoPE+FP8 write, nothing more.
4. **Routing hint → safe form, honest value.** Hit rate is not the issue; *bytes* are: at T ≥ 16 the routed experts weigh
   ≥ 250 MB ≫ 50 MB L2, so a prefetch covers ≤ 15% and attention at those shapes is itself bandwidth-heavy. Only decode-1
   full (attention 5–16 µs of idle HBM, 2 experts = 56 MB) can gain, ≤ 6–10 µs of 148 (≤ 7% on one shape). Better, guess-free
   variant with the same mechanism and 100% hit rate: during attention, prefetch the *un-routed* weights that run right after
   it (o-proj, router hi/lo, unpooler[:1024], first pooler slabs; ≤ 40 MB). Run T1 (free) for the record; spend no check on
   either before D1/D3 F1–F4 have landed.

## D5. Serving build order (Nov 5–19; 1 submission + 3 c=16 checks per day; 90-min node)
Framing from the rules (12): exactly 220 output tokens per request, hidden multi-turn coding at c = 1/4/16/32, score =
throughput at stock's interactivity. So a request = long prefill (prefix-cache hit or miss) + ~220/accepted-per-step
verify steps; c=16/32 verify batches are literally our `verify 16/32` shapes; c=1/4 are latency-bound (decode-1 shapes, PDL,
launch count). Kernel-track µs convert directly; the score simulator can be extended with a per-load step model.
- **S0 (Oct 30 → Nov 4, CPU only)**: `tools/diff_rules.py` (size, paths, no model_loader, line length); JIT packaging of our
  Triton kernels under `python/sglang/kernels/…` with import-time fallbacks to stock; a server-args default patch (DFlash on,
  block size, CUDA-graph max bs covering 32 × 9 tokens, chunked-prefill size, overlap scheduler, page size [Q3]) — first
  confirm which flags the organisers' launch command fixes [Q2]; a CPU smoke test that imports and shape-checks every path.
- **Nov 5 (S1)**: check 1 = stock + config defaults; check 2 = + sliding attention v4 + MoE epilogue + router GEMV as JIT
  (crow `serving-day-1`: A3 = "JIT works in the sandbox"); check 3 = + D1 GEMMs if landed. Submit the best green one.
- **Nov 6–8 (S2)**: one lever per day: DFlash block 8 vs 16 and draft window; CUDA-graph coverage for verify batches;
  prefill/decode overlap. Checks = {yesterday's best + lever, + lever at a second setting, re-run of the chosen} → the third
  check is the noise floor; submit only outside it.
- **Nov 9–12 (S3)**: n-gram → DFlash cascade (gated by D4-1's simulator); prefix-cache miss attribution (diagnose first).
- **Nov 13–15 (S4)**: stabilise, drop unconfirmed levers (PLAN §8), freeze Nov 15; Nov 17 12:00 PT last code.
- Since checks measure only c=16, predict c=1/4 from kernel-track decode-1 data and never submit a change that the
  simulator says costs TTFT (2x cap) at c=32.

## D6. Three-day allocation (Oct 25–27, UTC days; per problem 10 submits + 10–20 checks)
| day | moe_w4a16 (critical path, ≥ 50%) | attention_decode | decode_layer | bf16 |
|---|---|---|---|---|
| Oct 25 | 7: Marlin knob sweep (1) · grouped GEMM v1 tile sweep (2) · small-M v1 (2) · integrate (1) · submit (1) | 4: sliding single-launch sweep (1) + full check (1) · gather-ceiling microbench random vs sorted (1) · submit (1) | 3: PDL probe on v2 (1) · F2 routing fusion (1) · submit if green (1) | 0 |
| Oct 26 | 6: grouped GEMM v2 + SiLU fuse + stream-K tile list (3) · small-M v2 (2) · submit (1) | 4: verify-full WS v4 (2) · sorted gather if microbench says ≥ 15% (1) · submit (1) | 4: F1 (D1 kernels in the layer) (2) · F3/F4 (1) · submit (1) | 0 |
| Oct 27 | 4: skew/polish (2) · submit (2) | 3: v4 polish (2) · submit (1) | 6: integration (4) · submit (2) | 1: PDL + fixup-kernel port if F3/F4 exist (free rank points) |
Lanes: **Sonnet-A** = D1-2 grouped GEMM (Triton; CPU interpreter + RTX 2060 for correctness, judge for time). **Sonnet-B** =
D2-A then D2-B. **Argon** (read-only drafts/reviews) = D1-3 CUDA streaming kernel draft, every submit's diff review, the
exp-polynomial snippet. **Jules** (python only) = pivot-ratio sweep harness (D4-2), score_sim for moe/decode_layer, ngram_sim,
diff_rules. **Grok Build** = F2 routing kernel and F4 tail kernel drafts. **Grok Bot** = red team per submit, T1 hit-rate
script, daily scout, leaderboard delta digest. Director: hypotheses + predictions at 00:00Z, submits, 21:00Z digest.
Rules of the three days: ≥ 70% of checks on the day's critical path; refuted twice → drop; every check carries a predicted
per-shape µs; bf16 stays parked unless its kernels come free from F3/F4.

## D7. Assumptions [An] with cheapest tests, and open questions
| An | assumption | if false | cheapest test |
|---|---|---|---|
| A1 | HKV = 2 per GPU (TP8), K and V rows 128 B each, ~237 B/key from the floor arithmetic (avg ctx ~74k) | byte and MUFU budgets in D2 shift ~±15% | print `config` in a check log |
| A2 | SGLang a55ab7d5's MoE-Marlin = vLLM's: thread config auto, 1 CTA/SM at block ≥ 32 | option 1 gains less; options 2–3 unaffected | the knob-sweep check itself |
| A3 | E2M1 × E4M3 → BF16 exact (≤ 6 significant bits) | add one FP32 multiply, no design change | local bitwise compare vs `nvfp4.dequantize` |
| A4 | image Triton ≥ 3.4 (PDL API) and graph capture keeps programmatic edges; gaps ≈ 1–2 µs | F6 = 0 | `print(triton.__version__)` + the PDL probe check |
| A5 | Triton emits two independent wgmma chains for two heads at 8 warps (no cross-warpgroup shuffles) | D2-B via CUDA or via auto warp specialisation | offline sm_90a compile, count SASS instr/key (the builder's method) |
| A6 | Hopper MUFU = 16 ex2/clk/SM → 78 µs exp floor at verify-full b32 | B's ceiling moves | measurement-only exp2 loop kernel |
| A7 | cache rows dense in the allocated region (sorted gather ≈ sequential) | Move C dead (keep A/B) | the gather microbench (1 check) |
| A8 | server-arg defaults in the diff are honoured by the organisers' launch | S0 config lever dead | ask (Q2) |
| A9 | ≥ 30% of coding-output tokens copyable with L ≥ 6 | cascade small; DFlash tuning only | ngram_sim on public trajectories |
| A10 | leader anatomy (sliding ≤ 12 µs, full verify ≥ 1.8x) is inferred, not observed | our moves still stand on our own data | none needed |
| A11 | verify-32 local rows ≈ 1,080 (compute ≈ bytes floor) | large-M target shifts, still ≥ 2x Marlin | count local routes in local `generate_input` runs |
| A12 | 3–4 Triton kernels × config sweep compile inside 300 s prepare / 15-min run | fewer variants per check | log compile time per variant in the first sweep |
Open questions (put to the organisers or decide): **Q1** may P (softmax probabilities) be FP8 in the PV GEMM as FA3-FP8 does?
(rules name only queries) — until answered keep FP16. **Q2** are serving launch flags fixed by the harness or taken from
`server_args` defaults in the diff? **Q3** is a page size > 1 allowed in serving (model unchanged, attention gather changes)?
**Q4** `torch.zeros` (memset node) inside `kernel_fn` for the fused-reduce counters: confirm it passes the static checker
(shape-only allocation). **Q5** which Triton/CUDA versions are in `lmsysorg/sglang:dev-muse-spark-open` (A4/A5)?
**Q6** does the judge count the serving check's c=16 run with or without warm prefix cache (changes S3's value)?
Sources used beyond the pack: H200 4.8 TB/s, 989 TFLOP/s BF16 dense, 132 SMs (https://www.nvidia.com/en-us/data-center/h200;
https://www.techpowerup.com/gpu-specs/h200-sxm-141-gb.c4255); FA3 (https://arxiv.org/abs/2407.08608); FA4 exp emulation
(https://arxiv.org/abs/2603.05451); Hopper latencies (https://arxiv.org/html/2501.12084v1); Stream-K (https://arxiv.org/abs/2301.03598);
Machete/Marlin (https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel, https://github.com/IST-DASLab/marlin);
Triton PDL/WS (https://github.com/triton-lang/triton/releases, https://pytorch.org/blog/warp-specialization); megakernels
(https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles, https://www.usenix.org/conference/osdi26/presentation/cheng);
SGLang speculative docs (https://docs.sglang.io/docs/advanced_features/speculative_decoding); PLD (https://github.com/apoorvumang/prompt-lookup-decoding).
