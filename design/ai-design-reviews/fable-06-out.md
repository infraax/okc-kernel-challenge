# fable-06-okc-data output
executor: Fable (one-shot) · 2026-10-24 · design only · assumptions marked [An], register + cheapest test in D6.
Pack facts are cited as (NN) = file number in §4 of the brief; external facts carry a URL.

## Executive answer (read this if nothing else)
1. **Batch evaluation already works where prepare() exists; make it 4-5x denser, and give attention the same route
   via an import-time sweep in a measurement-only check** (kernel_fn = SGLang baseline). The v2 sweep (05, 06) spent
   most of its 200 s compiling: Triton re-specialises on every integer argument that is 1 or divisible by 16
   (https://triton-lang.org/main/python-api/generated/triton.jit.html, `do_not_specialize`), so m=1/8/16/32 each
   forced a recompile of every config (log 05: t jumps 27->57->87->117->143->171 s at the m and router boundaries).
   Fix: `do_not_specialize=["M","N","K"]`, read-only L2 flush like the judge, 11-rep screen + 51-rep finals, and a
   base64 result table in the 16 KB tail. Dispatching one variant per shape bucket inside one check is the second,
   zero-risk route and gives 12 (attention) / 30 (bf16) judge-grade numbers per check.
2. **bf16_proj: the gap is not only m <= 32.** From (04): verify-16/32 loads sit at 1.08x / 1.05x because cuBLAS at
   m=128 reaches only 52-59% of the floor (qkv_gate 17.01 vs 8.42, pooler 24.86 vs 14.64) and our Triton is off there.
   Fixing m <= 32 alone gives 1.33x; m=128/256 alone 1.21x; both ~1.40x (leader 1.51x). The m <= 32 design is a
   single-launch streaming kernel (no zero/copy graph nodes), TMA or 16-B loads, deep pipeline; m=128 is a full-M
   split-K GEMM with an in-kernel fixup.
3. **attention_decode: the full layers are latency-bound, not bandwidth-bound.** A model of "one 64-key block =
   ~2.4 us of serialised cold-L2 loads (page_table -> K -> V), no overlap across blocks" predicts 74 blocks x 2.4 =
   178 us for b16 (measured kernel 179.2) and 139 x 2.4 = 334 us for b32 (331.8) (08, 09 `_plan`). The fix is
   explicit prefetch/double-buffering and more programs, plus both KV heads per program. Verify shapes have an
   unexplained 12-14.5 us between the judge's time and the profile's kernel sum on exactly the 3 sliding-verify
   shapes; one diagnostic check settles it (D3). Removing it alone lifts 1.07x -> 1.20x.
4. **Serving (12): L2 "interleave" shrinks to a config lever** (overlap is on, mixed chunk only needs
   `--enable-mixed-chunk`), request-local n-gram needs our own trie (NgramCorpus is global) and stays gated by the
   simulator; kernels from the kernel track and CUDA-graph coverage move up.
5. **Checks per day:** ~70% on the critical-path problem (attention until Oct 28), the rest calibration + sweeps on
   the three others; Argon writes measurement harnesses and CUDA drafts, reviews diffs, never strategy.

## D1 Batch-evaluation design
Scoring frame (02, 10, 11): check runs are explicitly "for experiments"; prepare() is untimed (<= 300 s); print is
allowed, last 16,000 chars return; branching on shapes/config is allowed; forbidden: timing/harness tampering, work
outside timed calls that feeds the timed call, caching outputs, lower precision. The measurement checks below never
feed a timed call: kernel_fn is the unchanged baseline or a fixed per-bucket dispatch; results are read by humans
and hard-coded per shape bucket (same status as A5 in 01-PLAN; cuBLAS heuristics work the same way).

| option | problems | measurements/check | rule risk | do? |
|---|---|---|---|---|
| O1 prepare() sweep v2 as is (06) | all with weights | ~2,700 configs (05) | low (A5) | superseded by O2 |
| O2 prepare() sweep v3: `do_not_specialize`, read-only flush, 11-rep screen/51-rep finals, base64 table | all with weights | ~8-15k timings [A1] | low (A5) | **yes, first** |
| O3 per-shape-bucket variant dispatch in one candidate | all | 30 (bf16) / 12 (attention) judge numbers with CI + profile per shape | very low (explicitly allowed branching) | **yes, every check** |
| O4 import-time synthetic sweep in a MEASUREMENT-ONLY file (kernel_fn = baseline) | attention (no prepare) | ~100-300 variant x shape timings, compile-bound [A2] | low-medium: 15-min budget, must not run under capture | yes, 120 s budget, guard with `torch.cuda.is_current_stream_capturing()` (https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html) |
| O5 O3 + O4 in the same check | attention | 12 judge numbers + sweep | as O4 | yes once O4 is proven |
| O6 read the judge's per-shape profile (free with every run) | all | per-kernel GPU time per shape | none | yes, always (it found the verify gap) |
| O7 two variants computed back-to-back inside one timed call (sum) | all | sums only, halves the score | medium (extra work in timed call) | **no** |
| O8 device-side counter alternating variants across replays | all | n/a | **tampering-adjacent** (replays of one graph run different code) | **never** |
| O9 `triton.autotune` inside kernel_fn on the judge's inputs | all | n/a | borderline (tunes on judge data; sync under capture) | **no** |
| O10 `okc submit` used for measurement | all | 30/12 | none, but burns attempts, board noise | no |
| O11 reading harness buffers / frames for timing | all | n/a | **forbidden** (02) | **never** |

Recommended design per problem:
- **bf16_proj / decode_layer / moe / prefill (have weights): O2 + O3.** Sweep budget 200 s of the 300 s; the
  remaining run (tests + timing) took the v2 check ~6 min of queue+run (01-PLAN §2: 566 s observed) so stay at 200 s.
  Per-config cost after the compile fix: graph capture (~1 ms) + 11 reps x (128 MB read flush ~30 us + kernel) +
  double-precision error check -> ~3-5 ms, i.e. 40-60k timings in theory; plan on 8-15k [A1]. Log format: one
  human line per shape (top-5 + baseline) and one `T64 <shape> <base64 uint16 us x100>` block of all results; the
  parser lives in `tools/`. Keep 6-8 µs offset handling as now: same-method deltas applied to the judge's per-shape
  baseline (03, 04); the offset at m=1 in 05 vs 04 ranges 4.9 µs (o) to 9.4 µs (pooler), so never use ratios.
- **attention_decode: O3 now, O4 next check.** O3: `kernel_fn` dispatches on `(batch, q_len, window)` to one
  variant per scored shape; each shape's baseline time is known from 03/08 to +-1.5%, so each check yields 12
  independent (variant, shape) measurements against a fixed yardstick. Do not compare variants across shapes. O4:
  module-level `_measure()` builds synthetic inputs from shapes only (random FP8 cache rows, page table, positions),
  times variants as CUDA-graph replays after a read-only flush, prints compact lines, frees memory, and runs only if
  `not torch.cuda.is_current_stream_capturing()`; the file header says MEASUREMENT-ONLY; `tools/okc_run.py`
  refuses `submit` for any file containing `_measure`. Sweep runtime knobs first (split_len, nsplit, num_warps are
  runtime or cheap), constexpr knobs second (each BLOCK_N/stages combination is a ~1 s compile of two kernels).
- Honest limits: O3 measures one variant per shape per check, so a day gives ~120-240 (variant, shape) judge
  numbers for attention, which is enough for the 4-6 structural variants in D3; the fine config grid comes from O4.

## D2 bf16_proj plan
Diagnosis from (04, 05, 07):
- Score 1.152x = geo-mean of load ratios 1.122 / 1.143 / 1.141 / 1.408 / 1.079 / 1.050 (recomputed from the table;
  judge 1.152). Ceiling at the floors 2.03x. Per-load ceilings: 2.06 / 1.84 / 1.85 / 2.60 / 1.94 / 1.98.
- m <= 32 (decode 1/16/32, verify 1): our split-K configs add graph nodes: `acc.zero_()` (0.7 us in every router
  profile) and the fp32->bf16 copy, and each dependent graph node costs launch latency (NVIDIA measures ~2 us + per-
  node cost for straight-line graphs: https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/).
  cuBLAS's split-K also pays a 1.4-1.8 us `splitKreduce` node. Single-node kernels (pooler, o: SPLIT=1) are our
  only shapes above 1.1x besides the router. Pooler m=1 runs 18.95 us on a 13.99 us floor = 74% of 4.8 TB/s.
- m=128 (verify 16): cuBLAS picks 64x64 / 96x64 tiles -> each CTA re-reads the x tile [128 x 8192] (2 MB) from L2
  and the grid is 128-144 CTAs with split-K reduce nodes; 52-59% of floor. m=256 (verify 32): compute-bound
  (pooler/unpooler 17.2 GFLOP -> 17.4 us at 989 TFLOPS BF16 dense,
  https://www.nvidia.com/en-us/data-center/h200/), cuBLAS at 71% of MMA peak: leave cuBLAS except router/qkv/o.
- CUDA cores cannot carry m >= 8: pooler at m=32 is 2.1 GFLOP of FMA (~32 us at fp32 rate) > the 14 us floor, so
  every m >= 8 path needs tensor cores (Triton `tl.dot` or `mma.sync`); a pure-FMA CUDA GEMV is for m=1 only.

Technique (one launch, no memset/copy nodes, bandwidth first):
- **T1 streaming GEMV/GEMM (Triton, m <= 32):** BM = 16/32, BN in {8,16,32,64}, BK in {512,1024,2048},
  num_stages 3-6, 4-8 warps, SPLIT=1, direct BF16 store, grid = n/BN (pooler: 128 CTAs on 132 SMs, 1 wave;
  n=1024/256 shapes need T2). Two loaders to compare: masked `tl.load` with `tl.multiple_of(.,16)`/
  `tl.max_contiguous` hints (16-B vectors) vs a TMA descriptor over W
  (https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html, Triton 3.8 in
  the image per 05), which keeps 4-6 stages in flight at low issue cost. Target 85-90% of floor rate + ~1.5 us.
- **T2 split-K with in-kernel fixup (router n=256, unpooler n=1024 at m <= 8, qkv_gate m=1, all m=128):**
  partials to a workspace in `state` (from prepare, not from data) and a per-tile semaphore; the last-arriving CTA
  sums and writes the output and resets the semaphore, so the graph replays without a memset (stream-K style fixup,
  https://arxiv.org/abs/2301.03598). Workspace/semaphores are shape-indexed and allocated in prepare() [A3].
- **T3 CUDA `load_inline` FMA GEMV for m=1 (router hi+lo, unpooler n=1024, o):** 128-256 CTAs, 16-B loads, 4-8
  loads in flight per lane, warp-shuffle reduce, BF16 or FP32 store; compile at import (counts toward the 10 min,
  not the score, 02). Expected 3.5-4 us on 1.75 us floors where cuBLAS/our Triton take 6.6-10.5 us.
- **T4 m=128 full-M split-K GEMM (Triton):** BM=128, BN in {64,128,256}, SPLIT in {4,8} -> 144-288 CTAs, x
  slice read once per CTA (32-74 MB L2 traffic instead of 128+ MB), fixup from T2. m=256: cuBLAS stays.

Predicted per-shape µs (judge frame: floor/0.85 + 1.5 us launch/tail for m <= 32; m=128: max(mem/0.85, MMA at 60%
of peak) + 2-2.5 us; m=256 cuBLAS except router) and resulting loads:
| load | qkv_gate | o | pooler | unpooler | router | SGLang sum | ours sum | load ratio (now) |
|---|---|---|---|---|---|---|---|---|
| decode 1 (m=1) | 10.8 (1.32x) | 5.6 (1.26x) | 18.0 (1.18x) | 3.6 (1.85x) | 3.6 (2.95x) | 59.5 | 41.6 | 1.44 (1.12) |
| decode 16 | 10.8 | 5.7 | 18.0 | 18.0 (1.02x) | 3.6 (4.2x) | 76.4 | 56.2 | 1.36 (1.14) |
| decode 32 | 10.9 | 5.8 | 18.1 | 18.1 | 3.7 | 77.4 | 56.6 | 1.37 (1.14) |
| verify 1 (m=8) | 10.8 | 5.7 | 18.0 | 3.6 (1.96x) | 3.6 (7.3x) | 75.4 | 41.6 | 1.81 (1.41) |
| verify 16 (m=128) | 12.4 (1.37x) | 6.7 | 19.7 (1.26x) | 19.8 (cuBLAS) | 4.6 (3.5x) | 85.1 | 63.2 | 1.34 (1.08) |
| verify 32 (m=256) | 18.8 (1.05x) | 9.2 | 31.3 (cuBLAS) | 24.3 (cuBLAS) | 5.6 (3.1x) | 102.2 | 89.2 | 1.15 (1.05) |
**Predicted score 1.40x (range 1.30-1.48)**; leader 1.51x needs >= 90% of floor at m <= 32 and >= 70% of MMA
peak at m=128 (then ~1.50). The m=256 load caps at ~1.2x for everyone, so the board is decided on m <= 128.

First measurement check (one `okc check`, prepare() budget 200 s, kernel_fn = baseline):
1. **E1 T1 grid** on the 20 shapes with m <= 32: BN x BK x stages x warps as above, both loaders (masked 16-B vs
   TMA), SPLIT=1 only; read-only flush (`flush.sum()` over 128 MB or a read kernel), 11 reps, finals at 51.
   Decision rule: a config wins a bucket if it beats the judge's baseline by > 3% after the same-method delta.
2. **E2 T2 fixup vs zero+atomic** on router (all m) and qkv_gate m=1, unpooler m<=8: same tile configs, two
   reduction mechanisms; this quantifies the graph-node cost directly (expect 1.5-2.5 us per removed node).
3. **E3 T4 full-M split-K at m=128 and m=256** (BN in {64,128,256}, SPLIT in {2,4,8}) against cuBLAS measured the
   same way; m=256 only to confirm cuBLAS stays. T3 (CUDA) goes in a second check after Argon drafts it and the
   RTX 2060 proves correctness (sm_75 has no BF16 arithmetic; convert via shifts in the test build) [A4].

## D3 attention_decode plan
Diagnosis from (08, 09, 10):
- **Full layers are serialised gathers.** `_plan` gives nsplit 17 / split_len 4736 at b16 and nsplit 9 / 8896 at
  b32 (TARGET_PROGRAMS 528, HKV 2). Per program that is 74 and 139 blocks of 64 keys, each block = page_table load
  -> K row gather -> V row gather, all cold-L2 and address-dependent. At ~2.4 us per block with no overlap across
  blocks: 178 us (kernel 179.2 measured) and 334 us (331.8). The b1 full shape fits too (5 blocks -> 12 us vs 11.3).
  So the kernel moves 328 MB in 182 us = 1.8 TB/s = 37% of HBM; FA3 (SGLang) 1.63 TB/s. Little's law says 544
  programs x 3 stages x 4 KB would be enough bytes in flight if the pipeliner overlapped blocks; the fit says it does
  not (indirect, masked gathers feeding a dtype conversion rather than `tl.dot` directly) [A5].
- **Verify costs extra compute per block and more trig.** BLOCK_M=64 rows (8 tokens x 8 GQA heads) vs 16 for
  decode: the exp2/where/max work per block is 4x, and `_load_norm_rope` evaluates `tl.cos`/`tl.sin` on the whole
  [64, 64] tile (4,096 trig pairs per program, ~3 us of SFU/ALU on the critical path) instead of on
  [8 tokens x 32 frequencies]. That explains kernel 1 at 13.9 us (sliding verify) vs 7.8 (sliding decode).
- **The 14 us verify gap is between the judge's time and the profile's kernel sum**, only on the 3 sliding-verify
  shapes: 30.58 vs 16.2, 45.05 vs 30.5, 55.28 vs 43.7 (+14.4 / +14.6 / +11.6); all other shapes are within -1.3..+3.5
  us. The graph has exactly two kernel nodes and no memset/copy (torch.empty only), so launch gaps cannot be 14 us.
  Remaining explanations: (a) the profile pass runs warm-L2 or on different positions and the verify kernel is
  latency-exposed (register pressure at BLOCK_M=64 with 4 warps kills the little overlap that decode still has) so
  cold-L2 runs are much slower than the profiled run [A6]; (b) something in the judge's pairing specific to those
  shapes we cannot see. The pack cannot separate them; the first check below does.

Design (v2 -> v3), page-1 scattered FP8 K/V, same two-kernel skeleton first, single kernel later:
1. **Explicit software pipeline across blocks:** load the split's page ids in chunks of 512 (2 KB) up front, and
   issue block i+1's K and V gathers before block i's softmax/PV (manual double buffer in registers; 2 x 4 KB FP8
   tiles per program); target >= 3 blocks in flight per program. This turns 2.4 us/block into bandwidth share.
2. **Both KV heads per program:** rows of the two heads are adjacent (`row*(HKV*64) + h*64`), so one program
   consuming both heads reads each 128-B line whole instead of two programs fetching 64 B each at different times;
   q rows per program become L x G x 2 (16 decode, 128 verify -> two 64-row halves). Halves distinct requests [A7].
3. **More, shorter programs:** nsplit so that programs ~ 1,000 (8 per SM); partials grow to 4 MB (decode) / 17 MB
   (verify) which the reduce reads from L2/HBM in ~1-4 us.
4. **Trig hoist:** cos/sin on [tokens, 32] then broadcast; k rows likewise. Saves ~3 us per sliding-verify program
   and ~1 us on decode.
5. **Single launch for small shapes (v3):** last-arriving split merges (counter per (b,h), reset by the merger),
   applies RMSNorm x sigmoid(gate), writes BF16: removes one dependent graph node (~2 us) and the 32-program x 2-block
   latency chain becomes 1 block with 32 splits of 64 keys. Flash-decoding's split-KV + log-sum-exp merge is the
   reference (https://pytorch.org/blog/flash-decoding/).
6. Verify tiling: BLOCK_M 64 with 8 warps, or two 32-row programs sharing K/V tiles via L2 (test both).

Predicted per-load speedups (46 x sliding + 16 x full, SGLang times from 08):
| load | v1 (judge) | v2 (items 1-4) | v3 (+5, 6) |
|---|---|---|---|
| decode 1 | 1.65 | 2.26 (sliding 8.5 us, full 15) | 3.05 (6 / 12) |
| decode 16 | 1.23 | 1.90 (11 / 115) | 2.23 (7 / 105) |
| decode 32 | 1.21 | 1.77 (13 / 230) | 2.01 (9 / 210) |
| verify 1 | 0.90 | 2.24 (10 / 17) | 3.00 (7 / 14) |
| verify 16 | 0.89 | 1.75 (14 / 125) | 2.05 (9 / 115) |
| verify 32 | 0.75 | 1.64 (17 / 250) | 1.85 (12 / 230) |
| **geo-mean** | **1.07** | **1.91** | **2.32** |
Full-layer targets assume ~55-60% of 4.8 TB/s on 64-B scattered rows (115 us at b16 vs floor 63.2) [A8]; the
leader's 2.73x implies sliding shapes <= 5 us and full layers at >= 65% of HBM.

First 3 checks (all O3 dispatch; each shape gets one variant, baselines from 08):
1. **Verify-gap diagnostic:** sliding-verify shapes get v1 with (b1) num_warps=8, (b16) trig hoist only, (b32) two
   32-row programs; all other shapes get v1 unchanged. Read judged vs profile per shape: if the gap closes on any
   variant, (a) is confirmed and the fix is structural; if none closes it, (b) and we ask the organisers in D6.
2. **Pipelining proof on full layers:** full b16/b32 decode get item 1 (prefetch) with nsplit x2; full b1 and
   verify-full get item 1 + item 2 (both heads); sliding shapes get item 4. Prediction: b16 full 182 -> <= 120 us.
3. **v2 candidate assembled from the winners** + O4 import-time sweep of split_len/num_warps/BLOCK_N per shape;
   submit only if the check beats 1.070x by more than its CI and the red team acks (01-PLAN §7).

## D4 Serving re-ranking (what changes vs 01-PLAN §5)
Facts from (12): at a55ab7d5 the overlap scheduler is on for DFLASH (`scheduler.py:492`, only CPU disables it),
mixed chunk is allowed for DFLASH (`spec_info.py:155`) but off unless `--enable-mixed-chunk` (`scheduler.py:1325`);
chunked prefill defaults to 8192 on a 141 GB GPU (`memory_hook.py:152`); the NGRAM corpus is one trie per worker
(`ngram_worker.py:120`, `batch_put` without request id). Rules (02): drafter = DFlash as released, other drafting
only from the request itself; TTFT <= 2x stock; outputs must agree with stock.
| rank | lever | why now | before Nov 5 |
|---|---|---|---|
| 1 | **L3/L1 kernels from the kernel track** (attention decode+verify, GEMV, decode-layer glue as JIT Triton/.cu) | the only lever with judge-measured gains; decode step time is the score at stock interactivity | port v2 attention + bf16 v2 into `python/sglang` JIT paths behind a flag; one tiny JIT kernel in S1 (A3 crow) |
| 2 | **L5 CUDA-graph coverage + overlap sanity** | capture list already covers 1..8, 10..32 step 2 (`cuda_graph_hook.py:644`); confirm nothing at c=16/32 falls out of capture (verify batches, mixed chunk) | log capture hits/misses in a CPU dry run of the runner's bs logic; fix gaps in source |
| 3 | **L2 becomes config: mixed chunk on + chunk size** (was "interleave DFLASH and prefill" code work) | premise refuted; stock already overlaps; mixing only needs the flag and a size that keeps TTFT <= 2x | set in `_muse_spark_overrides`/launch defaults (source edit allowed); A/B on the Nov 5 check at c=16 |
| 4 | **DFLASH block size / acceptance tuning** (new) | verify shapes are half the spec step; block 8 vs 16 trades acceptance for verify cost; our verify kernel decides | measure acceptance on public coding trajectories with the released drafter on CPU [A9]; decide after kernel v2 |
| 5 | **L4 request-local n-gram drafting** | must be our own per-request trie (NgramCorpus is global and cross-request = not "from the request itself"); combining with DFLASH is a hybrid drafter = review risk | only the simulator (A7 >= 30%) before Nov 5; implementation only if it clears and kernels are frozen |
| 6 | prefill GEMM (large-M W4A16) | TTFT is a gate (<= 2x), not the score; prefix cache covers most of multi-turn | nothing before Nov 5 beyond moe_w4a16 kernel-track checks |
Net change vs 01-PLAN: L2 drops from code lever to flag; L4 is gated harder and loses the "reuse SGLang NGRAM" path;
add lever 4; S1 on Nov 5 = stock + flags + one validated JIT kernel, then one lever per day as planned.

## D5 Check allocation (Oct 25-27) and lane use
Limits are per problem (10 submissions + 10-20 checks each, 02), so the binding constraint is director time and the
~10-min queue per check (01-PLAN §2), not the per-problem caps. Assume checks can be queued in parallel [A10].
| day | attention_decode | bf16_proj | decode_layer | moe_w4a16 / prefill |
|---|---|---|---|---|
| Oct 25 | 6: D3 check 1 (diagnostic), check 2 (pipelining), 4 O3 dispatch rounds of structural variants | 3: E1-E3 measurement v3; v2 candidate if E1 wins | 1: baseline-vs-itself calibration (expect 1.00x) + profile read | 1: baseline calibration |
| Oct 26 | 6: v2 assembly, O4 sweep check, 4 dispatch rounds (verify tiling, both-heads) | 4: CUDA T3 check, T2 fixup candidate, v2 submit-gate check | 3: O2 sweep reusing the bf16 harness; glue profile | 1: O2 sweep (Marlin vs Triton W4A16 configs) |
| Oct 27 | 4: v3 single-launch variants; submit if > board + CI | 2: v2/v3 submit-gate; re-measure finals at 51 reps | 4: first composed candidate (attention v2 + bf16 v2 + glue) | 2: first candidate or defer |
Rules: every check has a hypothesis + predicted speedup in `experiments/log.jsonl` before it runs; a prediction
missing by > 30% stops that line until the simulator is fixed (01-PLAN §8); 21:00 digest reports hit rate.
Lanes:
- **Director:** hypotheses, dispatch files, submits, red-team ack, reading profiles (O6) the same hour they land.
- **Sonnet sub-agents (RTX 2060):** Triton correctness on the CPU interpreter / sm_75 for every variant before a
  check; PTX dumps to confirm 16-B vector loads (vectorisation is decided at compile time, not on the H200).
- **Argon (headless, strong, cold):** three card types, none strategic and none touching cp: (1) measurement
  infrastructure: sweep harness v3 (do_not_specialize, read-only flush, base64 table + parser, score-sim glue),
  attention synthetic-input builder for O4; (2) CUDA drafts with a spec and a CPU/sm_75 test: T3 FMA GEMV, T2
  semaphore fixup, the single-launch merge; (3) second-opinion review of every candidate diff against 02 rules and
  the correctness bounds (independent of the red team). Give it the pack files it needs and the done-when commands;
  one shot per card, k=1, director takes over on the first failed resubmit.
- **Jules:** only pure-python tools (parsers, simulators), no kernels (two drafts rejected, 03/brief §3).
- **Grok:** red team each submit; nearby-shape tests; cartography of FA3 verify path for lever 4.

## D6 Assumptions and open questions
| An | assumption | cheapest test |
|---|---|---|
| A1 | sweep v3 reaches >= 8k timings in 200 s once compile is removed (per-config ~5 ms: capture + 11 reps + error check) | time 50 configs of one specialization in a local dry run on the 2060 and extrapolate; confirm in E1's `t=` stamps |
| A2 | an import-time sweep of ~120 s fits the judge's 15-min attention run and runs before capture | first O4 check with a 60 s budget and a printed "MEAS done" line; abort rule: if the run times out once, O4 is dropped |
| A3 | semaphores/workspace in `state` and self-resetting counters are accepted as "repacking in prepare()" and graph-safe | red-team read of 02 + 11; functional test: 3 consecutive replays give identical output on the 2060 |
| A4 | `load_inline` compiles inside the judge image in < 60 s (nvcc present, 02 says native CUDA via load_inline is allowed) | a 10-line CUDA kernel in the first T3 check; log the compile time |
| A5 | the Triton pipeliner does not overlap the indirect masked gathers across blocks (the 2.4 us/block fit) | D3 check 2: if prefetch halves b16 full time, confirmed; local: compare TTGIR with/without manual prefetch |
| A6 | the sliding-verify gap is kernel runtime under cold L2 (profile pass is warm or differently timed), not a judge artefact | D3 check 1; if no variant moves judged time while profile moves, ask the organisers (public channel, no competitor contact) |
| A7 | consuming both KV heads in one program raises effective HBM bandwidth on 64-B rows by >= 1.3x | one dispatch slot in D3 check 2 (full b1) |
| A8 | 64-B scattered rows cap at ~55-60% of 4.8 TB/s on H200 (floor assumes 100%) | the b16 full shape after item 1-3; if we pass 70%, raise v3 targets |
| A9 | DFLASH acceptance with the released weights can be measured on CPU for block 8 vs 16 | run the draft model on 20 public coding trajectories on CPU; if it needs CUDA, drop lever 4 until Nov 5 |
| A10 | several checks can sit in the judge queue at once | queue two checks on different problems on Oct 25 and read `okc jobs` timestamps |
Open questions (cannot answer from the pack):
1. Does the judge's profile pass flush L2 and use the same inputs as the timed pass? (decides A6 fast)
2. Does the judge's "yours" time include anything besides graph replay (e.g. output copy for the correctness check)?
3. Is `okc check` concurrency limited per account? (A10)
4. Is the serving launch recipe (chunked prefill size, mixed chunk, DFLASH block size) fixed by the organisers or
   taken from our source defaults? 12 found no in-repo NVIDIA launch command.
5. Does the serving sandbox allow `load_inline`/Triton JIT with a writable cache (A3 in 01-PLAN)?
Not done on purpose: no full kernels; no okc runs; no reading outside the pack. The judged-vs-profile gap and the
2.4 us/block fit are the two facts the director should test first because each one changes the attention plan.
