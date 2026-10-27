# fable-05-okc output
executor: Fable (one-shot) · 2026-10-24 · pack read in §4 order; external facts carry URLs; assumptions are `[An]` and collected in D7.

## D1 Thesis

Pack-vs-brief corrections first (pack wins): (1) the run limit is **15 min** on decode_layer, moe_w4a16, attention_decode and prefill_layer and 10 min only on bf16_proj (TASK.md "A run has 15 minutes in all"); the rules page says 10. (2) **prefill_layer does not count toward the overall kernel board** (07/TASK.md: "has its own board and doesn't count toward the overall board"), so "4 of 5 problems" is *all* the problems that count. (3) Serving **check runs measure only concurrency 16, 1 rep, 2 conversations per slot, 35-min deadline** (09-serving-api.json `check.rungs`); only the one daily submission measures 1/4/16/32. (4) Serving compile/JIT time is inside the 90-min node slot, and .cu/.cuh/.cpp files under python/sglang are allowed (09 `allowed_suffixes`), so kernel-track work can ship as JIT kernels in the diff.

**Can we win money?** Top-3 ($10k+) is unlikely against teams with their own 8xH200 nodes who iterate locally; top-10 ($1k+) is reachable, and the top-50 kernel qualifier is cheap enough to take. My estimate: top-10 ≈ 20%, top-3 ≈ 5% `[A1]`. The way to that 20% is not a better model; it is picking the levers that stock SGLang leaves on the table *and that fit a team whose only H200 is the judge*.

What the stock numbers say (11-serving-stock.txt): at 1 user a draft+verify step takes 225.7/4.2 ≈ 18.6 ms; at 32 users it takes 36.2/4.3 ≈ 119 ms while the whole-layer kernel baseline is ≈ 22.6 ms per step averaged over loads (02: 10.77 ms × 2.101). Throughput saturates (12.98k → 13.48k tok/s/GPU from 16 → 32 users) and `ttft_first_p90` grows almost linearly with users (6.5 s, 22.7 s, 85.8 s, 192 s): first-turn prefills serialise and, because DFLASH disables SGLang's overlap scheduler and mixed chunked prefill (https://docs.sglang.ai/advanced_features/speculative_decoding.html, method table), every 16k-token prefill chunk stalls all decoders. So at 16/32 users the bottleneck is **prefill throughput and prefill/decode interleaving**, not decode batching; at 1/4 users it is **decode-step time and acceptance length**. Both are addressable with python/sglang changes plus JIT kernels.

**The single most important bet:** *use the kernel track as our H200.* Build the kernels the serving diff needs (fused attention-decode, fused layer glue, a Hopper-proper W4A16 grouped GEMM for the expert prefill) as kernel-track submissions, where each `okc check` gives us a real timed H200 run with a profile, and where an untimed 300 s `prepare()` plus the 16 KB log tail turns one check into a full autotuning sweep `[A2]`. The kernel board (top 50) falls out as a by-product; the serving diff ships those kernels plus two python levers: (i) prefill/decode interleave that respects TTFT ≤ 2x stock, (ii) request-local n-gram drafts that *replace* DFlash's block for a request when the conversation already contains the continuation (explicitly allowed by the rules; https://github.com/apoorvumang/prompt-lookup-decoding reports 2–4x on input-grounded tasks such as code editing).

Why this beats "write faster kernels with Claude": competitors with nodes will also port kernels; few will (a) treat prefill as the 32-user lever, (b) fix the DFLASH/overlap interaction, (c) run a disciplined one-submission-a-day campaign in which every node run is a predicted experiment. Our lean L1 is right (kernel track = qualifier + bench), L2 is half right (request-local drafting matters at 1–4 users; at 16/32 the gain is prefill + interleave, not batching), L3 is a necessity rather than an edge (D2).

## D2 Edge table

| edge | real / illusion | why | expected effect | cost | evidence that would change my mind |
|---|---|---|---|---|---|
| Kernel-track checks as a free H200 bench (profiles, 300 s prepare sweeps, 16 KB log) | **real, our biggest** | 10–20 checks/day/problem × 4 problems ≈ 50 timed H200 runs/day; nobody else needs this, we do | turns a GPU-less team into one with ~50 remote experiments/day | director time; strict log/sweep discipline | organisers forbid benchmarking inside `prepare()` or cut check limits `[A2]` |
| Prefill is the 16/32-user lever (Marlin is a small-batch kernel on the wrong architecture) | **real** | Marlin targets batch ≤ 16–32 and tails off toward 1.5x at 128 (https://github.com/IST-DASLab/marlin, https://arxiv.org/pdf/2408.11743); "designed for Ampere, struggles on Hopper" in compute-bound regimes, mma vs wgmma ≈ 37% peak loss (https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel); CUTLASS ships Hopper mixed-dtype wgmma GEMMs (https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu) | first-turn prefill ≈ 6.5 s → 3.5–4 s; at 32 users possibly +30–60% throughput `[A3]` | the hardest kernel we will write (NVFP4 E2M1 + FP8 block scales → BF16 in registers, grouped) | prefill_layer leaders stay at ≤ 1.75x after our first wgmma check; or serving logs show prefill < 30% of the 32-user load |
| Fix DFLASH ↔ overlap/mixed-chunk interleave; chunk policy under TTFT ≤ 2x | **real** | documented limitation; stock `ttft_later_p90` jumps 0.41 s → 2.18 s from 16 → 32 users; TTFT cap at 32 users is 2 × 2.67 s = 5.3 s, room to trade | +20–40% interactivity at 16/32 `[A4]` | python only; CPU-testable scheduler logic; risky to validate without a node | first serving check (c=16) shows no interactivity gain with smaller chunks / mixed batches |
| Request-local n-gram drafts replacing DFlash's block when the context already contains the continuation | **real but shared** | rules spell it out, so many will try; agentic-coding outputs re-emit file spans; keeping the block size fixed (8) means no shape or kernel change | 1–4 users: +15–40% tok/s on copy-heavy turns `[A5]`; ~0 at 32 users | 1 agent, CPU simulator first | offline simulator on public coding trajectories shows < 15% of output tokens predictable at n ≥ 4 |
| Kernel→serving bridge (JIT .cu under python/sglang) | **real** | allowed suffixes include .cu/.cuh; the image has nvcc (load_inline is how kernels are submitted) `[A6]` | decode step −25…−40% at 1–4 users | compile time inside the 90-min slot | serving sandbox lacks nvcc/JIT write access |
| Top-50 kernel priority | real, small | "priority" for serving nodes; backlog 1128/64 GPUs on Oct 24 | our one daily submission actually runs | one decent decode_layer entry | top-50 cut-off above 1.6x on decode_layer `[A7]` |
| Model quality | illusion | everyone has Claude-class | 0 | — | — |
| Profiles/floors as private information | illusion | every passing run returns a profile; floors are in every TASK.md | 0 | — | — |
| Parallel cheap agents on kernels | illusion unless gated | without a GPU, parallel agents cannot time anything; check budget is the binding resource; parallelism adds noise | ≤ 0 unless each agent owns a component with a CPU test and a prediction | — | — |
| "Disciplined loop" | necessity, not edge | competitors with nodes get 100× our evaluations; the loop makes our 50/day count, it does not beat their 5000 | keeps us in the game | ledger discipline | — |
| Larger DFlash block (16) | mostly illusion | drafter trained for its block; DFlash acceptance on code ≈ 4–5 at blocks 8–16 (https://inco.ai/blog/dflash2/, Tables 4–5); verify cost rises | ±10% | 1 check | one c=16 check shows +15% |
| Lower precision / sparse attention / caching | forbidden | rules | disqualification | — | — |

## D3 Serving plan

Metric mechanics we must respect: score = throughput at stock's interactivity per load, averaged; `TTFT ≤ 2x stock` per load (1: 0.32 s, 4: 0.61 s, 16: 3.9 s, 32: 5.3 s p90 `[A8]`); noise: stock `spread` is 4.6% at 1 user, 2.2% at 4, 0.7% at 16, 1 rep at 32. Gains < 5% at 1 user are invisible; stability beats brilliance in the last three days (final = medians of reruns). "tok/s per GPU" almost certainly counts prompt + output tokens as SGLang's bench_serving does (https://docs.sglang.ai/basic_usage/qwen3.html, "Total token throughput"), so the score is ≈ fixed tokens / wall time `[A9]`.

Levers ranked by expected gain per concurrency (H = high, M = medium, L = low, 0 = none):

| lever | c=1 | c=4 | c=16 | c=32 | buildable before Nov 5 w/o node | validation path |
|---|---|---|---|---|---|---|
| L1 expert prefill GEMM (wgmma W4A16 grouped, M ≥ 512/expert) | M (first turn ≈ 30% of wall) `[A10]` | M | H | H | yes, kernel-track `prefill_layer` checks | prefill_layer check speedup; serving check c=16 ttft_first |
| L2 prefill/decode interleave under DFLASH (chunk 4k–8k when running batch ≥ 8; admit prefill every k decode steps; keep TTFT ≤ 2x) | 0 | L | H | H | python; CPU unit tests on scheduler; small-model rental test | serving check c=16 (tok/s/user, ttft) |
| L3 decode-path JIT kernels (fused attention-decode; fused norm/residual/router/routing glue; skinny GEMV) | H | H | M | L | yes, kernel-track `attention_decode`/`decode_layer`/`bf16_proj` checks | kernel checks; serving submission c=1 |
| L4 request-local n-gram drafts replacing DFlash block (fixed block 8; per-request index rebuilt from the request's own tokens) | H on copy spans | H | L | 0 | CPU simulator + python; small-model rental test | simulator acceptance; serving check c=16 acceptance field |
| L5 CPU overhead: CUDA-graph coverage of the DFlash drafter + verify path for bs ≤ 32, scheduler recv interval | M | M | M | M | python; partially testable on CPU | serving check |
| L6 KV/prefix reuse | 0 | 0 | L | L | — | cache_hit already 0.92; skip |
| L7 MoE decode kernel swap (custom W4A16 small-M) | M | M | L | L | only if kernel-track moe work lands | moe_w4a16 checks |
| L8 bigger DFlash block / drafter tweaks | L | L | 0 | 0 | one flag | one check |

Pre-Nov-5 work (parallel, see D5): L3 kernels via kernel track (days 1–8); L1 prefill GEMM via prefill_layer checks (days 3–11); L2 + L4 + L5 as a python branch against commit a55ab7d5 with unit tests, read-only study of `srt/managers/scheduler*.py`, `srt/speculative/*dflash*`, `srt/layers/moe/*marlin*` `[A11]`; a static rule-checker for the diff (allowed prefix/suffixes, 400-char lines, 720 KB, forbidden words from 09). Optional rental (D4-bis): 1×H200 ≈ $5.29/h (https://www.runpod.io/articles/guides/nvidia-h200-gpu) running the a55ab7d5 image with Llama-3.1-8B-Instruct + its DFlash drafter (named in the SGLang docs) to integration-test L2/L4/L5 end-to-end on a synthetic multi-turn coding workload `[A12]`.

15-day schedule (1 submission + 3 checks/day; checks are c=16 only, so they validate L1/L2/L5 but not c=1 effects of L3/L4):

| day | submission (the only full measurement) | 3 checks (c=16) | decision rule |
|---|---|---|---|
| Nov 5 | **S1 = stock + L3 kernels only** (validated on kernel track) | C1 S1 sanity; C2 S1 + L2 chunk 8k; C3 S1 + L2 chunk 4k | S1 must score ≥ 1.0 at every load and TTFT ≤ 2x; else fix before features |
| Nov 6 | S2 = S1 + best L2 variant if its check beat C1 by > 3% tok/s/user at equal TTFT | L2 interval variants; L5 | record predicted vs observed per load |
| Nov 7–8 | S3 = S2 + L4 (n-gram) if simulator predicts ≥ +15% at c=1 | L4 correctness (outputs agree with stock), L1 if landed | L4 only ships with ≥ 2 clean checks |
| Nov 9–12 | S4..S7 = add L1 prefill GEMM when prefill_layer check ≥ 1.4x and a serving check shows ttft_first down | L1 shape robustness (nearby shapes), L2 re-tune with L1 | if S_k < S_{k-1} on the average, revert that day's lever |
| Nov 13–15 | S8..S10 = tuning of the best stack; one "robustness" submission with nothing new | hold-out style checks: vary conversation length, 1 rep each | freeze the stack on Nov 15 unless a lever is +5% with 2 confirming checks |
| Nov 16–18 | S11..S13 = frozen best; resubmit only on regression or a confirmed +5% | stability reruns of the frozen diff | no new code after Nov 17 12:00 PT |
| Nov 19 | S14 = frozen best (or the best historical if drift) | — | final; hand review readiness (D6) |

Expected stack at freeze, optimistic case `[A13]`: c=1 +45% (L3 +30%, L4 +15% blended), c=4 +40%, c=16 +45% (L1+L2), c=32 +60% (L1+L2). Pessimistic: +15/+15/+10/+15.

## D4 Kernel plan (Oct 24 – Nov 4, 11 days)

Is top 50 needed? Not for money; it buys node priority and it is the by-product of building the serving kernels, so yes, cheaply. Problem order and technique (floor math from the TASK files; H200 4.8 TB/s, 989 TFLOP/s dense BF16, 141 GB: https://www.nvidia.com/en-us/data-center/h200/):

| # | problem | days | technique | floor → baseline → target | why this order |
|---|---|---|---|---|---|
| 1 | `muse_attention_decode` | Oct 24–28 | one fused CUDA kernel per layer type: QK-norm + interleaved RoPE + FP8 cache store + flash-decoding over the FP8 paged cache (page 1) with split-KV sized to fill 132 SMs for 2 KV heads (FA3's split heuristic under-fills SMs at low head counts: https://arxiv.org/html/2604.00028v1) + fused per-head RMSNorm×sigmoid(gate); FP8 Q rounding allowed, FP32 accumulation | sliding floors 0.1–3.7 µs, full 4.3–127 µs; baseline ≈ 1.38 ms × 2.726 ≈ 3.8 ms/step over 62 layers; **target 2.0–2.4x** (leader 2.73x) | smallest surface, pure latency win at batch 1 (≥ 5 kernels → 1), reused in #2 and serving L3 |
| 2 | `muse_decode_layer` | Oct 27–Nov 2 | compose: #1 kernel; fused `attn_boundary`+router hi/lo+routing+pooler-prep into ≤ 2 kernels; GEMV from #3 for M ≤ 32; drop `F.pad`/copies by writing into preallocated outputs; keep Marlin experts initially; `prepare()` autotunes tile/split configs on synthetic inputs `[A2]` | floors 40–267 µs/layer; baseline ≈ 22.6 ms/step; **target v1 1.35–1.5x (predicted), v2 with #4 ≥ 1.7x** (leader 2.10x) | the overall board ranks "a decode step"; #2 of overall has only 3 problems, so this plus #1 likely clears top 50 `[A7]` |
| 3 | `muse_bf16_proj` | Oct 29–Nov 1 | skinny-M (1–32) split-K/stream-K BF16 GEMV reaching ≈ 90% of HBM BW (floors 1.7–14 µs; cuBLAS at M ≤ 32 ≈ 60%); router as hi+lo BF16 pair with FP32 accumulate (allowed, checked to 0.01%); M ≥ 128 stays cuBLAS | **target 1.3–1.45x** (leader 1.51x) | low effort, feeds #2 and serving L3 |
| 4 | `muse_moe_w4a16` | Nov 1–4, and continues as serving L1 | first the cheap fixes (fuse `silu_and_mul` and the post-expert norm/weighted-sum into GEMM epilogues; `moe_align` over 32 local experts instead of 256; `block_size_m` autotune); then the wgmma W4A16 grouped GEMM, written **large-M first** (prefill, M/expert ≈ 512) because that is where Marlin loses most (D2) | floors 11–109 µs; baseline ≈ 13.4 ms/step; **target 1.3–1.6x** (leader 2.35x) | hardest; its decode variant is optional, its prefill variant is the serving bet |
| 5 | `muse_prefill_layer` | Nov 2–4 | #4's large-M GEMM inside the layer; attention via FA3 prefill as is | floors 0.26–9.8 ms; **target 1.4–1.8x** (leader 1.75x) | no board weight; **it is our prefill benchmark for serving L1** |

Run/check allocation (per problem per day: 10 runs, 10–20 checks; reset 00:00 UTC). Rules: a check carries a written prediction; `submit` only when a check beats our board row by more than its 95% interval; never two submits of the same code (same code returns the earlier result).

| days | attention (10 chk) | decode_layer (10 chk) | bf16_proj (20 chk) | moe (20 chk) | prefill (10 chk) | submits |
|---|---|---|---|---|---|---|
| Oct 24–26 | 6/day: correctness ladder (16 tests), then timing | 2/day: baseline replica + profile read | 4/day: GEMV correctness | 2/day: measurement runs (sweeps in prepare) | 0 | first attention submit when check ≥ 1.5x |
| Oct 27–30 | 3/day: split-KV/full-layer tuning | 6/day: composition + fusions | 6/day: tiles/split-K | 4/day: cheap fixes | 2/day: large-M GEMM bring-up | attention, decode_layer v1, bf16 |
| Oct 31–Nov 2 | 1/day: regression guard | 6/day: v2 with moe fixes | 2/day | 8/day: wgmma GEMM | 6/day | decode_layer v2, moe |
| Nov 3–4 | 1/day | 4/day: robustness on test shapes | 1/day | 6/day | 8/day | final submits by Nov 4 18:00 PT (queue wait) |

Calibration: the first decode_layer hypothesis to test within a day (brief §9) is "fused attention-decode + fused post-attention boundary removes ≥ 40 µs per sliding layer at batch 1" → predicted decode-1 load speedup 1.15–1.25x, whole-problem 1.10–1.18x. Record predicted vs observed.

## D4-bis Test loop without an H200

- Daily budget rule: 70% of checks on the one problem whose kernel is on the critical path that day, 30% spread as regression guards; no check without a prediction and a hypothesis id; stop a line after two refuted predictions in a row and write why.
- Remote micro-bench: a *measurement-only* check (kernel_fn = baseline copy; `prepare()` times candidate kernels on synthetic inputs and prints a compact table into the log tail) gives 20–60 configs per check with no board effect `[A2]`. Keep timed `kernel_fn` free of any input-derived state.
- Local proxies that are worth it: (a) FP32 PyTorch references for every fused semantic (interleaved RoPE, QK-norm, routing tie rules, post-expert norm order) at tiny shapes on CPU/sm_75: catches 80% of correctness failures before a check; (b) a CPU simulator of the judge's scoring (weights × shapes, geometric mean over loads, 4 µs replay cost) to turn per-shape micro-times into predicted board speedups; (c) the n-gram acceptance simulator on public agentic-coding trajectories with a Llama-class tokenizer `[A14]`; (d) SGLang scheduler unit tests (python) for L2; (e) a roofline sheet per problem from the floors. What they cannot tell us: Hopper occupancy, FA3/Marlin actual timings, L2-cold effects, FP8 accumulation error at 80K (see https://vllm.ai/blog/2026-04-22-fp8-kvcache for FA3 FP8 long-context accumulation issues), CUDA-graph capture behaviour.
- Rent Hopper? **Yes, small and gated.** Ask Dex for 1×H200 (or H100; FA3/FP8 need sm_90) for ≈ 40 GPU-h ≈ $210 (price above): 15 h kernel iteration in week 1, 25 h in Nov 1–8 for the small-model serving integration test (L2/L4/L5 on Llama-3.1-8B + DFlash drafter at commit a55ab7d5 `[A12]`). Decision rule: rent on Oct 26 if the first 20 checks yield < 1.3x on attention_decode (we are blind), or on Nov 1 regardless for the serving integration; do **not** rent 8xH200 (≈ $40/h, ≥ $300 per full-model replay) unless a daily submission fails for a reason a check cannot reproduce twice.

## D5 Loop + fleet

Roles per lane (one component per agent; agents never call `okc`, the director's single submitter does):

| lane | owns | contract |
|---|---|---|
| Director (Claude, shell) | hypotheses, predictions, run/check spend, ledger, integrity review, daily submission choice, Dex updates | every node/judge run has a record before it runs |
| Sonnet A | fused attention-decode kernel (#1) | passes the 16-test ladder in FP32 reference locally; prediction per check |
| Sonnet B | layer glue fusions + composition (#2) | same, 13-test ladder |
| Sonnet C | skinny GEMV (#3) | 9-test ladder; BW model |
| Sonnet D | W4A16 wgmma grouped GEMM (#4/#5, serving L1) | correctness vs `nvfp4.py` reference at tiny shapes; roofline prediction |
| Sonnet E | python branch: L2 interleave + L5 graphs, unit tests | scheduler tests green on CPU; rule-checker green |
| Sonnet F | L4 n-gram drafter + offline simulator | simulator report (copy fraction, predicted acceptance) before any code lands |
| Grok 1 | source cartography of a55ab7d5 (where DFLASH verify, chunked prefill, Marlin runner live), public-kernel scouting (FlashInfer, Machete, CUTLASS 55) | maps with file:line, no opinions |
| Grok 2 | red team: scans every diff/kernel for D6 violations and for input-derived caching; reruns rule-checker | veto power, writes to ledger |
| NVIDIA free API | bulk test generation, docstring/diff summaries, second-opinion reviews (40 RPM) | never on the critical path |

Experiment record (one per check/run/submission): `id, date, track, problem/load, hypothesis, lever, change (diff hash, files), predicted (point, range), job id, observed (speedup + 95% CI, per-shape table / per-load table), profile top-3 kernels, verdict ∈ {confirmed, refuted, noise, failed-test, timeout}, cost (checks, runs, GPU-h, $), next action, author`.

Daily cadence (UTC): 00:00 limits reset → 00:30 director posts the day's 3–6 hypotheses with predictions → agents prepare candidates with local proxies → 06:00–20:00 checks in critical-path order (judge queue can be hours; never wait in a loop, poll with `--no-wait` and record) → 20:00 submit decisions → 22:00 ledger digest to Dex (one screen: board rank, best per problem, predicted-vs-observed hit rate, spend). Serving phase: same, with the one submission decided at 20:00 PT the previous day and uploaded at 06:00 PT to leave queue slack. The director keeps: spend (checks, runs, $), what ships, integrity sign-off, kill decisions; agents keep: their component's tests and predictions.

What makes parallel agents an edge rather than noise: distinct components with CPU-testable contracts; predictions before runs; a budget of checks per component per day set by the director; a refuted-twice rule; one writer per file; the red team off the critical path.

## D6 Integrity rules and kill criteria

Rules (hand review and held-out reruns decide the money):
1. Nothing in a timed call depends on anything derived from a previous call's inputs; `prepare()` touches weights/config only; no output caching or replay.
2. Precision: NVFP4 stays NVFP4, BF16 stays BF16, FP8 cache stays FP8, Q rounding E4M3 at most; router hi+lo only; no lower-precision activations anywhere stock keeps BF16.
3. Attention attends every key (full window / full context); no top-k, block-skip or weight-threshold tricks.
4. No touching timing, harness frames, I/O, env, loaded modules, reference imports, processes, network; the diff never contains the forbidden words in 09.
5. Serving: model unchanged (8 experts/token, 2,048 window, every norm, shapes); DFlash weights untouched (speed only); other drafts come only from the request's own tokens — **per request, never a cross-request n-gram cache** (SGLang's NGRAM trie is global, so we write our own); exactly 220 tokens per request; outputs agree with stock within stock's self-agreement.
6. Robustness over scored shapes: every kernel is checked on the hard tests and on "nearby" shapes (ctx 60–100k, batch 2–48, page 64) before it ships; anything tuned to 80k/220 only is rejected.
7. Red-team veto before any `submit`; `okc withdraw` immediately if a run is later found to violate 1–6.
8. No `okc login`, accounts, or spend by agents; Dex accepts rules and approves money.

Kill criteria:

| track | condition | date | action | keep anyway |
|---|---|---|---|---|
| kernel | no attention_decode check ≥ 1.5x | Oct 28 | request the 1×H200 rental; if refused, drop #3/#4, keep #1/#2 only | FP32 references, scoring simulator |
| kernel | decode_layer best < 1.25x or overall rank > 150 (`you.rank`) | Nov 1 | stop board work; keep only kernels needed for serving L1/L3 | kernels (Apache-2.0, upstreamable) |
| serving | S1 not ≥ 1.0 at all loads or TTFT > 2x | Nov 7 | freeze features, fix stability only | — |
| serving | best average < +10% over stock | Nov 13 | drop L4/L1 if not confirmed twice; submit the safe L3+L2 stack daily | ledger, simulator |
| serving | top-10 cut-off > our best × 1.25 | Nov 16 | no further spend or new code; frozen best to the end | the method and the write-up |

## D7 Assumptions and open questions

| id | assumption | cheapest observation that tests it |
|---|---|---|
| A1 | top-10 ≈ 20%, top-3 ≈ 5% | board after Nov 8: count entries > +20% over stock |
| A2 | timing candidate kernels on synthetic inputs inside untimed `prepare()` and printing to the log is within the rules ("work outside timed calls" means precomputing outputs) | one measurement-only check; director rereads rules; if in doubt use a baseline-kernel check whose `prepare()` only prints |
| A3 | first-turn prefill is ≥ 50% of wall time at 32 users and the expert GEMM is ≥ 60% of a 16k-chunk layer | 07/TASK.md says "most of the time goes to the routed experts"; first prefill_layer profile; serving check log (`ttft_first` share) |
| A4 | smaller chunks / interval admission under DFLASH raise interactivity at 16/32 within TTFT ≤ 2x | serving check C2/C3 on Nov 5 |
| A5 | ≥ 30% of output tokens in agentic-coding turns are 4-gram-predictable from the conversation | CPU simulator on public trajectories (Sonnet F, day 1) |
| A6 | the serving sandbox JIT-compiles .cu under python/sglang (nvcc present, writable cache) | S1 on Nov 5 includes one tiny JIT kernel with a log line |
| A7 | decode_layer ≈ 1.4x plus attention ≈ 2x places top 50 | `okc leaderboard --json you.rank` after our first two submits |
| A8 | the TTFT in the ≤ 2x rule is `ttft_p90_s` | read the serving leaderboard JSON once rows appear; otherwise design to p90 |
| A9 | "tok/s per GPU" counts prompt + output tokens, so score ≈ 1 / wall time | compare 6319 tok/s/GPU × 8 × stock turn latency with ~80k tokens/turn; ask in D7 → director |
| A10 | ≈ 12–13 turns per conversation (cache_hit 0.92 ⇒ (K−1)/K ≈ 0.93) | serving check log: requests per conversation |
| A11 | SGLang file paths for scheduler/DFLASH/Marlin at a55ab7d5 as named | Grok 1 cartography, day 1 |
| A12 | the a55ab7d5 image serves Llama-3.1-8B-Instruct with `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` on 1×H200 | first hour of the rental |
| A13 | stack gains as listed | predicted-vs-observed table per daily submission |
| A14 | copy fraction is tokenizer-robust (Llama-class proxy for the gated Muse tokenizer) | run the simulator with two tokenizers |

Open questions for the director/Dex (not guessable from the pack): how the overall board composes 4 problems into "a decode step" (#2 overall has 3 problems); the interpolation used for "throughput at stock's interactivity" when our interactivity exceeds stock's at the same load; the judge's sampling temperature (affects acceptance of n-gram drafts); whether server-arg overrides inside `server_args.py` count as "the same model" (they are code, not launch args); the real priority mechanism for top-50; whether `okc check` on serving exposes per-request logs.
