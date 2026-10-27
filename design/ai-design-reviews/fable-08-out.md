# fable-08-okc-next output
executor: Fable (one-shot) · 2026-10-24 · inputs: the 18-file pack + public web (URLs in §Sources) · design only, nothing run.
Status line: **Advice (not adopted)** · harness itachi/v0.1 (D1 follows pack 17 §1–9).
Conventions: numbers with a pack id in brackets (e.g. [03], [07], [16]) are measured; **every number marked `≈` or `est.` is mine**
(arithmetic on pack numbers or an engineering estimate). Shape ids are the judge's (#0–#11 for decode_layer/attention, #0–#5 moe, #0–#29 bf16).
Load formula used throughout [15]: load time = Σ weight × shape µs; problem speedup = geo-mean over the 6 loads of SGLang/ours. I re-derived
all four board numbers from the L0 tables [04] with this formula (1.689 / 1.436 / 1.227 / 1.155 to 3 decimals), so I2 (06b) is closed: the board
IS the weighted geo-mean of **loads**, not of shapes.

## 0. Headline (read this if nothing else)
1. **The time is in two kernels that share one defect.** `w4_largem_kernel` and `w4_smallm_kernel` both stream weights with ~3 KB in flight per
   warp and ~8 warps per SM (register double-buffer, 162–180 regs, 67 KB smem) [11]; every judged variant of the sweep [16] kept that
   geometry, which is why "knobs gave ≤ 2%". Marlin, with its 4-stage cp.async pipeline, reaches ≈ 2.5 TB/s on the layer's T=32 routing
   (101.6 µs for ≈ 255 MB, [03] #4) while ours sit at ≈ 1.6–2.3 TB/s. **Hypothesis H1: bytes-in-flight, not dequant / mma / fixup / chunk
   re-reads, caps both kernels.** One measurement check (stamps + a no-compute "pure stream" twin) decides it; the fix is a cp.async shared-
   memory ring (4–8 stages) in the same kernel skeleton, numerics bit-identical to v4.
2. **Two findings that change the lean.** (a) "GEMM2 slower than GEMM1 at M=1" is a PDL measurement artifact: GEMM2's CTAs launch early and
   wait at `griddepcontrol.wait`; its real time is end-to-end ≈ 13 µs (moe #0: ends 40.9, GEMM1 ends 27.6 [03]), the same ≈ 1.8 TB/s as GEMM1.
   (b) The chunk re-read is a 10–15 % effect (27 chunks for 24 experts at bs256, 24 for 23 at bs128 [16]), not the 3× gap. Both lean items
   (2) and (3) in the brief should be re-aimed at in-flight bytes.
3. **decode_layer is the largest overall lever** (its step time is ≈ 54 % of our summed step time, est.). Sure part of v5 (full_cat port):
   1.227 → ≈ 1.27. With the streaming GEMM replacing Marlin at T ≥ 32 and smallm at T ≤ 16: ≈ 1.62 (pess.) – 1.73 (cons.); with the four
   glue items (D4) ≈ 1.9 (opt.). The single biggest glue item is not a fusion: the side-stream router GEMV is stretched 4× (22 µs vs 5.5 µs
   standalone [03] #0 vs [04-bf16] #4) because the pooler's 192 KB-smem CTAs leave it 4 SMs; a config change saves ≈ 9 µs on six shapes.
4. **moe:** 1.436 → ≈ 1.83 (pess., GEMMs 1.3× faster) / ≈ 2.2 (cons., 1.6×) / ≈ 2.5 (opt.). Same kernel, both boards.
5. **attention:** do not reopen a builder lane. Verify-full is MUFU/convert/tensor-bound (355 µs vs 187 µs decode on the same bytes: 8× the
   exp2 per key), so porting the `full_cat` *structure* will not give 1.85×. One cheap combined check is worth it: ROWMAJOR partials for the
   sliding reducer (+1.2 %, est.) and, **only if Dex confirms legality (D9-Q1)**, P·V in FP8 (1.689 → ≈ 1.86).
6. **bf16:** two checks, _CFG + router fixup: 1.156 → ≈ 1.22. **Serving:** start S0 (CPU-only, Jules) on Oct 31; first three items in D8.
7. Top move for the next 24 h: C19 (layer v5 port, promote) + M1 (moe stamps + pure-stream twin, one measurement check). Everything after
   M1 is chosen by its decision table (D3 §3), not by taste.

---
## D1. itachi design: "streaming experts" (pack 17 §1–9)

### 1. True objective
**Overall kernel-board rank is as high as possible at 2026-11-05 07:59Z**, with every board entry legal (10) and reproducible on the judge's
reruns (held-out data, nearby shapes). Observable: `okc leaderboard --json` overall rank; per problem the judged speedup with CI-low above the
previous board entry (promote.py rule). Means test: if the rank rises because a competitor's entry is removed, or because bf16 alone moves
200 places, it still counts → objective, not means.
Non-goals: beating any single leader; serving-track code before Oct 31; elegance of kernels; anything the rules call tampering.

### 2. Constraints and failed solution classes
Hard limits: one judge account; per problem per day 10 submits + 20 checks, ~10 min each, 15 min run cap (10 for bf16), prepare() ≤ 300 s
[15]; no Hopper locally (RTX 2060 sm_75, NVRTC only, no nvcc) → CUDA correctness and speed are judge-only; ≈ 2 Sonnet builders, usage-limit
cuts; Argon / Grok Build read-only drafts; NVIDIA free models flaky; director usage limits; claude-remote 4 GB memory cap.
Failed classes (do not re-propose without the named new evidence; evidence = [07] rows / [08] §2):
| class | evidence | reopen only if |
|---|---|---|
| synthetic-routing sweeps as a predictor | moe sonnet_v2 1.098 vs predicted 1.2 (16:10Z); measure_grouped synthetic | never: generate_input routing only (A2) |
| grouped Triton W4A16 GEMM for large M | grouped_v1 0.970 (16:36Z) | never (CUDA path superseded it) |
| single-launch sliding attention | sonnet_v4 1.509, v4b 1.457 vs 1.689 (16:20Z, 16:37Z) | a trace shows reducer+gap > 40 % of a sliding shape AND a counter-free merge design |
| deeper pipelines / more programs on full-decode attention | v3 snapshot f68e4a3 1.475 (14:50Z) | decode-full measured < 2.5 TB/s on judge seeds |
| bf16 Triton streaming / full-M | measure_v3 1.001 (13:51Z), [08] | never in this window |
| routing-hint L2 prefetch | T1 (bytes argument, [09] D4) | never |
| draft cascade on kernel track | C12 +1 % | serving only |
| S2 fused routing in the layer | v3 1.077 vs v3_s1 1.194 (17:20Z/17:24Z) | fused kernel traced < 4 µs end-to-end incl. routing |
| PDL on Triton launches | v3_s13 1.203 vs v3_s1 1.194 = noise (17:31Z) | never (keep the free attribute on the CUDA kernels) |
| Marlin knob tuning | v2 refuted; S1 knobs = the only confirmed ones | never; Marlin is fallback only |
| largem / smallm knob sweeps (nw, mr, ks, nz) | [16]: ≤ 2 % | after the pipeline depth changes (one re-sweep) |
| megakernel / persistent whole layer | [09] D3 verdict | not in 10 days |

### 3. Environment / adversary
| what resists | cheapest early observation |
|---|---|
| the field moves up (overall 480→492 earlier today with no regression of ours) | daily `okc leaderboard --json` delta per problem; Grok Bot digest |
| judge queue near the deadline | `GET /queue` before each round; freeze new submits Nov 4 18:00Z |
| routing variance on the judge (moe bs16 p10–p90 112–173 µs [04]) hides ±15 % kernel changes | every measurement check prints `distinct`/`chunks` per seed and times ≥ 3 seeds; promote only on CI-low |
| our own sweeps overfit (A2) | predicted-vs-judged per-shape error > 15 % → line stops |
| PDL early-launch inflates CUPTI durations (H4) | report kernel end−previous end, never duration, for PDL kernels |
| judge-only CUDA debugging: a build error costs a 10-min check | NVRTC syntax compile for sm_80/sm_90 on the dev box before every check; `-Xptxas -v` in the check log |
| end-of-challenge reruns on held-out shapes | every candidate passes the problem's test list (bs 33 / 257 / skewed, ctx 1M …) in `okc check`; Marlin fallback stays |
| the overall formula is unknown (A1) | one isolated submit (bf16) and read the overall delta |

### 4. Two real alternatives (and the chosen one)
- **Alt A — incremental register pipeline on the existing kernels** (`__launch_bounds__(…, 3/4)`, a third prefetch buffer, D=4): reaches the
  objective only if occupancy alone is the limit. Cost ≈ 1 builder-day, 2 checks. Failure mode: register spills, or ceiling ≈ 2.5 TB/s.
  **Switch to A if** M1 shows ≥ +40 % from the launch-bounds / depth variants alone (then ship it as v5 the same day and stop the ring).
- **Alt B — CUTLASS / Machete-style wgmma mixed-input grouped GEMM** (image has CUTLASS [10]; Machete exists because "Marlin struggles on
  Hopper", https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel). Highest ceiling at bs 128/256.
  Cost ≈ 4–5 builder-days, compile time inside the 15-min run, unknown support for E2M1 × E4M3 group-16 scales and for grouped/EP routing.
  **Switch to B if** the ring (chosen) is judged twice and GEMM1 bs256 stays > 270 µs (≈ 1.9 TB/s) while the pure-stream twin is ≤ 180 µs
  (i.e. the memory pipeline is fine and the compute path is the stall).
- **Alt C — attention-first** (verify-full warp specialisation, fable-07 Move B). Switch to C only if A1 is falsified in the direction "overall =
  rank-sum" AND decode_layer/moe have both stalled for 2 days.
- **Chosen: "streaming experts"** = (i) cp.async smem ring in `w4_largem/smallm` (same dequant, mma, fixup → bit-identical), (ii) composition
  into decode_layer for every T (gated per bucket by prepare()-time A/B against Marlin), (iii) the four layer glue items, (iv) bf16 cheap fixes,
  (v) one conditional attention check. Rationale: one kernel change lands on two boards (≈ 81 % of our summed step time, est.); every step has
  a one-check kill test; nothing new in numerics.

### 5. Mechanism
- **K (enforced):** promote.py (check → submit only if speedup and CI-low beat the board; refuses measure_/trace_ files) [A5]; prepare()
  self-checks (bitwise v5-vs-v4 for the ring kernel; rel L2 < 1e-2 vs Marlin for anything touching numerics); per-bucket enable decided in
  prepare() by timing synthetic generate_input-style routing, never by judge inputs [A2]; Marlin fallback for M > 256 or failed self-check.
- **S (served):** C20 measurement file with level-2 stamps + variant twins; tools/make_trace.py for every candidate before promote;
  parse_trace.py reports end-to-end per PDL kernel [A6].
- **P (policy):** the izanami round picks the next move from D2 by (predicted Δ overall per check) with the daily cap ≥ 70 % of checks on
  decode_layer + moe until Oct 30; a hypothesis refuted twice is dropped; every check carries a predicted per-shape µs line in the log.
Assumptions inline: [A1] overall ≈ time-weighted decode step; [A2] generate_input routing ≈ judge routing; [A3] H1 (in-flight bytes) is the
limiter; [A4] cp.async ring with 4–8 stages reaches ≥ 3.2 TB/s on 1–2 KB contiguous per-warp streams on H200; [A5] promote.py CI rule holds;
[A6] PDL inflation explains GEMM2's duration; [A7] the layer's judge routing ≈ 2.5 routes/token, E ≈ 9/18/21 at T = 32/128/256 [05];
[A8] P in FP8 for P·V is legal (baseline FA3 FP8 path); [A9] pooler/router co-residency is the cause of the 22 µs router GEMV; [A10] builders
deliver ≈ 120 builder-hours in 10 days.

### 6. Assumption register
| An | assumption | if false, what breaks | watcher |
|---|---|---|---|
| A1 | overall rank ≈ time-weighted decode step (decode_layer 54 %, moe 27 %, bf16 12 %, attention 7 % of our step, est.) | priority order shifts toward rank-sum (bf16/attention cheap ranks) | **metric:** bf16-only isolated submit Oct 25 → overall delta vs the two models; director records which fits |
| A2 | generate_input routing ≈ judge routing (also for the layer: 2.5 routes/token) | prepare()-time A/B gates pick the wrong kernel per bucket | **law:** okc_run verdict "refuted" if judged per-shape µs deviates > 15 % from the measurement prediction → bucket gate reverts to Marlin |
| A3 | H1: in-flight bytes cap both GEMM kernels | ring gives < 1.3×; design pivots to Alt B or warp specialisation | **vector:** M1 pure-stream twin ≥ 0.85 × full-kernel time (confirm) / ≤ 0.55 (refute) |
| A4 | cp.async ring reaches ≥ 3.2 TB/s on this access pattern | moe ≤ 1.85, layer ≤ 1.62 | **crow** `streaming-gate` Oct 27 18:00Z: GEMM1 bs256 ≤ 230 µs in some check else switch to Alt A |
| A5 | promote.py CI rule prevents regressions | a worse run on the board | law: board keeps best (okc) + promote refuses; log review each round |
| A6 | PDL inflation explains GEMM2 > GEMM1 | small-M has a real GEMM2 defect worth 10 µs | **vector:** trace of M1 with `programmaticStreamSerializationAllowed=0` on GEMM2: duration drops to ≈ end-to-end, span unchanged |
| A7 | layer routing E ≈ 9/18/21 at T 32/128/256 | largem-in-layer predictions off by the ratio | metric: measurement check prints distinct/chunks on layer generate_input |
| A8 | P·V in FP8 is within the rules | removal risk | **accepted-risk only after Dex/organiser answer (D9-Q1)**; until then not built |
| A9 | router GEMV stretch = pooler smem monopoly | G1 saves < 3 µs | **vector:** C22 measurement: pooler stages 4→2 (or BK 512→256) → router GEMV ≤ 8 µs on the side stream |
| A10 | ≈ 120 builder-hours available | schedule slips 2–3 days; cut D7 items by rank (D2) | metric: handover's "builder hours used" vs plan at each decision point |
| A11 | cuBLAS/Triton pooler at T ≤ 32 stays the right choice (3.5 TB/s) | glue plan misses 5 µs | trace of v5b |
| A12 | the judge's 16 KB log tail holds the stamp summary | measurement data lost | law: print the summary first, per shape ≤ 12 lines, silence warnings (T5) |

### 7. Posthumous pack
- **Vectors / hidden tests** (path suggestion `<vectors-dir>/okc/fable-08/`, public subset = seeds 11–13, hidden = 5 more seeds):
  V1 ring kernel vs v4 kernel **bitwise equal** outputs for M ∈ {1,4,8,16,32,33,128,256,257}, 3 seeds (same dequant/mma/fixup order);
  V2 layer v5/v6 vs v4 rel L2 ≤ 2e-3 on the 13 test shapes' geometry (attention port changes summation order only);
  V3 pre-expert-norm fusion vs torch RMSNorm(1+w) in FP32 → one BF16 rounding: max |Δ| ≤ 1 BF16 ulp per element;
  V4 Triton epilogue 4-way grid vs 1-way: bitwise equal (same FP32 order per slot);
  V5 attention ROWMAJOR sliding: bitwise equal to v3 (layout only).
- **Task cards** (schema harness/taskcard.yaml; done-when = commands): C19 layer v5 port · C20 moe M1 stamps + twins · C21 ring kernel ·
  C22 layer G1 co-residency + unpooler + epilogue grid · C23 bf16 cheap · C24 layer v6 all-T gate · C25 G2 norm fusion · C26 attention
  combined check · C27 serving S0. Done-when lines are in D7's per-day rows.
- **Crows** (pre-kernel routines; each re-validates its assumptions at fire time, max_fires 1):
```json
[{"id":"crow.m1-decision","fire":"when C20's check lands (ledger task C20 → review)","expiry":"2026-10-26T12:00Z",
  "recheck":["A3","A6","kernel track open"],"action":"apply D3 §3 decision table; write chosen design id into handover"},
 {"id":"crow.streaming-gate","fire":"2026-10-27T18:00Z","expiry":"2026-10-28T06:00Z",
  "recheck":["A4: any check with GEMM1 bs256 <= 230 us","builders alive"],"action":"if false: switch moe lane to Alt A (launch_bounds+depth), move Sonnet-B to C22/C25"},
 {"id":"crow.a1-calibration","fire":"after the first isolated bf16 submit (C23)","expiry":"2026-10-27T00:00Z",
  "recheck":["overall rank changed"],"action":"record which overall model fits; if rank-sum: raise attention C26 priority one step"},
 {"id":"crow.serving-s0","fire":"2026-10-31T09:00Z","expiry":"2026-11-01T09:00Z",
  "recheck":["decode_layer board >= 1.45 (else fire again Nov 1)","Jules lane available"],"action":"dispatch C27 S0 (CPU-only)"},
 {"id":"crow.freeze","fire":"2026-11-03T12:00Z","expiry":"2026-11-05T08:00Z",
  "recheck":["kernel track open","no Dex stop"],"action":"no new kernels; only re-checks of board entries and promote of already-judged candidates; last submit 2026-11-04T18:00Z"}]
```
- **Gone-clause (if the director is unavailable):** (1) only promote.py from a director session submits; builders' branches wait. (2) Grok
  Bot keeps the red team per sha and the daily digest. (3) Continuity may run C20/C22/C23 measurement checks (kernel_fn = board kernel) but
  may not submit new kernels. (4) Decisions that wait for the director: switching lanes (crows above), reopening attention, anything on A8.
  (5) Dex may say "stop" in chat or ledger → routines disabled. (6) After Nov 5 07:59Z nothing on the kernel track; serving needs its own design.
  (7) Spend, accounts, organiser contact: Dex only. (8) If a board entry is found illegal: `okc withdraw` first, think second. (9) Handover's
  last block is the truth; (10) this file's D2 order is the queue.

### 8. Falsification and evaluation
- Wrong if the M1 pure-stream twin is ≤ 0.55 × the full kernel's time at bs 128/256 (then the compute path, not the memory pipeline, stalls;
  Alt B / warp specialisation).
- Wrong if the ring kernel, after one tuning check, is < 1.25 × faster than v4 on GEMM1 bs256 on judge seeds.
- Wrong if layer v5 (port only) judges < 1.25 (prediction 1.266 ± 0.02).
- Wrong if G1 (pooler smem change) leaves the side-stream router GEMV > 10 µs at T=1.
- Wrong if the overall rank on Oct 29 is not better than 336 while ≥ 2 promoted submits landed (then A1 or the field).
Replay set: the judge's own shapes (every check); metric = judged speedup with CI; threshold = CI-low above board. Cost rule: ≥ 70 % of
checks on decode_layer + moe through Oct 30; a hypothesis refuted twice is dropped; no check without a predicted per-shape µs in the log.

### 9. Adoption record
Decision id: ______ · risk class: ______ · adopted by: Dex (minister) / director · rollback: `okc withdraw <job>` per entry; S4/LARGEM enables are
module switches · first review: ______ (proposed 2026-10-26 12:00Z after C19 + C20 land).

---
## D2. Ranked hypotheses (by expected Δ overall per judge check; effort in builder-hours, est.)
| id | problem | shapes | mechanism | predicted µs → speedup (est.) | exact test | kill / confirm | effort | lane |
|---|---|---|---|---|---|---|---|---|
| H5 | decode_layer | #1,#3,#5 | port attention v3 `_full_cat_kernel`+ROWMAJOR reducer into the layer for full-decode (q_len 1) | #1 137→132.6, #3 340→280, #5 540→410 µs; loads decode16 1.347→1.445, decode32 1.138→1.273; problem 1.227→**1.266** | C19: `okc check` v5 then promote | kill if < 1.245; confirm ≥ 1.255 | 4 h | Sonnet-A |
| H1 | moe | #2,#4,#5 (+#0,#1,#3) | GEMMs are in-flight-bytes bound (~3 KB/warp, 8 warps/SM) | none directly; decides H2 vs Alt B | C20 measurement: stamps t0..t6 + twins {pure-stream, LB(…,3), LB(…,4), D=4} × bs 32/128/256 × 3 seeds | twin ≥ 0.85×full → H1 true; ≤ 0.55× → false | 6 h | Sonnet-B |
| H2 | moe (+layer) | all 6 | cp.async smem ring (4–8 stages) in largem/smallm, bit-identical numerics | GEMM1 bs256 322→≤200, GEMM2 162→≤100; moe 1.436→1.83 (1.3×) / 2.2 (1.6×) | C21: self-check bitwise vs v4, measurement sweep stages∈{3,4,6,8}, then promote | kill if after 2 checks GEMM1 bs256 > 270 µs | 30–40 h | Sonnet-B + Argon review |
| H10 | decode_layer | #8–#11 (#4,#5) | replace Marlin by the CUDA GEMM at T ≥ 32 when prepare()-time A/B on layer routing says so | Marlin 398/510 µs at T=128/256 → ≈ 147/170 with the ring (≈ 320/400 with v4 largem); verify16 1.053→≈1.7–2.0, verify32 1.034→≈1.6–1.9 | C24: measurement Marlin vs largem vs ring on layer generate_input, 3 seeds per T; gate per bucket | kill a bucket if ours ≥ 0.95 × Marlin | 10 h | Sonnet-A |
| H6 | decode_layer | #0–#3,#6,#7 (−9), #8–#11 (−8–11) | router GEMV on side stream gets only the 4 SMs the 192 KB-smem pooler leaves → 22 µs; cut pooler stages/BK so both co-reside | −9 µs ×6 shapes, −8–11 ×4; decode1 1.365→1.47 | C22 measurement variant: pooler (32,512,1,4,**2**) or (32,**256**,1,4,4); read router GEMV end−start on stream 1 | kill if router > 10 µs after change | 3 h | Sonnet-A |
| H7 | decode_layer | all after Marlin removal | pre-expert RMSNorm(1+w) folded into route kernel (per-token rms) + GEMM1 staging multiply | −6.7 µs/shape (layer_norm 6.3 + gap) | C25 + V3 vector; trace | kill if V3 fails or Δ < 4 µs | 8 h | Sonnet-B |
| H8 | decode_layer | #0,#1,#6,#7 | unpooler T≤8 via Triton GEMV (n=1024) writing stride 8192 into a zeroed [T,8192] buffer; drop splitK-reduce + F.pad memcpy | −5.4 µs ×4 | C22 variant | kill if Δ < 3 µs | 3 h | Sonnet-A |
| H9 | moe + layer | T ≤ 16 | Triton epilogue grid tokens×4 (each CTA recomputes sumsq from L2) | −3 µs on moe #0/#1/#3 and layer #0–#3,#6,#7; moe bs1 1.86→2.0 | C22 variant + V4 | kill if Δ < 1.5 µs | 2 h | Grok Build draft → Sonnet-A |
| H3 | moe | #0 (#3) | nz1=2 at M=1 forces 2 expert passes when n_act=3; flatten z ≥ n_act with early exit | bs1 GEMM1 27→≈18 µs when n_act=3 | C20 variant nz1∈{2,4,8} with n_act printed | kill if per-expert time not constant | 1 h | Sonnet-B |
| H13 | bf16 | #3,#18 (unpooler m 1/8); #4,#9,#14,#19,#24,#29 (router) | _CFG unpooler (32,256,1,4,4); router last-CTA fixup (no torch.zeros, no atomics); router m=128/256 BM=m | −3.5 ×2, −1.5 ×4, −4–6 ×2 → 1.156→≈1.22 | C23: measurement then promote (isolated submit for A1) | kill if < 1.19 | 4 h | director + Sonnet-A |
| H11 | attention | #8,#10 (#2,#4,#0,#6) | sliding partials [b,h,row,split] (ROWMAJOR) → reducer 5.5/8.1→≈2.5/3 µs | 1.689→≈1.71 | C26 (bundled) + V5 | noise floor ±1.5 % → confirm only if ≥ 1.70 | 2 h | Sonnet-A |
| H12 | attention | #9,#11 (#7) | P·V in FP8 (P e4m3, V as stored): removes 303 M V-conversions and halves PV tensor time | #9 181→≈125, #11 355→≈245; 1.689→≈1.86 | C26 only after D9-Q1 = yes | kill if error > checks.py bound or Δ < 15 % | 4 h | Sonnet-A |
| H4 | moe | #0,#1,#3 | GEMM2 "slow" = PDL early launch | report fix only | C20 prints end-to-end | — | 0.5 h | Jules (parser) |
| H14 | overall | — | overall ≈ time-weighted step | re-orders D2 if false | isolated bf16 submit | — | 0 | director |
| H15 | attention | #9,#11 | verify warp-specialisation + exp split (fable-07 Move B) | 1.69→≈2.1–2.3 | not in window | — | 40 h | parked |
| H16 | moe | #4,#5 | CUTLASS wgmma mixed-input (Alt B) | bs256 → ≈ 150 µs GEMM1 | only if H2 killed twice | — | 40 h | parked |

Reading the table under A1 (time-weighted): Δ overall per check ≈ H2 (≈ +0.25–0.45 overall-speedup, est.) ≫ H10 (needs H2) > H5 (+0.04) > H6
(+0.03) > H7/H8/H9 (+0.02 each) > H13 (+0.01) > H11/H12 (+0.005 / +0.02). Under rank-sum, H13 and H5 rise one step, H12 two steps.

---
## D3. moe plan: large-M and small-M

### 1. Where the measured time is and why (my reading of [03], [16], [11])
Per-GEMM effective bandwidth, counting distinct experts once (E from [05]; chunk re-reads counted as traffic where stated):
| shape | kernel | µs [03] | bytes (est.) | TB/s (est.) | note |
|---|---|---|---|---|---|
| moe #5 bs256 | GEMM1 | 322 | 27 chunks × 18.9 MB = 510 MB | 1.6 | re-read 27/24 = +12 % only |
| moe #5 | GEMM2 | 160 (end−end) | 27 × 9.45 = 255 MB | 1.6 | |
| moe #4 bs128 | GEMM1 / GEMM2 | 230 / 120 | 24 × 18.9 / 24 × 9.45 | 2.0 / 1.9 | |
| moe #2 bs32 | GEMM1 / GEMM2 | 147 / 81 | 16 × 18.9 / 16 × 9.45 | 2.1 / 1.9 | |
| moe #0 bs1 | GEMM1 / GEMM2 | 27.0 / **13.3 real** | 3 × 18.9 / 3 × 9.45 | 2.1 / 2.1 | n_act = 3 → 2 passes with nz1 = 2 (H3) |
| layer #4 T=32 **Marlin** | GEMM1 / GEMM2 | 64.6 / 37.0 | 9 × 18.9 / 9 × 9.45 | **2.6 / 2.3** | Marlin's cp.async pipeline, M-block re-reads small at 80 routes |
| layer #10 T=256 Marlin | pair | 510 | 21 × 28.3 MB | 1.2 | M-block re-reads (10+ blocks per hot expert) + 1 CTA/SM |
| sweep [16] m=128/256 | largem pair | 288 / 396 | 23 × 28.3 / 24 × 28.3 | 2.3 / 1.7 | harness (prepare) seeds |
Shared geometry of our two kernels [11]: per lane per prefetch batch 4 × uint4 + 4 × uint2 = 96 B → **3 KB per warp in flight** (one batch
consumed while one is in flight); `__launch_bounds__(NW*32)` without a min-blocks hint → 162–180 regs; smem 67 KB (largem, sized for 32 rows
× (Kc+8) × 2 B regardless of mc) → **2 CTAs × 4 warps or 1 CTA × 8 warps = 8 warps/SM**. In-flight bytes per SM ≈ 24 KB → ≈ 3.2 MB on 132 SMs.
Little's law at ≈ 0.7–0.9 µs loaded DRAM latency (Hopper far-miss ≈ 620–740 cycles, https://arxiv.org/html/2501.12084v1) gives 3.5–4.5 TB/s
*if every warp streamed all the time*; the CTA prologue (staging ≤ 64 KB of activations through 128 threads per chunk: a serialized L2 chain
of 32 uint4 per thread), the split-K fixup tail, cold TLB on a 1.5 GB weight bank and per-expert wave quantisation in smallm cut the duty
cycle to ≈ 40–60 %, which lands on the observed 1.6–2.3 TB/s. Compute is not it: ≈ 51 M warp-instructions for GEMM1 bs256 (est.) ≈ 0.6
instr/clk/SM out of 4; mma.sync at ≈ 0.08/clk/SM against ≈ 0.6 available even at Hopper's 63 % mma.sync ceiling
(https://arxiv.org/html/2402.13499v1, https://hazyresearch.stanford.edu/blog/2024-05-12-tk). The judge sweep [16] varied nw, mr, ks, nz:
every variant has the same 3 KB/warp and 8 warps/SM → the ≤ 2 % result is *predicted* by H1, not evidence against it.

### 2. Level-2 stamps (C20) — locations by function / loop, all `%globaltimer` (ns), tid 0 writes `[grid] × 8 u64` + smid
In `w4_largem_kernel` (and the same six in `w4_smallm_kernel`'s z-loop):
- **t0** entry after `pdl_wait` (first statement of the z/chunk loop body, before `prefetch_batch_l`);
- **t1** after the first `__syncthreads()` + lut copy (before the staging `for (idx …)` loop);
- **t2** after the staging loop's `__syncthreads()` → staging = t2−t1;
- **t3** in `chunk_gemm` after the `for (b …)` k-loop, before `if (KS > 1)` → stream = t3−t2 (bytes known: NW·MR·nit·576 B);
- **t4** after the fixup's second `__syncthreads()` (s_flag known) → arrival wait = t4−t3;
- **t5** last CTA only, after the partial `ldcg4` loop → fixup read = t5−t4;
- **t6** after the write loop → epilogue = t6−t5; also per-SM concurrency via an `atomicAdd(&occ[smid], ±1)` at t0/t6 with running max.
Printed per shape (≤ 12 lines, first in the log [A12]): kernel span (min t0 → max t6) and medians/p90 of each phase, per-CTA stream TB/s, max
concurrent CTAs per SM, chunks/distinct/n_act per seed, and for PDL kernels the end−previous-end time (H4).
```cuda
// stamp helper (device): one u64 per phase per CTA; tid 0 only; no effect on numerics
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define STAMP(k) do { if (LM_STAMPS && threadIdx.x == 0) stamps[(cta_linear_id) * 8 + (k)] = gtimer(); } while (0)
// pure-stream twin: same loads, no dequant / mma; fold prevents dead-code elimination
#if LM_NOCOMPUTE
  sink ^= w[r][d].x ^ w[r][d].y ^ w[r][d].z ^ w[r][d].w ^ s[r][d].x ^ s[r][d].y;   // replaces unpack/lut/mul/mma in do_batch_l
#endif
```
Twins in the same check (template/flag variants, ≤ 12 instantiations so nvcc stays < 2 min): {v4 as is, NOCOMPUTE, LB(128,3), LB(128,4),
D=4 third buffer (MR=2,D=2 → w0/w1/w2), LM_TC=64 with KS=8} × GEMM1/GEMM2 × bs 32/128/256 × seeds 11–13, kernel_fn = v4 (board).

### 3. Decision table (fires crow.m1-decision)
| observation | meaning | next build |
|---|---|---|
| NOCOMPUTE ≥ 0.85 × full time | memory pipeline is the cap (H1 true) | C21 ring (below) |
| NOCOMPUTE ≤ 0.55 × full | dequant/mma/ldmatrix ordering stalls the stream | C21 ring **with producer/consumer warps** (1 producer warp issues cp.async for 3 consumers) |
| LB(128,4) or D=4 alone ≥ +40 % | occupancy/depth alone suffices | ship as v5 same day (Alt A); ring continues only if < 3 TB/s |
| staging (t2−t1) > 25 % of (t6−t0) | activation staging dominates | ring + stage-once: CTA owns (chunk, ks) and loops over n-blocks |
| arrival+fixup (t6−t3) > 15 % | split-K tail | KS 4→2 (Kc 2048 → act smem 128 KB → ring stages must shrink) or FP32 atomic epilogue |
| max concurrent CTAs/SM = 1 | register/smem cap | LB min-blocks + act smem sized by NT bucket (3 launches by mc class) |
| n_act > nz passes visible (smallm) | wave quantisation (H3) | flatten z, early exit |

### 4. C21 — the ring kernel (what Sonnet-B builds; Argon reviews)
Keep: route kernels, chunk table, tile word layout, `unpack_word`/lut/`mul_bf16x2`, `mma16816`, split-K fixup order, SwiGLU epilogue
(→ bitwise equality V1). Change only how weight bytes arrive:
- per warp a ring of S stages × (MR·D) tiles of 576 B (512 B words + 64 B scales) in **shared memory**, filled with `cp.async.cg.shared.global`
  16 B per lane (sm_80+; NVRTC-compilable for syntax on the dev box), `cp.async.commit_group` per stage, `cp.async.wait_group S-2` before
  consuming; the dequant reads its word with one `ld.shared.v4` (and the scale pair with `ld.shared.v2`) — +2 LSU ops per tile, no registers.
- S = 6 → 6 × 1152 B (MR=2, D=1 tile each… choose D·MR = 2 per stage) ≈ 7 KB per warp, 28 KB per 4-warp CTA → in flight ≈ 5× today.
- smem budget: act 64 KB (NT=4) + ring 28 KB + lut 1 KB = 93 KB → 2 CTAs/SM (Hopper 227 KB); regs drop ≈ 48 (no w0/w1/s0/s1) → with
  `__launch_bounds__(128, 2)` safe; a 3-CTA variant uses Kc=512 (KS=8, act 32 KB).
- stage-once option (if staging > 25 %): grid (chunks, KS) persistent over n-blocks; act staged once; weights streamed for all 8192/4096 rows.
- Hopper-only twin (if time allows, C21b): `cp.async.bulk.shared::cluster.global` 1-D + mbarrier `expect_tx`, one elected thread per warp issues
  2–4 KB per stage (PTX ISA 8.0+); fewer issue slots, same ring. Not testable locally at all → second priority.
```cuda
// ring skeleton per warp (S stages, T = tiles per stage); addresses as today's wp/sp; consumed by do_batch from smem
for (int s = 0; s < S - 1; ++s) { issue_stage(s, s); }                 // prologue: cp.async 16 B/lane × T, commit_group
for (int b = 0; b < nbatch; ++b) {
    asm volatile("cp.async.wait_group %0;" :: "n"(S - 2));            // stage b has landed (this warp's own groups)
    __syncwarp();
    do_batch_smem<NT, MR>(ring + (b % S) * STAGE_BYTES, ...);        // ld.shared.v4 word, ld.shared.v2 scales, same dequant + mma
    __syncwarp();
    if (b + S - 1 < nbatch) issue_stage((b + S - 1) % S, b + S - 1); // refill the slot just consumed
}
asm volatile("cp.async.wait_group 0;");
```
Predicted (est.): weight stream 3.2–3.8 TB/s → GEMM1 bs256 322 → 150–185 µs, GEMM2 160 → 75–95; bs128 230/120 → 115–135/60–70; bs32
147/81 → 75–90/40–50; smallm GEMM1 at M=16 48 → 28–32, GEMM2 27 → 16–18 (real, end-to-end). Shape speedups (judge medians scaled):
bs1 1.86 → 2.3–2.7 (with H3+H9 → ≈ 3.0), bs8 1.46 → 1.9–2.3, bs16 1.39 → 1.8–2.2, bs32 1.10 → 1.4–1.7, bs128 1.48 → 1.9–2.3,
bs256 1.43 → 1.85–2.3 → **moe 1.436 → 1.83 (pess.) / 2.2 (cons.) / 2.5 (opt.)**; leader 2.52.

### 5. Small-M specifics (Q3)
- GEMM2 is not slower than GEMM1 (H4). Real small-M rates ≈ 1.8–2.1 TB/s on both.
- H3: SMALLM_CFG nz1=2 at M ≤ 1 means n_act = 3 → two passes of ≈ 13 µs; one wave can only hold ≈ 132 CTAs × 8 warps ≈ one expert's
  GEMM1 (128 CTAs) at today's register footprint, so the honest fix is the ring (fewer regs → 2 CTAs/SM) + z flattened to ≥ n_act.
- Epilogue 6.5 µs at bs1 is one Triton CTA doing two passes over 8 × 4096 (H9): grid tokens × 4 (each CTA recomputes `sumsq` for its 8 slots
  over the full row from L2, 64 KB, then writes its 1024 columns) → ≈ 3 µs; later fuse into GEMM2's last-CTA fixup with a per-token counter
  (≈ −5.5 µs) once the ring is stable.
- Target: small-M GEMMs at ≈ 3 TB/s → bs1 ≈ 1.4 route + 11 + 6 + 3 (epi) ≈ 22 µs (est.) vs 41.4 → 3.5× SGLang; bs8/16 ≈ 2.2×.

---
## D4. decode_layer plan
### 1. Port order and gates
P1 (C19, day 1): `_full_cat_kernel` + `_plan_full` + ROWMAJOR reducer from attention v3 into the layer for full layers with q_len = 1 (#1, #3,
#5); verify-full shapes keep v2's `_attn_kernel` (v3's `_full_kernel` is not faster: 181/355 vs 181/337 [03]). Sure gain: −4.4 / −60.7 / −130 µs.
P2 (C24, after C21): MoE path per T bucket chosen in prepare() by timing {Marlin+S1, largem v4, ring} on 3 seeds of layer-style routing
(2.5 routes/token, skewed) — the ring is expected to win every bucket; v4 largem wins only T=128/256 (≈ 320/400 µs vs 408/522 incl. glue,
est.) and loses T=32 (Marlin 109 µs vs ≈ 160). The layer's `largem_route` must read `local_start` from the device tensor with the wrap rule.
P3 (C22/C25): glue items G1–G4 below.
### 2. Fusion / glue map (T=1 sliding budget from [03] #0: 125 µs span, 20 launches, 7.2 µs of gaps)
| id | item | launches | µs saved per shape (est.) | shapes | legal note |
|---|---|---|---|---|---|
| G1 | pooler `_CFG` (32,512,1,4,**4**) = 48 KB/stage → 144–192 KB smem = 1 CTA/SM on 128 SMs, and router (64,128,16,4,**4**) = 36 KB/stage → 144 KB: the 64 router CTAs queue on the 4 free SMs (22 µs vs 5.5 standalone; the 3.4 µs fill is stretched the same way). Fix: stages 2 on both (96 + 72 KB co-reside) or BK halved | 0 | −9 (T ≤ 16), −8..−11 (T ≥ 128: router 28–38 µs beside the cuBLAS pooler) | all 12 | config only |
| G2 | pre-expert RMSNorm(1+w): per-token rms in the route kernel, applied in GEMM1 staging (FP32 → one BF16 rounding) | −1 | −6.7 | all (after Marlin removal) | same rounding points as reference |
| G3 | unpooler T ≤ 8 via Triton GEMV (n=1024) into a zeroed [T,8192] with stride 8192; drop cuBLAS splitK + reduce + F.pad memcpy | −2 | −5.4 | #0,#1,#6,#7 | BF16 out as today |
| G4 | Triton epilogue grid tokens×4 (H9); later fused into GEMM2 fixup | 0 / −1 | −3 / −5.5 | T ≤ 16 | FP32 order unchanged |
| G5 | o-proj / qkv_gate single-pass GEMV | — | ≤ −3 | — | **skip**: bf16 streaming class refuted; cuBLAS at 2.5–3.5 TB/s |
| G6 | PDL on Triton launches | 0 | ≈ 0 | — | **skip** (refuted) |
Persistent whole-post-attention kernel: **no** (unchanged from fable-07: weeks of work, judge-only debugging). The 6-launch layer is reached
instead by G1–G4 + the ring: at T=1 the chain becomes qkv → attn → reduce → o → boundary → [pooler ‖ router+routing] → route(+rms) →
GEMM1 → GEMM2 → epilogue → unpooler → boundary = 13 launches, ≈ 5 µs of gaps (est.).
### 3. Predicted per shape (µs, est.; judge medians [04] as base)
| # | now | P1 port | + ring GEMMs (cons.) | + G1–G4 | load effect |
|---|---|---|---|---|---|
| 0 decode1 sl | 125.2 | 125.2 | 114 | 90 | decode1 1.365 → 1.51 → 1.90 |
| 1 decode1 full | 137.0 | 132.6 | 121.6 | 97.6 | |
| 2 decode16 sl | 193.7 | 193.7 | 166 | 147 | decode16 1.347 → 1.445 → 1.65 → 1.84 |
| 3 decode16 full | 340.4 | 279.7 | 253 | 234 | |
| 4 decode32 sl | 240.9 | 240.9 | 207 | 198 | decode32 1.138 → 1.273 → 1.445 → 1.50 |
| 5 decode32 full | 539.8 | 409.5 | 376 | 367 | |
| 6 verify1 sl | 178.9 | 178.9 | 140 | 116 | verify1 1.495 → 1.91 → 2.29 |
| 7 verify1 full | 186.1 | 186.1 | 147 | 123 | |
| 8 verify16 sl | 502.2 | 502.2 | 241 | 232 | verify16 1.053 → 2.0 → 2.09 |
| 9 verify16 full | 664.2 | 664.2 | 403 | 394 | |
| 10 verify32 sl | 681.2 | 681.2 | 329 | 318 | verify32 1.034 → 1.91 → 1.96 |
| 11 verify32 full | 1017.4 | 1017.4 | 665 | 654 | |
| **problem** | **1.227** | **1.266** | **1.73** | **1.92** | pess. path (GEMMs 1.3×, Marlin 2× at 128/256, 1.1× at 32, + P1 + G1 only): **1.62** |

---
## D5. attention and bf16 verdicts
**attention (1.689, 156/589):** do **not** reopen a builder lane. Reasons with numbers: verify-full #11 moves 607 MB in 355 µs (1.7 TB/s)
while decode-full #5 moves 606 MB in 187 µs (3.2 TB/s) [03]; the difference is work per key, not gather structure: 128 exp2 and 128 FP8→FP16
conversions per key (64 rows × 2 heads) → 303 M each at b32 ≈ 72 µs apiece on a 16/clk/SM pipe (est.; FA4's Hopper MUFU figure,
https://arxiv.org/abs/2603.05451) plus ≈ 60–78 µs of tensor work (77.6 GFLOP [05]) with 8 warps/SM to overlap them. So "full_cat structure for
verify" (lean 4) is a non-starter; the real levers are (i) P·V in FP8 (removes the V conversion, halves PV tensor time; **legality open, D9-Q1**;
≈ −30 % on #9/#11 → 1.689 → ≈ 1.86), (ii) FA3-style warp specialisation + exp split (H15, ≈ 2.1–2.3, 40 builder-hours, high risk — not in the
window). Cheap rider: ROWMAJOR sliding partials (reducer 5.5/8.1 → ≈ 2.5/3 µs on #8/#10 → +1.2 %). One bundled check (C26) on Oct 30;
promote only if ≥ 1.70 (noise floor ±1.5 %). Under A1 the whole lane is worth ≈ +0.5–2 % overall; under rank-sum, one step more.
**bf16 (1.156, 549/796):** cuBLAS and our GEMV sit at 69–71 % of the byte floor (≈ 3.3–3.4 TB/s) [03]; nothing beyond the cheap fixes is
worth a check in this window (streaming/full-M class refuted). Cheap fixes (Argon §1.4 + one more): unpooler m 1/8 `_CFG` (6.8/7.2 → ≈ 3.3),
router last-CTA fixup without `torch.zeros`/atomics (6.4–7.4 → ≈ 5), router m=128/256 single M-block (9.9/12.3 → ≈ 6) → loads decode1 1.117
→ ≈ 1.23, verify1 1.394 → ≈ 1.54, others +2–5 % → **1.156 → ≈ 1.22** (est.), 2 checks + 1 submit; this submit doubles as the A1 calibration.

---
## D6. Leave behind (stop, reason, evidence that would reopen)
1. Single-launch sliding attention — refuted twice (v4 1.509, v4b 1.457); reopen only with a trace showing reducer+gap > 40 % and a merge
   design without a global counter spin.
2. Grouped Triton W4A16 for large M — 0.970; superseded.
3. Marlin knob tuning / Marlin as a target — ≤ 2 %; keep only as fallback (M > 256, self-check failure).
4. largem/smallm (nw, mr, ks, nz) sweeps on the present pipeline — ≤ 2 %; one re-sweep after the ring lands, then stop.
5. "Hot-expert re-read" as the primary moe explanation — 12 % of bytes at bs256 [16]; revisit only if stamps show stream TB/s ≥ 3.5 and
   span still > 1.3 × bytes/3.5 TB/s.
6. S2 fused routing, PDL on Triton, synthetic-routing sweeps, routing-hint prefetch, draft cascade (kernel track) — as [08].
7. Attention verify-full via `full_cat` structure — compute-bound (D5); reopen only as H15 with a dedicated builder for 4 days.
8. bf16 beyond `_CFG`/fixup — parked until Nov 5.
9. decode_layer S2/PDL/megakernel — parked; the 13-launch layer via G1–G4 covers the gap.
10. Attention decode-full deeper pipelines — at 3.0–3.2 TB/s; reopen only below 2.5 TB/s on judge seeds.

---
## D7. 10-day schedule (UTC days; checks ≤ 20 and submits ≤ 10 per problem per day — plan uses ≤ 5 / ≤ 2)
| day | moe | decode_layer | bf16 | attention | builders / lanes | decision point |
|---|---|---|---|---|---|---|
| Oct 25 | C20 M1 measurement (1–2 chk) | C19 v5 port: check + promote (2 chk, 1 sub); C22a G1 measurement (1 chk) | C23 measurement (1 chk) | — | Sonnet-A: C19→C22a; Sonnet-B: C20; Jules: stamp parser + end-to-end PDL report; Grok Bot: red team C19 | evening: crow.m1-decision |
| Oct 26 | C21 ring: build, NVRTC syntax, first judge compile + bitwise self-check (2 chk) | C22b v5b = v5 + G1 + G3 + G4: check + promote (2 chk, 1 sub) | C23 promote (1 sub, **isolated** → A1) | — | Sonnet-B: C21; Sonnet-A: C22b; Argon: ring review; Grok Build: epilogue grid draft | A1 calibration recorded |
| Oct 27 | C21 sweep stages/KS/LB (2 chk) → promote if > board (1 sub) | C24 measurement: Marlin vs v4 largem vs ring on layer routing (1 chk) | — | — | Sonnet-B: C21 tuning; Sonnet-A: C24 harness + layer largem route (device local_start) | 18:00Z crow.streaming-gate |
| Oct 28 | smallm on the ring + H3 flatten + H9 (2 chk, 1 sub) | C24 v6: ring for all T, per-bucket gate: check + promote (2 chk, 1 sub) | — | — | Sonnet-B: smallm ring; Sonnet-A: v6 | if v6 < 1.45: Argon diff review before anything else |
| Oct 29 | re-sweep configs once (1 chk) | C25 G2 norm fusion (+ V3): check + promote (2 chk, 1 sub) | — | — | Sonnet-B: G2 (CUDA route+staging); Sonnet-A: trace of v6 (make_trace) | 21:00Z: rank review; decide serving start date |
| Oct 30 | epilogue fused into GEMM2 fixup (1–2 chk) | trace-driven residual (1 chk) | — | C26 bundled check; promote if ≥ 1.70 (1 chk, ≤ 1 sub) | Sonnet-A: C26; Sonnet-B: epilogue fusion | A8 answer needed before C26's FP8 half |
| Oct 31 | buffer / fixes | buffer / fixes | — | — | Jules: C27 S0 (diff_rules, packaging, smoke test); Sonnet idle-time: regressions | crow.serving-s0 |
| Nov 1 | layer epilogue fusion port (1 chk, ≤ 1 sub) | same | — | — | Sonnet-A/B | |
| Nov 2 | hold-out robustness: run each board kernel once more (1 chk each) | same | 1 chk | 1 chk | Grok Bot: X scout for public Hopper W4A16 streaming posts (ideas only) | |
| Nov 3 | **freeze** new kernels (crow.freeze); promote only already-judged | same | — | — | director: handover, BEVINDINGEN | |
| Nov 4 | last submits ≤ 18:00Z; verify board rows | same | — | — | — | Nov 5 07:59Z close |
Dispatch order for builders: Sonnet-A = C19 → C22a → C22b → C24 → C26; Sonnet-B = C20 → C21 → smallm ring → G2 → epilogue fusion.
Capacity (est.): ≈ 30 checks moe, ≈ 25 decode_layer, 4 bf16, 3 attention = ≈ 62 of 800 available; the binding constraint is ≈ 120 builder-
hours (A10): ring 35, layer v5/v6 20, glue 16, measurement harnesses 10, bf16 4, attention 6, buffer 29.

---
## D8. Serving track (Nov 5 noon PT – Nov 19)
Start **S0 on Oct 31** (crow.serving-s0), CPU-only, Jules + director, zero judge checks and zero builder GPU time — it does not compete with the
kernel track, and the kernel-track kernels (ring MoE GEMM, attention v3, Triton projections) are the serving assets (c=16/32 verify batches
are literally our verify shapes [09] D5). First three things to build: (1) `tools/diff_rules.py` (paths under python/sglang only, no model_loader,
≤ 720 KB, line ≤ 400, no binaries) + JIT packaging of our kernels as `python/sglang/kernels/...` modules with import-time fallback to stock;
(2) a CPU smoke test that imports the patched modules and shape-checks every kernel path against the model config constants (H=8192,
I=4096, 16/2 heads, window 2048, 8 routed experts); (3) the server-args defaults patch (DFlash on, CUDA-graph max bs covering 32 × 9 tokens,
chunked-prefill size, overlap scheduler, page size [Q3 of fable-07]) plus the per-load step simulator — and the question to the organisers
whether launch flags are fixed (fable-07 Q2). Build order after Nov 5 as fable-07 D5 (A/B/noise triplet per day). If the kernel board reaches the
top 50 (priority), nothing changes in this plan; if not, S0 still starts Oct 31.

---
## D9. Open questions and assumptions with the cheapest test
| id | question / assumption | cheapest test | owner |
|---|---|---|---|
| Q1 (A8) | Is P (softmax probabilities) in FP8 E4M3 for P·V legal? Baseline calls FA3 with FP8 q and FP8 KV; FA3's FP8 path quantises P to e4m3 (https://arxiv.org/abs/2407.08608). Rules name only queries [10]. | read baseline.py's FA3 call and FA3's fp8 kernel source in the image; ask organisers in parallel; do not build before yes | Dex |
| Q2 (A1) | overall board formula | isolated bf16 submit (C23) → overall delta; compare with time-weighted (bf16 ≈ 12 %) vs rank-sum predictions | director |
| Q3 (A3/A6) | H1 and the PDL artifact | C20 (one check) | Sonnet-B |
| Q4 (A4) | does the ring reach ≥ 3.2 TB/s on 1–2 KB per-warp contiguous streams? DRAM row locality with ≈ 2000 concurrent streams is the unknown | C21 sweep; the NOCOMPUTE twin with S=8 is the ceiling measurement | Sonnet-B |
| Q5 (A7) | layer routing E/routes per T on the judge | print distinct/chunks in C24's measurement | Sonnet-A |
| Q6 | NVRTC on the dev box: can it compile `cp.async` (sm_80) and `cp.async.bulk`/mbarrier (sm_90) PTX for syntax? | `nvrtcVersion` + a 10-line compile on the 2060 box | Jules |
| Q7 | Triton version in the image (ROWMAJOR/PDL irrelevant now; needed for `tl.dot` fp8×fp8 with fp32 acc on sm_90 for H12) | `print(triton.__version__)` is already in the v3 log tail? else one line in C26 | Sonnet-A |
| Q8 (A9) | pooler CTA smem = 48 KB/stage × 4 stages (BN 32, BK 512) → 1 CTA/SM | `-Xptxas -v`-equivalent: Triton `kernel.metadata.shared` printed in C22a | Sonnet-A |
| Q9 | does `prepare()`-time A/B selection count as "branching on config" (yes: shape-only, decided before timing) | state it in the submission header; Grok Bot red team | director |
| Q10 (A12) | 16 KB tail holds C20's summary | print summary first; T5 rules | Jules |
| Q11 | judge reruns on "nearby shapes": per-bucket gates must be monotone in T (no cliff at 16/17 or 256/257) | C24 prints the chosen path for T ∈ {1..8,16,17,32,33,128,256,257} | Sonnet-A |

---
## Appendix A. Arithmetic behind the moe predictions (so the director can re-derive them)
Base = judge medians [04]; GEMM share = span minus route/epilogue from [03] (PDL-corrected for GEMM2). "fixed" = route + epilogue + gaps.
| shape | sglang | ours | fixed (est.) | GEMMs (est.) | ×1.3 → ours / speedup | ×1.6 → ours / speedup | ×1.9 (+H3/H9 at bs1) → ours / speedup |
|---|---|---|---|---|---|---|---|
| #0 bs1 | 76.8 | 41.4 | 7.9 | 33.5 | 33.7 / 2.28 | 28.9 / 2.66 | 22.5 / 3.41 |
| #3 bs8 | 142.3 | 97.6 | 5.2 | 92.4 | 76.3 / 1.87 | 62.9 / 2.26 | 53.8 / 2.64 |
| #1 bs16 | 194.6 | 140.1 | 7.5 | 132.6 | 109.5 / 1.78 | 90.4 / 2.15 | 77.3 / 2.52 |
| #2 bs32 | 204.1 | 186.2 | 7.4 | 178.8 | 145 / 1.41 | 119 / 1.72 | 101.5 / 2.01 |
| #4 bs128 | 465.9 | 314.0 | 11 | 303 | 244 / 1.91 | 200 / 2.33 | 170 / 2.74 |
| #5 bs256 | 622.1 | 434.2 | 16 | 418 | 338 / 1.84 | 277 / 2.25 | 236 / 2.64 |
| **geo-mean** | | **1.436** | | | **1.83** | **2.21** | **2.63** |
Sanity against the leader (2.52): the ×1.9 column would put us at the top of the moe board, so treat ×1.6 (≈ 3.3–3.6 TB/s on the weight
stream) as the planning number and ×1.3 as the floor the ring must clear to be worth promoting (CI-low > 1.436 is already met at ×1.1).
GEMM-time multipliers translate to stream rates (est.): today ≈ 1.6–2.1 TB/s → ×1.3 ≈ 2.1–2.7, ×1.6 ≈ 2.6–3.4, ×1.9 ≈ 3.0–4.0 TB/s.

## Appendix B. decode_layer launch sequences, before → after (T=1 sliding, [03] #0; µs)
Before (20 launches, span 125.3): qkv nvjet 12.6 + splitK-reduce 1.8 | attn 5.0 + reduce 2.4 | o nvjet 6.6 | boundary norm_residual 4.7 |
main: pooler `_gemv_splitk` 19.2 ‖ side: fill 3.4 → router `_gemv_splitk` **22.0** → routing 2.9 (side ends at 65.6, pooler at 55.3: **the side
stream is the critical path by ≈ 10 µs**) | layer_norm 6.3 | smallm_route 1.5 | GEMM1 16.9 | GEMM2 12.6 (real) | epilogue 5.9 | unpooler
nvjet 5.2 + splitK-reduce 1.7 | fill 0.8 + memcpy 0.9 (F.pad) | boundary 4.0 | ≈ 19 gaps × 0.38 = 7.2.
After P1 + ring + G1–G4 (13 launches, est.): qkv 12.6 + 1.8 | attn 5.0 + 2.4 | o 6.6 | boundary 4.7 | pooler 19.2 ‖ (router ≈ 5 → routing 2.9,
hidden) | route(+rms) 2.0 | GEMM1 ≈ 10.5 | GEMM2 ≈ 7.5 | epilogue ≈ 3 | unpooler Triton ≈ 3.2 (strided into zeroed buffer) | boundary 4.0 |
gaps ≈ 12 × 0.4 = 4.8 → **≈ 87–92 µs** (D4 table says 90). The remaining big items are then cuBLAS qkv (14.4 incl. reduce, floor 7.9) and the
pooler GEMV (19.2, floor 14.0) — bf16-class problems we have decided not to chase in this window.
T=256 sliding ([03] #10, span 683): the MoE block is Marlin 335 + act_and_mul 5.7 + Marlin 175 + align/count/ids 6.8 = 522 of 683; the ring
at ≈ 3.5 TB/s on E ≈ 21 experts (602 MB) is ≈ 172 + route 11 ≈ 183 → shape ≈ 344 (D4: 329 with G1); the rest is cuBLAS qkv 19.7, attention
17.1 + 7.8, pooler nvjet 33.5 ‖ router 38.4 (side, stretched; G1 → ≈ 12), unpooler 25.2, norms 8.5 + 7.5 + 6.2, routing 3.1.

## Appendix C. Task cards (schema harness/taskcard.yaml; done-when = commands the runner can execute)
- **C19 layer v5 (full_cat port)** — files: kernels/muse_decode_layer/sonnet_v5.py (= v4 + `_full_cat_kernel`, `_gather_cat`, `_step_cat`,
  `_plan_full`, ROWMAJOR reducer flag, FULL_CFG["decode"]; verify-full keeps v2 `_attn_kernel`). Done-when: `pytest tests/decode_layer -k
  "full and q_len1"` green locally (CPU interpreter / 2060 software FP8), V2 ≤ 2e-3 vs v4 on seeded inputs; `okc check` ≥ 1.255 with all 13
  tests passing; `tools/promote.py muse_decode_layer sonnet_v5.py --predict 1.266 --hyp "H5 port"`.
- **C20 moe M1 (stamps + twins)** — files: kernels/muse_moe_w4a16/measure_stamps_v1.py (kernel_fn = v4 verbatim; prepare() builds the
  stamped/twin variants, times GEMM1/GEMM2 per variant on generate_input routing bs 32/128/256 × seeds 11–13, prints ≤ 12 lines per
  shape, summary first). Done-when: NVRTC syntax compile of the device source for sm_80 passes on the dev box; `okc check` passes (speedup
  ≈ 1.436) and the log tail holds `M1 …` lines for 6 variants × 3 shapes; `tools/parse_stamps.py` (Jules) emits the D3 §3 table.
- **C21 ring kernel** — files: kernels/muse_moe_w4a16/sonnet_v5.py (+ measure_ring_v1.py). Done-when: V1 bitwise equality vs v4 for
  M ∈ {1,4,8,16,32,33,128,256,257} × 3 seeds inside prepare() (printed `ring bitwise=OK`); `okc check` of the measurement file shows GEMM1
  bs256 ≤ 230 µs for some stage count; promote when CI-low > 1.436.
- **C22 layer glue a/b** — a: measurement variant table for pooler/router `_CFG` smem (G1), reading stream-1 GEMV end−start; b: v5b with
  G1 + G3 (Triton unpooler, `out.stride(0)=8192`, `torch.zeros((T,8192))`) + G4 (epilogue grid tokens×4, V4 bitwise). Done-when: router
  GEMV ≤ 8 µs beside the pooler at T=1; `okc check` ≥ v5 + 0.03; promote.
- **C23 bf16 cheap** — `_CFG[("unpooler",1)] = _CFG[("unpooler",8)] = (32,256,1,4,4)`; router fixup kernel (counters in prepare(), last CTA
  reduces SPLIT partials in slice order and resets); router m=128/256 with BM=m. Done-when: measurement check shows #3/#18 ≤ 3.8 µs,
  #4/#9/#14/#19 ≤ 5.5, #24/#29 ≤ 7; promote ≥ 1.19 as an **isolated** submit (no other submit within 2 h) for A1.
- **C24 layer v6 (ring for all T, per-bucket gate)** — prepare() times {Marlin+S1, ring} per bucket on 3 seeds of layer-style routing
  (2.5 routes/token, skewed via the self-check generator with ROUTER_ALIGN-like concentration), chooses per bucket, prints the choice for
  T ∈ {1..8,16,17,32,33,128,256,257} (Q11); `largem_route` reads `local_start` from the device. Done-when: 13 tests pass; `okc check` ≥ 1.55;
  promote.
- **C25 G2 norm fusion** — route kernel computes per-token `inv = rsqrt(mean(x²)+eps)`; GEMM1 staging stores `bf16(x·inv·(1+w))`
  (FP32 math, one rounding). Done-when: V3 (≤ 1 BF16 ulp vs torch reference) and `okc check` ≥ v6 + 0.02; promote.
- **C26 attention bundled** — ROWMAJOR partials in `_attn_kernel` (pidx [b,h,row,split]) + reducer ROWMAJOR=True for sliding; **if Q1 = yes**
  `_step`: `acc = tl.dot(p.to(tl.float8e4nv), vu.to(tl.float8e4nv, bitcast=True), acc*alpha)` with FP32 acc for verify-full. Done-when: 16 tests
  pass; `okc check` ≥ 1.70; promote only above CI.
- **C27 serving S0** — D8 items (1)–(3); done-when: `python tools/diff_rules.py <diff>` passes on a diff that adds our JIT modules; CPU smoke
  test imports and shape-checks every path.

## Appendix D. Testing discipline (brief Q8): the cheapest judge test per class and how to batch
1. **Measurement check** (kernel_fn = board kernel verbatim; prepare() does the science): one check can hold ≈ 8–12 CUDA template
   instantiations (nvcc ≈ 5–10 s each) or ≈ 10–15 Triton configs (≈ 10–20 s each) × shapes × 3 seeds, inside prepare ≤ 300 s and run ≤ 15 min.
   Rules: `do_not_specialize` on sizes; compact lines (`B <id> m=<M> seed=<s> distinct=<E> chunks=<C> | <variant> g1=<µs> g2=<µs> …`),
   summary first, warnings silenced (16 KB tail); pivot = the board kernel in every round; score = median over seeds of pivot/variant, with
   the per-round pivot deviation subtracted (fable-07 D4-2); a difference inside the bootstrap CI is "noise", never "better".
2. **Trace check** (tools/make_trace.py): after a candidate passes correctness and before promote; read end−previous-end for PDL kernels;
   compare against the predicted per-shape µs line in the candidate's header (the A2 watcher).
3. **Promote** (tools/promote.py): only with a `--predict` and `--hyp`; refuses measurement files; CI-low rule.
4. **Batching hypotheses**: one measurement check per problem per day carries every variant whose change is local to one kernel or one
   config line (H1/H3/H4 together in C20; G1/G3/G4 together in C22a; H11/H12 in C26). Hypotheses that change numerics (G2, H12) get their own
   vector in prepare() first (V3, checks.py bound) and never share a check with a config sweep, so a failure is attributable.
5. **Kill thresholds**: a kernel-level hypothesis is killed when its best variant is < +8 % on the targeted kernel after one measurement check
   (noise ±1.5 % on the paired ratio, ±5 % on routing seeds), or when the judged speedup after promote is inside the CI of the previous
   board entry twice. A composition hypothesis (C19, C24) is killed when judged < predicted − 0.02 twice.
6. **Daily cap**: ≤ 5 checks per problem on build days, ≥ 70 % on decode_layer + moe through Oct 30; the remaining 15 checks are reserve
   for compile failures (a CUDA build error costs a full check) and for the Nov 2 hold-out re-runs.

## Appendix E. Attention verify-full: the per-key work that caps it (est.; b32 verify-full #11, 2.37 M key positions)
| pipe | per key (both KV heads) | total | floor at Hopper rates (est.) |
|---|---|---|---|
| tensor QK (FP8) | 2 × 64 rows × 64 d MACs | 38.8 GFLOP | ≈ 20 µs (FP8 1979 TFLOP/s dense) |
| tensor PV (FP16 today) | 2 × 64 × 64 MACs | 38.8 GFLOP | ≈ 39 µs (BF16/FP16 989) → ≈ 20 with FP8 P |
| MUFU exp2 | 128 | 303 M | ≈ 72 µs at 16/clk/SM |
| FP8→FP16 V convert | 128 | 303 M | ≈ 72 µs if 16/clk/SM → **0 with FP8 P·V** |
| FP32 softmax bookkeeping | ≈ 5 × 128 | 1.5 G | ≈ 45 µs at 128/clk/SM |
| HBM bytes | 256 B | 607 MB | 126 µs at 4.8 TB/s |
Measured 355 µs ≈ the sum, not the max → the pipes barely overlap at 8 warps/SM. Decode-full has 16 rows per pair (8× fewer exps and
converts per key) and sits at the HBM floor ×1.5. Hence D5: structure port ≠ fix; FP8 P·V removes one 72 µs pipe and halves another (≈ −30 %);
full overlap needs warp specialisation (H15).

## Appendix F. Snippets for the cheap items (each ≤ 25 lines)
```python
# G4 / H9: Triton epilogue grid (tokens, 4); each CTA recomputes sumsq over the full row (64 KB from L2) and stores its 1024 columns
@triton.jit
def _epilogue4_kernel(y_ptr, w_ptr, s_ptr, ids_ptr, out_ptr, eps, TOPK: tl.constexpr, KP: tl.constexpr, H: tl.constexpr,
                      BLOCK_H: tl.constexpr, NCHUNK: tl.constexpr):
    tok, part = tl.program_id(0), tl.program_id(1)
    slots = tl.arange(0, KP); in_k = slots < TOPK
    live = in_k & (tl.load(ids_ptr + tok * TOPK + slots, mask=in_k, other=-1) >= 0)
    s = tl.load(s_ptr + tok * TOPK + slots, mask=live, other=0.0).to(tl.float32)
    row = (tok * TOPK + slots)[:, None] * H
    sumsq = tl.zeros([KP], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):                       # full-row pass: identical FP32 order to the 1-CTA kernel (V4 bitwise)
        offs = c * BLOCK_H + tl.arange(0, BLOCK_H)
        w = tl.load(w_ptr + offs).to(tl.float32)
        x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None], other=0.0).to(tl.float32) * w[None, :]
        sumsq += tl.sum(x * x, axis=1)
    coef = tl.where(live, s * tl.rsqrt(sumsq / H + eps), 0.0)
    offs = part * (H // 4) + tl.arange(0, H // 4)            # this CTA's quarter of the output row
    w = tl.load(w_ptr + offs).to(tl.float32)
    x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, (tl.sum(x * coef[:, None], axis=0) * w).to(out_ptr.dtype.element_ty))
```
```python
# G3: unpooler at T <= 8 without F.pad: Triton GEMV writes the 1024-column slice straight into a zeroed [T, 8192] (stride_om = 8192)
ffn_output = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)           # one fill (0.8 us) replaces fill + memcpy + splitK-reduce
_gemv_splitk[(UNPOOL_SLICE // 32, 1, 1)](expert_output, dense["unpooler"], dense["unpooler"], ffn_output, tokens, UNPOOL_SLICE, D_EXPERT,
    expert_output.stride(0), dense["unpooler"].stride(0), HIDDEN, BM=16, BN=32, BK=256, SPLIT=1, HAS_LO=False, ATOMIC=False, SPLIT_X=False,
    num_warps=4, num_stages=4)
```
```python
# G1: co-residency (config only). Today ("pooler",1) = (32, 512, 1, 4, 4) and ("router",1) = (64, 128, 16, 4, 4) each need > 114 KB smem.
_CFG[("pooler", 1)] = (32, 512, 1, 4, 2)    # 96 KB; measure also (32, 256, 1, 4, 4)
_CFG[("router", 1)] = (64, 128, 16, 4, 2)   # 72 KB; both CTAs now fit one SM (227 KB)
```
```python
# H12 (only after Q1 = yes): verify-full P.V in FP8, V as stored; FP32 accumulate; q.K already FP8 (QK8)
p8 = p.to(tl.float8e4nv)                                  # P in [0, 1]: e4m3 keeps 3 mantissa bits, same as FA3's FP8 forward
v8 = vu.to(tl.float8e4nv, bitcast=True)                   # no conversion of V (removes 303 M cvt at b32)
acc = tl.dot(p8, v8, acc * alpha[:, None])                # wgmma e4m3 x e4m3 -> f32 on sm_90
```

## Sources (public web; pack files cited by id above)
H200 4.8 TB/s, 989 TFLOP/s dense BF16: https://www.nvidia.com/en-us/data-center/h200/ · Hopper mma.sync ≈ 63 % of peak, latencies:
https://arxiv.org/html/2402.13499v1, https://hazyresearch.stanford.edu/blog/2024-05-12-tk, https://arxiv.org/html/2501.12084v1 ·
PDL semantics: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html,
https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py · Machete/Marlin on Hopper:
https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel, https://github.com/IST-DASLab/marlin ·
FA3 (FP8 P·V, warp specialisation): https://arxiv.org/abs/2407.08608 · FA4 exp emulation / MUFU rate: https://arxiv.org/abs/2603.05451 ·
Stream-K (persistent tile scheduling): https://arxiv.org/abs/2301.03598.
