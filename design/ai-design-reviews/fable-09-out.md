# fable-09-okc-method output
executor: Fable (one-shot) · 2026-10-25 · inputs: the 37-file pack (ids below) + public web (URLs in §Sources) · design only, nothing run.
Status line: **Advice (not adopted)** · harness itachi/v0.1 (D1 follows pack 07a §1–9) · round loop izanami/v0.1 (D2 follows pack 07b).
Conventions: a number with a pack id in brackets is measured ([01] = fable-08, [02] REPORT, [02b] board, [03*] Hotz, [04a] Perplexity,
[04b] GPT, [04c]/[04d] X scouts, [05-03] L1 traces, [05-05] bytes model, [05-07] run log, [05-08] failed classes, [05-10] rules,
[05-11..14] board kernels, [05-15] TASKs, [05-16] sweeps, [06a-c] images, [F1]/[F2] = the brief's two new measured facts).
**Every number marked `est.` is mine** (arithmetic on pack numbers or an engineering estimate). Shape ids are the judge's.
Exclusion: a web search returned an excerpt of a competitor's public challenge repo (scout [04d] names it); I did not open it and
nothing from it is used here (D10 Q12).

## 0. Headline (read this if nothing else)
1. **fable-08's diagnosis survives; its first build does not.** The time is still inside two CUDA GEMM kernels with ≈ 3 KB per warp in
   flight and ≈ 8 warps per SM [01 D3] — but the X evidence ("fix latency with more parallelism, never with prefetch tricks", +66 %
   regression on hand double-buffering [04c]) and Little's-law arithmetic (24 KB in flight per SM at ≈ 55 % duty cycle ⇒ 1.7 TB/s, est.;
   matches the measured 1.6–2.1 TB/s) say the *cheap* lever is occupancy + duty cycle (launch bounds, smaller activation smem, one
   staging per several n-blocks, flattened z), not a 6-stage cp.async ring. **Order: measure (C20 twins) → parallelism (Alt A, 1 builder-
   day) → ring (Alt B) only if the twins say the per-warp stream, not occupancy, is the cap.** Decision table in D4.
2. **[F1] (Triton 3.8 emits no cp.async for our plain-load K loops) matters least where fable-08 looked and most where nobody looked.**
   The pooler GEMV already runs at 3.6 TB/s (67 MB in 18.5 µs [05-03 bf16 #2]); the TMA-descriptor rewrite is worth ≈ −2 µs per pooler
   shape and ≈ −2.7 µs per qkv shape (est.). The kernel that *can* lose a lot is attention's `_full_kernel` (verify-full: 355 µs for 607 MB
   = 1.7 TB/s [05-03 attn #11]): its docstring claims a cp.async pipeline [05-13] and it was never checked. **Day-0 move: PTX/SASS audit of
   every Triton kernel we ship (0 judge checks)**; the attention verify twins (gather-only, no-exp) go into one measurement check only if the
   audit shows 0 `cp.async` there.
3. **[F2] kills fable-08's G1 mechanism (smem monopoly) but not its symptom.** The side stream is the critical path by ≈ 10 µs at T = 1
   [01 App. B] because two bandwidth-bound kernels on two streams do not overlap — they share HBM and the small one (router, 64 CTAs, 1.5 TB/s
   alone) gets the leftover; the 1 KB `fill` taking 3.4 µs beside the pooler (0.67 µs alone [05-03 bf16 #4]) is the same starvation.
   Fix: router *before* the pooler on one stream + a router GEMV with ≥ 256 CTAs (or fused into the pooler kernel as extra N-blocks).
   Config-level, ≈ −6.5 µs on six layer shapes and ≈ −8..−12 µs at T = 128/256 (est.). One measurement check decides (D4 §2).
4. **The largest sure lever is still in decode_layer and needs no new kernel:** v4's `w4_largem` replacing Marlin at T = 128/256
   (Marlin runs at 1.0–1.3 TB/s there: 398/510 µs for 520/602 MB [05-03 layer #8/#10, 05-05]). With largem at today's 2.0 TB/s:
   verify16 1.053 → ≈ 1.41, verify32 1.034 → ≈ 1.42 ⇒ problem 1.227 → ≈ 1.36; with C19 (full_cat port) ≈ 1.40 (est.). fable-08 parked
   this behind the ring; it moves to day 2.
5. **Attention:** keep fable-08's verdict (no builder lane for a structure port) but add the audit (item 2). FP8 P·V: **dropped from the
   10-day queue** (legality unverified; [04c] saw FP8 operands fail a 1.5 % tolerance in bf16_proj; attention's bound is 3–15 % [05-15] so
   numerics may pass — a zero-check CPU tolerance test + reading baseline.py's FA3 call stays as a continuity card, build only on a yes).
6. **bf16:** C23 cheap fixes (1.156 → ≈ 1.19–1.22) + the TMA GEMV for qkv/pooler at m ≤ 32 if the audit confirms the pooler loses
   pipelining (→ ≈ 1.25, est.). m = 128/256 stays cuBLAS ([04c]: cuBLASLt hard to beat at m ≥ 64; our own measure_v3 refutation [05-07]).
7. **Method:** okcdbg L2 (per-kernel GB/s, % of calibrated bandwidth, x_floor), L7 (regs/spills/smem + PTX/SASS histogram), D3/D4
   detectors and three ratchet tests, ≤ 1 builder-day (D5). Every check carries a predicted per-shape µs line; the round loop (D2) spends
   ≤ 5 checks per problem per day and kills a hypothesis on its second miss. Top move for the next 24 h: audit + C19 promote + C20 build.

---
## D1. itachi design: "measure, then parallelism, then the ring" (pack 07a §1–9)

### 1. True objective
**Overall kernel-board rank as high as possible at 2026-11-05 07:59Z** [05-10], every board entry legal and reproducible on the judge's
reruns (held-out data, nearby shapes; medians of reruns are final [05-10]). Observable: `okc leaderboard --json` overall rank; per problem
the judged speedup with CI-low above the previous entry (promote.py rule). Means test: a rank gained by promoting one already-built
candidate, or because the field thins, counts. Non-goals: beating any single leader; serving-track code before Oct 31; kernel elegance;
anything the rules call tampering [05-10 §Validation].

### 2. Constraints and failed solution classes
Hard limits: one account; per problem per day 10 submits + 20 checks ([05-10]; [04c] "2 checks left of 20"), ≈ 10 min each, 15-min run cap
(10 for bf16), prepare() ≤ 300 s, 16 KB log tail [05-15]; no Hopper locally (RTX 2060 sm_75: NVRTC syntax only; Triton 3.8 AOT compile for
sm_90 works offline — that is how [F1] was measured); ≈ 2 Sonnet builders under the claude-remote 4 GB cap (heavy jobs via systemd-run);
Argon / Grok Build drafts only; NVIDIA free models flaky ([02]: 504s all day); director usage limits. The judge image is fixed at SGLang
a55ab7d5 with Triton 3.8 [05-10, 04c]; the kernel baseline cannot move inside the track (serving baseline may, D9).
Failed classes (do not re-propose without the named new evidence; evidence ids):
| class | evidence | reopen only if |
|---|---|---|
| synthetic-routing sweeps as predictor | moe v2 1.098 vs 1.2 predicted; grouped_v1 0.970 [05-07, 05-08] | never; generate_input routing only |
| grouped Triton W4A16 GEMM, large M | grouped_v1 0.970 [05-07] | never (CUDA path superseded it) |
| single-launch sliding attention | v4 1.509, v4b 1.457 vs 1.689 [05-07] | trace shows reducer+gap > 40 % of a sliding shape AND a counter-free merge |
| deeper pipelines / more programs on decode-full attention | v3 @f68e4a3 1.475 [05-07] | decode-full < 2.5 TB/s on judge seeds (it is at 3.2) |
| bf16 Triton streaming / full-M | measure_v3 1.001 [05-07] | never in this window |
| routing-hint L2 prefetch | T1 bytes argument [05-08] | never |
| S2 fused routing (router GEMV + top-k in one launch) | v3 1.077 vs v3_s1 1.194 [05-07] | the fused kernel traced < 4 µs end-to-end |
| PDL on Triton launches | v3_s13 = noise [05-07] | never (keep the free attribute on CUDA kernels) |
| Marlin knob tuning / Marlin as target | v2 refuted; S1 knobs only [05-07] | never; Marlin = fallback (M > 256, failed self-check) |
| largem/smallm knob sweeps on the present pipeline | ≤ 2 % [05-16] | after the occupancy/pipeline change: one re-sweep, then stop |
| **"pooler smem monopoly" (fable-08 G1 as a mechanism)** | **[F2]: 24 KB / 12 KB per CTA, not 192 KB** | never as stated; symptom reassigned to bandwidth sharing (D1 §5, D4 §2) |
| **hand double-buffering inside Triton kernels** | [04c]: +66 % regression; [F1]: the compiler drops cp.async anyway | never: pipeline Triton loads via TMA descriptors or by parallelism |
| **FP8 operands for bf16_proj / router** | [04c]: 2.7 % error vs 1.5 % tolerance; our bf16 bound is 1 %, router 0.01 % [05-15] | never |
| **TMA `gather4` for KV rows on H200** | SM100+ only ([04a] §C, vLLM PR 42436) | never on sm_90 |
| megakernel / persistent whole layer | [01 D4] verdict; [04a] rank 10 "do last" | not in 10 days |

### 3. Environment / adversary
| what resists | cheapest early observation |
|---|---|
| the field moves up ([04c]: a participant improved 1.32 → 1.50 and fell #137 → #170; our overall 492 → 335 in one day [02b]) | daily `okc leaderboard --json` delta per problem (Grok Bot digest); plan by Δ-speedup, never by rank |
| judge queue near the deadline ([04d]: ≈ 3 h behind, ≈ 1 h per run reported) | `okc problems` + queue probe before each round; freeze new submits Nov 4 18:00Z |
| routing variance (moe bs16 p10–p90 112–173 µs [01 §3]; trace span 83 µs vs judge median 140 µs on #1 [05-03 moe, 01 App. A]) | every measurement prints `distinct/chunks/n_act` per seed, times ≥ 3 seeds, paired with the board kernel as pivot |
| our own harness overfits or lies (synthetic routing; compiler deletes the twin's work [03a A6]) | reference.generate_input only; every twin folds its loads into a sink that is stored; `B` lines print bytes/µs and refuse > 4.5 TB/s (P6 [03c]) |
| PDL early launch inflates CUPTI durations (fable-08 H4) | report end − previous end for PDL kernels, never duration |
| judge-only CUDA debugging (a build error costs a 10-min check) | NVRTC syntax pass for sm_80/sm_90 before every check; `-Xptxas -v` in the build log; ≤ 12 template instantiations per check |
| post-challenge review by hand ([05-10]; [04d] "will still review top submissions"; "judge calls on copies of prepare() output" rumour) | nothing cached outside the returned state; prepare()-time A/B selection is shape-only and printed in the header; Grok Bot red-team per sha |
| the republishing arena ([04c] Cacheon SN14) | we never open or submit; watcher = Grok Bot scout on public posts only; if our code appears elsewhere, nothing changes for us |
| Triton 3.8's pipeliner silently doing nothing ([F1]) | L7 code stats: `cp.async`/`cp.async.bulk`/`wgmma` counts per kernel in every build; a pipelined kernel with 0 of each fails the ratchet |

### 4. Alternatives (≥ 2 besides the chosen one, each with a switch condition)
- **Alt B — cp.async smem ring (fable-08's choice)**: 4–8 stages per warp, bit-identical numerics. Reaches the objective if the per-warp
  stream depth is the cap. Cost 30–40 builder-hours, judge-only debugging. Failure: the ring lands at < 1.25× (occupancy or staging was the
  cap all along). **Switch to B if** the C20 twins show NOCOMPUTE ≥ 0.85× full AND the best occupancy/stage-once twin gains < 20 % AND
  the weight-stream twin (same bytes, 4 CTAs/SM, no compute) reaches ≥ 3.2 TB/s (so the layout can stream; our kernel's issue pattern cannot).
- **Alt C — CUTLASS / Machete-style wgmma mixed-input path for bs ≥ 128** ([04a] §A, rank 9: "accept only if end-to-end improves").
  Cost 4–5 builder-days; E2M1×E4M3 group-16 support unverified. **Switch to C if** after Alt A and B GEMM1 bs256 is still > 230 µs while
  NOCOMPUTE ≤ 0.55× full (compute path, not memory, is the stall).
- **Alt D — attention-first** (verify-full warp specialisation / exp split). **Switch to D if** the Day-0 audit shows `_full_kernel` has
  0 cp.async AND the gather-only twin runs ≤ 160 µs on #11 (then ≈ 190 µs of the 355 is a fixable pipeline, not MUFU), AND decode_layer +
  moe have both stalled for 2 days.
- **Chosen: "measure, then parallelism, then the ring"** = (i) C20 measurement with twins (one check per GEMM kernel family), (ii) Alt A
  build: `__launch_bounds__(NW·32, 3)`, activation smem sized by the mc class (Kc 512 → 32 KB act) so 3–4 CTAs/SM, CTA loops over
  2–4 n-blocks per staged activation tile ("stage-once"), z flattened ≥ n_act (H3); numerics bit-identical to v4; (iii) composition into
  decode_layer for T ≥ 128 **today** (v4 largem, gated per bucket in prepare()), (iv) the layer glue (router order/parallelism, norm fusion,
  unpooler, epilogue grid), (v) the audit-driven Triton fixes (TMA GEMV; attention verify pipeline), (vi) bf16 cheap fixes as the A1 probe.
  Rationale: every step is one measurement check from a kill; the first two builds change scheduling, not numerics; the layer gets +0.17
  (est.) from kernels that already exist on the moe board.

### 5. Mechanism
- **K (enforced):** promote.py (check → submit only if speedup and CI-low beat the board; refuses measure_/trace_ files; needs a red-team file
  per sha) [A5]; prepare() self-checks (bitwise v5-vs-v4 for scheduling-only changes; rel L2 < 1e-2 vs Marlin for anything touching numerics;
  per-bucket enable decided on generate_input-style routing, never judge inputs [A2]; Marlin fallback above 256 or on failure);
  **ratchet tests** (D5): kernel count per shape, declared bytes per step, code stats (spills = 0; cp.async/TMA count ≥ golden for kernels
  declared pipelined), board µs per shape (fail if a candidate is worse than the board by > CI).
- **S (served):** okcdbg L2/L3/L7 lines in every measurement and trace check; `tools/make_trace.py` before every promote; `parse_stamps.py`
  (Jules) emits the D4 decision rows; `tools/ptx_audit.py` runs offline (Triton AOT for sm_90) on every Triton kernel in a candidate file.
- **P (policy):** the izanami round (D2) picks the next move by predicted Δ overall per check with ≥ 70 % of checks on decode_layer + moe
  through Oct 30; a hypothesis refuted twice is dropped; no check without a predicted per-shape µs line; no builder dispatch without a
  decision-table row that names the kill threshold.
Assumptions inline: [A1] overall ≈ time-weighted decode step; [A2] generate_input routing ≈ judge routing; [A3] in-flight bytes × duty
cycle cap both GEMM kernels (not dequant/mma); [A4] occupancy + stage-once alone reach ≥ 2.8 TB/s on the weight stream; [A5] promote.py
CI rule holds; [A6] PDL inflation explains GEMM2 > GEMM1; [A7] layer routing E ≈ 18/21 experts at T 128/256 with ≈ 2.5 routes/token [05-05];
[A8] FP8 P·V legality unknown (not built); [A9] the router squeeze is HBM sharing + low router parallelism, not residency; [A10] ≈ 120
builder-hours in 10 days; [A11] cuBLAS stays right at m ≥ 64 for every projection; [A12] the 16 KB tail holds each summary; [A13] Triton
3.8 host TensorDescriptor loads pipeline (cp.async.bulk.tensor + mbarrier) on sm_90 in a `tl.dot` K loop with BM = 16; [A14] the Triton
AOT compile on the dev box reproduces the judge's PTX (same Triton 3.8 wheel, same target sm_90); [A15] calibrated achievable HBM
bandwidth on the judge ≈ 4.0–4.2 TB/s for a 64 MB cold-L2 copy (est.; [04a] §B: 84.5 % on H100 → 4.06 TB/s scaled).

### 6. Assumption register
| An | assumption | if false, what breaks | watcher |
|---|---|---|---|
| A1 | overall ≈ time-weighted step (layer ≈ 54 %, moe 27 %, bf16 12 %, attention 7 % of our summed step, est. [01]) | order shifts toward rank-sum (bf16/attention cheap ranks) | **metric:** isolated bf16 submit (C23, day 2) → overall delta vs both models; director records the fit |
| A2 | generate_input routing ≈ judge routing (layer: 2.5 routes/token) | prepare()-time gates pick the wrong kernel per bucket | **law:** okc_run verdict "refuted" if judged per-shape µs deviates > 15 % from the check's prediction → bucket reverts to Marlin |
| A3 | in-flight × duty cycle caps the GEMMs | Alt A gives < 1.15×; pivot to Alt C / warp specialisation | **vector:** C20: NOCOMPUTE ≥ 0.85× full (confirm) / ≤ 0.55× (refute) |
| A4 | occupancy + stage-once reach ≥ 2.8 TB/s | moe ≤ 1.7; ring (Alt B) becomes the build | **crow** `moe-gate` Oct 28 18:00Z: GEMM1 bs256 ≤ 240 µs in some check, else dispatch Alt B |
| A5 | promote.py CI rule prevents regressions | a worse run on the board | law: board keeps best (okc) + promote refuses; log review each round |
| A6 | PDL inflation explains GEMM2 > GEMM1 | small-M GEMM2 has a 10 µs defect | **vector:** C20 prints end − previous end; stamps t0..t6 per CTA |
| A7 | layer routing E ≈ 18/21, 17 tokens/expert at T = 128 [05-05] | largem-in-layer prediction off by the ratio | metric: C22a' prints distinct/chunks on layer generate_input, 3 seeds |
| A8 | FP8 P·V legality unknown | removal risk if built | **accepted-risk: not built**; continuity card = read baseline.py + CPU tolerance test only |
| A9 | router squeeze = HBM sharing + 64-CTA router | the order/parallelism fix saves < 3 µs | **vector:** C22a' variant "router first, SPLIT 64": side stream ends ≤ 24 µs after the pooler's start at T = 1 (today 29.5 µs) |
| A10 | ≈ 120 builder-hours | schedule slips 2–3 days; cut D6 items by rank | metric: handover "builder hours used" vs plan at each decision point |
| A11 | cuBLAS right at m ≥ 64 | bf16/layer plan misses 5–10 µs on 8 shapes | metric: C23 prints cuBLAS vs TMA-GEMV at m = 128/256 once; never a second check |
| A12 | 16 KB tail holds the summary | measurement lost | law: summary first, ≤ 12 lines per shape, warnings silenced, budgeted printer (D5) |
| A13 | Triton 3.8 host TensorDescriptor pipelines in a BM = 16 `tl.dot` loop on sm_90 | TMA GEMV is a no-op; keep plain loads, chase parallelism only | **vector:** offline PTX of the TMA variant shows `cp.async.bulk.tensor` + `mbarrier.try_wait` inside the K loop; else the variant is not shipped to a check |
| A14 | offline AOT PTX = judge PTX | the audit misleads | **vector:** the first measurement check prints the same histogram from the judge-compiled kernel (`CompiledKernel.asm["ptx"]`) and they must agree |
| A15 | achievable HBM ≈ 4.0–4.2 TB/s | every % of floor is off by the ratio | metric: `bw_ach` from the copy kernel printed in every prepare() (L1 line) |

### 7. Posthumous pack
- **Vectors / hidden tests** (`<vectors-dir>/okc/fable-09/`; public subset seeds 11–13, hidden 5 more seeds):
  V1 Alt A kernel vs v4 **bitwise equal** for M ∈ {1,4,8,16,32,33,128,256,257} × 3 seeds (scheduling only: same dequant/mma/fixup order);
  V2 layer v5/v5b/v6 vs v4 rel L2 ≤ 2e-3 on the 13 test geometries; V3 pre-expert norm fusion vs torch RMSNorm(1+w) FP32 → one BF16
  rounding: max |Δ| ≤ 1 BF16 ulp; V4 epilogue grid tokens×4 vs 1-way: bitwise; V5 router-first ordering: bitwise equal logits and ids;
  V6 **code-stats vector**: for each Triton kernel declared `pipelined`, PTX histogram has `cp.async` + `cp.async.bulk` ≥ 1 and
  `n_spills = 0` (law; fails the build, not the judge); V7 bytes-declared vs time-implied passes within ±25 % on every L2 line (D5 D3).
- **Task cards** (schema harness/taskcard.yaml; done-when = commands): C19 layer v5 port · C20 moe stamps + twins (extended) · C21 Alt A
  "occupancy + stage-once" · C21b ring (conditional) · C22a' layer measurement (router order/parallelism, largem-vs-Marlin per T, TMA
  pooler) · C22b v5b (= v5 + largem T ≥ 128 gate + router fix) · C23 bf16 cheap + TMA GEMV · C24 v6 (Alt A GEMMs in the layer, all T) ·
  C25 G2 norm fusion · C26' attention audit → measurement (conditional) · C27 serving S0 · C28 okcdbg L2/L7 + ratchets · C29 ptx_audit ·
  C30 hold-out re-runs. Done-when lines in D8's rows.
- **Crows** (pre-kernel routines; re-validate at fire time; max_fires 1):
```json
[{"id":"crow.c20-decision","fire":"when C20's check lands (ledger task C20 -> review)","expiry":"2026-10-27T06:00Z",
  "recheck":["A3","A6","kernel track open"],"action":"apply D4 §1 decision table; write chosen build id (C21 Alt A | C21b ring | Alt C) into handover"},
 {"id":"crow.moe-gate","fire":"2026-10-28T18:00Z","expiry":"2026-10-29T06:00Z",
  "recheck":["A4: any check with GEMM1 bs256 <= 240 us","builders alive"],"action":"if false: dispatch C21b ring to Sonnet-B; Sonnet-A continues layer glue"},
 {"id":"crow.a1-calibration","fire":"after the isolated bf16 submit (C23)","expiry":"2026-10-28T00:00Z",
  "recheck":["overall rank changed","no other submit within 2 h"],"action":"record which overall model fits; if rank-sum: raise C26'/C23 one step"},
 {"id":"crow.audit-attention","fire":"when C29 ptx_audit output exists for 05-13","expiry":"2026-10-26T12:00Z",
  "recheck":["_full_kernel cp.async count known"],"action":"if 0: schedule C26' measurement (gather-only, no-exp, programs/BN twins) for day 2; else close C26' as 'pipelined, MUFU-bound'"},
 {"id":"crow.serving-s0","fire":"2026-10-31T09:00Z","expiry":"2026-11-01T09:00Z",
  "recheck":["decode_layer board >= 1.40 (else fire again Nov 1)","Jules lane available"],"action":"dispatch C27 S0 (CPU-only)"},
 {"id":"crow.freeze","fire":"2026-11-03T12:00Z","expiry":"2026-11-05T08:00Z",
  "recheck":["kernel track open","no Dex stop"],"action":"no new kernels; only hold-out re-checks and promote of already-judged candidates; last submit 2026-11-04T18:00Z"}]
```
- **Gone-clause (director unavailable):** (1) only promote.py from a director session submits; builders' branches wait. (2) Grok Bot
  keeps the red team per sha and the daily digest. (3) Continuity may run measurement checks whose kernel_fn is a board kernel verbatim and
  may promote a candidate whose check is already judged above CI; it may not submit new kernels. (4) Decisions that wait: lane switches
  (crows), reopening attention, anything on A8, any spend/accounts/organiser contact (Dex). (5) "stop" in chat or ledger disables the
  routines. (6) After Nov 5 07:59Z nothing on the kernel track. (7) If an entry is found illegal: `okc withdraw` first, think second.
  (8) The handover's last block is the truth; (9) D6's order is the queue; (10) D4's tables are the decisions — nobody re-plans them alone.

### 8. Falsification and evaluation
- Wrong if C20's NOCOMPUTE twin is ≤ 0.55× the full kernel at bs 128/256 (compute path stalls; Alt C / warp specialisation, not ring).
- Wrong if the best occupancy/stage-once twin gains < 10 % AND the ring (after two checks) < 1.25× on GEMM1 bs256 (then the layout or the
  split-K tail is the cap: re-aim at Kc/KS and the fixup).
- Wrong if layer v5 (C19) judges < 1.245 (prediction 1.266 ± 0.02) or v5b (largem T ≥ 128 + router fix) < 1.33 (prediction 1.40 ± 0.03).
- Wrong if the router-first ordering leaves the side stream ending > 26 µs after the pooler's start at T = 1 (today 29.5 µs).
- Wrong if the TMA GEMV variant shows `cp.async.bulk.tensor` in PTX but gains < 5 % on the pooler (then the GEMV was never pipeline-bound).
- Wrong if the overall rank on Oct 29 is not better than 335 while ≥ 2 promoted submits landed (then A1 or the field).
Replay set: the judge's shapes (every check); metric = judged speedup with CI; threshold CI-low above board; cost rule ≥ 70 % of checks on
decode_layer + moe through Oct 30; refuted twice → dropped; every check has a predicted per-shape µs line.

### 9. Adoption record
Decision id: ______ · risk class: ______ · adopted by: Dex (minister) / director · rollback: `okc withdraw <job>` per entry; bucket
gates and LARGEM_ENABLE are module switches · first review: ______ (proposed 2026-10-27 06:00Z after C19 + C20 + the audit land).

---
## D2. izanami round spec for okc (pack 07b format)
**Open (every session):** read handover's last two blocks, inbox, ledger items; write the header
`## <UTC> session <id> · objective: raise overall rank; stop when: <N checks dispatched and their verdicts anchored | 20 rounds | 90 min>`.
**Round inputs** (one tool pass, no browsing): (1) `okc problems` → checks/submits left per problem; (2) `okc jobs` → landed runs since the
last anchor with speedup, CI, log tail; (3) `okc leaderboard --json` → per-problem board and rank delta; (4) ledger: builder cards in
phase (building / needs-review / judged); (5) the D6 queue with each item's state (unmeasured / measured / killed / promoted).
**One round = one move** (artifact | dispatch | convert | review | stop). okc-specific instances:
- **review**: one landed check vs its predicted line → verdict {confirmed, better, refuted, noise} written as `{"shape":…,"pred":…,"obs":…}`
  per shape; apply the D4 decision row; a refuted item gets its second-strike mark or is dropped.
- **dispatch (check)**: one `okc check` with `--predict` and `--hyp`, only for a file whose NVRTC syntax pass and local tests are green and
  whose header prints its predicted per-shape µs; ≤ 1 check per problem per round, ≤ 5 per problem per day on build days (reserve 15 for
  compile failures and hold-out re-runs).
- **dispatch (builder)**: one task card to Sonnet-A/B via /shisui, with a kill threshold copied from D4/D6; Argon/Grok Build get the
  read-only draft/review of the same card; Jules gets tools only (never cp, never kernels on the judge path).
- **convert**: an `okc check` in flight → one send-later at +12 min with the job id ("read verdict, apply row X"); a builder → a ledger
  monitor on the card id; a human decision (A8, spend) → message to Dex + stop.
**Per round:** predict (`after this round, <job id> has a verdict row | card <id> is in phase building | …`), act, anchor
`{"r":N,"move":…,"target":…,"predict":…,"observed":…,"refs":[…],"head":…,"next":…}` to handover + ledger, judge (noop if nothing observed).
**Budgets:** session ≤ 20 rounds or 90 min; per day: ≤ 5 checks and ≤ 2 submits per problem; ≥ 70 % of checks on decode_layer + moe
through Oct 30; ≤ 2 builders dispatched at once; one Fable call per major idea with a /shisui brief Dex sees first.
**Decision rules (fixed, not re-planned in the round):** promote iff judged speedup > board AND CI-low > board CI-low (promote.py);
kill a kernel hypothesis when its best variant is < +8 % on the targeted kernel after one measurement check (noise ±1.5 % paired, ±5 % across
routing seeds), or when its judged speedup is inside the previous entry's CI twice; kill a composition when judged < predicted − 0.02 twice;
a bucket gate reverts to Marlin when judged per-shape µs deviates > 15 % from the prediction (A2); a check whose log tail lacks the
summary block is a tooling failure, not evidence (fix the printer before the next check).
**Stop/switch rules:** two consecutive noop rounds → convert or stop; a usage stop → the next session resumes from the last round's `next`;
crow.moe-gate / crow.freeze switch lanes (D1 §7); Dex "stop" disables everything.
**Recorded per round:** the anchor line; per judged check: `experiments/log.jsonl` row (already the practice [05-07]) + the per-shape
pred/obs table in the card; per decision: the D4 row id applied; per day: `BEVINDINGEN.md` entry (date, agent, change, result).

---
## D3. What changed vs fable-08 (keep / change / drop, with evidence ids)
| fable-08 item | status | why (evidence) |
|---|---|---|
| H1: in-flight bytes cap both GEMMs; C20 stamps + NOCOMPUTE twin first | **keep, extend** | [01 D3]; [03a A6] bisect-by-deleting; [04a] diagnostic ladder; add occupancy, stage-once, NOSTAGE, weight-stream and BW-calibration twins |
| H2: cp.async ring as the first build (C21) | **change → second build (Alt B)** | [04c]: parallelism over prefetch tricks (+66 % regression); Little's law: 8 warps × 3 KB = 24 KB/SM ≈ needed 24–36 KB (est.) — occupancy is the cheaper half; [04a] "register double-buffering is not a pipeline" still stands |
| Alt A (launch bounds / depth) as fallback | **change → first build (C21)** | same; plus stage-once (CTA prologue stages 64 KB of activations per 72 KB of weights at bs256, est. from [05-11] geometry) |
| H10: largem in the layer at T ≥ 32 "after the ring" | **change → day 2 with v4 largem, T ≥ 128 only** | Marlin 1.0–1.3 TB/s at T 128/256 [05-03 layer, 05-05]; largem 1.5× Marlin at m 128/256 on moe routing [05-16]; T = 32 stays Marlin (101.6 µs vs ≈ 127 est.) |
| H5 / C19 full_cat port | **keep (day 0)** | unchanged [05-03 attn #3/#5 vs layer #3/#5] |
| H6 / G1 pooler smem co-residency | **drop mechanism, keep symptom → H6'** | [F2]: 24 KB/12 KB smem; new mechanism = HBM sharing + 64-CTA router (D4 §2 arithmetic) |
| H7 / G2 norm fusion | keep | layer_norm 5.6–6.3 µs on every shape [05-03 layer] |
| H8 / G3 unpooler, H9 / G4 epilogue grid, H3 z-flatten | keep (cheap) | unchanged |
| H11 ROWMAJOR sliding partials | keep as rider | unchanged |
| H12 FP8 P·V | **drop from the window; continuity card only** | legality open [01 D9-Q1]; [04c] FP8 tolerance failure (different problem); [04a] "FA3's FP8 accuracy depends on its method" |
| H13 bf16 cheap fixes (isolated submit = A1 probe) | keep | unchanged |
| new H17: PTX/SASS audit of every Triton kernel (0 checks) | **add (day 0)** | [F1]; [03a A4] read the generated code; [03b §5 L7] |
| new H18: TMA-descriptor GEMV (pooler, qkv at m ≤ 32, router) | **add (audit-gated)** | [F1], [04c] TMA unlocked multistage smem; API verified (Sources: Triton docs, CUDA driver API) |
| new H19: attention verify-full pipeline twins | **add (audit-gated)** | [05-13] docstring claims cp.async; never checked; 1.7 TB/s on 607 MB [05-03] |
| new H20: router GEMV ≥ 256 CTAs + router-first ordering + (later) fused pooler+router | **add** | D4 §2; [04d] router 10.2 → 3.9 µs by a participant (idea only) |
| new H21: cuBLAS/split path for the router at m = 128/256 in the layer | **add** | router 28–38 µs on the side stream at T 128/256 [05-03 layer #8–#11]; Hotz §3 item 1 [03] |
| new H22: o-proj one-shot GEMV (BK = 1024, all loads before compute) at m ≤ 32 | add (cheap variant in C23/C22a') | o at 2.6 TB/s on 16.8 MB, floor 3.5 µs [05-03 bf16 #1] — ramp-bound |
| persistent grouped scheduling ([04b] #1) | **partial**: our largem grid already loops z over chunks [05-11]; stage-once = the useful half | [04b] numbers are not ours; [04a] "no defensible multiplier" |
| Humming / FlashInfer SM90 FP4 as drop-in ([04d]) | drop | image fixed [05-10]; FlashInfer's SM90 FP4 path uses FP8 activations [04a] — illegal here |
| serving S0 on Oct 31 | keep | unchanged |

---
## D4. Diagnostic ladder per problem (what each twin removes, what the reading means, which decision it fires)
General rules for a measurement check (fable-08 App. D still applies): kernel_fn = the board kernel verbatim; prepare() does the science;
L2 flush (read > 64 MB) before every timed trial; CUDA events; median of 5 (min of 3 for ≥ 100 µs kernels); pivot = board kernel in every
round, score = pivot/variant per seed; ≥ 3 generate_input seeds; print the summary first; ≤ 12 lines per shape; `B` lines carry bytes and
GB/s and refuse any GB/s > 4500 (P6). One check per problem answers the table below. Budget per check: ≤ 12 nvcc instantiations (≈ 2 min)
or ≤ 8 Triton configs (≈ 2–3 min) inside prepare ≤ 300 s; run ≤ 15 min.

### §1 moe (C20, Sonnet-B; GEMM1/GEMM2 × bs {32,128,256} largem, bs {1,8,16} smallm × seeds 11–13)
| twin / variant | removes or changes | expected reading (est.) | reading → decision |
|---|---|---|---|
| CAL copy kernel (64 MB, 4 CTAs/SM, 16 B/lane) | everything but streaming | `bw_ach` 4.0–4.2 TB/s | all `%ach` below use it (A15); if < 3.6 the judge box is throttled: compare ratios only |
| WSTREAM: plain kernel that reads exactly the packed weight bytes of the active experts (words + scales, same addresses), folds into a sink | dequant, mma, staging, fixup | ≥ 3.2 TB/s | the ceiling for THIS layout; if < 2.8 TB/s the layout itself (576 B tiles, 2 streams per warp) is the cap → repack (words+scales adjacent, 1 KB per warp-load) before any pipeline work |
| v4 + stamps t0..t6 (fable-08 D3 §2 locations) | nothing | staging (t2−t1) 15–35 % of CTA time; fixup tail 5–15 % | staging > 25 % → stage-once is mandatory in C21; tail > 15 % → KS 4→2 or FP32 atomic epilogue |
| NOCOMPUTE (loads kept, dequant/mma replaced by XOR fold) | compute path | 0.8–0.95× full | ≥ 0.85 → memory pipeline is the cap (A3 true) → C21 Alt A; ≤ 0.55 → compute/ordering stalls → Alt C or producer/consumer warps |
| NOSTAGE (s_act = zeros; no L2 reads of activations) | prologue L2 chain | −10..−30 % | if ≥ −20 %: stage-once ranks above occupancy in C21 |
| LB(128,2) / LB(128,3) (`__launch_bounds__(NW·32, k)`), MR=1·D=2 (fewer regs) | occupancy limit | +10..+40 % | ≥ +25 % alone → ship as v5 this week; < +10 % → occupancy is not the limiter (then the ring) |
| KC512 (Kc = 512, KS = 8; act smem 33 KB) | smem-limited residency | +5..+30 % | combined with LB: if CTAs/SM (printed via `atomicAdd(occ[smid])` max) rises to 3–4 and TB/s ≥ 2.8 → A4 true |
| NBLOOP=2/4 (CTA owns (chunk, ks), loops n-blocks with one staged s_act) | re-staging per n-block | +5..+25 % at bs 128/256 | > +10 % → in C21 |
| ZFLAT (smallm: z ≥ n_act, early exit) [H3] | wave quantisation at bs 1 | GEMM1 bs1 27 → ≈ 18 µs | keep if per-expert time is constant |
| GEMM2 with `programmaticStreamSerializationAllowed = 0` | PDL early launch | duration ≈ end-to-end, span unchanged | A6 confirmed → parser reports end − previous end only |
Readings print as `B moe m=<M> seed=<s> distinct=<E> chunks=<C> n_act=<n> | <variant> g1=<us> g2=<us> tb=<TB/s> ach=<%> occ=<CTAs/SM>`.
Decision table (fires crow.c20-decision): (i) NOCOMPUTE ≥ 0.85 and (LB|KC512|NBLOOP best) ≥ +25 % → **C21 Alt A = the best combination**,
predicted GEMM1 bs256 322 → ≤ 240 µs; (ii) NOCOMPUTE ≥ 0.85 and best occupancy twin < +15 % and WSTREAM ≥ 3.2 TB/s → **C21b ring** (fable-08
D3 §4 unchanged, S = 6, 2 CTAs/SM); (iii) NOCOMPUTE ≤ 0.55 → **Alt C** scoping + producer/consumer variant; (iv) WSTREAM < 2.8 TB/s → repack
first (one builder-day) then re-run C20; (v) staging > 25 % in any case → NBLOOP into whichever build wins.

### §2 decode_layer (C22a', Sonnet-A; T ∈ {1, 8, 16, 32, 128, 256} on layer generate_input, 3 seeds; plus C24-lite)
Arithmetic behind H6' (shape #0, [05-03 layer]): pooler 36.1 → 55.3 (19.2 µs, 67 MB, 3.5 TB/s); side stream: fill 36.0–39.5 (1 KB in
3.4 µs; 0.67 alone), router 40.8 → 62.8 (22.0 µs; 5.5 alone), routing → 65.6; main idles until layer_norm at 66.1. The router made almost
no progress while the pooler streamed (if it had run at even 0.6 TB/s leftover it would have ended ≈ 55; it ended 7.5 µs after the pooler,
≈ its whole standalone time + ramp). Sum of bytes 75.4 MB at 3.5 TB/s = 21.5 µs: two bandwidth-bound kernels on two streams serialise in
HBM, and the one with fewer outstanding requests (router: 64 CTAs = 4 n-blocks × 16 splits [05-14 `_CFG`]) starves (A9).
| variant | changes | expected (est.) | decision |
|---|---|---|---|
| ORDER: router (+ fill-free fixup) then pooler on the main stream; routing on the side stream after the router | stream topology only (V5 bitwise) | side stream ends ≈ 36.1 + 5.5 + 2.9 ≈ 44.5; main free at 55.3; layer_norm starts at 55.7 vs 66.1 → **−10 µs** at T = 1 | if side ends ≤ 48 → adopt now in v5b |
| RPAR: router `_CFG` (16,128,64,4,·) or (64,128,64,·) → 256 CTAs; last-CTA fixup instead of `torch.zeros` + atomics (H13) | router parallelism | router alone 5.5 → ≈ 3–3.5 µs; fill gone (−0.7..−3.4) | adopt if ≤ 4 µs alone |
| RFUSE: router hi+lo as 8 extra N-blocks of the pooler kernel (N = 4096 + 2·256 … or two tensors via one grid with a block-id switch) | one launch, 75.4 MB | ≈ 21–22 µs for both; routing after | later (C25-era) if ORDER + RPAR leave > 2 µs of idle |
| RBIG: router at T = 128/256 via SGLang's split path (two `torch.mm` hi/lo, FP32 out) or BM = m single block | 4 M-blocks re-streaming 16 MB [03 §3.1] | 28–38 µs → ≈ 6–8 µs; side stream no longer the critical path at T ≥ 128 (**−8..−12 µs**) | adopt if ≤ 10 µs |
| TMAPOOL: `_gemv_splitk` with host TensorDescriptor on W, BK = 256, 4–6 stages (H18) | Triton pipelining ([F1]) | pooler 19.2 → ≈ 16.5–17 µs (3.6 → 4.0 TB/s) | adopt if ≥ −1.5 µs AND PTX shows `cp.async.bulk.tensor` (A13) |
| OSHOT: o-proj at m ≤ 32 as Triton GEMV BK = 1024, 1 stage, 256 CTAs (H22) | cuBLAS nvjet 6.6 µs (2.6 TB/s) | ≈ 4.5–5 µs | adopt if ≤ 5.5 |
| C24-lite: {Marlin+S1, v4 largem} per T ∈ {32, 128, 256} on layer routing | the T ≥ 128 gate (H10') | largem 0.6–0.7 × Marlin at 128/256; ≥ 1.2 × Marlin at 32 | gate largem on for 128/256 if ≤ 0.8 × Marlin on all 3 seeds; print the path for T ∈ {1..8,16,17,32,33,128,256,257} (monotone) |
Readings: `B layer T=<T> seed=<s> E=<distinct> | <variant> side_end=<us> main_idle=<us> pool=<us> rout=<us> g1=<us> g2=<us>`.
Predicted after C22b (v5 + ORDER + RPAR + RBIG + largem T ≥ 128): #0 125 → 115, #1 137 → 122 (port + order), #2 194 → 184, #3 340 → 270,
#4 241 → 236, #5 540 → 405, #6 179 → 169, #7 186 → 176, #8 502 → 355, #9 664 → 515, #10 683 → 462, #11 1017 → 795 (est.) ⇒ loads decode1
1.365 → 1.48, decode16 1.347 → 1.49, decode32 1.138 → 1.30, verify1 1.495 → 1.58, verify16 1.053 → 1.45, verify32 1.034 → 1.46 ⇒
**problem 1.227 → ≈ 1.46 (est.; kill if < 1.38)**. Then G2 (−6 µs × 12), G3/G4 (−8 µs on T ≤ 8), Alt A GEMMs in the layer (C24) → ≈ 1.65–1.8.

### §3 attention (C29 audit first, 0 checks; C26' only if `_full_kernel` shows 0 cp.async; shapes #9/#11 + #8/#10, seeds ×3)
| twin / variant | removes or changes | expected (est.) | decision |
|---|---|---|---|
| GATHER-only: `_full_kernel` with QK/softmax/PV replaced by a fold of the K/V tiles | all compute | #11: 130–170 µs (607 MB at 3.6–4.6 TB/s) | the gather ceiling; if ≤ 160 and full = 355 → ≥ 190 µs is compute/overlap, not bytes |
| NOEXP: `exp2` replaced by a cheap linear op (result folded) | MUFU | −60..−100 µs | MUFU share; if ≥ 80 µs → exp emulation (FA4-style poly on FMA, [01 App. E]) becomes a card for Nov 1–2 |
| NOCVT: V consumed as raw FP8 bits (wrong numbers, folded) | 303 M converts | −40..−80 µs | convert share; if ≥ 60 → the P·V FP8 question is worth asking Dex (A8), still not built |
| PROG512 / BN128 / stages 2 vs 4 | residency vs per-program depth | ±10 % | more programs win → "parallelism" again; depth wins → cp.async must be real (audit) |
| ROWMAJOR sliding partials (H11) | reducer layout | #8 reducer 5.7 → ≈ 2.5, #10 8.1 → ≈ 3 µs | rider in any attention promote |
Decision: if GATHER-only ≤ 160 AND the audit shows 0 cp.async → one fix card (force the pipeline: `tl.load` with `tl.multiple_of`/unmasked
inner tiles so Triton lowers to cp.async, or restructure to 4 programs/SM) with kill at verify-full > 300 µs; else attention stays closed
except H11. Under A1 the whole lane is ≈ +0.5–2 % overall; it is the first thing cut if builders are short (A10).

### §4 bf16 (C23, director + Sonnet-A; one measurement check then one isolated submit)
Variants: `_CFG` unpooler m 1/8 (32,256,1,4,4) (6.8/7.2 → ≈ 3.3 µs); router fixup (no zeros/atomics; 6.4–7.4 → ≈ 5); router m = 128/256
BM = m (9.9/12.3 → ≈ 6); RPAR router configs at m ≤ 32; TMAPOOL at m ≤ 32 for pooler and **qkv** (cuBLAS 13.2 µs + reduce for 37.8 MB
= 2.9 TB/s → ≈ 10.5 µs at 3.6 TB/s); OSHOT for o at m ≤ 32. Readings as fable-08 C23. Predicted 1.156 → 1.19–1.22 (cheap only) / ≈ 1.25 with
TMA qkv+pooler (est.); kill the TMA half if < −1.5 µs on the pooler. This submit is isolated (no other submit within 2 h) → A1 calibration.

---
## D5. okcdbg minimum spec (≤ 1 builder-day) + ratchet tests
Scope for day 0–1 (Jules for parsers/tests, Sonnet-A for the prepare()-side library; ≤ 400 lines so it ships inside a submission):
**Levels** (cumulative, `OKC_DEBUG=0..7`, `OKC_DEBUG_BUDGET=14000` chars, `OKC_DEBUG_ONLY=<regex>`): build **L1, L2, L3, L7** and detectors
**D3, D4** now; L4/L5(D1, D2)/L6 later (03b §5.1 table).
Output lines (fixed width, units in the header only, tags greppable; verbose first, summary last; never cut mid-line):
```
#OKC v1 problem=muse_moe_w4a16 file=sonnet_v4 shape=5 T=256 E=24 dev=H200 bw_ach=4120 tf_ach=610 peak_bw=4800
#K  i name                                grid         blk  smem   reg spl n    us  %tot  GB/s %ach  TF  floor x_fl passes_t
#K  1 moe_g1_w4a16_T256_N8192_K4096_e24   (64,4,48)    128  67584  180   0 1 322.1  65.3  1584   38  86  123.7 2.60    2.4
#K  2 moe_g2_w4a16_T256_N4096_K8192_e24   (32,4,48)    128  67584  162   0 1 161.8  32.8  1576   38  85   61.9 2.61    2.4
#B  g1 w=510MB(uniq 453) x=2.1MB out=4.2MB traffic/uniq=1.13  g2 w=255MB(uniq 227) ...
#C  moe_g1  ld.global=144 cp.async=0 cp.async.bulk=0 ldmatrix=96 mma.sync=512 wgmma=0 cvt=0 ex2=0 bar.sync=6 atom=1 src=ab12cd
#D3 moe_g1 passes_t=2.4 vs declared 1.13 -> re-reads are not the cap; in-flight/duty cycle is (see #K x_fl)
#D4 gaps=0.3us copies=0 memsets=0 dead_writes=0 nonaligned_args=0
#S  kernels=4 gpu_us=499.8 wall_us=493.4 busy=98.7% gaps_us=0.3 floor_us=191 x_floor=2.58 recoverable_us=297 bound=2.55x
```
Definitions: `floor` = max(declared unique bytes / bw_ach, flops / tf_ach) per kernel; `x_fl` = us / floor; `passes_t` = us × bw_ach /
unique bytes (time-implied passes: ≈ 1 at the floor; ≈ traffic/uniq means re-reads miss L2; < 1 means the declaration is too big);
`recoverable_us` = Σ (us − floor) over kernels + gaps; `bound` = wall / (wall − recoverable) — the speedup if everything hit the floor
(a ceiling, never a prediction; "say the number first" [03 §1.7]).
**How the data is produced on the judge:**
- Timing: the existing trace path (`tools/make_trace.py` + CUPTI under graph replay [02]) gives per-kernel us; `bw_ach`/`tf_ach` from a
  64 MB cold-L2 copy kernel and one 4096³ BF16 GEMM run in prepare() (printed at L1; 3 trials, median).
- Bytes model: a small registry `okcdbg.kernel(name=…, bytes=lambda a: …, traffic=lambda a: …, flops=lambda a: …)` evaluated on real
  shapes and the real routing tensor at warm-up (never a formula for n_active); seeded from [05-05] rows; names carry T/N/K/E
  (`moe_g1_w4a16_T256_N8192_K4096_e24`) so the 16 KB tail stays readable.
- L7 code stats: Triton → after the first launch, `fn.device_caches[dev][0][key]` (or `fn.warmup(...)`) yields a `CompiledKernel` with
  `.asm["ptx"]`, `.asm["cubin"]`, `.n_regs`, `.n_spills`, `.metadata.shared` (Triton compiler.py, Sources); histogram = regex counts on PTX
  (`ld.global`, `ld.global.nc`, `cp.async`, `cp.async.bulk`, `cp.async.bulk.tensor`, `mbarrier`, `ldmatrix`, `mma.sync`, `wgmma`, `cvt`,
  `ex2.approx`, `bar.sync`, `atom`, `st.global`) and, where `cuobjdump` exists in the image (ships with nvcc), `cuobjdump -sass` on the
  cubin written to /tmp → counts of `LDGSTS` (cp.async), `UTMALDG`/`UBLKCP` (TMA), `LDG`, `HMMA`, `MUFU.EX2`, `F2FP`, `LDSM`, `STL/LDL`
  (spills). CUDA (`load_inline`) → `extra_cuda_cflags=["-O3","-Xptxas","-v"]` with `verbose=True` and the build log captured into the
  budgeted printer (regs/spills/smem per template instance), plus `cuobjdump -sass` on the built `.so` for the histogram. Offline: the
  same histogram from `triton.compile(src, target=GPUTarget("cuda", 90, 86))` on the dev box (= today's [F1] method; A14 says the two must agree).
- D3: `passes_t` vs declared `traffic/uniq` per kernel (table above). D4: from the trace: gaps (sum, max, count > 3 µs), kernels matching
  `copy_|elementwise_kernel|FillFunctor|Memcpy|Memset` > 64 KB, buffers written and never read (needs arg roles from the registry; skip if
  not declared), non-contiguous or `ptr % 16 != 0` args on kernels declaring vector loads.
**Ratchet tests** (`tests/ratchet/test_<problem>.py`, golden in `tests/ratchet/<problem>.json`, updated only by promote.py when a submit
lands; fail-if-worse and "too far from" = fail if a candidate is > 20 % *better* than the limit without updating it, tinygrad's rule [03b §3.4]):
R1 kernel count per shape ≤ golden; R2 declared unique bytes per shape ≤ golden × 1.02 (catches re-read regressions when a config changes);
R3 code stats: `n_spills = 0` for every kernel; every kernel declared `pipelined` has `cp.async + cp.async.bulk* ≥ 1`; regs ≤ golden + 8;
R4 board µs per shape: a trace check's per-shape span ≤ golden × (1 + CI) else promote refuses. Smoke: `OKC_DEBUG=100` on a stored trace
on CPU parses and stays ≤ 14000 chars. Build order (hours, est.): registry + L2/L3 printer 2 · trace join + calibration 1 · L7 1.5 ·
D3/D4 1 · ratchets + smoke 1.5 = 7 h.

---
## D6. Revised ranked hypothesis queue (by expected Δ overall per judge check; effort = builder-hours, est.)
| id | problem | shapes | mechanism | predicted µs → speedup (est.) | exact test | kill / confirm | effort | lane |
|---|---|---|---|---|---|---|---|---|
| H17 | all Triton | — | PTX/SASS audit: cp.async / TMA / wgmma / spills per shipped Triton kernel (offline) | none directly; gates H18/H19 | C29: `tools/ptx_audit.py 05-11..14` → table | — | 3 h | director (+Jules) |
| H5 | decode_layer | #1,#3,#5 | port attention v3 `_full_cat_kernel` + ROWMAJOR reducer for q_len 1 full | #1 137 → 132.6, #3 340 → 280, #5 540 → 410; problem 1.227 → 1.266 | C19: check then promote | kill < 1.245; confirm ≥ 1.255 | 4 h | Sonnet-A |
| H10' | decode_layer | #8–#11 | v4 `w4_largem` instead of Marlin at T ≥ 128 (prepare()-time gate on layer routing; `largem_route` reads device `local_start`) | Marlin 398/510 → ≈ 260/300 (2.0 TB/s on 520/602 MB); verify16 1.053 → 1.41, verify32 1.034 → 1.42; problem → 1.36 (1.40 with H5) | C24-lite in C22a', then C22b promote | kill a bucket if largem ≥ 0.9 × Marlin on any seed; composition kill < 1.33 | 8 h | Sonnet-A |
| H1' | moe | all 6 | C20 twins: NOCOMPUTE, NOSTAGE, LB, KC512, NBLOOP, ZFLAT, WSTREAM, CAL, stamps | decides C21 vs C21b vs Alt C | C20 one measurement check | D4 §1 table | 8 h | Sonnet-B (+Jules parser) |
| H2' | moe (+layer) | all 6 | **Alt A**: launch bounds (·,3), act smem by mc class, stage-once over n-blocks, z-flatten; bitwise = v4 | GEMM1 bs256 322 → ≤ 240, GEMM2 162 → ≤ 120; moe 1.436 → ≈ 1.75–1.85 (×1.3 column of [01 App. A]) | C21: V1 bitwise in prepare(), sweep LB×Kc×NBLOOP, promote | kill if < +15 % on GEMM1 bs256 after one check | 10 h | Sonnet-B + Argon review |
| H6' | decode_layer | #0–#3,#6,#7 (+#8–#11 via H21) | router first + ≥ 256-CTA router + fill-free fixup (HBM sharing, not residency) | −10 µs at T = 1, −6 at T = 16/32 (est.); decode1 1.365 → 1.48 | C22a' ORDER/RPAR variants; V5 bitwise | kill if side stream ends > 48 µs after pooler start at T = 1 | 3 h | Sonnet-A |
| H21 | decode_layer | #8–#11 | router at T ≥ 128 via two cuBLAS hi/lo GEMMs or BM = m | 28–38 → 6–8 µs; −8..−12 µs/shape | C22a' RBIG | kill if > 10 µs | 1 h | Sonnet-A |
| H2b | moe (+layer) | all 6 | **Alt B** cp.async ring (fable-08 D3 §4), only on decision (ii) | GEMM1 bs256 → 150–185; moe → ≈ 2.2 | C21b | kill if after 2 checks GEMM1 bs256 > 270 | 30–40 h | Sonnet-B |
| H7 | decode_layer | all 12 | pre-expert RMSNorm(1+w) folded into route kernel + GEMM1 staging (FP32, one BF16 rounding) | −6.7 µs/shape | C25 + V3 | kill if V3 fails or Δ < 4 µs | 8 h | Sonnet-B |
| H24 | decode_layer | all T | Alt A/B GEMMs into the layer for all T with per-bucket gate (T = 32 included once GEMMs ≥ 2.6 TB/s) | T = 32 Marlin 101.6 → ≈ 85; T ≥ 128 another −60..−100 µs | C24 | kill a bucket if ours ≥ 0.95 × Marlin | 10 h | Sonnet-A |
| H18 | bf16 + layer | pooler m ≤ 32, qkv m ≤ 32 | TMA-descriptor GEMV (BK 256, 4–6 stages, 2 CTAs/SM) | pooler 18.5 → ≈ 16.7; qkv 13.2 → ≈ 10.5; bf16 +0.03, layer −4 µs on 8 shapes | C23 / C22a' TMAPOOL; A13 vector first | kill if < −1.5 µs on pooler or no TMA in PTX | 6 h | Sonnet-A (draft: Grok Build) |
| H13 | bf16 | #3,#18,#4,#9,#14,#19,#24,#29 | `_CFG` unpooler m 1/8; router fixup; router m 128/256 BM = m | 1.156 → ≈ 1.19–1.22 | C23 then isolated submit (A1) | kill if < 1.19 | 4 h | director + Sonnet-A |
| H8/H9/H3 | layer + moe | T ≤ 16 | unpooler Triton into zeroed [T,8192]; epilogue grid tokens×4; z-flatten | −5.4, −3, −9 µs on their shapes | C22b variants + V4 | kill if Δ < 3 / 1.5 µs | 2+2+1 h | Grok Build draft → Sonnet-A |
| H22 | bf16 + layer | o m ≤ 32 | one-shot GEMV, BK 1024, 256 CTAs | 6.6 → ≈ 4.8 µs | C23 OSHOT | kill if > 5.5 | 1 h | Sonnet-A |
| H19 | attention (+layer verify-full) | #9,#11 | verify-full pipeline twins (gather-only, no-exp, no-cvt, programs/BN) | if gather-only ≤ 160: fix → 355 → ≈ 260 (est.), attention 1.689 → ≈ 1.85 | C26' (audit-gated) | kill if gather-only > 200 or fix > 300 µs | 4 h + 8 h fix | Sonnet-A late |
| H11 | attention | #8,#10 | ROWMAJOR sliding partials | +1.2 % | rider in C26' | confirm only ≥ 1.70 | 2 h | Sonnet-A |
| H4 | moe | #0,#1,#3 | GEMM2 "slow" = PDL early launch | parser fix | C20 | — | 0.5 h | Jules |
| H14 | overall | — | overall ≈ time-weighted step | re-orders the queue if false | isolated bf16 submit | — | 0 | director |
| H12 | attention | #9,#11 | P·V in FP8 | 1.689 → ≈ 1.86 | **not in window**; continuity: baseline.py read + CPU tolerance test | — | 4 h | parked |
| H15 | attention | #9,#11 | warp specialisation + exp split | → 2.1–2.3 | not in window | — | 40 h | parked |
| H16 | moe | #4,#5 | CUTLASS/Machete wgmma mixed-input (Alt C) | bs256 GEMM1 → ≈ 150 | only on decision (iii) | — | 40 h | parked |
Dropped: G1 as stated (smem monopoly), hand double-buffering in Triton, FP8 operands for projections, TMA gather4, Humming/FlashInfer
drop-ins, megakernel. Reading the table under A1: H10' + H5 (≈ +0.09 overall-speedup, est.) ≥ H2' (+0.1–0.2 if Alt A reaches ×1.3) >
H6'+H21 (+0.03) > H7 (+0.02) > H18/H13 (+0.01 each) > H19 (+0.01, audit-gated). Under rank-sum, H13 and H19 rise one step.

---
## D7. Per-problem technique notes (snippets ≤ 30 lines each)
### moe — Alt A, the scheduling change (C21)
Geometry today [05-11]: largem CTA = NW 4 warps × MR 2 → 128 weight rows, one K slice (Kc = 1024), one 32-token chunk; smem = 1 KB lut +
32 × (1024 + 8) × 2 B = 67 KB; regs 162–180 → 2 CTAs/SM (regs), 8 warps/SM, 3 KB in flight per warp. Per CTA: 72 KB of weights from
HBM vs 64 KB of activations staged from L2 through 128 threads (32 × 16 B loads per thread, a serialised latency chain) — the prologue is
of the same order as the streaming. Alt A changes four things, none in the numerics:
```cuda
// 1. residency: ask for 3 CTAs/SM (ptxas caps regs at 170 for 128 threads); spills are a ratchet failure (R3)
template <bool FUSE, int NW, int MR>
__global__ void __launch_bounds__(NW * 32, 3) w4_largem_kernel(...)
// 2. activation smem by chunk class: rows = 8/16/32 by mc, KP = Kc + 8 -> 17/33/66 KB; dynamic smem set per launch from ctab's max mc
// 3. stage-once: a CTA owns (chunk z, k-slice ks) and loops NB n-blocks over the SAME s_act (one staging per NB weight groups)
for (int nbi = 0; nbi < NB; ++nbi) {
    const int nb = blockIdx.x * NB + nbi, g0 = (nb * NW + warp) * MR;        // grid.x = nbn / NB
    const uint4* wp = Wq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 32 + lane;
    const uint2* sp = Sq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 8 + gid;
    prefetch_batch_l<MR, D>(w0, s0, wp, sp, NIT * 32, NIT * 8);               // first batch of this n-block in flight
    chunk_gemm<FUSE, NW, MR, NT>(w0, s0, wp, sp, ..., nb, ...);                 // unchanged body: same dequant/mma/fixup order (V1)
}
// 4. smallm: flatten z >= n_act (grid.z = n_act_max), early exit `if (z >= n_act) return;` -> no second pass at bs 1 (H3)
```
Prediction (est.): 3 CTAs/SM × 4 warps = 12 warps × 3 KB = 36 KB in flight per SM; duty cycle 55 → 75 % with NB = 4 ⇒ 1.7 → ≈ 2.6–2.9 TB/s
on the weight stream ⇒ GEMM1 bs256 322 → 190–230 µs. If C20 shows ptxas cannot hold 170 regs without spills, use MR = 1, D = 2 (half the
batch registers, 1.5 KB/warp in flight) with 4 CTAs/SM — same bytes per SM. The ring (Alt B) stays exactly fable-08 D3 §4 and is only built
on decision (ii). Twin construction rule for all C20 variants: every removed stage folds its inputs into `sink ^= …` which is stored by lane 0
at the end (compiler-deletion trap [03a A6]).

### decode_layer — ordering and the T ≥ 128 gate (C22a'/C22b)
```python
# H6'/H21: router first (own fixup, no fill), then the pooler; only the 3 us routing goes to the side stream.
logits = _router_logits(h_in, router_input, state, path)          # 256-CTA cfg at T <= 32; two cuBLAS hi/lo mm at T >= 128
side.wait_stream(main)
with torch.cuda.stream(side):                                       # routing_kernel reads logits only (2.9 us, not bandwidth-bound)
    topk_w, topk_ids = state["routing"](logits, router_bias, ...)
pooled = _pooler(h_in, state, pooler_path)                          # 19.2 us at T = 1, overlaps the routing, no HBM competitor
main.wait_stream(side)
# H10': the MoE path per T bucket, chosen in prepare() on layer-style generate_input routing (3 seeds), printed in the header
MOE_PATH = {1: "smallm", 8: "smallm", 16: "smallm", 32: "marlin", 128: "largem", 256: "largem"}   # prepare() may flip 32 later (C24)
```
Rules that keep the review clean: the choice depends on T and config only; `largem_route` reads `local_start` from the device tensor
with the wrap rule (fable-08 P2); the self-check compares largem vs Marlin on seeded synthetic inputs in prepare() (never judge inputs);
Marlin stays for T > 256 and on a failed self-check. Predicted per shape in D4 §2.

### attention — only what the audit allows
Already folded: verify uses 64 rows per KV head (8 q-tokens × 8 GQA heads, `MAX_ROWS = 128`, "sep" kernel [05-13]) — Perplexity's
"dedicated verify kernel folding 8 × 8" [04a] is our current design, so no new kernel for that. The question is whether its loads are
pipelined. Audit lines to produce (C29): `#C _full_kernel  ld.global=.. cp.async=.. ldmatrix=.. mma.sync=.. ex2=.. cvt=.. regs=.. spills=..`
and the same for `_full_cat_kernel`, `_attn_kernel`, `_reduce_kernel`, `_gemv_splitk`, `_epilogue_kernel`. If cp.async = 0 on
`_full_kernel`: the Hopper-legal pipeline for 128-B random rows is cp.async (16 B per lane, 8 lanes per row) or plain loads with more rows
in flight per SM; TMA gather4 is SM100+ ([04a] §C, Sources). Try, in this order, as C26' twins: unmasked inner `tl.load` on `[BN, 128]`
tiles with `tl.multiple_of(off, 16)` (lets Triton lower to cp.async), 4 programs/SM (regs ≤ 128), BN 128 with 2 stages. H11 rides along.

### bf16 — TMA GEMV (H18) and the cheap fixes
```python
from triton.tools.tensor_descriptor import TensorDescriptor         # host-side descriptor (Triton >= 3.3; 3.8 in the image)
@triton.jit
def _gemv_tma(x_ptr, w_desc, out_ptr, M, N, K, stride_xm, stride_om,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n, pid_k, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM); rk0 = pid_k * (K // SPLIT)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K // SPLIT, BK):                               # Triton pipelines desc.load with mbarriers (num_stages)
        w = w_desc.load([pid_n * BN, rk0 + kk])                       # [BN, BK] bf16 tile via cp.async.bulk.tensor (A13: verify in PTX)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + (rk0 + kk + tl.arange(0, BK))[None, :], mask=rm[:, None] < M, other=0.0)
        acc += tl.dot(x, tl.trans(w))
    # split-K: write FP32 partials; last CTA (counter) sums in slice order and writes BF16 -> no torch.zeros, no atomics (H13 fixup)
# host: w_desc = TensorDescriptor.from_tensor(W, [BN, BK]); BK <= 256 (boxDim <= 256), BN*? strides 16-B aligned, base 16-B aligned
```
Constraints verified against the CUDA driver API (Sources): `boxDim[i] ≤ 256`, inner box bytes a multiple of 16, global strides a
multiple of 16 B, base 16-B aligned — so BK = 256 (512 B) with 4–6 stages (16 KB per stage) and `num_warps = 4`; weights live in state
(stable pointers) so the descriptor is capture-safe; inputs are fresh values in the same buffers under graph replay (as today).
Cheap fixes as fable-08 App. F (G4 epilogue grid, G3 unpooler, router fixup) unchanged.

---
## D8. 10-day schedule (UTC days; checks ≤ 20 and submits ≤ 10 per problem per day; plan uses ≤ 5 / ≤ 2)
| day | moe | decode_layer | bf16 | attention | builders / lanes | decision point |
|---|---|---|---|---|---|---|
| Oct 25 (D0) | C20 build starts (stamps + the 9 twins) | **C19 v5 port: check + promote** (2 chk, 1 sub) | — | **C29 PTX/SASS audit** (0 chk) | director: audit; Sonnet-A: C19 → C22a' harness; Sonnet-B: C20; Jules: C28 okcdbg L2/L7 + parse_stamps; Grok Bot: red team C19 | 22:00Z crow.audit-attention |
| Oct 26 (D1) | **C20 measurement** (1–2 chk) | **C22a' measurement**: ORDER/RPAR/RBIG/TMAPOOL/OSHOT + C24-lite largem-vs-Marlin (1–2 chk) | C23 build (cheap + TMA/OSHOT variants) | — | Sonnet-B: C20 → start C21 scaffolding; Sonnet-A: C22a' → C22b; Argon: review C20 twins for deletion traps; Grok Build: router fixup + epilogue grid drafts | evening: crow.c20-decision → C21 Alt A or C21b |
| Oct 27 (D2) | **C21 Alt A** build + NVRTC + V1 bitwise; first judge compile (1–2 chk) | **C22b v5b** = v5 + ORDER + RPAR + RBIG + largem T ≥ 128 gate: check + promote (2 chk, 1 sub) | **C23 measurement** (1 chk) → **isolated submit** (1 sub) | — | Sonnet-B: C21; Sonnet-A: C22b; Jules: ratchet tests R1–R4 | A1 calibration recorded (crow.a1-calibration) |
| Oct 28 (D3) | C21 sweep LB × Kc × NB (2 chk) → promote if > board (1 sub) | C25 G2 norm fusion + G3/G4 (+V3/V4): check + promote (2 chk, 1 sub) | — | C26' measurement if audit said 0 cp.async (1 chk) | Sonnet-B: C21 tuning; Sonnet-A: C25; Argon: C21 diff review | 18:00Z crow.moe-gate (Alt A ≤ 240 µs on bs256?) |
| Oct 29 (D4) | smallm on Alt A + ZFLAT + H9 (2 chk, 1 sub) **or** C21b ring day 1 | **C24 v6**: Alt A GEMMs for all T, per-bucket gate; trace (make_trace) of v5b/v6 (2 chk, ≤ 1 sub) | — | — | Sonnet-B: smallm / ring; Sonnet-A: C24 | 21:00Z rank review; order of D5–D7 by Δ/check |
| Oct 30 (D5) | ring tuning or epilogue fused into GEMM2 fixup (1–2 chk) | trace-driven residuals (1 chk, ≤ 1 sub) | — | C26' fix check if gather-only ≤ 160 (1 chk, ≤ 1 sub) | Sonnet-A: attention fix or layer residuals; Sonnet-B: moe | A8 answer needed only if NOCVT ≥ 60 µs |
| Oct 31 (D6) | buffer / re-sweep once (1 chk) | buffer / fixes | — | — | Jules: C27 S0 (diff_rules, packaging, smoke); Sonnet idle time: regressions | crow.serving-s0 |
| Nov 1 (D7) | ring → layer port if ring won (1 chk, ≤ 1 sub) | same | — | H11 rider promote if ≥ 1.70 | Sonnet-A/B | |
| Nov 2 (D8) | **C30 hold-out**: every board kernel once more (1 chk each problem) | same | 1 chk | 1 chk | Grok Bot: X scout (ideas only); interns: nearby-shape lists | |
| Nov 3 (D9) | **freeze** (crow.freeze): promote only already-judged | same | — | — | director: handover, BEVINDINGEN, ratchet goldens | |
| Nov 4 (D10) | last submits ≤ 18:00Z; verify board rows | same | — | — | — | Nov 5 07:59Z close |
Dispatch order: Sonnet-A = C19 → C22a' → C22b → C25 → C24 → (C26' fix) → residuals; Sonnet-B = C20 → C21 → (C21b) → smallm → epilogue
fusion → layer port. Capacity (est.): ≈ 28 checks moe, ≈ 24 decode_layer, 4 bf16, 4 attention = 60 of 800; binding constraint ≈ 120
builder-hours (A10): C20 8, C21 10 (+ ring 35 if chosen), layer C19/C22/C24 24, glue 12, okcdbg + audit + ratchets 10, bf16 6, attention
4–12, buffer ≥ 20. Done-when lines: C19 = fable-08 App. C; C20 = log tail holds `B moe` lines for 9 variants × 2 GEMMs × 3 shapes × 3 seeds
+ `#K/#C` lines, `parse_stamps.py` emits the D4 §1 table; C21 = `ring/altA bitwise=OK` printed for 9 M values × 3 seeds, GEMM1 bs256 ≤ 240 µs
in some variant, promote when CI-low > 1.436; C22a' = `B layer` lines for 6 variants × 6 T × 3 seeds with side_end/main_idle and the
largem-vs-Marlin ratio per T; C22b = 13 tests pass, `okc check` ≥ 1.38, promote; C23 = fable-08 C23 + TMAPOOL/OSHOT rows, isolated submit
≥ 1.19; C25 = V3 ≤ 1 ulp and check ≥ v5b + 0.02; C28 = `OKC_DEBUG=100` smoke green, R1–R4 green on the board kernels; C29 = audit table
for the 6 Triton kernels with cp.async/TMA/spills columns.

---
## D9. Risk register with watchers
| risk | what it does to us | watcher / response |
|---|---|---|
| board moves faster than we do (overall 492 → 335 in a day was us; others do the same [04c]) | rank targets slip while speedups rise | daily digest (Grok Bot); plan by Δ-speedup; never spend a check for rank alone |
| SGLang upstream MegaMoE/MegaAttention [04c] | serving baseline may be newer SGLang → margins shrink | kernel track: image fixed (a55ab7d5) → no effect; serving: S0 starts Oct 31 anyway; ask organisers (Dex) which commit serves |
| hand review of top entries / reruns on held-out shapes [05-10] | removal or re-scoring | every candidate passes the full test list (bs 33/257/skewed, ctx 1M …); prepare()-time choices are shape-only and printed; nothing cached outside state; Grok Bot red team per sha; `okc withdraw` first if in doubt |
| "judge calls on copies of prepare() output" [04d, unconfirmed] | stale closure buffers pass locally, mis-time or fail on review | audit each kernel file for tensors referenced outside the returned state dict (counters, LUTs, streams); make the self-check run on the returned state only |
| republishing arena [04c] | our code copied; exposure if anyone on our side reads it | policy: never open, never submit; scout reads public posts only; no response needed if copied |
| tolerance: bf16 1 % / router 0.01 % / moe 1.5 % / attention & layer 3–15 % [05-15] | FP8 anything in projections fails [04c]; attention P·V unknown | R-rule: numerics-changing variants get their own vector before any check; FP8 P·V stays unbuilt (A8) |
| Triton 3.8 pipeliner silently inert ([F1]) in a new kernel | a "pipelined" Triton candidate ships with 0 cp.async | R3 ratchet + audit table on every candidate; TMA variants must show `cp.async.bulk.tensor` offline before a check |
| compiler deletes a twin's work (NOCOMPUTE/NOEXP) | a twin reads as "free" and misleads the decision | sink fold stored by lane 0; a twin faster than WSTREAM/CAL is rejected as invalid (P6) |
| routing-seed variance (moe bs16 112–173 µs p10–p90 [01]) | ±15 % kernel changes invisible | ≥ 3 seeds, paired ratios vs pivot, distinct/chunks printed; CI-low promote rule |
| judge queue / 15-min cap near the deadline | lost checks | freeze Nov 3; last submit Nov 4 18:00Z; ≤ 12 nvcc instantiations per check |
| builders short (usage limits, 4 GB cap) [05-08] | schedule slips | heavy.sh scopes; cut by D6 rank (attention first, then bf16 TMA, then ring) |
| A1 wrong (overall = rank-sum) | bf16/attention worth more than planned | isolated bf16 submit D2 → crow.a1-calibration re-orders D6 |
| Hopper residency surprises (3 CTAs/SM asked, 2 given) | Alt A under-delivers | C20 prints max concurrent CTAs/SM per variant; `-Xptxas -v` regs in the log |

---
## D10. Open questions + assumptions `[An]` with the cheapest test
| id | question / assumption | cheapest test | owner |
|---|---|---|---|
| Q1 (A8) | FP8 P·V legality (does baseline.py's FA3 call quantise P to e4m3?) | read baseline.py + FA3 fp8 path in the image (10 min, Dex/director); CPU tolerance test with P e4m3 vs the attention bound; no build before both say yes | Dex |
| Q2 (A1) | overall board formula | isolated bf16 submit (C23) → overall delta vs time-weighted vs rank-sum | director |
| Q3 (A3/A6) | memory pipeline vs compute; PDL artefact | C20 (one check) | Sonnet-B |
| Q4 (A4) | do occupancy + stage-once reach ≥ 2.8 TB/s on this layout? | C20 LB/KC512/NBLOOP twins + WSTREAM ceiling | Sonnet-B |
| Q5 (A7) | layer routing E / tokens per expert at T 128/256 on the judge | C22a' prints distinct/chunks per seed | Sonnet-A |
| Q6 (A9) | router squeeze = HBM sharing + 64 CTAs? | C22a' ORDER + RPAR: side_end ≤ 48 µs at T = 1 | Sonnet-A |
| Q7 (A13) | Triton 3.8 host TensorDescriptor pipelines a BM = 16 `tl.dot` K loop on sm_90 | offline AOT PTX of `_gemv_tma`: `cp.async.bulk.tensor` + `mbarrier` present; `n_spills = 0` | director |
| Q8 (A14) | offline AOT PTX = judge PTX | first measurement check prints the judge-side histogram; diff | Jules |
| Q9 (A15) | achievable HBM on the judge box | CAL copy kernel in every prepare() | Sonnet-B |
| Q10 | is `cuobjdump` in the image (SASS counts) and does `-Xptxas -v` reach the log tail via `load_inline(verbose=True)`? | one line in C20's prepare(): `shutil.which("cuobjdump")`; print the build log through the budgeted printer | Sonnet-B |
| Q11 | does `__launch_bounds__(128, 3)` compile without spills for largem (ptxas cap 170 regs)? | NVRTC `-Xptxas -v` equivalent on the dev box for sm_90 (regs/spills in the output) before C21's check | Sonnet-B |
| Q12 | web-search hygiene: a search surfaced an excerpt of a competitor's public repo (named in [04d]); excluded here | add that repo/domain to the agents' search exclude list; note in the scout's rules | director |
| Q13 | per-bucket gates monotone in T (reruns on nearby shapes) | C22b prints the chosen path for T ∈ {1..8,16,17,32,33,128,256,257} | Sonnet-A |
| Q14 | does the judge's graph replay keep input buffer pointers stable (TMA descriptors on x would need it; we only put W in descriptors) | not needed if descriptors cover weights only; if ever used on x: a check that prints `x.data_ptr()` across warm-up calls | — |
| Q15 (A12) | 16 KB tail holds C20's 9-variant summary | summary first, ≤ 12 lines/shape; `OKC_DEBUG=100` smoke on a stored log | Jules |
| Q16 | trace span vs judge median discrepancy on moe #1 (83 vs 140 µs) | print the judge's per-seed times next to the trace seed in C20; if the judge's seeds are heavier, predictions scale by the ratio | director |

## Appendix A. Arithmetic behind the two load-bearing predictions (so the director can re-derive them)
### A.1 Little's-law budget for the MoE GEMMs (est. unless marked)
| quantity | value | source |
|---|---|---|
| per-SM share of 4.8 TB/s | 36.4 GB/s | 4.8 TB/s / 132 SMs |
| per-SM share at a calibrated 4.1 TB/s | 31 GB/s | A15 |
| loaded DRAM round trip | 0.8–1.0 µs (unloaded "a few hundred ns") | arXiv 2608.15764; 2501.12084 |
| bytes in flight per SM needed at 4.1 / 4.8 TB/s | 25–31 KB / 29–36 KB | share × latency |
| largem per-warp prefetch batch | MR 2 × D 2 × (16 + 8) B × 32 lanes = 3 KB | [05-11] `prefetch_batch_l` |
| warps per SM today | 8 (2 CTAs × 4 warps; regs 162–180, smem 67 KB) | [01 D3] |
| raw in flight per SM | 24 KB | 8 × 3 KB |
| duty cycle (staging prologue + fixup tail + wave ends) | ≈ 55 % | stamps will measure; implied by 1.6–2.1 TB/s observed |
| effective in flight | ≈ 13 KB → 13/31 × 4.1 ≈ 1.7 TB/s | matches [05-03 moe] 1.6–2.1 |
| Alt A: 12 warps × 3 KB, duty 75 % | 27 KB → ≈ 3.5 TB/s cap; planning number 2.6–2.9 after tails | D7 |
| Alt B: 8 warps × 7 KB (S = 6), duty 75 % | 42 KB → bandwidth-capped (≥ 3.2 TB/s) | [01 D3 §4] |
Bytes per CTA at bs256 (largem, Kc 1024, 128 rows): weights 128 × 1024 × 0.5625 B = 72 KB (HBM); activations staged 32 rows × 2 KB = 64 KB
(L2, serialised 16-B loads: 32 per thread) → the prologue is not negligible, which is why NOSTAGE and NBLOOP are twins, not afterthoughts.

### A.2 decode_layer loads with H10' (v4 largem at T ≥ 128) and H5 (C19), judge medians [01 D4 §3] as base
Largem time in the layer scaled by expert count from the moe problem's judged kernels ([05-03 moe]: bs128 GEMM pair 350 µs at E ≈ 22.8;
bs256 484 µs at E ≈ 24.4 [05-05]) to the layer's E (18.4 at T 128, 21.3 at T 256 [05-05]): 350 × 18.4/22.8 ≈ 282 µs; 484 × 21.3/24.4 ≈ 422 µs.
Marlin pair today: 398 (T 128) / 510 (T 256) [05-03 layer #8, #10]. Savings: ≈ 116 / 88 µs per shape (conservative; the D4 §2 table uses
the bytes-at-2.0 TB/s variant, 138 / 209 µs, as the optimistic bound — the check decides which holds).
| load | weights × shapes today (µs) | with H10' (conservative) | SGLang = today × board speedup | new speedup |
|---|---|---|---|---|
| verify16 | 46 × 502.2 + 16 × 664.2 = 33 728 | 46 × 386 + 16 × 548 = 26 524 | 33 728 × 1.053 = 35 516 | 1.34 |
| verify32 | 46 × 681.2 + 16 × 1017.4 = 47 614 | 46 × 593 + 16 × 929 = 42 142 | 47 614 × 1.034 = 49 233 | 1.17 |
Geo-mean over the six loads with ln values [0.311, 0.298, 0.129, 0.402, 0.293, 0.157] → 1.30 (conservative) ; with the optimistic
savings → verify16 1.41 / verify32 1.42 → 1.36; adding C19 (decode16 1.347 → 1.445, decode32 1.138 → 1.273) → 1.34 / 1.40. D4 §2's
1.46 adds H6' + H21 (−10 µs on the six small shapes, −8..−12 on #8–#11). Kill threshold for v5b (1.33) sits at the conservative case.

### A.3 Router squeeze (shape #0) — the numbers behind A9
Pooler 67.1 MB [05-05] in 19.2 µs = 3.5 TB/s on stream 0 (36.1 → 55.3). Router hi+lo 8.4 MB [05-05 bf16 #4] alone: 5.5 µs (1.5 TB/s,
64 CTAs). Overlapped: 40.8 → 62.8 = 22.0 µs, i.e. 7.5 µs after the pooler ended — at least its own standalone time — so during the 14.5 µs
of overlap it moved ≈ 8.4 − 7.5 × 1.5 TB/s ≈ 0 MB (est.): bandwidth starvation, not residency ([F2]: 24 KB + 12 KB smem fit one SM
together with room to spare). Serialised sum 75.5 MB / 3.5 TB/s = 21.6 µs vs the observed 26.7 µs (36.1 → 62.8): the extra ≈ 5 µs is the
router's own low parallelism once alone + the 3.4 µs fill. ORDER + RPAR therefore predicts side_end ≈ 36.1 + 3.5 + 2.9 ≈ 42.5 µs and the
main stream free at 55.3 instead of 65.6 (−10 µs at T = 1); at T = 16/32 the router is 14.3/10.4 µs beside the pooler [05-03 layer #2/#4]
→ ≈ −6 µs. The 1 KB fill at 3.4 µs beside the pooler (0.67 µs alone) is the same effect on a kernel with one CTA.

### A.4 What one measurement check costs and holds (prepare() ≤ 300 s, run ≤ 15 min, tail 16 000 chars)
C20: 9 CUDA variants × 2 GEMMs = 18 template instantiations — too many for one nvcc build at ≈ 5–10 s each if compiled separately;
compile them as ONE extension (`load_inline` once, ≈ 2–3 min) with variant selected by template flags; timing: 9 variants × 2 GEMMs ×
3 shapes × 3 seeds × 5 trials × ≈ 0.4 ms ≈ 2 s plus L2 flushes (64 MB read ≈ 20 µs each) — negligible; printing: 3 shapes × 12 lines ≈
3.6 KB + `#K/#C` ≈ 2 KB → fits. C22a': 6 Triton configs (≈ 15 s each JIT ≈ 90 s) + the layer's existing self-check ≈ 50 s + largem vs
Marlin at 3 T × 3 seeds ≈ 10 s → ≈ 150 s. If a check's prepare() approaches 240 s, split by shape class, never by dropping seeds.

## Sources (public web; pack files cited by id above)
Triton `make_tensor_descriptor` (16-B aligned base, 16-B stride multiples, 2–5-D): https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html ·
host `TensorDescriptor.from_tensor` and "strides of TMA descriptors must be 16-byte aligned": https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html ·
Triton 3.8.0 release notes (tensor descriptors as kernel args, incl. in tuples): https://newreleases.io/project/pypi/triton/release/3.8.0 ·
`CompiledKernel.n_regs / n_spills / metadata.shared / asm`: https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py ·
CUDA driver `cuTensorMapEncodeTiled` constraints (boxDim ≤ 256, inner box bytes multiple of 16, strides multiple of 16): https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html ·
Triton cp.async vs `cp.async.bulk.tensor` PTX on Hopper: https://pytorch.org/blog/hopper-tma-unit/ ·
Little's law for plain global loads on H100 ("order of ten kilobytes per SM in flight", unloaded latency a few hundred ns): https://arxiv.org/pdf/2608.15764 ·
Hopper tuning guide (228 KB smem/SM, 64 warps, TMA): https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html · H200 4.8 TB/s, 132 SMs, 50 MB L2: https://www.nvidia.com/en-us/data-center/h200/ ·
Marlin (4-stage cp.async pipeline; "not yet optimized for Hopper"): https://github.com/IST-DASLab/marlin · Machete: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/Readme.md ·
FlashInfer page-size-1 decode 83–85 % BW on H100 (MLSys 2025 supplement): https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Supplemental-Conference.pdf ·
TMA `gather4` requires SM100+: https://github.com/vllm-project/vllm/pull/42436 · FA3 (FP8 P·V method): https://arxiv.org/abs/2407.08608 · FA4 exp emulation: https://arxiv.org/abs/2603.05451 ·
tinygrad DEBUG ladder / ratchets / process replay: https://github.com/tinygrad/tinygrad (docs/developer/speed.md; test/external/process_replay) ·
Hopper mma.sync ceiling and latencies: https://arxiv.org/html/2402.13499v1, https://arxiv.org/html/2501.12084v1 · H100 empirical roofline 84.5 %: https://par.nsf.gov/servlets/purl/10539988 ·
PDL semantics: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html · okc rules: https://openkernelchallenge.dev/rules.

---
## Director erratum (2026-10-25 ~14:00Z, after the C29 audit — read before D4 §2/§3, D6 H18/H19, D7)
- **[F1] is retracted.** The director's probe (tools/ptx_probe_gemv.py) compiled without Triton's 16-byte divisibility attributes. A real
  launch of `_gemv_splitk` pipelines: cp.async 24–96, ld.global 0 (design/breakdown/ptx-audit.md, tools/ptx_audit.py). No shipped Triton
  kernel was found with num_stages > 1 and 0 cp.async. → H18 (TMA GEMV) drops to "only if it wins on its own"; R3 ratchet keeps.
- **[F2] is retracted too:** the probe's 24 KB was the no-attribute compile; real pooler configs reserve **131–147 KB smem per CTA (1 CTA/SM)**.
  The smem co-residency mechanism (fable-08 G1) is back as a live explanation next to HBM sharing (A9); C22a' tests ORDER, RPAR and POOLSMEM.
- **Attention `_full_kernel` issues cp.async (32, wait_group 1, 4 stages)** → crow.audit-attention condition ("0 cp.async") is false: C26' is
  not scheduled; attention verify-full is pipelined and compute/MUFU-bound as fable-08 D5 argued.
- **MoE CUDA GEMMs: 0 cp.async (register-prefetch streaming), regs 162–180 largem / 96–98 smallm** — consistent with the Alt A / ring analysis.
