# itachi-corridor: decode_layer handovers between kernels ("the corridor")
Status: **Adopted by Dex 2026-10-26 (chat: "Build the tests on these thesis … /itachi /izanami")** · harness: itachi/v0.1 · author: director
Inputs: design/research/construction-analogy-answers.md (the builder's answers + the corridor answers), judge job efc7af96cfa347ca (board
v5f 1.433), design/breakdown/muse_decode_layer.md, fable-09 D6 queue, fable-08/09 PDL notes.

## 1. True objective
Raise the muse_decode_layer board entry above 1.433 (target ≥ 1.47 by 2026-11-04) by cutting time lost BETWEEN kernels, measured by
the judge's paired speedup. Non-goals: MoE kernel internals (parked, C31), attention internals (done today), new numerics.
Means test: any other way to ≥ 1.47 counts; this design is the cheapest route we see because T = 1 runs at 2.7× its floor.

## 2. Constraints and failed classes
- Judge: 10 submits + 20 checks per problem per day; prepare() ≤ 300 s; no state across calls; cold L2; CUDA graph replay.
- ≤ 2 Sonnet builders; Fable theory-only; Grok Build ≤ $3/day (read-only drafts); Jules (pure Python, GitHub main); Argon quota guard.
- Failed classes (do not re-propose without new evidence):
  - "PDL on Triton launches with gdc_wait as the first statement" = noise (fable-09 table, v3_s13 [05-07]). **New here:** work before the wait.
  - Megakernel / single launch (fable-09 dropped); S2 fused routing; TMA GEMV (C23); OSHOT; FP8 operands for projections.
  - "two crews in one room" smem co-residency (router/pooler squeeze, fixed by ps1; do not re-grow pooler smem).

## 3. Environment model (what resists)
| resistance | cheapest early observation |
|---|---|
| cuBLAS (nvjet) kernels cannot take our gdc_wait/prefetch: 2–3 of the T = 1 chain | profile 0 lists them; count only our kernels as targets |
| L2 prefetch of weights competes with the finishing kernel for DRAM ("the corridor") | variant TOOLS_FULL vs TOOLS_LIGHT: if FULL < LIGHT, congestion is real |
| CUDA graph capture may drop/ignore stream priorities or PDL attributes | measurement prints whether the PDL edge exists (stamps: next kernel's first CTA start < previous end) |
| 50 MB L2: prefetched lines may be evicted before use at T ≥ 16 | gain shrinking T = 1 → 32 |
| judge noise ±1.5 % per problem | promote rule (CI-low > board CI-low) |

## 4. Alternatives (with switch conditions)
- **ALT-1 Fusion first (L-B, "never go out for one thing").** Fold norm/residual/gate glue into the consuming GEMV or producing epilogue.
  Cost 1–2 builder-days, numerics risk (one BF16 rounding). Switch to it first if L-A1 gains < 3 µs at T = 1 (handovers are not the loss,
  kernel count is).
- **ALT-2 Bigger carry per trip in the small GEMVs (q5).** BK/BN/SPLIT sweeps for _gemv_splitk / unpool / router at T ≤ 16. Cheap (config),
  but fable-09 sweeps already gave ≤ 2 % on most; switch if the corridor stamps show the kernels themselves (not the gaps) dominate.
- **Chosen: L-A1 + L-A2 measured together in ONE check, then L-B on its result.**

## 5. Mechanism
The builder's rules → mechanisms (K = enforced by the judge/tests, S = served, P = policy):
- **R1 "tools in the corner while the other crew finishes" (L-A1, S):** launch each of OUR kernels with PDL; before `gdc_wait` it issues a
  LIGHT L2 prefetch of the weight bytes it will stream first (its first k-stages, `cp.async.bulk.prefetch.L2` / `prefetch.global.L2`),
  then waits for the activations [A1][A2]. Weights never depend on the previous kernel; activations always do.
- **R2 "the crew that exits goes first" (L-A2, S):** the previous kernel keeps priority: (a) the prefetch is capped (LIGHT = first ≤ 64 KB
  per CTA, est.), (b) PDL places the dependent's CTAs only on SMs the previous grid frees [A3], (c) the side stream (router ‖ pooler at
  T ≤ 8) gets low stream priority, the critical path high [A4].
- **R3 "prepare everything that is not a specialty, with high confidence; don't do their job" (S):** pre-wait work = address math, weight
  prefetch, smem/barrier init, scale LUT loads. Never activations, never partial results (that would be the job itself — and racy).
- **R4 "in one motion, also prepare for others" (L-B one-motion, S):** a kernel that already has x in registers writes the normalized copy
  its next consumer needs (epilogue writes rmsnorm(s) / gated input) instead of a separate norm kernel.
- **R5 "never go out for just one thing" (L-B, S):** a kernel < 5 µs at T = 1 that only reads/writes [T, 8192] is fused into a neighbour.
- **R6 "meister in the boiler room" (L-C, S):** the expert GEMMs (the specialty) start their weight prefetch while the router/routing glue
  runs; for experts only the SHARED parts are known before routing (scales LUT, the first expert slots once routing writes them) [A5].
- **K:** every variant bitwise or within the layer tolerance vs v5f; tests on the 2060; promote rule; no work outside the timed call
  (all prefetches happen inside kernel_fn's launches, cold L2 respected).

## 6. Assumption register
| id | assumption | if false | watcher |
|---|---|---|---|
| A1 | at T = 1 each weight-streaming kernel loses ≈ 1–2 µs ramping DRAM from cold (est.) | L-A1 gains < 3 µs | check C33: TOOLS_LIGHT vs PDL_OFF at T = 1 (kill line) |
| A2 | Triton 3.8 can emit an L2 prefetch before gdc_wait (inline asm) and keep PDL in graph capture | variant does not build / no overlap | C33 build + stamps (first CTA start < previous end) |
| A3 | PDL dependents occupy only SMs freed by the previous grid | prefetch steals the finishing kernel's SMs | C33 TOOLS_FULL vs TOOLS_LIGHT |
| A4 | stream priority survives graph capture on CUDA 13 | L-A2 is a no-op | C33 PRIO variant vs none (≥ 1 µs at T ≤ 8 or drop) |
| A5 | expert weight prefetch before routing completes is impossible except shared parts | L-C needs routing first | accepted-risk (director): L-C limited to post-routing early start |
| A6 | 3–5 µs glue kernels at T = 1 are fusable without a second BF16 rounding | numerics fail | 2060 reference tests per fusion (V3-style) |
| A7 | gains add up across T = 1/16/32 enough for +2–3 % problem | promote fails on CI | TASK load formula prediction on each check |

## 7. Posthumous pack
**Task cards**
- **C33 (Sonnet-B): measure_corridor_v1.py** (MEASUREMENT-ONLY, kernel_fn = sonnet_v5f verbatim). Variants of the whole layer at decode
  T = 1 / 16 / 32 (sliding + full) and verify 1: PDL_OFF (board), WAITFIRST (S3_PDL as built), TOOLS_LIGHT, TOOLS_FULL, PRIO,
  TOOLS_LIGHT+PRIO. 3 seeds, cold-L2 flush, graph replay, overhead subtracted; per-variant layer µs + stamps of handover gaps
  (next first-CTA start − previous last-CTA end) per kernel pair; S lines first, ≤ 14 KB. Done when: tests green (tools/heavy.sh,
  set -o pipefail), PTX shows the prefetch before griddepcontrol.wait, "C33 READY <sha>".
- **C34 (Grok Build draft → Sonnet-A): fusion map L-B** — from the T = 1/16 profiles list each glue kernel, its consumer/producer, fusion
  target, bytes saved, numerics impact; read-only draft in drafts/, then a Sonnet build card for the top 2.
- **C35 (Jules): tools/chain.py** — per shape from a judge profile JSON/txt: kernel chain, our-vs-library, µs, floor share, "ramp suspects"
  (kernels < 8 µs streaming weights); pure Python + tests.
- **C36 (Argon, theory):** "what cuts T = 1 decode from 108 to ≤ 80 µs" given profile 0 and the builder's rules; ideas only, ≤ 40 lines.
**Decision table after C33** (T = 1 decode µs, vs PDL_OFF):
| reading | action |
|---|---|
| TOOLS_LIGHT ≥ 3 µs better, TOOLS_FULL ≤ LIGHT | build sonnet_v5g with TOOLS_LIGHT on all our kernels → check → promote |
| TOOLS_FULL clearly worse than LIGHT | corridor congestion confirmed: keep LIGHT, cap smaller |
| WAITFIRST ≈ TOOLS_LIGHT ≈ OFF | handovers are not the loss → ALT-1 fusion first (C34) |
| PRIO ≥ 1 µs at T ≤ 8 | add priority to v5g |
**Gone-clause:** if the director is unavailable: the 3-hourly routine (<routine>) may merge green C33–C35, run
measurement checks via okc_run.py and promote non-attention entries with OKC_DEX_OVERRIDE when the promote rule holds; nothing on
attention without a red-team file; no new Fable call; decisions outside the table wait for Dex.

## 8. Falsification
- L-A1 is wrong if TOOLS_LIGHT at T = 1 decode is not ≥ 3 µs faster than PDL_OFF (3 seeds, paired).
- L-A2 is wrong if PRIO changes T ≤ 8 by < 1 µs.
- R4/R5 fusions are wrong per fusion if Δ < 1.5 µs or the reference check fails.
- The design is wrong overall if, after C33 + one fusion build, the board stays < 1.45.
Evaluation set: the judge's 12 decode_layer shapes; metric: paired speedup + per-shape µs; cost: ≤ 4 checks/day on decode_layer.

## 9. Adoption record
Decision: adopted by Dex (minister) in chat 2026-10-26; risk class: low (measurement + config/kernel ports, reversible, promote rule).
Rollback: board entry stays v5f until a candidate beats it. First review: after C33's check (same day) and at the 2026-10-29 routine round.
