# fable-10-moe-ws output
executor: Fable (one-shot, design only, nothing run) · 2026-10-26 · inputs: pack 01–14 + public web (URLs in §Sources).
Conventions: a number with a pack id in brackets is measured ([03a] = 03-log-a78c1b9c, [03b] = 03-log-698aa839, [03c] = ab6bedeb, [03d] = 7e8e3866,
[01]/[02] = the kernel sources, [04] ptx audit, [05] trace, [06] bytes model, [08] fable-09, [10] X scout, [11] Perplexity, [14] history).
**Every number marked `est.` is mine** (arithmetic on pack numbers or an engineering estimate). GEMM1 bs256 numbers are seed 11 unless stated.
Reading of the S line used throughout (the stamp code in [01] lines 1589–1653, 1695–1728): `item` = first n-block incl. staging (st[6]−st[0]),
`cta` = CTA lifetime (st[7]−st[0]), the % shares are of `item`, `bw/cta` ≈ weight bytes of one n-block / its stream phase (73.7 KB / (0.67 × 12.0 µs) = 9.2 GB/s
vs the printed 9.6). If the measurement file defines it differently (D6 Q1) the ratios below move by < 10 %, the diagnosis does not.

## 0. Headline
1. **The per-CTA rate (≈ 10 GB/s) is not the mystery; it is the DRAM share of 396 resident CTAs (4.0 TB/s / 396 = 10.1 GB/s).** v5's warps
   stream at ≈ 2.3 GB/s each = 9.2–9.6 GB/s per 4-warp CTA, i.e. exactly the share. What is lost is **time not streaming**: 33 % of each CTA's
   life (stage 19 %, split-K wait 10 %, fix 5 %, epi 4 % [03a S line]) and 29 % of the slot-time (4.36 waves × 39.4 µs = 172 µs of CTA time
   inside a 243 µs span, est. from [03a]). 4.0 × 0.67 × 0.71 ≈ 1.9 TB/s ≈ the measured 1.9–2.2 [03a]. The stream itself never pauses in WSTREAM
   (no staging, no fixup, 4 CTAs/SM, 8 KB per warp in flight) → 4.0 TB/s with the same bytes and addresses [03a]: layout and DRAM locality are not the cap.
2. **Per warp, the stream is a latency chain of ≈ 1 µs per 2.3-KB batch** (two batches in flight, so a ≈ 2 µs effective round trip of which ≈ 0.5 µs
   is the warp's own dequant+mma chain, est.; unloaded Hopper global latency is 656 clk ≈ 0.35 µs [arXiv 2501.12084], the rest is load queueing).
   The ring removed the memory wait (7 batches in flight) and the warp immediately hit its own chain: 410 clk per 576-B tile per warp with only
   2 warps per scheduler [03b: 10.7 GB/s/CTA at 2 CTAs/SM] — and it paid a third of the CTAs for 101 KB of smem. **So "deeper prefetch inside the same warp"
   cannot win; neither can "more CTAs" (capped at 3 by 67 KB smem + 168 regs).** The X evidence ("parallelism, never prefetch tricks" [10]) is the same finding.
3. **Design (D2):** one persistent CTA per SM, 13 warps: 1 producer warp that only issues 1-D `cp.async.bulk` copies (weights 512 B + scales 64 B per tile,
   activation row-slices 128 B) into an 8-stage smem ring with mbarriers, running ahead across work items without ever draining; 12 consumer warps
   (3 per scheduler) that only do lds → NVFP4 unpack → ldmatrix → `mma.sync`, with v5's exact dequant/mma/fixup order (**bitwise = v5**). No staging
   phase, no wave tail (device-side item queue), ≈ 97 KB of weights in flight per SM, consumer regs ≤ 152, smem 148–186 KB. No wgmma, no setmaxnreg,
   no tensor maps in step 1 (plain sm_90 PTX; sm_75 fallback = synchronous copies + spin flags, bitwise-testable on the 2060).
4. **Prediction (est.):** GEMM1/GEMM2 bs256 234/124 → 160/82 µs, bs128 184/95 → 142/73, bs32 103/56 → 79/42, bs16 85/47 → 69/36, bs8 72/42 → 58/31,
   bs1 29/20 → 20/13 ⇒ moe 1.486 → ≈ 1.9 (1.75–2.05). **One judge check decides** (D3): kill the mma.sync-consumer design if the best variant leaves
   GEMM1 bs256 above 205 µs while its own no-compute twin streams ≥ 3.5 TB/s (then the consumer side is the cap → wgmma consumers, D5).

---
## D1. Diagnosis with arithmetic

### 1.1 The budget (GEMM1 bs256, v5 `base` [03a], seed 11)
| quantity | value | source |
|---|---|---|
| weight bytes streamed | 27 chunks × 18.87 MB = 510 MB (unique 453 MB = 24 experts; the 3 two-chunk experts re-read 57 MB) | geometry [01]: 512 groups × 64 it × 576 B; `uniq1=453MB` [03a] |
| time / rate | 233.8 µs → 2.18 TB/s streamed (1.94 unique = the printed `tb`) | [03a] |
| ideal at the WSTREAM ceiling | 510 MB / 4.0 TB/s = 128 µs | [03a] WSTREAM 4.00 TB/s |
| CTAs | grid 16 × 4 × 48 = 3072, active 16 × 4 × 27 = 1728 (z ≥ 27 exit at once), 3/SM × 132 = 396 slots → 4.36 waves | [01] host, `occ=3/3.0` [03a] |
| CTA lifetime / span | 39.4 µs / 243 µs → slot-time used 1728 × 39.4 / (396 × 243) = **71 %** | [03a] S line (est.) |
| per CTA | 4 n-blocks × 73.7 KB = 295 KB in 39.4 µs = 7.5 GB/s over its life; stream phase 9.2–9.6 GB/s | [03a] |
| per warp in the stream phase | 18.4 KB per n-block / (0.67 × 12.0 µs) = **2.3 GB/s = one 2304-B batch per 1.0 µs** | [03a], batch = MR 2 × D 2 × 576 B [01] |
| per SM if every slot streamed | 12 warps × 2.3 = 28 GB/s ≈ the 30 GB/s share of 4.0 TB/s | est. |
| DRAM share per CTA at 3/SM | 4.0 TB/s / 396 = 10.1 GB/s | est. |
So: **stream-phase rate × duty × slots = 3.7 TB/s × 0.67 × 0.71 ≈ 1.8 TB/s** (est.; phases of different CTAs overlap, hence the measured 2.2). The two
loss terms are the non-streaming phases and the empty slot-time, not the stream rate per CTA.

### 1.2 What limits one warp to 2.3 GB/s (the per-batch 1.0 µs)
v5's k-loop ([01] lines 1581–1588) keeps two batches in flight: batch b+1 is issued, batch b consumed. Steady state T = max(C, (L + C)/2) with L = load
round trip as seen by the warp, C = the warp's own chain per batch (lds LUT, unpack, bf16x2 fma, ldmatrix, mma). Readings:
- full: T = 1.0 µs; NOCOMPUTE (dequant+mma removed, loads/ldmatrix/fold kept): stream phase ≈ 6.2 µs per n-block, est. from +15–21 % [03a] → T ≈ 0.78 µs, C ≈ 0.1.
- ⇒ **L ≈ 1.45 µs, C ≈ 0.5 µs** (est.): the loaded round trip is ≈ 4× the unloaded 656 clk [2501.12084] (queueing at 2 TB/s + the `.nc` path), and the
  dequant+mma chain is ≈ 250 clk per 576-B tile per warp (≈ 90 instructions: 10 int ops + 2 bfe + 2 lds + 4 fma.bf16x2 + NT mma per word, × 4 words,
  + NT ldmatrix; mma.sync m16n8k16 latency 24 clk [2501.12084], 4 independent accumulators at MR 2 × NT 2).
- Hence NOCOMPUTE gains only 15–23 %: C is a quarter of the round trip; **the warp is latency-bound with 4.6 KB in flight**, and Little's law per SM
  (12 × 4.6 KB = 55 KB; 55 KB / 1.45 µs = 38 GB/s) says the SM could just about feed its share — if it streamed all the time.
- **The ring [03b]**: 7 × 1152 B = 8 KB per warp in flight, so the memory wait is gone (8 KB / 1.45 µs = 5.5 GB/s available) — yet 2.7 GB/s per warp =
  0.41 µs per 1152-B batch = **410 clk per tile**: the warp's own chain grew (wait_group → syncwarp → 2 × (lds.128 + lds.64) → 64-bit address math →
  2–4 cp.async → commit → dequant → mma; the lds→unpack dependency is exposed with only 2 warps per scheduler at 2 CTAs/SM). Per SM: 8 warps × 2.7 = 22 GB/s
  stream-phase × 57 % duty [03b] ≈ 12 GB/s ⇒ 1.6 TB/s ✓ (measured 1.62). The ring moved the limiter from memory latency to the issue chain and lost 33 % of the warps.
- **WSTREAM**: 16 independent 16-B loads per lane = 8 KB per warp, no chain, 77 regs → 4 CTAs/SM [03a]: 16 warps × 8 KB = 128 KB per SM in flight and
  100 % duty ⇒ 128 KB / 1.45 µs = 88 GB/s ≫ 30 ⇒ DRAM-bound at 4.0 TB/s ✓. The same bytes and addresses: **the 576-B tile layout and DRAM page locality are not the cap.**

### 1.3 The other readings, each explained by the same model (est. where not bracketed)
| reading | explanation |
|---|---|
| LB3 alone +1–3 % [03a] (8 → 12 warps/SM) | the `lb3` twin keeps NB 1: every 73.7-KB n-block carries a 64-KB L2 staging prologue through 128 threads (STAGE8: 8 loads per thread in flight); a third CTA adds a third serialized staging chain more than a third stream. Occupancy is necessary, not sufficient. |
| NB2/NB4 +8–13 %, NOSTAGE +14 % [03a] | the two halves of the staging loss (19 % of the item at NB 4; ≈ 45 % at NB 1). Still 19 % because the first n-block of every CTA stages, and every CTA has only one chunk (nz 48 ≥ 27 chunks: no reuse across chunks). |
| KC (Kc 512, KS 8) −15 % [03a] | halves the stream per item, keeps the fixed costs (stage, fill/drain of the 2-deep pipeline ≈ one L per n-block, fixup with 8 slices, partials ×2). Any design with per-item fixed costs loses at small Kc. |
| NB4 ≈ NB2 at bs 256, NB4 worse for GEMM2 [03a] | fewer, longer CTAs → worse wave quantisation (GEMM2: 16 × 4 × 27 = 1728 CTAs at NB 2 → 864 at NB 4 = 2.2 waves). |
| ring 82 (D 1 RS 8 × 2 groups) < ring, nb1 < nb2 < nb4 [03b] | more work per item per warp lengthens the chain; fewer n-blocks per CTA shortens the drain… both second-order next to the −33 % residency. |
| sweeps ≤ 2 % [03c][03d] | knobs that do not change the duty cycle or the per-warp structure cannot move a kernel whose stream phase already runs at the share. |
| v5 faster at small M (2.4–2.7 TB/s at bs 8–32 [03a]) | NT 1–2 ⇒ C ≈ 0.25 µs ⇒ T ≈ 0.85 µs; fewer chunks ⇒ fewer waves ⇒ less tail. Consistent with C being part of the chain. |
| regs 162–180, 0 cp.async [04] | register prefetch costs 48 regs per thread (2 × MR × D × 6) and forces LB 3 to spill 20 B [03a ptxas]: the registers are the price of in-warp buffering, which is the thing to remove. |

### 1.4 Verdict on the brief's lean
"Bounded by the serial per-warp chain, not by bytes in flight" is half right: v5 is latency-bound per warp (4.6 KB in flight, L ≈ 1.45 µs), and the moment
depth is added (ring) the warp's own chain binds at ≈ 410 clk per tile. Both bounds sit at ≈ 2.3–2.8 GB/s per warp, so the only ways up are **(a) more compute
warps per SM that never wait for memory, (b) a loader that is not a warp's instruction stream, (c) a stream that never pauses for staging, fixup, epilogue or
wave tails.** A producer/consumer split gives (a) and (b); the item queue + act-slice streaming give (c). `setmaxnreg` is not needed at 13 warps (D2 §2.3).
What else it is not: ldmatrix/smem (≤ 50 % of 128 B/clk at NT 4, est.), DRAM locality (WSTREAM), the fixup order (5 %), PDL (pdloff = noise [03a]).

---
## D2. Design: `w4_moe7_kernel<FUSE, D, DBG>` — persistent, warp-specialised, bitwise v5

### 2.1 Work decomposition
- **Item** = (chunk z, K slice ks, group range ng): 24 consecutive 16-row weight groups (GEMM1: 512 groups → 22 items per (z, ks), the last with 8 groups;
  GEMM2: 256 → 11 items, the last with 16), one K slice of Kc = K/KS (KS 4 default, runtime). Bytes per full item: 24 × (Kc/64) × 576 B = 221 KB (Kc 1024).
- **Items per call**: n_chunks × KS × NI (NI shape-only). bs256 GEMM1: 27 × 4 × 22 = 2376 items ≈ 18 per SM → tail ≤ 1 item ≈ 7 µs (est.). bs1 GEMM1: 264 (2 per SM);
  GEMM2 bs1: 132 (1 per SM). Small M may prefer KS 8 (more items, trivial fixup at NT 1): a runtime knob for the check.
- **Grid** = `multiProcessorCount` CTAs (132), 1 CTA/SM (smem ≥ 148 KB guarantees it), block = 13 warps = 416 threads. Items are handed out by a device
  counter (`atomicAdd` on `ctr[gemm]`, zeroed by the route kernel every call → graph-safe, no host sync) in (z, ks, ng) order, z-major: concurrently running
  CTAs work on the same expert (its activation slices stay hot in L2, its weights stream from the same DRAM pages).
- **Stage** (one ring slot) = the 24 groups' tiles for D k64 iterations (D × 13.8 KB) **plus the chunk's activation slice for the same k range** (32 rows × D × 128 B
  = D × 4 KB, rows at pitch 144 B so ldmatrix stays bank-conflict-free as today's KP = Kc + 8 trick). No staged activation tile, no staging phase: the
  activations ride in the ring (they are L2-resident: x = 2 MB, inter = 16 MB at bs 256).
- Per item: Kc/64 = 16 (D 1) or 8 (D 2) stages. The producer walks items continuously; the ring never drains at an item or chunk boundary (v6's continuity, now free).

### 2.2 Warp roles
```
warp 0  = producer (one elected lane per copy; 24 lanes issue the 24 groups' word + scale copies, lanes 0..31 the act rows)
  for (seq = 0;; ++seq) {                              // item sequence of THIS CTA
    i = atomicAdd(ctr, 1); if (i >= n_items) { write sentinel desc; arrive(full[s]) with expect_tx 0; break; }
    (z, ks, ng) = split(i); desc[seq & 1] = {z, ks, ng, e, mc, slot_ptr, fscale};   // read by consumers after their first full-wait of the item
    for (q = 0; q < nstages_per_item; ++q) {           // ring slot s = (global stage counter) % S, phase = (counter / S) & 1
      mbarrier.try_wait(empty[s], phase ^ 1);          // consumers released this slot (12 arrivals)
      if (lane == 0) mbarrier.arrive.expect_tx(full[s], bytes_w + bytes_a);
      if (lane < 24 && ng*24+lane < G) { cp.async.bulk words  512*D B  -> slot.w[lane];  cp.async.bulk scales 64*D B -> slot.s[lane]; }
      if (lane < mc)                 cp.async.bulk act row 128*D B (act + (slot[lane] >> act_shift)*K*2 + (ks*Kc + q*64*D)*2) -> slot.a[lane];
    }
  }
warps 1..12 = consumers (warp w owns groups ng*24 + 2(w-1), +1 : MR = 2, B fragments shared by both)
  for (seq = 0;; ++seq) {
    wait full of the item's first stage; read desc[seq & 1]; if sentinel break;  NT = mc<=8 ? 1 : mc<=16 ? 2 : 4;  acc = 0
    for (q ...) { mbarrier.try_wait(full[s], phase);
                  for d in 0..D-1: w = lds.128(slot.w[g][d] + lane*16); sc = lds.64(slot.s[g][d] + gid*8); (both groups)
                                   m5_batch body verbatim [01 1546-1560] with bf = ldmatrix(slot.a + lane_addr_rel + (d*4+jp*2)*32);
                  __syncwarp(); if (lane == 0) mbarrier.arrive(empty[s]); }
    m6_finish verbatim [02 2160-2227] (partials, fixup by the last CTA of (z, ng), SwiGLU/BF16 epilogue) with bar.sync 1,384 instead of __syncthreads
  }
```
(≤ 40 lines; the real code reuses `m5_batch`'s inner loop and `m6_finish` unchanged — that is what keeps it bitwise.)

### 2.3 Residency budget (1 CTA/SM; H200: 65 536 regs, 228 KB smem per SM, 227 KB per block)
| resource | per CTA | note |
|---|---|---|
| threads / warps | 416 / 13 | `__launch_bounds__(416, 1)` → ptxas cap ⌊65536/416⌋ = 157 → **152 regs** for every warp, no `setmaxnreg` needed (sm_90a-only, 24–256 in steps of 8 [Colfax CUTLASS tutorial]) |
| consumer regs (est.) | acc 2×NT×4 ≤ 32, bf 16, w+s 12, a/q/sc 10, addresses/loop ≈ 30, temps ≈ 20 → **≈ 120** | v5's 168 included 48 regs of in-warp prefetch buffers [01]; spills must be 0 (R3 ratchet) |
| producer regs | ≈ 40 used of 152 | waste 32 × 112 = 3.6 k regs = 5 %; irrelevant |
| smem ring | D 1, S 8: 8 × (13.8 + 4.6) = **147 KB**; D 2, S 5: 5 × (27.6 + 9.2) = 184 KB | in flight (S−1 stages): 97 KB / 110 KB of weights + act |
| smem other | lut 1 KB, 16 mbarriers 128 B, 2 item descriptors 64 B, s_flag | total **148 KB (D1S8) / 186 KB (D2S5)** ≤ 227 KB → 1 CTA/SM, carveout max as today |
| in flight vs need | need 30 GB/s × 1.45 µs = 44 KB per SM (est.; WSTREAM ran with 128 KB) → 97–110 KB gives 2.2–2.5× margin |
| consumer TLP | 12 warps = 3 per scheduler; required rate 576 B per 38 clk per SM = 152 clk per tile per scheduler; chain ≈ 250–300 clk per tile per warp (no load issue, no wait_group) → 3 warps give 83–100 clk ⇒ **1.5–1.8× margin** (est.) |
| scheduler issue (est., NT 2) | 3 warps × 175 instr per k64 step (2 tiles) = 525 per 920 clk (= 13.8 KB at 30 GB/s) → IPC 0.57 | NT 4: ≈ 0.68 |
| INT pipe (16 lanes/clk/scheduler) | ≈ 96 int ops × 2 clk × 3 warps = 576 of 920 clk → **63 %** | the next cap after DRAM: "unpack the unpack" [10] is the follow-up, not step 1 |
| tensor cores (mma.sync ≈ 490 TFLOPS on H800 = half of wgmma peak [2501.12084]) | NT 2: 3 × 16 mma × 7 clk = 336 / 920 = 37 %; NT 4: 73 % | NT 4 chunks (mc 17–32) are ≈ 30 % of the bytes at bs 256 (est.) |
| smem bandwidth (128 B/clk) | per step: ring lds 13.8 KB + bulk writes 18.4 KB + ldmatrix 12 × 2 × NT × 512 B (24–49 KB) = 56–81 KB / 920 clk → **48–69 %** | MR 2 halves ldmatrix vs MR 1; MR 4 would need 64 acc regs (no) |
| L2 → SM act traffic | 64 KB per 221-KB item (29 %; v5: 64 per 295 KB) = 1.2 TB/s GPU-wide at 4 TB/s weights (est.) | L2 hits; 128-B bulk copies, 32 per stage |
Expected sustained weight stream ≈ 3.3–3.6 TB/s (est.: DRAM 4.0 × tails, INT/TC/smem at 60–73 % with 3-way TLP). The consumer-only twin (D3) measures the consumer ceiling directly.

### 2.4 Synchronisation (CUTLASS `PipelineTmaAsync` pattern, 1-D copies instead of tensor maps)
- `full[s]`: init count 1; producer lane 0 `mbarrier.arrive.expect_tx` (tx ≤ 36.9 KB ≪ 2^20−1 [PTX ISA]); copies complete-tx; consumers `try_wait.parity`.
- `empty[s]`: init count 12; each consumer warp's lane 0 arrives after its last smem read of the slot (`__syncwarp` first); producer waits parity^1.
- Act rows: `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` needs 16-B aligned src/dst and size % 16 == 0 [PTX ISA 8.5+, sm_90]: rows are at
  slot × K × 2 (8 KB multiples) + ks × Kc × 2 + q × 128 ✓, words 512 B ✓, scales 64 B at 64-B offsets ✓ (`Sq[g][it][8][2][4]` [01 pack_w4_tiles]), dst pitches 16-B multiples ✓.
- Visibility: async-proxy writes → mbarrier complete → consumer wait (acquire): no `fence.proxy.async` (consumers never write the ring; the WAR on slot reuse is
  ordered by arrive(empty) → producer wait → next bulk copy, exactly CUTLASS's consumer_release).
- Fixup/epilogue among consumers only: `bar.sync 1, 384` (named barrier; the producer never joins; `__syncthreads` is banned in the kernel body).
- Item descriptor ring of 2: the producer is ≤ S−1 stages ahead ≤ 1 item; the consumer reads `desc[seq&1]` after its first full-wait of the item.
- End: sentinel descriptor + an empty stage (expect_tx 0, one arrival) wakes the consumers; `cp.async.bulk.wait_group` is not needed (no bulk stores).
- PDL: producer `griddepcontrol.launch_dependents` at start; GEMM2 issues its first S−1 weight stages **before** `griddepcontrol.wait`, then the act copies
  (expect_tx already counted both; tx completes when all bytes land regardless of issue order) → GEMM2's ramp overlaps GEMM1's tail legally (weights do not depend on GEMM1).

### 2.5 Numerics
Per (group, ks): k64 iterations in order 0..Kc/64−1, words j 0..3, `unpack_word` → `s_lut` scale → `mul_bf16x2` → `mma16816` into `acc[r][nt]` — v5's `m5_batch` body
unchanged; split-K partials and the last-CTA sum in slice order 0..KS−1 — v6's `m6_finish` unchanged; `write_out` math unchanged ⇒ **bitwise = v5 = v4 at the same KS**
(v6 proved the trick: `moe6 bitwise vs v5 per bucket: all OK` [03b]). The existing prepare() self-check (10 M values × seeded inputs, per-bucket gate, v5 fallback) is reused.
If a bucket needs KS ≠ v5's KS (e.g. KS 8 at bs ≤ 8), it is not bitwise but inside tolerance (rel L2 vs Marlin 5.7e-3 [03c] ≪ 1.5 %): gate that bucket on the L2 check instead.

### 2.6 What stays out of step 1 (and why)
wgmma (needs sm_90a codegen via `TORCH_CUDA_ARCH_LIST=9.0a`, a canonical swizzled B layout — our row-major act slices are not one — and is untestable on sm_75);
`setmaxnreg` (not needed at 152 regs); tensor-map TMA (the row gather is per-row anyway; 1-D bulk copies need no descriptor); clusters/multicast (experts are not shared
across CTAs' items at the same time by construction). Step 2, only if D3 says "consumer-bound": wgmma `m64nNk16` RS form (A = the dequantised fragments in registers, exactly
mma.sync's A layout per warp; B from smem via descriptor; act slices written in the 128-B-swizzle K-major layout by the producer with `cp.async.bulk.tensor.2d` per row),
which removes the 24–49 KB/step of ldmatrix traffic and doubles the TC rate (wgmma reaches peak only at N ≥ 64 [2501.12084]; N = 32 here, still ≥ mma.sync).

---
## D3. The judge measurement check (`measure_ws_v1.py`: kernel_fn = v5 verbatim; prepare() builds + times; 3 seeds × bs {256,128,32,16,8,1}; B/S lines as [03a])
Instantiations (≤ 8, build ≤ 150 s est.; v6's 12 took 157 s [03b] — print the build time): `w4_moe7_kernel<FUSE, D, DBG>` × FUSE {T, F} × D {1, 2} × DBG {0, 1};
S runtime (slot index by compare, not modulo), KS runtime, DBG = 1 selects at runtime one of STAMPS / NOCOMPUTE / CONSONLY (branches outside the hot loop; NOCOMPUTE's fold is a uniform branch per stage).
| variant | removes / changes | expected (est.) | reading → decision |
|---|---|---|---|
| WS (D1,S8) | the design | GEMM1 bs256 **≤ 175 µs** (≥ 2.9 TB/s), GEMM2 ≤ 90; bs1 GEMM1 ≤ 22 | ≤ 175 and bitwise OK on every bucket → promote as sonnet_v7 |
| WS (D2,S5) | 1-KB copies, 110 KB in flight | within ±5 % of D1S8 | if > +5 % → DRAM-side locality/depth mattered: ship D2 |
| WS + STAMPS | per-CTA: items, consumer time in `try_wait(full)` (%), producer time in `try_wait(empty)` (%), fix+epi (%), bytes/µs per SM | cons_wait_full 5–20 %, prod_wait_empty 5–20 %, fix+epi ≤ 8 % | **cons_wait_full ≥ 25 %** → producer/DRAM side binds; **prod_wait_empty ≥ 25 %** → consumers bind |
| WS + NOCOMPUTE | consumers fold (sink) instead of dequant+mma; producer real | **≥ 3.5 TB/s** (the WS stream ceiling incl. mbarrier/bulk overhead; WSTREAM 4.0) | < 3.2 → the bulk-copy producer cannot stream (small copies / TMA issue rate): producer = 4 cp.async warps instead (½ day), design unchanged |
| WS + CONSONLY | producer issues every stage from one fixed 36.9-KB region (L2-hot), consumers real | the **consumer ceiling**: ≥ 4.5 TB/s-equivalent | < 3.5 → consumer-bound: step 2 (wgmma) or unpack trim before anything else |
| KS 2/4/8 (runtime) at bs ≤ 8 and 256 | item count vs fixup | KS 8 wins at bs 1/8 by 5–10 %; KS 4 at ≥ 32 | per-bucket KS in MOE7_CFG; buckets with KS ≠ v5 gate on rel-L2 |
Decision table (fires the director's crow): (i) best WS ≤ 175 µs on GEMM1 bs256 and bitwise → promote; (ii) 175–205 and cons_wait_full ≥ 25 % and NOCOMPUTE ≥ 3.5 → the
producer side: D2S5 / 4-warp cp.async producer, one more check; (iii) 175–205 and prod_wait_empty ≥ 25 % → consumer side: CONSONLY says how far; next build = wgmma consumers
(step 2) — mma.sync consumers are then "confirmed but not enough"; (iv) NOCOMPUTE < 3.2 → fix the producer first, no verdict on the design; (v) any bucket `DIFF` → bug, that
bucket stays v5 (the gate does this), fix before promote; **(vi) kill**: best WS > 205 µs (< +15 % vs v5's 234) while NOCOMPUTE ≥ 3.5 and CONSONLY ≥ 3.5 → the sync protocol
itself costs the time (unlikely) → D5-A. Noise: paired ratios vs v5 per seed, ±1.5 % paired, ±5 % across seeds [08 D2]; a twin above 4.5 TB/s is invalid (P6).

---
## D4. Build plan for a Sonnet builder (≤ 1 builder-day, est. 10.5 h)
File: `sonnet_v7.py` = [01] verbatim + a `MOE7` block (as [02] added MOE6); `measure_ws_v1.py` = [02]'s `measure_ring_v1` pattern with the D3 variants.
Reuse from [01]: every helper (ldmatrix_x4, unpack_word, mul_bf16x2, mma16816 + sm_75 `mma_emul`, f2bf, pdl_*), the pack/LUT, both route kernels (+ one int: zero `ctr[2]`),
`m5_batch`'s inner body, the host template (`m5_launch` → `m7_launch` with grid = SM count, dynamic smem, `cudaFuncAttributeMaxDynamicSharedMemorySize` 227 KB, carveout),
`_moe5_selfcheck`/bucket gate. From [02]: `m6_finish`, the item-walk/continuity idea, the DBG-twin + instantiation-list pattern, `cp_async16` (for the 4-warp producer fallback).
| step | what | hours | verify on the RTX 2060 (sm_75, NVRTC) |
|---|---|---|---|
| 1 | smem map (ring slots: words 24 × D × 512, scales 24 × D × 64, act 32 × D × 144 B; lut; mbarriers; desc) + helpers `mbar_init/arrive_expect_tx/arrive/try_wait`, `bulk_g2s(dst, src, bytes, mbar)` under `#if __CUDA_ARCH__ >= 900`; portable fallback: `bulk_g2s` = lane loop of 16-B ld.global/st.shared; `full/empty` = smem counters (`volatile` + `__threadfence_block`), same call sites | 2 | NVRTC compiles for sm_75 and sm_90 (`-arch=sm_90`, no 90a); the fallback deadlocks nowhere (DBG spin counter with a bound) |
| 2 | producer loop: item split (z, ks, ng) from `ctr`, descriptor write, stage issue (words/scales/act rows), PDL order for GEMM2 (weights before `griddepcontrol.wait`), sentinel | 2 | DBG coverage bitmap: every (z, ks, ng) issued exactly once per call |
| 3 | consumer loop: slot lds (16 B at lane × 16; 8 B at gid × 8), `lane_addr` relative to the slot's act area (pitch 144 B), `m5_batch` body, arrive(empty), `m6_finish` with `bar.sync 1,384`, NT dispatch per item | 2 | **bitwise vs v5** on M ∈ {1,2,4,8,12,16,32,33,128,256} × 3 seeds (the existing self-check), incl. `local_start` ≠ 0 rejection as today |
| 4 | host: `moe7_gemm` dispatch, scratch sizing (`part` by (z, ng, ks, warp, r, nt); `cnt` by (z, ng)), `MOE7_CFG` per bucket (KS, D), prepare() gate + v5 fallback, build-time print | 1 | prepare() < 200 s locally without SGLang (torch path) |
| 5 | `-Xptxas -v` on the dev box for sm_90: consumer path ≤ 152 regs, **0 spills**, smem static ≤ 1 KB | 0.5 | ratchet R3 |
| 6 | `measure_ws_v1.py`: variants of D3, S lines (cons_wait_full, prod_wait_empty via `%globaltimer` stamps per warp role), B lines, KS sweep, seeds 11–13 | 2 | parse offline on a stored log; ≤ 12 lines per shape (16 KB tail) |
| 7 | red-team file + predicted-µs header (below) + handover | 1 | `okc check` once; promote only on D3 (i) |
Fallback path on sm_75 is bitwise-testable (as v6's was [02]); timing there is meaningless. Known sm_75 gaps: no mbarrier (sm_80+), no `cp.async.bulk` (sm_90+): both are behind the `#if`.
Predicted per shape (est.; v5 = [03a]/[03b] seed-11 medians; "other" = route + epilogue from [05]):
| shape | v5 g1 / g2 / other | WS g1 / g2 | shape total old → new | ratio |
|---|---|---|---|---|
| bs 1 | 29.0 / 19.8 / 8 | 20 / 13 | 57 → 41 | 1.39 |
| bs 8 | 72.3 / 41.5 / 5 | 58 / 31 | 119 → 94 | 1.26 |
| bs 16 | 84.8 / 47.4 / 7.5 | 69 / 36 | 140 → 113 | 1.24 |
| bs 32 | 103.0 / 56.1 / 7 | 79 / 42 | 166 → 128 | 1.30 |
| bs 128 | 183.7 / 95.2 / 11 | 142 / 73 | 290 → 226 | 1.28 |
| bs 256 | 233.8 / 123.7 / 16 | 160 / 82 | 374 → 258 | 1.45 |
Model: bytes / 3.3 TB/s (bs ≥ 128) or 3.5 TB/s (bs ≤ 32) + 4–5 µs ramp. Geomean 1.32 ⇒ **moe 1.486 → ≈ 1.9** (1.75–2.05; the judge's medians, not seed 11, decide).
Header line for the check: `# PREDICTION: GEMM1 bs256 234 -> 160 (kill > 205), GEMM2 124 -> 82; moe 1.486 -> 1.9; bitwise v5 on all buckets`.

---
## D5. If not warp specialisation
- **D5-A "WS-lite" (same ring, no producer warp, ½ the novelty):** 12 warps, 1 CTA/SM, persistent item queue, act slices in the ring — but each warp's lane 0 issues its own
  groups' bulk copies S−1 stages ahead on per-warp mbarriers (2 instructions per stage; the TMA engine does the work, no `wait_group`, no per-lane cp.async chain). Keeps the
  issue→compute coupling (continuity logic in every warp, as v6) but the chain cost per tile is ≈ v5's 250 clk, not the ring's 410, and the in-flight bytes are 97 KB per SM.
  Expected 0.9× of D2's result (est.); choose it if the producer/consumer protocol fails on the judge twice (two checks) — the consumer code is identical, so the switch is ½ day.
- **D5-B wgmma consumers** (D2 §2.6 step 2): the right end state if D3 says consumer-bound; 1–2 builder-days, judge-only debugging, needs `9.0a` codegen through `load_inline`
  (set `TORCH_CUDA_ARCH_LIST=9.0a` before the build; verify `cuobjdump` shows sm_90a — D6 A10).
- **Not recommended now:** CUTLASS 3.x `MainloopSm90TmaGmmaWarpSpecializedMixedInput` / Machete — Hopper mixed-input exists (int4/fp8 × bf16, scales per K-block), but E2M1 values
  with E4M3 group-16 scales, chunked variable-M experts and our bitwise contract are not covered [11 §A]; a port is 4–5 builder-days [08 D1 §4 Alt C] with no sm_75 path.
  Marlin/MoE-Marlin is the baseline's own kernel family (2.5 TB/s on this layout at best [12 §2]); a Triton/Gluon warp-specialised port is the same design in a DSL the builder
  does not know and cannot test locally; "different chunking" alone keeps the duty-cycle losses (§1.3 KC/NB rows).

---
## D6. Assumptions `[An]` with the cheapest test, and open questions
| id | assumption | if false | cheapest test |
|---|---|---|---|
| A1 | the loaded round trip is ≈ 1.5 µs and ≥ 45 KB per SM in flight suffices at 4 TB/s (est. from NOCOMPUTE vs full, §1.2) | S 8 too shallow → stream < 3 TB/s with cons_wait_full high | D3 STAMPS + D2S5 twin; if both wait on full: S 12 (D1: 221 KB smem, still 1 CTA/SM) |
| A2 | consumer chain ≈ 250–300 clk per tile with 3 warps per scheduler ⇒ ≥ 3.6 TB/s-equivalent | consumer-bound at ≈ 2.8–3.0 TB/s | D3 CONSONLY twin (direct ceiling) |
| A3 | 512-B / 64-B / 128-B 1-D bulk copies stream at DRAM speed (TMA issue rate ≥ 48 + 32 copies per 920 clk per SM) | NOCOMPUTE-WS < 3.2 | D3 NOCOMPUTE-WS; fallback = D 2 (1-KB copies) or 4 cp.async producer warps |
| A4 | consumer path ≤ 152 regs, 0 spills at MR 2 / NT 4 without the 48 prefetch regs | spills → ratchet fail; then MR 1 for the NT-4 class (ldmatrix traffic ×2, still < 128 B/clk) | `-Xptxas -v` sm_90 on the dev box (step 5) |
| A5 | smem 148/186 KB → exactly 1 CTA/SM and the 2 descriptors + 16 mbarriers fit static smem | 0 CTAs resident (launch fails) or 2/SM at small act classes (harmless) | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` printed in prepare(); STAMPS occ max |
| A6 | all bulk-copy sources are 16-B aligned: `act + slot×K×2 + (ks×Kc + q×64×D)×2`, `Wq` 512-B tiles, `Sq` 64-B tiles | illegal-instruction fault | static asserts + a DBG alignment check in the producer (sm_75 path too) |
| A7 | mbarrier tx-count per phase ≤ 36.9 KB ≪ 2^20−1; arrival counts 1 / 12 | protocol hangs | PTX ISA 8.5 table [docs]; DBG bounded spins |
| A8 | one global atomic per item (2376 at bs 256) and 12 `empty` arrivals per stage cost < 1 % | measurable overhead | STAMPS prod/cons wait shares; NOCOMPUTE ceiling |
| A9 | the activation slices are L2 hits (x 2 MB; inter 16 MB written by GEMM1 just before) | L2 misses add 29 % DRAM bytes | NOSTAGE-style twin (act copies from a fixed 36 KB region) if NOCOMPUTE-WS < 3.5 |
| A10 | `load_inline` on the judge compiles `cp.async.bulk` + `mbarrier` for plain sm_90 (PTX 8.0+, CUDA 13) — no `9.0a` needed in step 1 | build error = one lost check | NVRTC `-arch=sm_90` on the dev box compiles the same source (step 1) |
| A11 | bitwise = v5 because the per-(group, ks) k order and the fixup order are unchanged | a bucket shows DIFF | prepare() self-check (reused); the sm_75 run shows it before the check |
| A12 | the dynamic item queue + z-major order keeps DRAM page locality ≥ WSTREAM's (which had 8 KB per stream in flight) | D1S8 (3.5 KB per group-stream in flight) < D2S5 by > 5 % | the D2S5 twin |
| A13 | item tail ≤ 1 item (≈ 7 µs) and the 8/16-group last items (4 idle consumer warps for 1/22 of items) cost ≤ 4 % | wave/tail loss | STAMPS: per-CTA item counts and end times |
| A14 | planning numbers 3.3–3.5 TB/s sustained (DRAM 4.0 × caps) | predictions off by the ratio; the kill line (205) still holds | D3 |
Open questions for the director: **Q1** the exact definition of `bw/cta`, `item`, `cta` in measure_altA_v1/measure_ring_v1 (my reading in the header; the stamp sites are clear,
the printer is not in the pack). **Q2** chunk size distribution on judge routing (how many chunks have mc > 16 → NT 4 share; est. 30 % of bytes at bs 256). **Q3** is `cuobjdump`
in the image (to confirm `LDGSTS`/`UBLKCP`-free consumer SASS and the bulk copies in the producer)? **Q4** does the judge's torch set `-gencode` for sm_90 only (A10) — print
`torch.cuda.get_arch_list()` in prepare(). **Q5** the budget: v5's 5 fallback instantiations + 8 of v7 in one prepare() (est. 200–250 s of the 300) — if the first build log says > 240 s,
drop D 2 and the DBG twins from the board file (the measurement file keeps them). **Q6** for the layer composition (decode_layer T ≥ 17): the same kernel with `local_start` from the
device table — not in this brief, flag only.

## Sources (public web)
PTX ISA (cp.async.bulk 1-D forms, mbarrier expect_tx, tx-count limit 2^20−1, sm_90 target notes): https://docs.nvidia.com/cuda/parallel-thread-execution/index.html and the 8.5 archive
https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html · setmaxnreg range 24–256 in multiples of 8, warp-specialised GEMM pipeline (producer TMA, consumer wgmma,
full/empty mbarriers): https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel · TMA + mbarrier transaction bytes: https://research.colfax-intl.com/tutorial-hopper-tma ·
mbarrier phase/tx semantics: https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_async_barriers/index.html · Hopper microbenchmarks (global latency 656 clk H800; mma.sync m16n8k16
latency 24 clk, ≈ 490 TFLOPS dense; wgmma peak only at N ≥ 64; SS/RS forms): https://arxiv.org/abs/2501.12084 and https://arxiv.org/abs/2402.13499 · FlashAttention-3 producer/consumer
warp specialisation on Hopper: https://arxiv.org/abs/2407.08608 · CUTLASS CuTe mbarrier arrive/expect_tx API: https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/arch/mbar.py ·
Hopper tuning guide (228 KB smem/SM, 64 warps): https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html · H200 4.8 TB/s, 132 SMs: https://www.nvidia.com/en-us/data-center/h200/
