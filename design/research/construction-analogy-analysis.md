# Construction analogy (discussed with an experienced builder) → okc — director's analysis, 2026-10-26
Source: (private notes) (Grok's 12-point mapping + 3 raw transcripts). Grok's mapping
is sound; this note adds what the analogy says about our *measured* bottlenecks (not general MoE practice).

## The three ideas that hit our real problems
1. **"Let the misassigned plumber lay the pipes; the electrician only does the specialist part" → warp specialisation (MoE GEMMs).**
   Measured: every warp of our expert kernel does load → dequant → mma → fixup in series, and the per-CTA stream rate is stuck at
   ≈ 10 GB/s whatever we change (C20 a78c1b9c: occupancy 3/SM +1–3 %; C21b 698aa839: 2.7× deeper prefetch → 9.6→10.7 GB/s/CTA but
   residency 3→2 → net −22 %). The analogy's structure: generalists do the preparation continuously (producer warps: load + unpack
   NVFP4 into a ready-to-use tile), specialists do only the certified work (consumer warps: mma). Hopper adds the "tool you didn't
   know existed": `setmaxnreg` moves registers from producer to consumer warpgroups, so the crew fits without losing residency —
   exactly what killed the ring. This is the best candidate for the next Fable THEORY brief (Dex: Fable theory only).
2. **Same idea for attention verify-full (lane reopened by Dex 2026-10-26): the specialist (MUFU, 16 exp2/clk/SM) is the bottleneck,
   128 exp2 + 128 FP8→FP16 converts per key (fable-08 D5/App. E: 355 µs for 607 MB = 1.7 TB/s). Give part of the "simple" work to the
   general crew: compute a share of the exp2 with an FP32 polynomial on the FMA units (FA4-style split). Legal: FP32 math, error ≈ 1 ulp,
   far inside the 3–15 % attention bound — and `ex2.approx` is itself approximate ("the expensive wood is just as crooked").
   First test (one measurement check): NOEXP / NOCVT / GATHER-only twins (fable-09 D4 §3) + an exp-split variant (25/50 % on FMA).
3. **"Rooms too small for all the material" → our ring failure in one sentence.** A bigger material pile per crew (101 KB smem ring)
   meant fewer crews fit on the site (3 → 2 CTAs/SM). Rule for every next design: show the residency budget (regs, smem per CTA,
   CTAs/SM) before building; ptx_audit / -Xptxas -v first.

## Confirmations (already in our method)
- "The label at the door may not replace the work list" = exact router decides; cached/predicted routing stays illegal.
- "Too quiet / two men in one room" = okcdbg D4 (idle gaps) and D1–D3 (duplicate work / two kernels competing — the pooler/router
  squeeze was literally two crews in one room).
- "Three deep questions, then a second round" = successive halving; we already batch variants in one measurement check. Refinement:
  two stages inside one prepare() (1 seed × 3 shapes screen → top-k get all shapes × 3 seeds) → more variants per judge check.
- "Trust after the down-payment; one bad step → back to full checks" = promote.py CI rule + two-strike kill. Never in a kernel
  (state across calls is illegal).
- "The 120 vs 140 sills you didn't know existed" = Hopper features our kernels don't use yet: wgmma (MoE uses mma.sync only),
  TMA bulk copies, setmaxnreg, clusters/multicast. Ask the "Timmerman": CUTLASS/Machete code in the public domain (ideas only).
- "Farmers' market of finished stalls; know when to stop walking" = test library kernels present in the judge image first
  (cuBLASLt won at m ≥ 64; FA3 is the attention baseline); stop rule after 3 variants < 1 %.

## Follow-up questions for the builder
1. When the plumber prepares for the electrician: does the electrician work in parallel in another room, or wait until the prep is
   done? How many helpers per specialist before they get in each other's way?
2. On a big job, do you bring all material for the day into the room at once, or have it delivered in batches? What do you do when the
   room is too small for the pile?
3. When there is only one certified specialist and many rooms: do you get a faster specialist, or split off every simple step of his
   work to the others? Which steps of a "specialist" task can a helper safely pre-do?
4. How do you find out which standard sizes and tools exist that you don't know yet (the 140 sills)? Whom do you ask first?
5. How many beams do you inspect before you stop — a fixed number, or until you have seen enough "straight on the side that matters"?
