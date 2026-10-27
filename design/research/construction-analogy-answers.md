# An experienced builder's answers to the 12 NL/PL questions → okc thesis (director analysis, 2026-10-26)
Source: inbox/claude-agent/20261026-claude-questions-2.txt (Grok transcript, 18:32, Dutch; the [PL] lines are whisper noise, ignored).
Questions: design/research/construction-analogy-questions.md. Analysis checked against our code (sonnet_v5f, sonnet_v5 moe/attention) and judge data.

## Thesis (one line)
**The specialist's work is rarely the bottleneck; trips, handovers and waiting between trades are.** Make every trip carry more, let the
next trade prepare its tools before the room is free, and hand over room by room instead of house by house.

## Per question: answer → okc reading → status
| q | his answer (paraphrase) | okc reading | status / evidence |
|---|---|---|---|
| 1 helpers per specialist, wait or work elsewhere | vague (replace the plumber when the electrician arrives) | — | no usable signal |
| 2 all material at once or in batches; room too small | bring everything the room needs at once; when nearly done, already bring the next batch and clear leftovers so there is room | prefetch the next work item while finishing this one; free buffer space early | **tested** (MoE v7 continuity ring): the idea worked, but the hallway (ring protocol) became the cap ≈ v5 rate (C31) |
| 3 one master, many rooms: faster master or split his steps | master gives a blueprint and does the final certified step; helpers do the rest | producer/consumer split | **tested** (v7): helpers fine, handover point too slow. Same conclusion as q2 |
| 4 one room 10× the work | spread: two helpers per room continuously, more on the big room | load balance hot experts / split-K | already in v5 (chunks ≤ 32 + split-K fixup) |
| 5 80,000 parts at known addresses: one bus, many cyclists or bundles | the key number is how much one person carries per trip; too few trips per carrier is waste; it is a calculation | bytes per load × number of loaders (Little's law); bigger carry per trip | **confirmed today**: attention BN128 (twice the keys per trip) +13–16 % on verify; in decode_layer −8 % / −4 % on verify-full (board 1.423 → 1.433) |
| 6 inspection: helper with a simple ruler | straight pipes without bends: quick check; the master only checks the joints | fast path for full tiles, slow path only for tails | already (attention masks only the tail tile). No new lever |
| 7 chain of trades: start the next trade in a finished room | finish room by room so the next trade can start there; no need to wait for the whole house | per-tile / per-block handover between kernels instead of whole-kernel barriers | **open, big** (see Lever A) |
| 8 time lost switching trades | tell the next trade "we're almost done, prepare your tools" | start the next kernel early and let it do everything that does NOT depend on the previous result (load its weights) before it waits | **open, big** — our PDL test waited FIRST (gdc_wait as first statement) and was noise; "tools first, then wait" was never tried (see Lever A) |
| 9 tiny jobs: combine or one person in series | always combine a tiny job with what follows | fuse small kernels into the next one | **open, medium** (Lever B) |
| 10 how to learn standard sizes you don't know | experience; ask colleagues | read public kernels (CUTLASS, FA3, Marlin) | in our method already |
| 11 how many beams before you stop | stop when you have what you need | stop rule | matches successive halving; no change |
| 12 two failures, try a third time? | the third time always differently: change the process | two-strike rule | matches; v7 parked after the 3rd reading (a diagnostic, not a retry) |
His own top picks: material vs room (residency) and small bundles + pick-up addresses (carry size per trip).

## What we can use (ranked)
**Lever A — "tools ready before the room is free" (decode_layer, small T).** At T = 1 the layer is ~20 kernels in a chain; the judge has
decode 1 at 125 µs vs a 40 µs floor (3.1×, the largest ratio we have; decode-1 shapes carry weight 46 + 16). Every GEMV's WEIGHTS do not
depend on the previous kernel; only its activations do. Today PDL is off (S3_PDL = False) and, when it was on, each kernel called
gdc_wait as its first statement, so nothing overlapped (the measured "noise"). New variant: launch the next kernel early, let it stream
its weight tile into smem/registers (or prefetch to L2) BEFORE gdc_wait, then wait for the activations. That is q8 + q7 literally.
Test: one measurement check with the variant per kernel group (ORDER: wait-first vs tools-first), on T = 1 / 16 / 32 shapes.
Kill if decode 1 gains < 3 µs. Not a failed class: the failed one is "PDL with wait first".
**Lever B — "combine tiny jobs with what follows" (decode_layer, T ≤ 16).** Fuse the small glue kernels (norms, gates, residual updates,
epilogues) into the GEMV that consumes them (queue items H7, H8/H9). Each removed kernel saves its ramp (≈ 2–4 µs at T = 1).
**Lever C — "carry more per trip" everywhere a kernel streams small pieces.** Already won on attention; next candidates are the GEMVs with
small BK at T ≤ 8 (queue H18/H22 style), judged by the same twin method (one check, several tile sizes).
Already tested and closed by our own measurements: q2/q3 as a MoE ring (C30–C31).

## Follow-up for the builder (optional)
- q7/q8 detail: when the next trade starts early in a finished room, what does he do so the two crews don't block the corridor?
  (okc: two kernels sharing memory bandwidth — our "router/pooler squeeze" earlier).

## Update 2026-10-26 evening: own transcript + Dex's corridor answer
Own transcription (faster-whisper large-v3, Dutch forced; files in (transcripts not published)/: VAD pass + no-VAD pass for the quiet answers; the no-VAD
pass loops at 14:19–14:28 and 16:00–16:12 — ignored). Grok's version was right in substance; corrections:
- q6 inspection, full answer (10:30–11:26): "straight, simple pieces — no elbows, no T-pieces, nothing — you check quickly, you assume no
  broken ones; the joints, where they come together, are what you check." → confirms the fast path / tail-only masking reading.
- q8 (14:13–14:30, partly garbled): "how does it work now? … in practice you have to go and look how it can be done" → no rule of thumb;
  measure it. q10 (15:58): "collega's, people in your trade" (Grok's "[PL] Konkolegas" was this). q11: "as many as you need for the jobs".
- q1 has no recorded answer ("ja oké, duidelijk").
**Corridor (Dex, chat): "the crew that needs to exit always goes first; as soon as it is out, the other enters."**
→ the kernel on the critical path has priority; the next kernel may prepare, but must not take bandwidth from the one finishing.
Consequences for Lever A: (1) the pre-wait work of the early kernel stays LIGHT — prefetch its weights into L2 (or the first 1–2 stages
only), not a full smem stream; (2) CUDA already lets the dependent kernel's blocks land only on SMs the previous kernel has freed
(PDL scheduling = "enter as the other exits"); (3) on our side stream (router ‖ pooler at T ≤ 8, the measured "two crews in one room"),
give the critical-path stream the higher stream priority.

## Tests (one judge measurement check each, decode_layer, shapes T = 1 / 16 / 32)
| id | test | twins | kill / confirm |
|---|---|---|---|
| L-A1 | tools-before-wait: PDL on, each GEMV prefetches its weight tile to L2 before gdc_wait | WAITFIRST (old, noise), TOOLS_L2, TOOLS_SMEM (first stage into smem), PDL off | kill if decode 1 gains < 3 µs vs PDL off |
| L-A2 | corridor priority: critical-path stream high priority, side stream (router) low | PRIO_MAIN, PRIO_SIDE, none | kill if < 1 µs on T ≤ 8 |
| L-B | combine tiny jobs: fuse norm/gate/residual glue into the consuming GEMV (H7/H8/H9) | per fusion on/off | kill a fusion if Δ < 1.5 µs |
