# C37 result: decode_layer T = 1 quick wins from Argon C36 (job f31c983903fd4930, 2026-10-26 17:34Z)
Layer µs (3 seeds, reps 9): base dec1s 105.2 / dec1f 113.2 / dec16s 195.1; noise (base2 − base) ≈ 0.7 µs.
| variant | dec1s | dec1f | dec16s | verdict |
|---|---|---|---|---|
| nz8 (A1, dormant: E = 2–3 local experts at T = 1) | −1.4 | −1.1 | −0.4 | no |
| trig_late (A1b) | −0.5 | +0.2 | −1.3 | no — the early trigger is fine |
| rows2 (A1c, 8-row act staging) | **−6.0** | **−5.8** | −0.4 | worse: more GEMM2 CTAs/SM compete |
| route_side (A2b) | +1.1 | +1.2 | +0.6 | small; rider only |
| combo | +1.0 | +1.6 | +0.9 | = route_side |
D: no winner ≥ 2 µs. A2 (pooler/router smem) was refuted in code before the run (router_fix 24 KB; C22a' had measured ps1 at T = 1).
Conclusion with C33: at T = 1 the loss is neither in the handovers (+2 µs light prefetch) nor in the small-M knobs; what is left is
kernel count / glue (fusions, C38) and the cuBLAS projections (qkv 12.5 + splitKreduce 1.9, o 6.7 µs vs ~8 and ~3.5 µs floors; Argon #4).
