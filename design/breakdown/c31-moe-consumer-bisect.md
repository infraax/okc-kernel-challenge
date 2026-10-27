# C31 result: v7 consumer-chain bisect (job d31e4c24cc07420b, 2026-10-26 ~12:58Z; direct okc call after an okc_run transient error, logged by hand)
Twins on top of CONSONLY2 (D2S5, producer copies from one L2-hot region). GEMM1 µs, % = time saved vs consonly2:
| bs | consonly2 | nounpack | nomma | noldm | nofix | ldsonly (ring + lds, no math) | smem (lds + ldmatrix) |
|---|---|---|---|---|---|---|---|
| 256 | 294 | 238 (19 %) | 274 (7 %) | 273 (7 %) | 245 (17 %) | **214 (27 %)** | 221 (25 %) |
| 128 | 240 | 19 % | 6 % | 6 % | 17 % | 25 % | 23 % |
| 16 | 111 | 85 (24 %) | 4 % | 5 % | 12 % | **78 (30 %)** | 82 |
| 1 | 41 | 13 % | −1 % | 5 % | 5 % | 34 (17 %) | 35 |
Verdict (all bs): **ring-bound**. With no unpack and no mma at all, the ring alone (bulk copy → mbarrier → lds → release) takes 214 µs
at bs 256 = 73 % of the consumer time and ≈ the producer-only nocomp2 (201–216, C30d). That floor (≈ 2.4 TB/s) is about v5's own rate,
so no consumer-side fix (wgmma: nomma saves only 7 %; dequant: 19 %; fixup overlap: 17 %) can lift v7 clearly above v5.
Decision: **v7 / warp-specialised ring parked** (3 measurements: C30b copy-issue, C30d consumer cap, C31 ring floor). Do not re-propose
without a new reading that the 1-CTA/SM ring protocol can stream ≥ 3.3 TB/s (e.g. a ring-only microbenchmark at > 3.3 TB/s).
Side fact: muse_prefill_layer does not count for the overall board (fable-05 pack correction (2)); its board exists (21 entrants, leader 2.0x).
