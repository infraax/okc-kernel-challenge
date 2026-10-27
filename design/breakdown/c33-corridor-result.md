# C33 result: corridor measurement on decode_layer (job fcbc85a47127453a, 2026-10-26 17:29Z)
Layer µs (3 seeds, cold L2, graph replay), vs PDL_OFF (= board v5f):
| shape | PDL_OFF | WAITFIRST | TOOLS_LIGHT | TOOLS_FULL | PRIO | LIGHT+PRIO |
|---|---|---|---|---|---|---|
| d1 sliding | 116.2 | 114.7 | **113.8 (+2.4)** | 120.0 (−3.8) | 116.9 | 114.6 |
| d1 full | 123.4 | 123.3 | 122.6 (+0.8) | 127.6 (−4.2) | 123.9 | 122.2 |
| v1 full | 172.3 | 172.0 | 170.8 (+1.5) | 178.6 (−6.3) | 172.4 | 171.3 |
| d16 s / f | 180.7 / 262.4 | 179.5 / 262.8 | 180.1 / 262.3 | 185.5 / 268.4 | 180.5 / 263.8 | 179.4 / 261.6 |
| d32 s / f | 262.2 / 424.9 | 260.7 / 424.4 | 261.7 / 424.3 | 265.4 / 428.4 | 261.5 / 425.2 | 261.3 / 425.6 |
D (T = 1, weighted 46:16): WAITFIRST +1.1, TOOLS_LIGHT +2.0, TOOLS_FULL −3.9, PRIO −0.7 µs.
Readings:
1. **The builder's corridor rule confirmed**: prefetching the whole weight slice early (TOOLS_FULL) is 3.5–6 µs SLOWER — it takes bandwidth from
   the kernel that is finishing. Light prefetch (≤ 64 KB/CTA) helps +2 µs at T = 1, ~0 at T ≥ 16. Below the 3 µs build line → no stand-alone
   build; carry TOOLS_LIGHT as a free rider in the next decode_layer version (switch, bitwise math).
2. PRIO survives graph capture (14/16 nodes high) but gives nothing: the side stream is not starved by priority.
3. PDL edges 6 → 13 of 16 with TOOLS_LIGHT; library kernels (cuBLAS, SGLang) still break the chain.
4. H stamps: _router_fix_kernel (side stream) starts 34–44 µs after its stream's previous kernel and runs 13–21 µs when overlapped with the
   pooler (7 µs alone in the board profile): the router ‖ pooler pair is the T = 1 critical section → Argon's idea 2 (C37 POOL_PS1_T1) and
   Grok's pooler-prologue fusion target exactly this.
Decision (itachi-corridor §7): no row ≥ 3 µs → ALT-1 fusion first (C34 map) + C37 quick wins; TOOLS_LIGHT kept as a rider.
