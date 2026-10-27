# C39 result: WSTREAM-style tensor-core GEMV on bf16 (job e2a5099d5c5149de, 2026-10-26 17:53Z) — KILLED
All 20 cells (5 layers × m 1/8/16/32) correct but SLOWER than the board: x_board 0.17–0.73; WS streams 0.2–2.5 TB/s vs board 2.6–3.7.
Prediction (≈ 4 TB/s, qkv 10 µs) missed by ~2×: the pure-stream WSTREAM rate (4.0–4.5 TB/s in the MoE twin) does not survive mma +
x staging + split-K fixup in this form; small cells are dominated by ramp/fixup. Do not retry this kernel family without a new reading.
Useful fact: the board's Triton `_gemv_splitk` already streams the pooler at 3.6–3.7 TB/s while qkv_gate sits at 2.6–2.8 TB/s
(13.4–14.6 µs) and o at 2.9 → a qkv/o config gap of ~3 µs (est.) if qkv reached the pooler's rate. Check measure_cheap_v1 history first.
