# C5 kernels/muse_bf16_proj/jules_v1.py — skinny-M BF16 GEMV/GEMM in Triton (lane: Jules)
Read first: `problems/muse_bf16_proj/TASK.md`, `baseline.py`, `reference.py`, `checks.py`, root `AGENTS.md`;
`experiments/profiles/muse_bf16_proj.728c9344dc3b4559.txt` (the judge's per-shape timings of SGLang: e.g. router m=1
10.5 µs vs floor 1.75 µs; qkv_gate m=1 14.45 vs 7.87; pooler m=1 21.6 vs 14.0).
Goal: a candidate (first line `# okc: problem=muse_bf16_proj`) that streams the weights at close to HBM bandwidth for
m ≤ 32 (split-K / many CTAs so 132 SMs are busy, FP32 accumulation, BF16 out) and keeps cuBLAS (`torch.mm`) for
m ≥ 128 unless clearly better. Router: FP32 weights may be split once in prepare() into BF16 hi + lo with FP32
accumulation (allowed at any m); must meet the router's 0.01% bound and the 0.1% "single power-of-two entry" rows.
Precompute nothing from inputs. Branch only on shapes/config. Triton + torch only.
Tests: `tests/test_bf16_jules_v1.py` builds inputs/weights with the reference on CPU at the TASK.md test shapes (m 1, 5,
7, 8, 9, 33, 130 …; k, n as given) and checks with the reference's check function, under `TRITON_INTERPRET=1`.
Done when that test passes and the header comment states the predicted per-shape µs (from bytes / 4.8 TB/s × your
expected efficiency) and the resulting speedup (use the TASK.md weights and geo-mean over loads).
