# C9 batch measurement checks (lane: director) — Dex's question "test many at once instead of one run"
A check run is off the board. On problems with weights, prepare() runs once, untimed, ≤ 300 s, on the judge's H200.
A measurement-only candidate keeps SGLang's baseline as kernel_fn (scores ~1.00x) and uses prepare() to time a grid of
kernel configurations per scored shape on synthetic activations (CUDA-graph replay after an L2 flush, like the judge),
printing `MEAS ...` lines into the 16 KB log tail. One check = dozens of configs instead of one.
- v1: kernels/muse_bf16_proj/measure_v1.py (Triton split-K GEMV, 12 configs × 30 shapes, budget 240 s).
- Limits: log tail 16 KB (print compactly); attention_decode has no prepare() (needs another route: ask Fable);
  measurement checks are never submitted (they would just score ~1.00x anyway).
- Integrity: synthetic inputs only; nothing measured feeds kernel_fn of this file; the chosen configs go into a
  separate real candidate whose config choice depends on shapes only.
