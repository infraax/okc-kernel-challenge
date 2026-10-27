# C4 kernels/muse_attention_decode/<lane>_v1.py — fused attention decode in Triton (k=2: Sonnet "sonnet_v1", Jules "jules_v1")
Read first: `problems/muse_attention_decode/TASK.md`, `baseline.py` (SGLang: FusedQKNorm → store_cache → q→FP8 → FA3 →
rms_norm_gate), `reference.py`, `checks.py`, `AGENTS.md` (project root, hard rules).
Goal: one candidate file, first line `# okc: problem=muse_attention_decode`, that computes the step with fewer launches:
(1) one kernel: per-head RMSNorm of q,k (q × 5.473), interleaved RoPE on sliding layers, FP8 E4M3 cast + store of the new
k,v at `cache_loc` (apply k/v scales), q to FP8 or BF16; (2) split-KV flash-decoding over the FP8 paged cache (page 1 and
64), causal inside an 8-token verify block, window 2048 on sliding layers / whole context on full layers, FP32
accumulation, sized to fill 132 SMs with only 2 KV heads (split along the sequence) + a reduce kernel that also applies
per-head RMSNorm × sigmoid(gate) and writes BF16 [B, L, Hq·64]. Must attend every key it should (rules).
- Triton only (+ torch). No CUDA-graph-breaking ops in kernel_fn (no .item(), no value-dependent shapes: size the split
  grid from shapes/config only; use the provided metadata tensors on device).
- Tests: CPU via `TRITON_INTERPRET=1` with small cases, GPU (dev box only) via tools/local_check.py when available.
  Jules: write `tests/test_attention_jules_v1.py` that builds inputs with `problems/muse_attention_decode/reference.py`
  `generate_input(..., device="cpu")` at small ctx (≤ 4096) for sliding+full, decode+verify, page 1 and 64,
  kv_scale 1 and 1/64, and checks with the reference's `check_implementation`.
Done when: your test file passes (`TRITON_INTERPRET=1 pytest -q tests/test_attention_<lane>_v1.py`), the kernel file
contains no input-value-dependent host logic, and a 10-line header comment states the expected speedup and why.
Do not touch problems/**, tools/okc_run.py, other lanes' files. No scratch files.
